from typing import Any, List, Optional, Set, Tuple
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


class SecurityError(Exception):
    """Raised when generated SQL violates safety or parameterization rules."""


BLOCKED_AST_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Grant,
    exp.Revoke,
    exp.Command,
    exp.Pragma,
)


def _extract_int_limit(limit_expr: Optional[exp.Limit]) -> Optional[int]:
    """Safely extracts an integer LIMIT value if present as a literal."""
    if not limit_expr or not isinstance(limit_expr.expression, exp.Literal):
        return None
    if not limit_expr.expression.is_int:
        return None
    try:
        return int(limit_expr.expression.this)
    except (ValueError, TypeError):
        return None


def _apply_limit_cap(expression: exp.Expression, max_row_limit: int) -> exp.Expression:
    """Injects LIMIT if missing, or caps an existing LIMIT to max_row_limit."""
    if not isinstance(expression, (exp.Select, exp.SetOperation)):
        return expression

    current_limit = expression.args.get("limit")
    if current_limit is None:
        return expression.limit(max_row_limit)

    parsed_limit = _extract_int_limit(current_limit)
    if parsed_limit is None or parsed_limit > max_row_limit:
        expression.set("limit", exp.Limit(expression=exp.Literal.number(max_row_limit)))

    return expression


def enforce_sql_guardrails(
    sql_query: str,
    max_row_limit: int = 100,
    params: Optional[List[Any] | Tuple[Any, ...]] = None,
    allowed_tables: Optional[Set[str]] = None,
    dialect: str = "sqlite",
) -> Tuple[str, Tuple[Any, ...]]:
    """
    Validates a read-only SQL query via AST inspection, checks parameter placeholders,
    enforces table allowlists (if provided), and injects/caps LIMIT.

    Returns:
        (sanitized_sql, normalized_params_tuple)
    """
    if max_row_limit < 1:
        raise ValueError("max_row_limit must be at least 1.")

    if not sql_query or not sql_query.strip():
        raise SecurityError("Security Violation: Empty SQL query is not allowed.")

    if params is None:
        params_list: List[Any] = []
    elif isinstance(params, (list, tuple)):
        params_list = list(params)
    else:
        raise SecurityError("Security Violation: params must be a list or tuple.")

    # 1. Parse SQL into AST
    try:
        raw_expressions = sqlglot.parse(sql_query, read=dialect)
    except ParseError as exc:
        raise SecurityError(
            f"SQL parsing failed: Invalid syntax or malformed query. Details: {exc}"
        ) from exc

    expressions = [expr for expr in raw_expressions if expr is not None]
    if len(expressions) != 1:
        raise SecurityError(
            f"Security Violation: Expected exactly 1 SQL statement, got {len(expressions)}."
        )

    expression = expressions[0]

    # 2. Top-level statement must be SELECT or SetOperation (e.g., UNION / INTERSECT)
    if not isinstance(expression, (exp.Select, exp.SetOperation)):
        raise SecurityError(
            f"Security Violation: Query type '{type(expression).__name__}' is blocked. "
            "Only read-only SELECT queries are permitted."
        )

    # 3. Walk full AST to block write/DDL/command nodes and count '?' placeholders
    placeholder_count = 0
    for node in expression.walk():
        if isinstance(node, BLOCKED_AST_NODES):
            raise SecurityError(
                f"Security Violation: Disallowed SQL construct '{type(node).__name__}' detected."
            )
        if isinstance(node, exp.Placeholder):
            placeholder_count += 1

    # 4. Verify '?' placeholder count matches supplied parameters
    if placeholder_count != len(params_list):
        raise SecurityError(
            f"Parameter mismatch: query has {placeholder_count} '?' placeholder(s), "
            f"but {len(params_list)} parameter(s) were provided."
        )

    # 5. Optional physical table allowlist check (excluding CTE aliases)
    if allowed_tables is not None:
        allowed_lower = {t.lower() for t in allowed_tables}
        cte_names = {
            cte.alias_or_name.lower()
            for cte in expression.find_all(exp.CTE)
            if cte.alias_or_name
        }
        referenced_tables = {
            tbl.name.lower()
            for tbl in expression.find_all(exp.Table)
            if tbl.name
        }
        physical_tables = referenced_tables - cte_names
        unauthorized = physical_tables - allowed_lower
        if unauthorized:
            raise SecurityError(
                f"Security Violation: Access to unauthorized table(s): {', '.join(sorted(unauthorized))}."
            )

    # 6. Inject or cap LIMIT safely
    expression = _apply_limit_cap(expression, max_row_limit=max_row_limit)

    sanitized_sql = expression.sql(dialect=dialect)
    return sanitized_sql, tuple(params_list)