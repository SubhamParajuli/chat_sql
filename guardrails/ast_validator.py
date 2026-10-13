import sqlglot
from sqlglot import exp


class SecurityError(Exception):
    """Raised when generated SQL violates safety constraints."""
    pass

def enforce_sql_guardrails(sql_query:str,max_row_limit:int=100)->str:
    """
    Enforces read-only operations (Blocks INSERT,UPDATE,DELETE, DROP ,etc)
    Automatically injects pagination limits to prevent memory exhaustion on large datasets.
    """
    try:
        parsed_expressions =sqlglot.parse(sql_query,read="sqlite")

    except Exception as e:
        raise SecurityError(f"Sql parsing failed :Invalid syntax or malformed query. Details:{e}")

    sanitized_queries=[]

    for expression in parsed_expressions:
        #Check 1: Strict Read-Only enforcement

        if not isinstance(expression,exp.Select):
            raise SecurityError(
                f"Security Violation: Query type '{type(expression).__name__}' is blocked. "
                "Only read-only SELECT queries are permitted."
            )

        #Check 2: Mandatory Pagination/Limit Injection

        current_limit=expression.args.get("limit")
        if not current_limit:
            expression=expression.limit(max_row_limit)

        else:
            limit_value= int(current_limit.expression.this)
            if limit_value> max_row_limit:
                expression.args["limit"]=exp.Limit(expression=exp.Literal.number(max_row_limit))

        sanitized_queries.append(expression.sql(dialect="sqlite"))

    return ";".join(sanitized_queries)   