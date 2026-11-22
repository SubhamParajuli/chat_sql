import json
import sqlite3
import asyncio
from pathlib import Path

from langchain_core.messages import HumanMessage


SQL_ERROR_MARKERS = ["SQL Execution Error:", "Security Violation:", "Parameter mismatch:"]
API_ERROR_MARKERS = ["429", "rate_limit", "rate limit", "tokens per minute"]


def run_readonly_query(db_path, sql, params=None):
    """Run a reference query and return both its rows and column names."""
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        cursor = conn.execute(sql, params or [])
        rows = cursor.fetchall()
        columns = [item[0] for item in cursor.description] if cursor.description else []
        return rows, columns


def normalize_value(value):
    if isinstance(value, float):
        return round(value, 4)
    return value


def normalize_rows(rows, preserve_order=False):
    normalized = [tuple(normalize_value(value) for value in row) for row in rows]
    if not preserve_order:
        normalized.sort(key=str)
    return normalized


def project_actual_columns(actual_rows, actual_columns, expected_columns):
    """Ignore extra output columns only when expected names match uniquely."""
    if not expected_columns or not actual_columns:
        return actual_rows, False

    actual_names = [name.lower() for name in actual_columns]
    expected_names = [name.lower() for name in expected_columns]
    indexes = []

    for name in expected_names:
        if actual_names.count(name) != 1:
            return actual_rows, False
        indexes.append(actual_names.index(name))

    if len(actual_rows) == 0:
        return actual_rows, len(actual_columns) != len(expected_columns)

    projected = [tuple(row[index] for index in indexes) for row in actual_rows]
    return projected, len(actual_columns) != len(expected_columns)


def get_failure_category(message):
    text = (message or "").lower()
    if any(marker in text for marker in API_ERROR_MARKERS):
        return "api_rate_limit"
    if "security violation" in text:
        return "guardrail_rejection"
    if "sql execution error" in text or "parameter mismatch" in text:
        return "sql_execution_error"
    if not text:
        return "missing_execution_result"
    return "other_workflow_error"


async def evaluate_case(case, graph, retriever, db_path):
    state = {
        "messages": [HumanMessage(content=case["question"])],
        "db_path": db_path,
        "sql_query": "",
        "sql_params": [],
        "execution_result": "",
        "query_columns": [],
        "query_rows": [],
        "error_count": 0,
        "retrieved_tables": [],
        "attempts": 0,
    }

    try:
        result = await graph.ainvoke(
            state,
            config={"configurable": {"schema_retriever": retriever, "eval_mode": True}},
        )
    except Exception as error:
        message = str(error)
        category = get_failure_category(message)
        return {
            "id": case["id"],
            "level": case.get("level", "unknown"),
            "question": case["question"],
            "passed": False,
            "result_match": False,
            "execution_success": False,
            "failure_category": category,
            "failure": message,
            "api_error": category == "api_rate_limit",
        }

    predicted_sql = result.get("sql_query", "")
    predicted_params = result.get("sql_params", [])
    actual_rows = result.get("query_rows", [])
    actual_columns = result.get("query_columns", [])
    retrieved_tables = result.get("retrieved_tables")
    required_tables = case.get("required_tables", [])
    execution_text = result.get("execution_result", "")

    try:
        expected_rows, expected_columns = run_readonly_query(db_path, case["expected_sql"])
        projected_rows, columns_projected = project_actual_columns(
            actual_rows, actual_columns, expected_columns
        )
        preserve_order = "ORDER BY" in case["expected_sql"].upper()
        result_match = normalize_rows(projected_rows, preserve_order) == normalize_rows(expected_rows, preserve_order)
        reference_error = None
    except Exception as error:
        expected_rows = []
        expected_columns = []
        projected_rows = actual_rows
        columns_projected = False
        result_match = False
        reference_error = str(error)

    # The execution node writes SQL errors here. A summary API failure should
    # not turn an already successful SQL execution into an execution failure.
    execution_success = not any(marker.lower() in execution_text.lower() for marker in SQL_ERROR_MARKERS)

    if retrieved_tables is None:
        table_recall = None
        table_precision = None
        retrieved_set = set()
    else:
        retrieved_set = {table.lower() for table in retrieved_tables}
        required_set = {table.lower() for table in required_tables}
        matched_tables = retrieved_set & required_set
        table_recall = len(matched_tables) / len(required_set) if required_set else 1.0
        table_precision = len(matched_tables) / len(retrieved_set) if retrieved_set else 0.0

    passed = result_match and execution_success
    failure_category = None if passed else ("sql_execution_error" if not execution_success else "result_mismatch")

    # Mark a possible truncation explicitly; do not count a truncated result as correct.
    possible_truncation = len(actual_rows) == 1000 and len(expected_rows) > len(actual_rows)

    return {
        "id": case["id"],
        "level": case.get("level", "unknown"),
        "question": case["question"],
        "required_tables": sorted(table.lower() for table in required_tables),
        "retrieved_tables": sorted(retrieved_set) if retrieved_tables is not None else None,
        "table_recall": round(table_recall, 4) if table_recall is not None else None,
        "table_precision": round(table_precision, 4) if table_precision is not None else None,
        "generated_sql": predicted_sql,
        "generated_params": predicted_params,
        "reference_sql": case["expected_sql"],
        "expected_columns": expected_columns,
        "actual_columns": actual_columns,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "comparison_rows": projected_rows,
        "columns_projected_for_comparison": columns_projected,
        "possible_row_limit_truncation": possible_truncation,
        "execution_success": execution_success,
        "result_match": result_match,
        "passed": passed,
        "failure_category": failure_category,
        "failure": None if passed else ("SQL execution failed" if not execution_success else "Result mismatch"),
        "error": execution_text if not execution_success else None,
        "reference_error": reference_error,
        "retry_count": max(0, result.get("attempts", 1) - 1),
    }


async def run_evaluation(dataset_path, db_path, graph, retriever):
    with open(dataset_path, "r", encoding="utf-8") as file:
        cases = json.load(file)

    results = []
    for index, case in enumerate(cases):
        print(f"Evaluating {case['id']}: {case['question']}")
        results.append(await evaluate_case(case, graph, retriever, db_path))
        if index < len(cases) - 1:
            await asyncio.sleep(2)

    total = len(results)
    api_errors = [item for item in results if item.get("api_error") or item.get("failure_category") == "api_rate_limit"]
    evaluable = [item for item in results if item not in api_errors]
    result_correct = sum(item.get("result_match", False) for item in evaluable)
    execution_success = sum(item.get("execution_success", False) for item in evaluable)
    workflow_passed = sum(item.get("passed", False) for item in evaluable)

    retrieval_results = [item for item in results if item.get("table_recall") is not None]
    retrieval_recall = sum(item["table_recall"] for item in retrieval_results) / len(retrieval_results) if retrieval_results else None
    retrieval_precision = sum(item["table_precision"] for item in retrieval_results) / len(retrieval_results) if retrieval_results else None

    report = {
        "total_questions": total,
        "evaluable_questions": len(evaluable),
        "api_rate_limit_errors": len(api_errors),
        "result_accuracy": round(result_correct / len(evaluable), 4) if evaluable else 0,
        "execution_success_rate": round(execution_success / len(evaluable), 4) if evaluable else 0,
        "workflow_pass_rate": round(workflow_passed / len(evaluable), 4) if evaluable else 0,
        "average_table_recall": round(retrieval_recall, 4) if retrieval_recall is not None else None,
        "average_table_precision": round(retrieval_precision, 4) if retrieval_precision is not None else None,
        "retrieval_tracking_questions": len(retrieval_results),
        "failed_questions": [item for item in results if not item.get("passed", False)],
        "api_errors": api_errors,
        "results": results,
    }

    output_path = Path("evaluation_report.json")
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n========== ChatSQL Evaluation ==========")
    print(f"Questions:                 {total}")
    print(f"Evaluable questions:       {len(evaluable)}")
    print(f"API rate-limit errors:     {len(api_errors)}")
    print(f"Result accuracy:           {report['result_accuracy']:.1%}")
    print(f"Execution success rate:    {report['execution_success_rate']:.1%}")
    print(f"Workflow pass rate:        {report['workflow_pass_rate']:.1%}")
    if retrieval_recall is not None:
        print(f"Average table recall:      {retrieval_recall:.1%}")
        print(f"Average table precision:   {retrieval_precision:.1%}")
    print(f"Failed questions:          {len(report['failed_questions'])}")
    print(f"Report saved to:            {output_path.resolve()}")
    return report
