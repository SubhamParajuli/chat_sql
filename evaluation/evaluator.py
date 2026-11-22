import json
import sqlite3
from pathlib import Path

from langchain_core.messages import HumanMessage


ERROR_MARKERS = [
    "SQL Execution Error:",
    "Security Violation:",
    "429",
    "rate limit",
    "tokens per minute",
]


def run_readonly_query(db_path, sql, params=None):
    """Run a reference SQL query using a read-only SQLite connection."""
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        return conn.execute(sql, params or []).fetchall()



def normalize_rows(rows, preserve_order=False):
    """Round floats and optionally ignore result row ordering."""
    normalized = []
    for row in rows:
        normalized_row = []
        for value in row:
            if isinstance(value, float):
                value = round(value, 4)
            normalized_row.append(value)
        normalized.append(tuple(normalized_row))

    if not preserve_order:
        normalized.sort(key=str)
    return normalized



def project_rows(actual_rows, actual_columns, expected_columns):
    """Remove extra output columns when expected column names are unambiguous."""
    if not actual_columns or not expected_columns:
        return actual_rows, False


    # Avoid unsafe projection when names are duplicated.
    if len(set(actual_columns)) != len(actual_columns) or len(set(expected_columns)) != len(expected_columns):
        return actual_rows, False


    indexes = []
    for column in expected_columns:
        if column not in actual_columns:
            return actual_rows, False
        indexes.append(actual_columns.index(column))



    projected = [tuple(row[index] for index in indexes) for row in actual_rows]
    return projected, True



def ordered_rows_match(expected_rows, actual_rows, reference_sql, expected_columns):
    """Compare ordered results, allowing different order only within tied ORDER BY values."""
    sql = reference_sql.upper()
    if "ORDER BY" not in sql:
        return normalize_rows(expected_rows) == normalize_rows(actual_rows)



    # For simple ORDER BY column/alias expressions, treat rows with the same
    # ordering value as a tie group. Their internal order is not meaningful.
    order_part = sql.split("ORDER BY", 1)[1].split("LIMIT", 1)[0].split("OFFSET", 1)[0]
    first_order = order_part.split(",", 1)[0].strip()
    order_name = first_order.split()[0].strip('`"').lower()
    column_names_lower = [name.lower() for name in expected_columns]
    if order_name not in column_names_lower:
        return normalize_rows(expected_rows, True) == normalize_rows(actual_rows, True)

    key_index = column_names_lower.index(order_name)
    def grouped(rows):
        result = []
        i = 0
        while i < len(rows):
            key = rows[i][key_index]
            group = []
            while i < len(rows) and rows[i][key_index] == key:
                group.append(rows[i])
                i += 1
            result.append(normalize_rows(group))
        return result

    return grouped(expected_rows) == grouped(actual_rows)



def get_failure_category(execution_text):
    text = (execution_text or "").lower()
    if "429" in text or "rate limit" in text or "tokens per minute" in text:
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
            config={"configurable": {"schema_retriever": retriever}},
        )
    except Exception as error:
        return {
            "id": case["id"],
            "question": case["question"],
            "passed": False,
            "result_match": False,
            "execution_success": False,
            "failure_category": "workflow_exception",
            "failure": f"Workflow failed: {error}",
        }


    predicted_sql = result.get("sql_query", "")
    predicted_params = result.get("sql_params", [])
    actual_rows = result.get("query_rows", [])
    retrieved_tables = result.get("retrieved_tables")
    required_tables = case.get("required_tables", [])



    try:
        expected_rows = run_readonly_query(db_path, case["expected_sql"])
        reference_sql = case["expected_sql"]
        with sqlite3.connect(db_path) as conn:
            expected_columns = [column[0] for column in conn.execute(reference_sql).description or []]
        actual_columns = result.get("query_columns", [])
        compared_rows, projected = project_rows(actual_rows, actual_columns, expected_columns)
        if "ORDER BY" in reference_sql.upper():
            result_match = ordered_rows_match(expected_rows, compared_rows, reference_sql, expected_columns)
        else:
            result_match = normalize_rows(expected_rows) == normalize_rows(compared_rows)
        reference_error = None
    except Exception as error:
        expected_rows = []
        result_match = False
        reference_error = str(error)



    execution_text = result.get("execution_result", "")
    execution_success = not any(marker.lower() in execution_text.lower() for marker in ERROR_MARKERS)



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
    failure_category = get_failure_category(execution_text) if not execution_success else None
    if not execution_success and result_match:
        failure_category = "workflow_rejected_correct_result"



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
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "columns_projected_for_comparison": projected if reference_error is None else False,
        "possible_row_limit_truncation": len(actual_rows) == 50 and len(expected_rows) > 50,
        "execution_success": execution_success,
        "result_match": result_match,
        "passed": passed,
        "failure_category": failure_category,
        "failure": None if passed else ("Workflow execution failed" if not execution_success else "Result mismatch"),
        "error": execution_text if not execution_success else None,
        "reference_error": reference_error,
        "retry_count": max(0, result.get("attempts", 1) - 1),
    }



async def run_evaluation(dataset_path, db_path, graph, retriever):
    with open(dataset_path, "r", encoding="utf-8") as file:
        cases = json.load(file)



    results = []
    for case in cases:
        print(f"Evaluating {case['id']}: {case['question']}")
        results.append(await evaluate_case(case, graph, retriever, db_path))



    total = len(results)
    api_errors = [item for item in results if item.get("failure_category") == "api_rate_limit"]
    evaluable_results = [item for item in results if item.get("failure_category") != "api_rate_limit"]
    result_correct = sum(item.get("result_match", False) for item in evaluable_results)
    execution_success = sum(item.get("execution_success", False) for item in results)
    workflow_passed = sum(item.get("passed", False) for item in evaluable_results)



    retrieval_results = [item for item in results if item.get("table_recall") is not None]
    retrieval_recall = (
        sum(item["table_recall"] for item in retrieval_results) / len(retrieval_results)
        if retrieval_results else None
    )
    retrieval_precision = (
        sum(item["table_precision"] for item in retrieval_results) / len(retrieval_results)
        if retrieval_results else None
    )



    report = {
        "total_questions": total,
        "result_accuracy": round(result_correct / len(evaluable_results), 4) if evaluable_results else 0,
        "accuracy_denominator": len(evaluable_results),
        "api_rate_limit_errors": len(api_errors),
        "execution_success_rate": round(execution_success / total, 4) if total else 0,
        "workflow_pass_rate": round(workflow_passed / len(evaluable_results), 4) if evaluable_results else 0,
        "average_table_recall": round(retrieval_recall, 4) if retrieval_recall is not None else None,
        "average_table_precision": round(retrieval_precision, 4) if retrieval_precision is not None else None,
        "retrieval_tracking_questions": len(retrieval_results),
        "failed_questions": [item for item in results if not item.get("passed", False)],
        "results": results,
    }



    output_path = Path("evaluation_report.json")
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")



    print("\n========== ChatSQL Evaluation ==========")
    print(f"Questions:                 {total}")
    print(f"Result accuracy:            {report['result_accuracy']:.1%} ({report['accuracy_denominator']} evaluable questions)")
    print(f"API rate-limit errors:      {report['api_rate_limit_errors']}")
    print(f"Execution success rate:     {report['execution_success_rate']:.1%}")
    print(f"Workflow pass rate:         {report['workflow_pass_rate']:.1%}")
    if retrieval_recall is not None:
        print(f"Average table recall:       {retrieval_recall:.1%}")
        print(f"Average table precision:    {retrieval_precision:.1%}")
    else:
        print("Table recall/precision:    not tracked yet")
    print(f"Failed questions:           {len(report['failed_questions'])}")
    print(f"Report saved to:            {output_path.resolve()}")
    return report
