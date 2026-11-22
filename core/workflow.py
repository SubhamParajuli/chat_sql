import json
import sqlite3
import asyncio
import random
from typing import Annotated, Any, List, TypedDict, Optional, Set


from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph, add_messages
from langchain_core.runnables import RunnableConfig


from core.vector_retriever import VectorSchemaRetriever
from guardrails.ast_validator import SecurityError, enforce_sql_guardrails


async def invoke_with_retry(model, messages, max_attempts=5):
    """Retry temporary Groq rate-limit errors with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return await model.ainvoke(messages)
        except Exception as error:
            message = str(error).lower()
            is_rate_limit = "429" in message or "rate_limit" in message or "rate limit" in message or "tokens per minute" in message
            if not is_rate_limit or attempt == max_attempts - 1:
                raise
            wait_seconds = min(30, 2 ** (attempt + 1)) + random.uniform(0, 1)
            print(f"  [Rate Limit] Waiting {wait_seconds:.1f}s before retry {attempt + 2}/{max_attempts}...")
            await asyncio.sleep(wait_seconds)



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    db_path: str
    sql_query: str
    sql_params: List[Any]
    execution_result: str
    query_columns: List[str]
    query_rows: List[tuple[Any, ...]]
    error_count: int
    retrieved_tables: List[str]
    attempts: int




def _extract_sql_and_params(raw_text: str) -> tuple[str, List[Any]]:
    """Parse JSON {sql, params}; otherwise treat response as plain SQL."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()



    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "sql" in parsed:
            sql = str(parsed["sql"]).strip()
            params = parsed.get("params", [])
            if not isinstance(params, list):
                params = list(params) if isinstance(params, tuple) else [params]
            return sql, params
    except json.JSONDecodeError:
        pass


    return cleaned, []




def sql_workflow(model: ChatGroq, schema_retriever: VectorSchemaRetriever, checkpointer=None):
    async def generate_sql_node(state: AgentState, config: RunnableConfig):
        configurable = (config or {}).get("configurable", {})
        active_retriever = configurable.get("schema_retriever") or schema_retriever

        latest_user_msg = state["messages"][-1].content
        db_schema = active_retriever.retrieve_relevant_schema(latest_user_msg)
        retrieved_tables = getattr(active_retriever, "last_retrieved_tables", []).copy()

        retry_instruction = ""
        if state.get("error_count", 0) > 0:
            retry_instruction = (
                "\n\n[CORRECTION REQUIRED]\n"
                f"Previous SQL: {state.get('sql_query', '')}\n"
                f"Previous Params: {state.get('sql_params', [])}\n"
                f"Issue: {state.get('execution_result', '')}\n"
                "Fix the SQL query or parameters using only valid tables/columns from the schema. "
                "If text filtering returned no rows due to casing, use LOWER(col) = LOWER(?) or LIKE."
            )

        prompt = (
            "You are an expert SQLite SQL generator.\n"
            f"Relevant Schema:\n{db_schema}\n\n"
            f"User request: {latest_user_msg}"
            f"{retry_instruction}\n\n"
            "RULES:\n"
            "1. Output ONLY a single read-only SELECT or WITH...SELECT statement.\n"
            "2. Use ? placeholders for literal filter values and put their values in 'params' in order.\n"
            "3. Respond as JSON with keys 'sql' and 'params' (no markdown fences)."
        )

        response = await invoke_with_retry(model, [HumanMessage(content=prompt)])
        sql_text, sql_params = _extract_sql_and_params(response.content)
        attempt_count = state.get("attempts", 0) + 1
        print(f"  [Agentic Loop] Generated SQL: {sql_text} | Params: {sql_params}")
        return {
            "sql_query": sql_text,
            "sql_params": sql_params,
            "retrieved_tables": retrieved_tables,
            "attempts": attempt_count,
        }



    async def execute_sql_node(state: AgentState, config: RunnableConfig):
        raw_sql = state.get("sql_query", "").strip()
        raw_params = state.get("sql_params", [])
        current_errors = state.get("error_count", 0)


        configurable = (config or {}).get("configurable", {})
        active_retriever = configurable.get("schema_retriever") or schema_retriever

        allowed_tables: Optional[Set[str]] = None
        if active_retriever is not None and hasattr(active_retriever, "get_allowed_tables"):
            tables = active_retriever.get_allowed_tables()
            if tables:
                allowed_tables = tables

        try:
            eval_mode = configurable.get("eval_mode", False)
            row_limit = 1000 if eval_mode else 50
            safe_sql, safe_params = enforce_sql_guardrails(
                raw_sql,
                max_row_limit=row_limit,
                params=raw_params,
                allowed_tables=allowed_tables,
            )

            ro_uri = f"file:{state['db_path']}?mode=ro"
            conn = sqlite3.connect(ro_uri, uri=True, timeout=5.0)
            try:
                cursor = conn.cursor()
                cursor.execute(safe_sql, safe_params)
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
            finally:
                conn.close()

            # Empty rows, NULL aggregates, and zero are valid SQL results.
            print(f"  [Agentic Loop] Query executed. Columns: {columns} | Rows: {rows}")
            return {
                "sql_query": safe_sql,
                "sql_params": list(safe_params),
                "query_columns": columns,
                "query_rows": rows,
                "execution_result": json.dumps({"columns": columns, "rows": rows}, default=str),
                "error_count": 0,
            }

        except SecurityError as sec_err:
            error_msg = f"Security Violation: {sec_err}"
            print(f"  [Guardrail Blocked] {error_msg}")
            return {"execution_result": error_msg, "error_count": current_errors + 1}

        except sqlite3.Error as db_err:
            error_msg = f"SQL Execution Error: {db_err}"
            print(f"  [DB Error] {error_msg}")
            return {"execution_result": error_msg, "error_count": current_errors + 1}



    def should_retry_or_summarize(state: AgentState):
        result = state.get("execution_result", "")
        errors = state.get("error_count", 0)
        is_retryable_issue = (
            "SQL Execution Error:" in result
            or "Parameter mismatch:" in result
            or "Security Violation:" in result
        )
        if is_retryable_issue and errors < 3:
            print(f"  [Router] Retrying self-correction (Attempt {errors + 1}/3)...")
            return "retry"
        return "summarize"



    async def summarize_node(state: AgentState):
        user_query = state["messages"][-1].content
        result = state["execution_result"]

        mutation_keywords = ["delete", "remove", "drop", "update", "insert", "modify", "clear"]
        is_mutation_attempt = any(kw in user_query.lower() for kw in mutation_keywords)
        if is_mutation_attempt:
            refusal_msg = (
                "I cannot delete, modify, or clear database records because this system is read-only. "
                f"Based on your query, here is what was retrieved:\n\n{result}"
            )
            return {"messages": [AIMessage(content=refusal_msg)]}

        prompt = (
            f"User question: '{user_query}'\n"
            f"Query result: {result}\n"
            "Write a concise, helpful plain-English response grounded strictly in the query result above."
        )
        try:
            response = await invoke_with_retry(model, [HumanMessage(content=prompt)])
            answer = response.content
        except Exception as error:
            # SQL has already executed. Keep its result even if the optional
            # natural-language summary cannot be generated because of an API error.
            answer = f"The SQL query executed successfully, but I could not generate a summary right now. Query result: {result}"
            print(f"  [Summary Warning] {error}")

        return {"messages": [AIMessage(content=answer)]}



    workflow = StateGraph(AgentState)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("summarize", summarize_node)

    workflow.add_edge(START, "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_conditional_edges(
        "execute_sql",
        should_retry_or_summarize,
        {"retry": "generate_sql", "summarize": "summarize"},
    )
    workflow.add_edge("summarize", END)

    return workflow.compile(checkpointer=checkpointer)
