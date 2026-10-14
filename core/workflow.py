import sqlite3
import json
from typing import TypedDict,List
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import START,END,StateGraph
from guardrails.ast_validator import enforce_sql_guardrails,SecurityError



class AgentState(TypedDict):
    messages: List[BaseMessage]
    db_path:str
    sql_query:str
    execution_result: str
    error_count: int


# PILLAR 1: Schema Context Optimization (Simple Version)

def get_lightweight_schema(db_path: str) -> str:
    """Reads table schemas so we only pass what's needed."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        schema[table] = [col[1] for col in cursor.fetchall()]
    conn.close()
    return json.dumps(schema)


def sql_workflow():
    model=ChatGroq(model="qwen/qwen3.8-27b",temperature=0)

    # STEP 1: The LLM writes the SQL query based on user prompt

    async def generate_sql_node(state: AgentState):
        db_schema = get_lightweight_schema(state["db_path"])
        
        # Clean, general error feedback passed directly to the LLM
        error_context = ""
        if state.get("execution_result") and ("Error" in state["execution_result"] or state["execution_result"] == "[]"):
            error_context = (
                f"\n\n[NOTICE FROM PREVIOUS ATTEMPT]\n"
                f"Your last query produced this result/error: {state['execution_result']}\n"
                "Please analyze why this happened (check column names, table schemas, or case sensitivity like LOWER()) and write a corrected SQL query."
            )

        prompt = (
            f"You are an expert SQL generator. Database schema: {db_schema}\n"
            f"User request: {state['messages'][-1].content}"
            f"{error_context}\n\n"
            "Return ONLY a valid SQL SELECT statement. No markdown formatting, no code blocks, no explanations."
        )
        
        response = await model.ainvoke([HumanMessage(content=prompt)])
        raw_sql = response.content.strip()
        
        print(f"  [Agentic Loop] LLM Generated SQL: {raw_sql}")
        return {"sql_query": raw_sql}

    
    # STEP 2: Pillar 2 (AST Guardrails) & Database Execution

    async def execute_sql_node(state: AgentState):
        raw_sql = state["sql_query"]
        current_errors = state.get("error_count", 0)
        
        try:
            safe_sql = enforce_sql_guardrails(raw_sql, max_row_limit=50)
            print(f"  [Agentic Loop] Guardrail Passed. Executing: {safe_sql}")
        except SecurityError as e:
            return {
                "execution_result": f"Security Error: {e}",
                "error_count": current_errors + 1
            }

        try:
            conn = sqlite3.connect(state["db_path"])
            cursor = conn.cursor()
            cursor.execute(safe_sql)
            rows = cursor.fetchall()
            conn.close()

            is_empty_result = not rows or rows == [()] or rows == [(None,)]
            # CRITICAL FIX: If rows are empty, treat it as a failure so the LLM retries!
            if is_empty_result:
                print(f"  [Agentic Loop] Query returned empty/null results {rows}. Triggering self-correction.")
                return {
                    "execution_result": f"Query returned empty or null results: {rows}. Please review the schema, table values, or filtering conditions and try a different query.",
                    "error_count": current_errors + 1
                }

            print(f"  [Agentic Loop] Query executed successfully. Results: {rows}")
            return {"execution_result": str(rows)}
        except Exception as db_err:
            error_msg = f"SQL Execution Error: {db_err}"
            print(f"  [Agentic Loop] Caught error: {error_msg}")
            return {
                "execution_result": error_msg,
                "error_count": current_errors + 1
            }
# CONDITIONAL ROUTER: Decides whether to loop back or finish
    def should_retry_or_summarize(state: AgentState):
        result = state.get("execution_result", "")
        errors = state.get("error_count", 0)
        
        # Now ANY mention of failure or 0 rows will trigger a retry up to 3 times
        if ("Error" in result or "0 rows" in result or result == "[]") and errors < 3:
            print(f"  [Router] Issue detected. Looping back to self-correct (Attempt {errors + 1})...")
            return "retry"
        
        return "summarize"

    # STEP 3: Final Answer generation
    async def summarize_node(state:AgentState):
        result=state["execution_result"]
        prompt=f"The user asked: '{state['messages'][-1].content}. The query result was: {result}. Write a short, friendly plain-English answer."
        response=await model.ainvoke([HumanMessage(content=prompt)])
        return {"messages":[AIMessage(content=response.content)]}

    

    
    #Build Linear Graph Flow

    workflow=StateGraph(AgentState)
    workflow.add_node("generate_sql",generate_sql_node)
    workflow.add_node("execute_sql",execute_sql_node)
    workflow.add_node("summarize",summarize_node)

    workflow.add_edge(START,"generate_sql")
    workflow.add_edge("generate_sql","execute_sql")
    workflow.add_edge("execute_sql","summarize")
    # Add the conditional edge loop!
    workflow.add_conditional_edges(
        "execute_sql",
        should_retry_or_summarize,
        {
            "retry": "generate_sql",       # Loops back if it fails
            "summarize": "summarize"       # Moves forward if successful or max retries hit
        }
    )
    workflow.add_edge("summarize",END)

    return workflow.compile()