import sqlite3
from typing import TypedDict,List,Annotated
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import START,END,StateGraph,add_messages
from guardrails.ast_validator import enforce_sql_guardrails,SecurityError
from core.vector_retriever import VectorSchemaRetriever


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]
    db_path:str
    sql_query:str
    execution_result: str
    error_count: int


def sql_workflow(model:ChatGroq,schema_retriever:VectorSchemaRetriever,checkpointer=None):
    # STEP 1: The LLM writes the SQL query based on user prompt
    async def generate_sql_node(state: AgentState):
        db_schema = schema_retriever.retrieve_relevant_schema(state["messages"][-1].content)
        
        # If this is a retry (error_count > 0), forcefully instruct the model to fix the case matching
        retry_instruction = ""
        if state.get("error_count", 0) > 0:
            retry_instruction = (
                "\n\n[URGENT CORRECTION REQUIRED]\n"
                f"Your previous query failed or returned no data (Result: {state.get('execution_result')}).\n"
                "The database text values might have different capitalization (e.g. 'Engineering' instead of 'engineering'). "
                "You MUST use LOWER(department) = LOWER('engineering') or LIKE '%engineering%' in your WHERE clause!"
            )

        prompt = (
            f"You are an expert SQL generator. Relevant Schema: {db_schema}\n"
            f"User request: {state['messages'][-1].content}"
            f"{retry_instruction}\n\n"
            "CRITICAL RULES:\n"
            "1. Return ONLY a valid SQL SELECT statement. No markdown, no explanations.\n"
            "2. Always use LOWER() on string columns for filtering."
        )
        
        response = await model.ainvoke([HumanMessage(content=prompt)])
        raw_sql = response.content.strip()
        print(f"  [Agentic Loop] LLM Generated SQL: {raw_sql}")
        return {"sql_query": raw_sql}

    
    # STEP 2: Pillar 2 (AST Guardrails) & Database Execution

    async def execute_sql_node(state: AgentState):
        raw_sql = state.get("sql_query", "").strip()
        
        try:
            # 1. Run your robust ast_validator guardrail & limit injector
            safe_sql = enforce_sql_guardrails(raw_sql, max_row_limit=50)
            
            # 2. Execute the sanitized SQL
            conn = sqlite3.connect(state["db_path"])
            cursor = conn.cursor()
            cursor.execute(safe_sql)
            rows = cursor.fetchall()
            conn.close()
            
            is_empty_result = not rows or rows == [()] or rows == [(None,)]
            if is_empty_result:
                return {
                    "execution_result": f"Query returned empty or null results: {rows}.",
                    "error_count": state.get("error_count", 0) + 1
                }

            print(f"  [Agentic Loop] Query executed successfully. Results: {rows}")
            return {"execution_result": str(rows), "sql_query": safe_sql}
            
        except SecurityError as sec_err:
            # Catch security violations explicitly so the LLM knows it was blocked
            error_msg = str(sec_err)
            print(f"  [Guardrail Blocked] {error_msg}")
            return {
                "execution_result": error_msg,
                "error_count": state.get("error_count", 0) + 1
            }
        
        except Exception as db_err:
            error_msg = f"SQL Execution Error: {db_err}"
            return {
                "execution_result": error_msg,
                "error_count": state.get("error_count", 0) + 1
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
    async def summarize_node(state: AgentState):
        user_query = state["messages"][-1].content.lower()
        result = state["execution_result"]
        
        # 1. Programmatic guardrail: Intercept mutation requests instantly
        mutation_keywords = ["delete", "remove", "drop", "update", "insert", "modify", "clear"]
        is_mutation_attempt = any(keyword in user_query for keyword in mutation_keywords)
        
        if is_mutation_attempt:
            refusal_msg = (
                f"I cannot perform delete or modification operations because this system is strictly read-only. "
                f"However, based on your search query, here is the record found:\n\n{result}"
            )
            return {"messages": [AIMessage(content=refusal_msg)]}
        
        # 2. Standard summarization for read queries
        prompt = (
            f"The user asked: '{user_query}'. "
            f"The query result was: {result}. "
            "Write a short, friendly plain-English answer based strictly on the data."
        )
        response = await model.ainvoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=response.content)]}

    

    
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

    return workflow.compile(checkpointer=checkpointer)