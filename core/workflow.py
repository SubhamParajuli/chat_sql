import os
import operator
from dotenv import load_dotenv
from typing import Annotated, TypedDict,List,Any
from langchain_core.messages import HumanMessage,BaseMessage,ToolMessage,AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph , START,END
from guardrails.ast_validator import enforce_sql_guardrails,SecurityError
import json
import sqlite3

load_dotenv()

#define state tracked across langgraph state machine

class AgentState(TypedDict):
    messages:Annotated[List[BaseMessage],operator.add]
    db_path:str
    mcp_session : Any
    error_count:int



#schema context optimization (for big database)
def get_relevant_schema(db_path:str, user_prompt:str) ->str:
    """
    Introspects the database and filters relevant table schemas.
    Alternative Rejected: Dumping all table schemas into the prompt. 
    Why rejected: Causes token bloat, high cost, and severe accuracy degradation 
    on databases with hundreds of tables.
    """

    conn=sqlite3.connect(db_path)
    cursor=conn.cursor()
    cursor.execute("SELECT name FROM sqlit_master WHERE type='table';")
    tables=[row[0] for row in cursor.fetchall()]

    schema_info={}

    prompt_words=set(user_prompt.lower().split())

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns=[{"column":col[1],"type":col[2]} for col in cursor.fetchall()]
