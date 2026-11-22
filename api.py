import os
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from core.vector_retriever import VectorSchemaRetriever
from core.workflow import sql_workflow
from main_db_setup import setup_local_database

load_dotenv()


# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    db_path: Optional[str] = None


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    generated_sql: str
    sql_params: List[Any] = Field(default_factory=list)
    execution_result: str
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)


class UploadDBResponse(BaseModel):
    status: str
    db_path: str
    filename: str


# Shared global application state
resources: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages heavy resources across the application startup and shutdown cycle.
    Prevents reloading models or reconnecting to state DBs on every HTTP request.
    """
    default_db_path = setup_local_database()
    os.makedirs("temp_uploads", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    model = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
    schema_retriever = VectorSchemaRetriever(default_db_path)

    checkpoints_path = os.path.join("data", "checkpoints.db")
    checkpointer_cm = AsyncSqliteSaver.from_conn_string(checkpoints_path)
    checkpointer = await checkpointer_cm.__aenter__()

    graph = sql_workflow(
        model=model,
        schema_retriever=schema_retriever,
        checkpointer=checkpointer,
    )

    resources["graph"] = graph
    resources["default_db_path"] = default_db_path
    resources["model"] = model
    resources["checkpointer"] = checkpointer
    resources["checkpointer_cm"] = checkpointer_cm
    resources["retrievers"] = {default_db_path: schema_retriever}

    print("\n[*] FastAPI Agent Service ready!")
    try:
        yield
    finally:
        await resources["checkpointer_cm"].__aexit__(None, None, None)
        print("[*] FastAPI Service stopped cleanly.")


app = FastAPI(title="Text-to-SQL Agent API", lifespan=lifespan)


@app.post("/upload-db", response_model=UploadDBResponse)
async def upload_database(file: UploadFile = File(...)):
    """
    Receive a custom .db or .sqlite file, save it, and dynamically create/index
    a VectorSchemaRetriever for it.
    """
    filename = file.filename or ""
    if not (filename.endswith(".db") or filename.endswith(".sqlite")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only .db or .sqlite files are supported.",
        )

    saved_path = os.path.join("temp_uploads", f"{uuid.uuid4().hex[:8]}_{filename}")
    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        retriever = VectorSchemaRetriever(saved_path)
        resources["retrievers"][saved_path] = retriever

        return UploadDBResponse(
            status="success",
            db_path=saved_path,
            filename=filename,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process database: {exc}")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Main multi-turn conversation endpoint. Handles execution through LangGraph
    and returns SQL, bound parameters, execution results, and natural language summaries.
    """
    active_db_path = request.db_path or resources["default_db_path"]

    if not os.path.exists(active_db_path):
        raise HTTPException(status_code=404, detail=f"Database not found at: {active_db_path}")

    if active_db_path not in resources["retrievers"]:
        resources["retrievers"][active_db_path] = VectorSchemaRetriever(active_db_path)

    active_retriever = resources["retrievers"][active_db_path]

    thread_id = request.thread_id or f"session_{uuid.uuid4().hex[:8]}"
    config: Dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "schema_retriever": active_retriever,
        }
    }

    initial_state: Dict[str, Any] = {
        "messages": [HumanMessage(content=request.message)],
        "db_path": active_db_path,
        "sql_query": "",
        "sql_params": [],
        "execution_result": "",
        "query_columns": [],
        "query_rows": [],
        "error_count": 0,
    }

    final_sql = ""
    final_params: List[Any] = []
    final_result = ""
    final_columns: List[str] = []
    final_rows: List[List[Any]] = []
    final_response = ""

    try:
        async for event in resources["graph"].astream(initial_state, config=config, stream_mode="updates"):
            for _, update in event.items():
                if not isinstance(update, dict):
                    continue
                if "sql_query" in update:
                    final_sql = update["sql_query"]
                if "sql_params" in update:
                    final_params = list(update["sql_params"] or [])
                if "execution_result" in update:
                    final_result = update["execution_result"]
                if "query_columns" in update:
                    final_columns = list(update["query_columns"] or [])
                if "query_rows" in update:
                    final_rows = [list(r) for r in (update["query_rows"] or [])]
                if "messages" in update and update["messages"]:
                    last_msg = update["messages"][-1]
                    final_response = getattr(last_msg, "content", str(last_msg))

        return ChatResponse(
            thread_id=thread_id,
            response=final_response,
            generated_sql=final_sql,
            sql_params=final_params,
            execution_result=final_result,
            columns=final_columns,
            rows=final_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {exc}")