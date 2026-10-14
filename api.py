import os
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import Dict,Any,Optional

from fastapi import FastAPI,HTTPException,UploadFile,File
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.vector_retriever import VectorSchemaRetriever
from core.workflow import sql_workflow
from main_db_setup import setup_local_database


load_dotenv()


#pydantic model for i/o validation

class ChatRequest(BaseModel):
    message:str
    thread_id:Optional[str]=None
    db_path:Optional[str]=None

class ChatResponse(BaseModel):
    thread_id:str
    response:str
    generated_sql:str
    execution_result:str

class UploadDBResponse(BaseModel):
    status: str
    db_path: str
    filename: str


#Shared Global Application State
resources: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    Manages heavy resources across the application startup and shutdown cycle.
    Prevents reloading models or reconnecting to state DBs on every HTTP request.
    """

    default_db_path=setup_local_database()
    os.makedirs("temp_uploads",exist_ok=True)
    os.makedirs("data",exist_ok=True)

    #initialize model
    model=ChatGroq(model="qwen/3.8-27b",temperature=0)

    #initialize default vector schema retriever
    schema_retriever=VectorSchemaRetriever(default_db_path)

    #Initialize AsyncSqliteSaver for persistent state tracking
    checkpoints_path=os.path.join("data","checkpoints.db")
    checkpointer_cm=AsyncSqliteSaver.from_conn_string(checkpoints_path)
    checkpointer=await checkpointer_cm.__aenter__()

    #compile langgrap agent workflow
    graph = sql_workflow(
        model=model, 
        schema_retriever=schema_retriever, 
        checkpointer=checkpointer
    )
    
    # Cache in shared dict
    resources["graph"] = graph
    resources["default_db_path"] = default_db_path
    resources["model"] = model
    resources["checkpointer"] = checkpointer
    resources["checkpointer_cm"] = checkpointer_cm
    resources["retrievers"] = {default_db_path: schema_retriever}
    
    print("\n[*] FastAPI Agent Service ready!")
    yield
    
    # Clean shutdown of checkpointer connection
    await resources["checkpointer_cm"].__aexit__(None, None, None)
    print("[*] FastAPI Service stopped cleanly.")

app = FastAPI(title="Text-to-SQL Agent API", lifespan=lifespan)



@app.post("/upload-db",response_model=UploadDBResponse)
async def upload_database(file: UploadFile=File(...)):
    """
    Receive a custom .db or .sqlite file,save it and dynamically creates/indexes a VectorSchemaRetreiver  for it.
    """
    if not (file.filename.endswith(".db") or file.filename.endswith(".sqlite")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only .db or .sqlite files are supported."
        )

    saved_path=os.path.join("temp_uploads",f"{uuid.uuid4().hex[:8]}_{file.filename}")
    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Dynamically create vector retriever for the uploaded database
        retriever = VectorSchemaRetriever(saved_path)
        resources["retrievers"][saved_path] = retriever
        
        return UploadDBResponse(
            status="success",
            db_path=saved_path,
            filename=file.filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process database: {str(e)}")
    
@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Main multi-turn conversation endpoint. Handles execution through LangGraph
    and returns SQL, execution results, and natural language summaries.
    """
    # Fallback to default DB if no custom DB specified
    active_db_path = request.db_path or resources["default_db_path"]
    
    if not os.path.exists(active_db_path):
        raise HTTPException(status_code=404, detail=f"Database not found at: {active_db_path}")
    
    # Ensure retriever exists for this database
    if active_db_path not in resources["retrievers"]:
        resources["retrievers"][active_db_path] = VectorSchemaRetriever(active_db_path)

    # Session thread management
    thread_id = request.thread_id or f"session_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "db_path": active_db_path,
        "sql_query": "",
        "execution_result": "",
        "error_count": 0
    }
    
    final_sql = ""
    final_result = ""
    final_response = ""
    
    try:
        # Stream workflow node updates
        async for event in resources["graph"].astream(initial_state, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                if not isinstance(update, dict):
                    continue
                if "sql_query" in update:
                    final_sql = update["sql_query"]
                if "execution_result" in update:
                    final_result = update["execution_result"]
                if "messages" in update:
                    final_response = update["messages"][-1].content
                    
        return ChatResponse(
            thread_id=thread_id,
            response=final_response,
            generated_sql=final_sql,
            execution_result=final_result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")