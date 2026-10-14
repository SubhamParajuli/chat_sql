import asyncio
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.vector_retriever import VectorSchemaRetriever
from core.workflow import sql_workflow
from main_db_setup import setup_local_database

load_dotenv()

async def main():
    db_path = setup_local_database()
    
    print("\n" + "=" * 60)
    print("  Text-to-SQL Agent CLI (LangGraph + Groq + SQLite Persistence)")
    print("=" * 60)
    print("[*] Initializing LLM client (qwen/qwen3.8-27b)...")
    model = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
    
    print("[*] Initializing Vector Schema Retriever...")
    schema_retriever = VectorSchemaRetriever(db_path)
    
    checkpoints_path = os.path.join("data", "checkpoints.db")
    os.makedirs("data", exist_ok=True)
    
    async with AsyncSqliteSaver.from_conn_string(checkpoints_path) as checkpointer:
        print(f"[*] Persistent Checkpointer ready at: {checkpoints_path}")
        
        # Build workflow with persistence injected
        app = sql_workflow(model=model, schema_retriever=schema_retriever, checkpointer=checkpointer)
        
        # Interactive session thread setup
        thread_id = "session_cli_user"
        config = {"configurable": {"thread_id": thread_id}}
        
        print("\n" + "-" * 60)
        print("  Type your SQL request below.")
        print("  Commands: 'exit' or 'quit' to end session | 'reset' to switch thread.")
        print("-" * 60 + "\n")
        
        while True:
            try:
                # Use asyncio.to_thread for non-blocking console input in async loop
                user_input = await asyncio.to_thread(input, "\nUser > ")
                user_input = user_input.strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["exit", "quit"]:
                    print("\n[*] Exiting Text-to-SQL Agent. Goodbye!")
                    break
                    
                if user_input.lower() == "reset":
                    import uuid
                    thread_id = f"session_{uuid.uuid4().hex[:8]}"
                    config = {"configurable": {"thread_id": thread_id}}
                    print(f"\n[*] Switched to new conversation thread: {thread_id}")
                    continue

                initial_state = {
                    "messages": [HumanMessage(content=user_input)],
                    "db_path": db_path,
                    "sql_query": "",
                    "execution_result": "",
                    "error_count": 0
                }

                print("\n  [Workflow Execution]")
                async for event in app.astream(initial_state, config=config, stream_mode="updates"):
                    for node_name, update in event.items():
                        if not isinstance(update, dict):
                            continue
                        
                        # Print intermediate node updates cleanly
                        if node_name == "generate_sql" and "sql_query" in update:
                            print(f"  ├── [Generated SQL]: {update['sql_query']}")
                            
                        elif node_name == "execute_sql" and "execution_result" in update:
                            res = update["execution_result"]
                            if "Security Violation" in res or "SECURITY GUARDRAIL" in res:
                                print(f"  ├── [Guardrail Blocked]: {res}")
                            else:
                                print(f"  ├── [Execution Result]: {res}")
                                
                        elif node_name == "summarize" and "messages" in update:
                            latest_msg = update["messages"][-1]
                            print(f"\nAssistant > {latest_msg.content}")

            except (KeyboardInterrupt, EOFError):
                print("\n\n[*] Session interrupted. Exiting.")
                break
            except Exception as e:
                print(f"\n[!] Error during workflow execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())