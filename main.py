import os
import sqlite3
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from core.workflow import sql_workflow


load_dotenv()

def setup_local_db(db_path="./data/enterprise.db"):
    os.makedirs(os.path.dirname(db_path),exist_ok=True)
    conn=sqlite3.connect(db_path)
    cursor=conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS  employees(
        id INTEGER PRIMARY KEY,
        name TEXT,
        department TEXT,
        salary REAL
        )
    """)
    cursor.execute("INSERT OR IGNORE INTO employees VALUES (1, 'Alice Smith', 'Engineering', 95000.0);")
    cursor.execute("INSERT OR IGNORE INTO employees VALUES (2, 'Bob Jones', 'Marketing', 75000.0);")
    cursor.execute("INSERT OR IGNORE INTO employees VALUES (3, 'Charlie Brown', 'Engineering', 105000.0);")
    
    conn.commit()
    conn.close()
    return os.path.abspath(db_path)

async def main():
    db_path=setup_local_db()
    print(f"[*] Local SQLite database ready at: {db_path}")
    app = sql_workflow()
    user_query = "What is the average salary of employees in the engineering department?"
    print(f"\nUser Prompt: '{user_query}'\n")

    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "db_path": db_path,
        "sql_query": "",
        "execution_result": ""
    }

    print("--- Streaming Workflow Execution ---")


    # Asynchronously stream updates node-by-node
    async for event in app.astream(initial_state,stream_mode="updates"):
        for node_name,node_output in event.items():
            print(f"  [Stream Event] Node '{node_name}' finished processing.")

    # Get final aggregated result
    final_output = await app.ainvoke(initial_state)
    final_message = final_output["messages"][-1].content
    
    print(f"\n==================================================")
    print(f"Final Answer:\n{final_message}")
    print(f"==================================================")

if __name__ == "__main__":
    asyncio.run(main())