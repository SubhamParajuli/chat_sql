import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Text-to-SQL AI Agent",
    page_icon="⚡",
    layout="wide"
)

API_BASE_URL = "http://127.0.0.1:8000"

# Initialize Session States
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_db_path" not in st.session_state:
    st.session_state.active_db_path = None

if "active_db_name" not in st.session_state:
    st.session_state.active_db_name = "Default (enterprise.db)"

# ==========================================
# Sidebar: Database & Thread Controls
# ==========================================
st.sidebar.title("⚡ Agent Settings")

# 1. Conversation Thread Reset
if st.sidebar.button("🔄 New Conversation Thread"):
    st.session_state.thread_id = None
    st.session_state.messages = []
    st.sidebar.success("Thread reset! Started fresh session.")

st.sidebar.markdown(f"**Current Thread:**\n`{st.session_state.thread_id or 'New Session'}`")
st.sidebar.divider()

# 2. Database Uploader
st.sidebar.subheader("🗄️ Database Connection")
st.sidebar.info(f"**Active Database:**\n`{st.session_state.active_db_name}`")

uploaded_file = st.sidebar.file_uploader(
    "Upload custom SQLite DB (.db / .sqlite)", 
    type=["db", "sqlite"]
)

if uploaded_file is not None:
    if st.sidebar.button("Load Uploaded Database"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        with st.sidebar.status("Uploading & Indexing Schema...", expanded=True) as status:
            try:
                res = requests.post(f"{API_BASE_URL}/upload-db", files=files, timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.active_db_path = data["db_path"]
                    st.session_state.active_db_name = data["filename"]
                    st.session_state.thread_id = None  # Reset thread for new DB
                    st.session_state.messages = []
                    status.update(label="Database Ready!", state="complete")
                    st.rerun()
                else:
                    status.update(label="Upload Failed", state="error")
                    st.sidebar.error(res.json().get("detail", "Error uploading file"))
            except Exception as e:
                status.update(label="Connection Error", state="error")
                st.sidebar.error(f"Could not connect to backend: {e}")

# Option to revert to default
if st.session_state.active_db_path is not None:
    if st.sidebar.button("Revert to Default DB"):
        st.session_state.active_db_path = None
        st.session_state.active_db_name = "Default (enterprise.db)"
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.rerun()

# ==========================================
# Main Chat Interface
# ==========================================
st.title("🤖 Enterprise Text-to-SQL Agent")
st.caption("Resilient Agent with AST Guardrails, ChromaDB Schema Indexing & LangGraph Memory")

# Display previous messages in chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sql" in msg and msg["sql"]:
            with st.expander("🔍 Generated SQL & Query Result"):
                st.code(msg["sql"], language="sql")
                st.text(msg["result"])

# Chat Input Box
if user_input := st.chat_input("Ask a question about your data or request an query..."):
    # Render user prompt immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Processing AST guardrails & executing query..."):
            payload = {
                "message": user_input,
                "thread_id": st.session_state.thread_id,
                "db_path": st.session_state.active_db_path
            }
            
            try:
                response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Store thread ID from server
                    st.session_state.thread_id = data["thread_id"]
                    
                    # Render AI response
                    st.markdown(data["response"])
                    
                    if data["generated_sql"]:
                        with st.expander("🔍 Generated SQL & Query Result"):
                            st.code(data["generated_sql"], language="sql")
                            st.text(data["execution_result"])
                    
                    # Save to UI history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["response"],
                        "sql": data["generated_sql"],
                        "result": data["execution_result"]
                    })
                else:
                    st.error(f"Error {response.status_code}: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Failed to communicate with API backend: {e}")