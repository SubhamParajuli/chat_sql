
import streamlit as st
import requests

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Text-to-SQL AI Agent",
    page_icon="⚡",
    layout="wide",
)

API_BASE_URL = "http://127.0.0.1:8000"


# ==========================================
# Session State
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_db_path" not in st.session_state:
    st.session_state.active_db_path = None

if "active_db_name" not in st.session_state:
    st.session_state.active_db_name = "Default (enterprise.db)"


# ==========================================
# Helper Functions
# ==========================================
def reset_conversation():
    st.session_state.thread_id = None
    st.session_state.messages = []


def display_query_details(msg):
    sql = msg.get("sql", "")
    params = msg.get("params", [])
    columns = msg.get("columns", [])
    rows = msg.get("rows", [])
    result = msg.get("result", "")

    if not sql:
        return

    with st.expander("Query Details", expanded=False):

        st.markdown("#### Generated SQL")
        st.code(sql, language="sql")

        if params:
            st.markdown("#### SQL Parameters")
            st.json(params)

        if columns and rows:
            st.markdown("#### Query Results")

            try:
                st.dataframe(
                    rows,
                    column_config={
                        column: column
                        for column in columns
                    },
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception:
                st.write(rows)

        elif result:
            st.markdown("#### Execution Result")
            st.write(result)

        if columns and not rows:
            st.info("The query returned no rows.")


# ==========================================
# Sidebar
# ==========================================
st.sidebar.title("Agent Settings")

# Conversation controls
st.sidebar.subheader("Conversation")

if st.sidebar.button("New Conversation", use_container_width=True):
    reset_conversation()
    st.rerun()

st.sidebar.caption(
    f"Thread ID: {st.session_state.thread_id or 'New session'}"
)

st.sidebar.divider()

# Database controls
st.sidebar.subheader("Database")

st.sidebar.info(
    f"Active database: {st.session_state.active_db_name}"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload SQLite database",
    type=["db", "sqlite"],
)

if uploaded_file is not None:

    if st.sidebar.button("Load Database", use_container_width=True):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
            )
        }

        with st.sidebar.status("Uploading and indexing database...") as status:

            try:
                response = requests.post(
                    f"{API_BASE_URL}/upload-db",
                    files=files,
                    timeout=60,
                )

                if response.status_code == 200:

                    data = response.json()

                    st.session_state.active_db_path = data["db_path"]
                    st.session_state.active_db_name = data["filename"]

                    reset_conversation()

                    status.update(
                        label="Database loaded successfully!",
                        state="complete",
                    )

                    st.rerun()

                else:

                    status.update(
                        label="Database upload failed",
                        state="error",
                    )

                    try:
                        error_message = response.json().get(
                            "detail",
                            response.text,
                        )
                    except Exception:
                        error_message = response.text

                    st.sidebar.error(error_message)

            except requests.exceptions.RequestException as exc:

                status.update(
                    label="Connection failed",
                    state="error",
                )

                st.sidebar.error(
                    f"Could not connect to FastAPI: {exc}"
                )


# Revert to default database
if st.session_state.active_db_path is not None:

    if st.sidebar.button("Use Default Database", use_container_width=True):

        st.session_state.active_db_path = None
        st.session_state.active_db_name = "Default (enterprise.db)"

        reset_conversation()

        st.rerun()


# ==========================================
# Main Interface
# ==========================================
st.title("Text-to-SQL AI Agent")

st.caption(
    "Ask questions about your database using natural language. "
    "The agent retrieves relevant schema, generates SQL, "
    "validates it, and executes the query."
)

# Current database information
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Active Database",
        st.session_state.active_db_name,
    )

with col2:
    st.metric(
        "Conversation Messages",
        len(st.session_state.messages),
    )

st.divider()


# ==========================================
# Display Chat History
# ==========================================
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            display_query_details(msg)


# ==========================================
# Chat Input
# ==========================================
user_input = st.chat_input(
    "Ask a question about your database..."
)

if user_input:

    # Save and display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare API request
    payload = {
        "message": user_input,
        "thread_id": st.session_state.thread_id,
        "db_path": st.session_state.active_db_path,
    }

    # Get response from backend
    with st.chat_message("assistant"):

        with st.spinner("Analyzing schema, generating and validating SQL..."):

            try:

                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json=payload,
                    timeout=120,
                )

                if response.status_code == 200:

                    data = response.json()

                    # Update thread
                    st.session_state.thread_id = data["thread_id"]

                    # Display answer
                    st.markdown(data["response"])

                    # Prepare assistant message
                    assistant_message = {
                        "role": "assistant",
                        "content": data.get("response", ""),
                        "sql": data.get("generated_sql", ""),
                        "params": data.get("sql_params", []),
                        "result": data.get("execution_result", ""),
                        "columns": data.get("columns", []),
                        "rows": data.get("rows", []),
                    }

                    # Show SQL and query results
                    display_query_details(assistant_message)

                    # Save conversation
                    st.session_state.messages.append(
                        assistant_message
                    )

                else:

                    try:
                        error_detail = response.json().get(
                            "detail",
                            response.text,
                        )
                    except Exception:
                        error_detail = response.text

                    st.error(
                        f"API Error {response.status_code}: {error_detail}"
                    )

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. "
                    "The agent may still be processing the query."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI backend. "
                    "Make sure the API is running."
                )

            except requests.exceptions.RequestException as exc:

                st.error(f"Request failed: {exc}")

            except Exception as exc:

                st.error(f"Unexpected error: {exc}")
