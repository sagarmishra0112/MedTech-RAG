import streamlit as st
import requests

# Set page config for a cleaner look
st.set_page_config(page_title="MedTech RAG", page_icon="🏥", layout="centered")

# Custom UI Styling
st.title("🏥 MedTech RAG: X-Ray Assistant")
st.markdown("Ask technical questions about X-ray documentation. The AI retrieves exact diagnostic chunks and tables.")

# Sidebar Configuration
st.sidebar.header("⚙️ Pipeline Mode")
pipeline_mode = st.sidebar.radio(
    "Select RAG Pipeline:",
    ["🤖 Agentic RAG (/agent-query)", "⚡ Classic RAG (/query)"],
    help="Agentic RAG uses multi-step tool calls and self-critique grading. Classic RAG is a single pass."
)

is_agentic = "Agentic" in pipeline_mode

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input box at the bottom of the screen
if prompt := st.chat_input("Ask a question about Calibration, Overload Settings, etc..."):
    # 1. Display User Message Immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Show assistant "Thinking..." indicator while querying the API
    with st.chat_message("assistant"):
        spinner_msg = "🤖 Agent thinking & invoking tools..." if is_agentic else "⚡ Searching Vector Database..."
        with st.spinner(spinner_msg):
            try:
                # 3. Call your FastAPI Backend!
                endpoint = "http://127.0.0.1:8000/agent-query" if is_agentic else "http://127.0.0.1:8000/query"
                payload = {"question": prompt} if is_agentic else {"question": prompt, "top_k": 3}
                
                response = requests.post(endpoint, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer found.")
                    sources = data.get("sources", [])
                    
                    # 4. Format the response to show text + source info (+ agent metrics if agentic)
                    full_response = f"{answer}\n\n"
                    full_response += f"**Sources retrieved:** {', '.join(sources)}"
                    
                    if is_agentic:
                        steps = data.get("steps_taken", 0)
                        grade = data.get("grade", 0)
                        full_response += f"\n\n---\n*🤖 Agent Metadata: `{steps}` tool calls made | Self-Critique Quality Score: `{grade}/5`*"
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"API Error {response.status_code}: Make sure FastAPI is running on port 8000.")
            
            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Error: Could not connect to the Backend API. Make sure FastAPI is running on port 8000.")
