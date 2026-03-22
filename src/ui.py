import streamlit as st
import requests

# Set page config for a cleaner look
st.set_page_config(page_title="FinanceRAG", page_icon="📈", layout="centered")

# Custom UI Styling
st.title("📈 FinanceRAG Assistant")
st.markdown("Ask questions about your enterprise documents. The AI retrieves exact chunks and tables.")

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
        with st.spinner("Searching Vector Database..."):
            try:
                # 3. Call your FastAPI Backend! (Assuming it's running on port 8000)
                response = requests.post(
                    "http://127.0.0.1:8000/query",
                    json={"question": prompt, "top_k": 3}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer found.")
                    sources = data.get("sources", [])
                    
                    # 4. Format the response to show the text + source info
                    full_response = f"{answer}\n\n"
                    full_response += f"**Sources retrieved:** {', '.join(sources)}"
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"API Error {response.status_code}: Make sure FastAPI is running on port 8000.")
            
            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Error: Could not connect to the Backend API. Make sure you ran 'python -m uvicorn src.api:app' in another terminal.")
