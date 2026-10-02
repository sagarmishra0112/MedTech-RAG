import streamlit as st
import requests

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="MedTech RAG", page_icon="🏥", layout="centered")

st.title("🏥 MedTech RAG: X-Ray Assistant")
st.markdown(
    "Ask technical questions about the Allengers 100 X-ray service manual. "
    "The AI retrieves exact diagnostic chunks and tables."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
pipeline_mode = st.sidebar.radio(
    "Pipeline:",
    ["🤖 Agentic RAG", "⚡ Classic RAG"],
    help=(
        "**Agentic**: Multi-step tool calls + Claude judge + flaw-driven retry.\n\n"
        "**Classic**: Single retrieve → generate pass."
    ),
)
is_agentic = "Agentic" in pipeline_mode

# ── Chat state ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "_processing" not in st.session_state:
    st.session_state["_processing"] = False

# Replay previously displayed messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Helper: call API and render response ─────────────────────────────────────
def ask(prompt: str):
    # Guard: prevent duplicate calls triggered by Streamlit re-renders
    if st.session_state["_processing"]:
        return
    st.session_state["_processing"] = True

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        spinner_msg = (
            "🤖 Agent thinking & invoking tools…"
            if is_agentic
            else "⚡ Searching Vector Database…"
        )
        with st.spinner(spinner_msg):
            try:
                endpoint = (
                    "http://127.0.0.1:8000/agent-query"
                    if is_agentic
                    else "http://127.0.0.1:8000/query"
                )
                payload = (
                    {"question": prompt}
                    if is_agentic
                    else {"question": prompt, "top_k": 3}
                )
                response = requests.post(endpoint, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    answer  = data.get("answer", "No answer found.")
                    sources = data.get("sources", [])

                    full_response = f"{answer}\n\n"
                    full_response += f"**Sources:** {', '.join(sources)}"

                    if is_agentic:
                        steps    = data.get("steps_taken", 0)
                        grade    = data.get("grade", 0)
                        attempts = data.get("correction_attempts", 0)
                        provider = data.get("judge_provider", "unknown")
                        conf     = data.get("confidence", "unknown")
                        critique = data.get("judge_critique", "")

                        # Confidence emoji
                        conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
                        retry_note = f" · `{attempts}` retr{'y' if attempts == 1 else 'ies'}" if attempts > 0 else ""

                        full_response += (
                            f"\n\n---\n"
                            f"*{conf_icon} Judge: `{grade}/5` ({conf}) · "
                            f"`{steps}` tool calls{retry_note} · graded by `{provider}`*"
                        )

                        # ── Only shown when the answer STILL failed after all retries ──
                        if critique:
                            full_response += (
                                f"\n\n> ⚠️ **Answer could not be corrected. Claude's diagnosis:**\n"
                                f"> `{critique}`"
                            )

                    st.markdown(full_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                else:
                    st.error(
                        f"API Error {response.status_code}: "
                        "Make sure FastAPI is running on port 8000."
                    )
            except requests.exceptions.ConnectionError:
                st.error(
                    "🚨 Connection Error: Could not reach the backend API. "
                    "Make sure FastAPI is running on port 8000."
                )

    st.session_state["_processing"] = False

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about calibration, fuses, overload settings…"):
    ask(prompt)


