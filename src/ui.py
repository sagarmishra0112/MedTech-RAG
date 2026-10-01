import streamlit as st
import requests

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="MedTech RAG", page_icon="🏥", layout="centered")

st.title("🏥 MedTech RAG: X-Ray Assistant")
st.markdown(
    "Ask technical questions about X-ray documentation. "
    "The AI retrieves exact diagnostic chunks and tables."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Pipeline Mode")
pipeline_mode = st.sidebar.radio(
    "Select RAG Pipeline:",
    ["🤖 Agentic RAG (/agent-query)", "⚡ Classic RAG (/query)"],
    help=(
        "Agentic RAG uses multi-step tool calls, a cross-provider Claude judge, "
        "and flaw-type–driven retry. Classic RAG is a single pass."
    ),
)
is_agentic = "Agentic" in pipeline_mode

# ── Low-score replay panel (only visible in Agentic mode) ───────────────────
LOW_SCORE_QUERIES = [
    "What is the output power of the Allengers 100 X-Ray generator?",
    "What is the kVp range of the machine?",
    "What is the timer range of the Allengers 100?",
    "What mAs value is required when running at 100 mA and 80 kVp?",
    "For a 100 mA, 60 kVp exposure, what mAs should be used?",
    "What is the most important parameter to monitor during preventive maintenance?",
]

if is_agentic:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔁 Replay Low-Score Queries")
    st.sidebar.caption(
        "These questions scored 0 in the last RAGAS eval. "
        "Click one to inject it into the chat and see Claude's verdict."
    )
    for q in LOW_SCORE_QUERIES:
        if st.sidebar.button(q[:55] + ("…" if len(q) > 55 else ""), key=f"replay_{q[:20]}"):
            st.session_state["injected_query"] = q

# ── Chat state ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay previously displayed messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Helper: call API and render response ─────────────────────────────────────
def ask(prompt: str):
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
                    full_response += f"**Sources retrieved:** {', '.join(sources)}"

                    if is_agentic:
                        steps    = data.get("steps_taken", 0)
                        grade    = data.get("grade", 0)
                        attempts = data.get("correction_attempts", 0)
                        provider = data.get("judge_provider", "unknown")
                        conf     = data.get("confidence", "unknown")
                        critique = data.get("judge_critique", "")

                        # ── Core agentic metadata line ──
                        full_response += (
                            f"\n\n---\n"
                            f"*🤖 Agent · `{steps}` tool calls · "
                            f"Judge score: `{grade}/5` (`{conf}`) · "
                            f"Retries: `{attempts}` · "
                            f"Judge: `{provider}`*"
                        )

                        # ── Judge critique block — only shown when a retry occurred ──
                        if critique:
                            full_response += (
                                f"\n\n> **🔍 Claude's rejection diagnosis:**\n"
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
                    "🚨 Connection Error: Could not reach the backend. "
                    "Make sure FastAPI is running on port 8000."
                )

# ── Handle sidebar replay injection ─────────────────────────────────────────
if "injected_query" in st.session_state:
    injected = st.session_state.pop("injected_query")
    ask(injected)

# ── Normal chat input ────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about Calibration, Overload Settings, etc…"):
    ask(prompt)

