import time
import json
import re
import requests
import psutil
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 1. GPU Telemetry Initialization
# ─────────────────────────────────────────────────────────────
HAS_GPU = False
try:
    import pynvml
    pynvml.nvmlInit()
    if pynvml.nvmlDeviceGetCount() > 0:
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
        gpu_name = gpu_name.decode("utf-8") if isinstance(gpu_name, bytes) else gpu_name
        HAS_GPU = True
except Exception:
    gpu_handle, gpu_name = None, "N/A (CPU Only)"


def get_hardware_telemetry():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    if HAS_GPU and gpu_handle:
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            return {
                "cpu": cpu, "ram_used": ram.used / (1024**3), "ram_total": ram.total / (1024**3),
                "ram_pct": ram.percent, "gpu_util": util.gpu, "temp": temp,
                "vram_used": mem.used / (1024**2), "vram_total": mem.total / (1024**2),
                "vram_pct": (mem.used / mem.total) * 100, "gpu_name": gpu_name
            }
        except Exception:
            pass
    return {
        "cpu": cpu, "ram_used": ram.used / (1024**3), "ram_total": ram.total / (1024**3),
        "ram_pct": ram.percent, "gpu_util": 0, "temp": 0, "vram_used": 0, "vram_total": 0,
        "vram_pct": 0, "gpu_name": "N/A"
    }


def extract_thinking_and_response(text):
    """Separates <think>...</think> reasoning from the final answer."""
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        response = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return thinking, response
    elif "<think>" in text:
        parts = text.split("<think>", 1)
        return parts[1].strip(), parts[0].strip()
    return None, text.strip()


# ─────────────────────────────────────────────────────────────
# 2. UI Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="LLM Chat & Telemetry", page_icon="⚡", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Telemetry
with st.sidebar:
    st.title("⚡ Telemetry")
    try:
        models = [m["name"] for m in requests.get("http://localhost:11434/api/tags", timeout=3).json().get("models", [])]
    except Exception:
        models = []
    selected_model = st.selectbox("Model", models if models else ["qwen3.5:9b"])

    with st.expander("⚙️ Settings", expanded=False):
        system_prompt = st.text_area("System Prompt", "You are a helpful and concise technical assistant.")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    stats = get_hardware_telemetry()
    st.caption("🖥️ **System Resources**")
    st.metric("CPU Load", f"{stats['cpu']:.1f} %")
    st.metric("System RAM", f"{stats['ram_used']:.2f} / {stats['ram_total']:.1f} GB ({stats['ram_pct']}%)")
    st.progress(stats['ram_pct'] / 100.0)

    st.caption(f"🎮 **GPU ({stats['gpu_name'][:14]})**")
    c1, c2 = st.columns(2)
    c1.metric("Compute", f"{stats['gpu_util']} %")
    c2.metric("Temp", f"{stats['temp']} °C")
    st.metric("VRAM", f"{stats['vram_used']:.0f} / {stats['vram_total']:.0f} MB", f"{stats['vram_pct']:.1f}%")
    if stats['vram_total'] > 0:
        st.progress(min(stats['vram_pct'] / 100.0, 1.0))


# ─────────────────────────────────────────────────────────────
# 3. Main Chat with Collapsible Thoughts & Metrics
# ─────────────────────────────────────────────────────────────
st.title("💬 Chat with Thought Inspector")
st.caption(f"Active Model: **{selected_model}** | Turns: **{len(st.session_state.messages)}**")

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            thinking = msg.get("thinking")
            if thinking:
                with st.expander("💭 Thought Process (Click to Expand)", expanded=False):
                    st.markdown(f"```text\n{thinking}\n```")
            
            st.markdown(msg["content"])
            
            if "metrics" in msg and msg["metrics"]:
                m = msg["metrics"]
                with st.expander("📊 Turn Eval Metrics", expanded=False):
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("⚡ Gen Speed", f"{m['gen_tps']:.1f} tok/s")
                    c2.metric("⏱️ TTFT", f"{m['ttft_ms']:.0f} ms")
                    c3.metric("📥 Prompt Speed", f"{m['prompt_tps']:.1f} tok/s")
                    c4.metric("🧠 Context Tokens", f"{m['context_tokens']}")
                    c5.metric("⏳ Latency", f"{m['total_time_s']:.2f} s")
                    st.caption(f"Peak VRAM: **{m['peak_vram_mb']:.0f} MB** | Output: **{m['eval_count']} tokens**")
        else:
            st.markdown(msg["content"])

# User Input
if user_prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    payload_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    for m in st.session_state.messages:
        # Pass clean assistant response to context history
        payload_messages.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        thought_container = st.empty()
        response_container = st.empty()
        
        raw_stream = ""
        ttft_recorded, ttft_ms, t_start = False, 0.0, time.perf_counter()
        final_metrics = {}

        try:
            res = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": selected_model,
                    "messages": payload_messages,
                    "options": {"temperature": temperature},
                    "stream": True
                },
                stream=True,
                timeout=120
            )

            for line in res.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    
                    # Capture reasoning from thinking tag or field
                    msg_obj = chunk.get("message", {})
                    token = msg_obj.get("content", "")
                    
                    if not ttft_recorded and token:
                        ttft_ms = (time.perf_counter() - t_start) * 1000
                        ttft_recorded = True

                    raw_stream += token
                    
                    curr_thinking, curr_response = extract_thinking_and_response(raw_stream)
                    
                    # Show live streaming thought inside expander
                    if curr_thinking:
                        thought_container.markdown(
                            f"🧠 *Thinking Process:*\n> {curr_thinking.replace(chr(10), chr(10)+'> ')}"
                        )
                    if curr_response:
                        response_container.markdown(curr_response + "▌")

                    if chunk.get("done", False):
                        total_time = time.perf_counter() - t_start
                        ec, ed = chunk.get("eval_count", 0), chunk.get("eval_duration", 1)
                        pc, pd = chunk.get("prompt_eval_count", 0), chunk.get("prompt_eval_duration", 1)

                        final_metrics = {
                            "ttft_ms": ttft_ms,
                            "gen_tps": (ec / (ed / 1e9)) if ed > 0 else 0,
                            "prompt_tps": (pc / (pd / 1e9)) if pd > 0 else 0,
                            "total_time_s": total_time,
                            "eval_count": ec,
                            "context_tokens": pc + ec,
                            "peak_vram_mb": get_hardware_telemetry()["vram_used"]
                        }

            # Finalize clean render
            final_thinking, final_answer = extract_thinking_and_response(raw_stream)
            thought_container.empty()
            
            if final_thinking:
                with st.expander("💭 Thought Process (Click to Expand)", expanded=False):
                    st.markdown(f"```text\n{final_thinking}\n```")
            
            response_container.markdown(final_answer if final_answer else raw_stream)

            if final_metrics:
                with st.expander("📊 Turn Eval Metrics", expanded=False):
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("⚡ Gen Speed", f"{final_metrics['gen_tps']:.1f} tok/s")
                    c2.metric("⏱️ TTFT", f"{final_metrics['ttft_ms']:.0f} ms")
                    c3.metric("📥 Prompt Speed", f"{final_metrics['prompt_tps']:.1f} tok/s")
                    c4.metric("🧠 Context Tokens", f"{final_metrics['context_tokens']}")
                    c5.metric("⏳ Latency", f"{final_metrics['total_time_s']:.2f} s")
                    st.caption(f"Peak VRAM: **{final_metrics['peak_vram_mb']:.0f} MB** | Generated: **{final_metrics['eval_count']} tokens**")

            st.session_state.messages.append({
                "role": "assistant",
                "thinking": final_thinking,
                "content": final_answer if final_answer else raw_stream,
                "metrics": final_metrics
            })

        except Exception as e:
            st.error(f"Error: {e}")
