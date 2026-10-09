"""
Rui - a warm, expressive AI chat assistant.
Optimized for Hugging Face Spaces FREE CPU Basic tier (2 vCPU, 16 GB RAM).

Model : bartowski/Llama-3.2-1B-Instruct-GGUF  (Llama-3.2-1B-Instruct-Q4_K_M.gguf)
Engine: llama-cpp-python (CPU inference)
UI    : Gradio ChatInterface
"""

import os

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
REPO_ID = "bartowski/Llama-3.2-1B-Instruct-GGUF"
FILENAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

N_CTX = 2048          # keep memory usage low
N_THREADS = 2         # free tier = 2 vCPUs
MAX_NEW_TOKENS = 512  # reply budget (history is trimmed to fit around this)

# Monetization placeholders -- replace via Space "Variables" or edit directly
BMC_URL = os.getenv("BMC_URL", "https://www.buymeacoffee.com/your-username")
SPONSOR_URL = os.getenv("SPONSOR_URL", "https://github.com/sponsors/your-username")
UPI_ID = os.getenv("UPI_ID", "yourname@upi")

SYSTEM_PROMPT = """You are Rui, a warm, expressive and helpful AI assistant.

Rules you must always follow:
1. LANGUAGE: Reply in the exact same language and script the user writes in. \
If they write Punjabi in Gurmukhi (ਪੰਜਾਬੀ), answer in Punjabi in Gurmukhi. \
If they write Hindi in Devanagari (हिन्दी), answer in Hindi in Devanagari. \
If they write Hinglish or Roman-script Punjabi, answer in the same Roman script. \
Never switch to English unless the user does.
2. EMOTION: Notice how the user feels and match it. Be cheerful when they are \
happy, gentle and comforting when they are sad or stressed, and enthusiastic \
when they are excited.
3. STYLE: Be friendly, natural and conversational. Keep answers clear and \
reasonably short unless the user asks for detail. Use an occasional emoji when it fits.
4. HONESTY: If you are not sure about something, say so instead of making things up."""

# --------------------------------------------------------------------------
# Model loading (runs once at startup)
# --------------------------------------------------------------------------
print("Downloading / locating model...")
MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

print("Loading model into memory...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_batch=256,
    use_mmap=True,
    verbose=False,
)
print("Model ready.")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _text_of(content) -> str:
    """Gradio may pass message content as a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(parts)
    return str(content or "")


def _count_tokens(text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))


def build_messages(message: str, history: list) -> list:
    """System prompt + as much recent history as fits inside the context window."""
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    user_msg = {"role": "user", "content": message}

    # Reserve room for system prompt, new message, reply, and chat-template overhead
    budget = (
        N_CTX
        - MAX_NEW_TOKENS
        - _count_tokens(SYSTEM_PROMPT)
        - _count_tokens(message)
        - 64
    )

    kept = []
    for msg in reversed(history):
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _text_of(msg.get("content"))
        cost = _count_tokens(text) + 8
        if cost > budget:
            break
        budget -= cost
        kept.append({"role": role, "content": text})
    kept.reverse()

    return [system_msg, *kept, user_msg]


# --------------------------------------------------------------------------
# Chat handler (streaming)
# --------------------------------------------------------------------------
def chat(message: str, history: list):
    messages = build_messages(message, history)

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stream=True,
    )

    partial = ""
    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        token = delta.get("content")
        if token:
            partial += token
            yield partial


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
CSS = """
.gradio-container {max-width: 1100px !important; margin: auto;}
#rui-side {border-radius: 14px; padding: 8px 14px;}
"""

SIDEBAR_MD = f"""
## 🌸 Meet Rui
Rui is a warm, expressive AI friend who understands your mood and replies in
**your own language and script** — English, Hindi (हिन्दी), Punjabi (ਪੰਜਾਬੀ) and more.

---
### ☕ Support Rui
Rui is free to use. If it made your day better, you can help keep it running:

- [☕ Buy Me a Coffee]({BMC_URL})
- [💖 Sponsor on GitHub]({SPONSOR_URL})
- UPI: `{UPI_ID}`

---
### 📢 Your Ad Here
Interested in sponsoring or advertising on Rui?
Contact: `your-email@example.com`

---
<sub>Runs on a small 1B-parameter model on free CPU hardware, so replies can be
slow and may occasionally be imperfect — especially in Punjabi and Hindi.</sub>
"""

EXAMPLES = [
    "Hi Rui! Tell me about yourself 😊",
    "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਰੂਈ, ਅੱਜ ਮੈਂ ਬਹੁਤ ਖੁਸ਼ ਹਾਂ!",
    "नमस्ते रूई, आज मन थोड़ा उदास है।",
    "Give me 3 tips to stay focused while studying.",
]

with gr.Blocks(title="Rui — AI Chat", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown("# 🌸 Rui — Your Warm AI Companion")
    with gr.Row():
        with gr.Column(scale=1, min_width=260, elem_id="rui-side"):
            gr.Markdown(SIDEBAR_MD)
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=chat,
                type="messages",
                chatbot=gr.Chatbot(type="messages", height=520, show_copy_button=True),
                textbox=gr.Textbox(placeholder="Message Rui in any language...", scale=7),
                examples=EXAMPLES,
                cache_examples=False,
            )

if __name__ == "__main__":
    # queue() keeps requests orderly so the 2 vCPUs aren't overloaded
    demo.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=7860
    )
