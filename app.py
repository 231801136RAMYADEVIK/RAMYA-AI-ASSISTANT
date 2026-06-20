import streamlit as st
from groq import Groq
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
import os
import io
import base64
import tempfile

st.set_page_config(page_title="Voice Assistant", page_icon="🎙️", layout="centered")

# ---------------- Config ----------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"
STT_MODEL = "whisper-large-v3-turbo"

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set. Set it as an environment variable before running.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful, concise voice assistant. Keep replies short and conversational since they will be read aloud."}
    ]
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None
if "pending_tts" not in st.session_state:
    st.session_state.pending_tts = None

# ---------------- Core functions ----------------
def transcribe(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(path), f.read()),
                model=STT_MODEL,
            )
        return result.text.strip()
    finally:
        os.remove(path)


def get_llm_reply(user_text: str) -> str:
    st.session_state.messages.append({"role": "user", "content": user_text})
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=st.session_state.messages,
        temperature=0.7,
        max_tokens=300,
    )
    reply = response.choices[0].message.content.strip()
    st.session_state.messages.append({"role": "assistant", "content": reply})
    return reply


def synthesize_speech(text: str) -> bytes:
    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def autoplay_audio(audio_bytes: bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,
    )


def handle_user_text(user_text: str):
    with st.spinner("Thinking..."):
        reply = get_llm_reply(user_text)
    with st.spinner("Generating speech..."):
        audio_bytes = synthesize_speech(reply)
    st.session_state.pending_tts = audio_bytes


# ---------------- UI ----------------
st.title("🎙️ Voice Assistant")
st.caption("Powered by Groq (Whisper + Llama 3.3 70B)")

chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    audio = mic_recorder(
        start_prompt="🎤 Speak",
        stop_prompt="⏹ Stop",
        just_once=True,
        use_container_width=True,
        key="recorder",
    )

with col2:
    typed_text = st.chat_input("Or type your message...")

# Handle mic input
if audio and audio.get("bytes"):
    audio_id = audio.get("id", str(len(audio["bytes"])))
    if audio_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio_id
        with st.spinner("Transcribing..."):
            user_text = transcribe(audio["bytes"])
        if user_text:
            with st.chat_message("user"):
                st.write(user_text)
            handle_user_text(user_text)
            st.rerun()

# Handle typed input
if typed_text:
    with st.chat_message("user"):
        st.write(typed_text)
    handle_user_text(typed_text)
    st.rerun()

# Play pending TTS reply (after rerun, last assistant message is on screen)
if st.session_state.pending_tts:
    autoplay_audio(st.session_state.pending_tts)
    st.session_state.pending_tts = None

if st.button("🗑️ Clear conversation"):
    st.session_state.messages = st.session_state.messages[:1]
    st.session_state.pending_tts = None
    st.session_state.last_audio_id = None
    st.rerun()