# frontend/app.py
import streamlit as st
import requests
from datetime import datetime

# ======================
# Page config
# ======================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ======================
# State initialization
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "transcribed_audio_text" not in st.session_state:
    st.session_state.transcribed_audio_text = None
if "transcribed_audio_filename" not in st.session_state:
    st.session_state.transcribed_audio_filename = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Backend API URL
BACKEND_URL = "http://localhost:8000/api/v1"

# ======================
# Header
# ======================
st.title("🤖 RAG Chatbot")
st.caption("Chat with your PDF, Audio, and Video (under 15MB) files")

# ======================
# Sidebar (Upload & Clear)
# ======================
with st.sidebar:
    st.header("📁 Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF, Audio, or Video files",
        type=["pdf", "mp3", "mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("🚀 Process Files"):
            for uploaded_file in uploaded_files:
                # Check if the file has already been processed in the current session
                if uploaded_file.name not in st.session_state.uploaded_files:
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{BACKEND_URL}/upload", files=files)

                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ {result['message']} for {uploaded_file.name}")

                            st.session_state.uploaded_files.append(uploaded_file.name)

                            if result.get("transcribed_text"):
                                st.session_state.transcribed_audio_text = result["transcribed_text"]
                                st.session_state.transcribed_audio_filename = uploaded_file.name
                        else:
                            st.error(f"Error uploading {uploaded_file.name}: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Connection error: {str(e)}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.transcribed_audio_text = None
        st.session_state.uploaded_files = []
        st.rerun()

    if st.button("🗑️ Clear DB"):
        try:
            response = requests.post(f"{BACKEND_URL}/clear_db")
            if response.status_code == 200:
                st.success("✅ Vector store cleared successfully!")
                st.session_state.uploaded_files = []
            else:
                st.error(f"Error clearing vector store: {response.text}")
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")

# ======================
# Tabs
# ======================
tab_chat, tab_files, tab_transcriptions, tab_help = st.tabs(
    ["💬 Chat", "📂 Files", "🎤 Transcriptions", "ℹ️ Help"]
)

# ----------------------
# Chat Tab
# ----------------------
with tab_chat:
    st.subheader("Chat with your documents")

    # Display messages
    for msg in st.session_state.messages:
        timestamp = datetime.now().strftime("%I:%M %p")
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
                st.caption(timestamp)
        else:
            with st.chat_message("assistant"):
                if msg.get("error"):
                    st.error(msg["content"])
                else:
                    st.markdown(msg["content"])
                    st.caption(timestamp)
                    if msg.get("detailed_sources"):
                        with st.expander("📚 Sources"):
                            for src in msg["detailed_sources"]:
                                score_text = f" (Relevance: {src['relevance_score']:.3f})" if src.get("relevance_score") else ""
                                st.write(f"- {src['source']}{score_text}")

    # Chat input (always visible)
    user_input = st.chat_input("Ask a question...")
    if user_input:
        # Append user's message to display immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        try:
            with st.spinner("🤔 Thinking..."):
                # Send query and session_id to the backend. Backend manages chat_history.
                response = requests.post(
                    f"{BACKEND_URL}/query", 
                    json={
                        "query": user_input,
                        "session_id": st.session_state.session_id
                    }
                )
            if response.status_code == 200:
                result = response.json()
                # Update session_id if provided by the backend (first query)
                if "session_id" in result:
                    st.session_state.session_id = result["session_id"]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "detailed_sources": result.get("detailed_sources", [])
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"API Error: {response.text}",
                    "error": True
                })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Connection error: {str(e)}",
                "error": True
            })
        st.rerun()

# ----------------------
# Files Tab
# ----------------------
with tab_files:
    st.subheader("Uploaded Files")
    if st.session_state.uploaded_files:
        for file in st.session_state.uploaded_files:
            st.write(f"📄 {file}")
    else:
        st.info("No files uploaded yet.")

# ----------------------
# Transcriptions Tab
# ----------------------
with tab_transcriptions:
    st.subheader("Transcribed Audio")
    if st.session_state.transcribed_audio_text:
        st.info(f"**File:** {st.session_state.transcribed_audio_filename}")
        st.write(st.session_state.transcribed_audio_text)
    else:
        st.info("No audio transcriptions yet.")

# ----------------------
# Help Tab
# ----------------------
with tab_help:
    st.subheader("How to use this RAG Chatbot")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **1. Upload Files**  
        📄 PDF documents  
        🎵 Audio files (MP3, WAV, M4A, OGG)  
        🎥 Video files (MP4, AVI, MOV, MKV)
        """)

    with col2:
        st.markdown("""
        **2. Process Files**  
        Click the 'Process File' button to extract text and prepare for chatting
        """)

    with col3:
        st.markdown("""
        **3. Ask Questions**  
        Type questions about your documents and get AI-powered answers with source references
        """)

# ======================
# Footer
# ======================
st.markdown("""
---
Built with ❤️ using Streamlit, FastAPI, and Hugging Face models
""")
