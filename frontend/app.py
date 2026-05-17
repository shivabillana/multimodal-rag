import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Multimodal RAG", page_icon="📄", layout="wide")
st.title("📄 Multimodal Document Intelligence")
st.caption("Upload PDFs with charts, tables, and images — ask questions in plain English")

# Session state
if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar — upload
with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_filenames:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.uploaded_filenames.append(uploaded_file.name)
                    st.success(f"✅ {uploaded_file.name} — {data['pages_ingested']} pages ingested")
                else:
                    st.error(f"❌ Failed to upload {uploaded_file.name}")

    if st.session_state.uploaded_filenames:
        st.divider()
        st.markdown("**Active documents:**")
        for fname in st.session_state.uploaded_filenames:
            st.markdown(f"- {fname}")

        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.uploaded_filenames = []
            st.session_state.chat_history = []
            st.rerun()

# Main — chat
if not st.session_state.uploaded_filenames:
    st.info("Upload one or more PDF documents from the sidebar to get started.")
else:
    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": question,
                        "filenames": st.session_state.uploaded_filenames
                    }
                )

            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                pages_used = data["pages_used"]
                st.write(answer)
                st.caption(f"📄 Answer based on {pages_used} page(s)")
            else:
                answer = "Sorry, something went wrong. Please try again."
                st.error(answer)

        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("assistant", answer))