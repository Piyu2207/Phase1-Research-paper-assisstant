import io
import requests
import streamlit as st

st.set_page_config(page_title="Single-Source RAG", page_icon="📄", layout="wide")
st.title("📄 Single-Source RAG Assistant")
st.caption("PDF → local embeddings → FAISS → query routing → cross-encoder reranking → Gemini")

api_url = st.sidebar.text_input("Backend URL", "http://localhost:8000")
uploaded = st.sidebar.file_uploader("Upload one PDF", type=["pdf"])

if uploaded and st.sidebar.button("Load PDF", type="primary"):
    with st.spinner("Extracting, chunking, embedding and indexing..."):
        response = requests.post(
            f"{api_url}/ingest",
            files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            timeout=120,
        )
    if response.ok:
        st.session_state["indexed_file"] = uploaded.name
        st.sidebar.success(f"Indexed {uploaded.name}")
    else:
        st.sidebar.error(response.text)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(f"**{source['id']} — page {source['page']}**")
                    st.caption(source["snippet"])
                    st.divider()
        elif message.get("retrieved_sources"):
            with st.expander("🔎 Retrieved candidates (not used as evidence)"):
                for source in message["retrieved_sources"]:
                    st.markdown(f"**{source['id']} — page {source['page']}**")
                    st.caption(source["snippet"])
                    st.divider()

question = st.chat_input("Ask a question about the uploaded PDF...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking and generating..."):
            response = requests.post(
                f"{api_url}/query",
                json={"question": question},
                timeout=120,
            )
        if response.ok:
            data = response.json()
            st.markdown(data["answer"])
            if data.get("sources"):
                with st.expander("📚 Sources"):
                    for source in data["sources"]:
                        st.markdown(f"**{source['id']} — page {source['page']}**")
                        st.caption(source["snippet"])
            elif data.get("retrieved_sources"):
                with st.expander("🔎 Retrieved candidates (not used as evidence)"):
                    st.caption("The retriever found these chunks, but the answer was not supported by them. They are shown for transparency and are not presented as citations.")
                    for source in data["retrieved_sources"]:
                        st.markdown(f"**{source['id']} — page {source['page']}**")
                        st.caption(source["snippet"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": data.get("sources", []),
                "retrieved_sources": data.get("retrieved_sources", []),
            })
        else:
            st.error(response.text)
