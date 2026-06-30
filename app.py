import os
import re
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.title("🎥 YouTube Video Q&A")

# ── Extract video ID from any YouTube URL format ─────────
def extract_video_id(url):
    patterns = [
        r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# ── Fetch transcript and convert to Documents with timestamps ──
def fetch_transcript(video_id):
    try:
        yt_api = YouTubeTranscriptApi()
        transcript = yt_api.fetch(video_id)
    except Exception as e:
        return None, str(e)

    documents = []
    for segment in transcript:
        doc = Document(
            page_content=segment.text,
            metadata={"start": segment.start, "video_id": video_id}
        )
        documents.append(doc)

    return documents, None

# ── Merge segments into one text, then split with timestamp lookup ──
def chunk_documents(documents):
    # Step 1 - Join all segment text together, tracking where each segment starts
    full_text = ""
    checkpoints = []  # list of (character_position, timestamp)

    for doc in documents:
        checkpoints.append((len(full_text), doc.metadata["start"]))
        full_text += doc.page_content + " "

    # Step 2 - Split the joined text into chunks (like your notebook did)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    text_chunks = splitter.split_text(full_text)

    # Step 3 - For each chunk, find its starting position in full_text,
    # then look up the closest timestamp checkpoint
    chunks_with_metadata = []
    search_start = 0

    for chunk_text in text_chunks:
        chunk_pos = full_text.find(chunk_text, search_start)
        if chunk_pos == -1:
            chunk_pos = full_text.find(chunk_text)  # fallback

        # find the last checkpoint at or before this position
        timestamp = 0
        for pos, ts in checkpoints:
            if pos <= chunk_pos:
                timestamp = ts
            else:
                break

        chunks_with_metadata.append(
            Document(
                page_content=chunk_text,
                metadata={"start": timestamp, "video_id": documents[0].metadata["video_id"]}
            )
        )
        search_start = chunk_pos + 1

    return chunks_with_metadata

# ── Load embedding model once ─────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ── Build vector store from video ID (cached per video) ──
@st.cache_resource
def build_vector_store(video_id):
    documents, error = fetch_transcript(video_id)
    if error:
        return None, error

    chunks = chunk_documents(documents)
    embeddings = load_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, None

# ── Main UI ────────────────────────────────────────────────
video_url = st.text_input("Paste a YouTube URL")

if video_url:
    video_id = extract_video_id(video_url)

    if video_id is None:
        st.error("Invalid YouTube URL. Please check and try again.")
    else:
        with st.spinner("Fetching transcript and building index..."):
            vector_store, error = build_vector_store(video_id)

        if error:
            st.error(f"Could not fetch transcript: {error}")
        else:
            st.success("Video processed! Ask your question below.")

            question = st.text_input("Ask a question about the video:")

            if question:
                prompt_template = PromptTemplate.from_template("""
Use the following context from a YouTube video transcript to answer the question.
Be specific and accurate.
If the answer is not in the context, say "I could not find this information in the video."

Context:
{context}

Question: {question}

Answer:
""")

                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    api_key=os.getenv("GROQ_API_KEY")
                )

                retriever = vector_store.as_retriever(search_kwargs={"k": 6})

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                with st.spinner("Thinking..."):
                    answer = chain.invoke(question)
                    source_docs = retriever.invoke(question)

                st.markdown("### Answer")
                st.write(answer)

                # ── Show clickable timestamp links, deduplicated and sorted ──
                st.markdown("### Watch the relevant moments")
                timestamps = sorted(set(doc.metadata["start"] for doc in source_docs))

                for ts in timestamps:
                    seconds = int(ts)
                    minutes = seconds // 60
                    secs = seconds % 60
                    link = f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"
                    st.markdown(f"- [{minutes}:{secs:02d}]({link})")