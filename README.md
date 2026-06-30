# 🎥 YouTube Video Q&A — RAG Application

A Retrieval-Augmented Generation (RAG) application that lets you paste any YouTube video link and ask natural language questions about its content — complete with clickable timestamps that jump straight to the relevant moment in the video.

🔴 **[Live Demo](https://youtube-qna-chatbot-rag-by-geetanshu.streamlit.app/)** &nbsp;|&nbsp; ⭐ **[GitHub Repository](https://github.com/geetanshusinghrajawat/Youtube-QnA-ChatBot-RAG/)**

> **Note:** Due to YouTube blocking transcript requests from cloud server IPs (a known restriction affecting all cloud-hosted apps using `youtube-transcript-api`), this app runs reliably on `localhost` but may fail to fetch transcripts when deployed on shared cloud infrastructure unless a residential proxy is configured. See [Known Limitations](#known-limitations) below.

---

## What it does

Paste any YouTube video URL and ask questions like:

- *"What is this video about?"*
- *"What did they say about pricing?"*
- *"Summarize the main argument."*

The app fetches the video's transcript, finds the most relevant moments for your question, and generates a grounded answer — along with clickable links to jump directly to the exact timestamps in the video where that information was discussed.

---

## How it works

```
YouTube URL  →  Extract Video ID  →  Fetch Transcript (with per-segment timestamps)
                                                ↓
                      Merge segments → Re-chunk (1000 chars, 100 overlap)
                                    → Recover nearest timestamp per chunk
                                                ↓
                              Embedding (all-MiniLM-L6-v2)
                                                ↓
                               FAISS Vector Store (in-memory, cached per video)
                                                ↓
              User Question  →  Similarity Search (top 6 chunks)
                                                ↓
                          Prompt Template + Groq LLM (Llama 3.1)
                                                ↓
                     Grounded Answer + Sorted, Deduplicated Timestamp Links
```

This is a RAG (Retrieval-Augmented Generation) pipeline adapted for video transcripts instead of documents:

1. **Indexing** — The transcript is fetched as small timestamped segments, merged into one continuous text, then re-split into properly-sized chunks. Each chunk recovers the timestamp of its original position via a checkpoint lookup, since merging text for better chunking would otherwise lose per-segment timestamp metadata.
2. **Retrieval** — The user's question is converted to a vector and FAISS finds the most semantically similar chunks.
3. **Generation** — The retrieved chunks and question are sent to Llama 3.1 via Groq API, which generates a grounded answer. The timestamps of the chunks used are shown as clickable "watch this moment" links.

---

## Tech Stack

| Component | Tool |
|---|---|
| Framework | LangChain (LCEL) |
| Transcript Source | `youtube-transcript-api` |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (free, local) |
| Vector Store | FAISS (in-memory, cached per video) |
| LLM | Llama 3.1 8B via Groq API (free) |
| Frontend | Streamlit |
| Secret Management | python-dotenv |

---

## Project Structure

```
youtube-qna-chatbot-rag/
├── app.py              ← complete RAG application
├── requirements.txt    ← all dependencies
├── .env                ← API keys (not committed to GitHub)
├── .gitignore           ← keeps secrets out of version control
└── README.md
```

---

## Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-here.git
cd youtube-qna-chatbot-rag
```

### 2. Create and activate virtual environment
```bash
conda create -n yt-rag python=3.10
conda activate yt-rag
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

### 5. Run the app
```bash
streamlit run app.py
```

---

## Key Learnings & Design Decisions

**Reconstructing timestamps after merge-and-resplit chunking** — A YouTube transcript arrives as hundreds of tiny (2–3 second) segments, each individually too short to be a useful retrieval unit. Splitting each segment independently (`split_documents()`) left chunk count unchanged and chunks too small for meaningful context. The fix: merge all segment text into one continuous string (preserving a position-to-timestamp checkpoint list while doing so), re-chunk that string into properly-sized ~1000-character pieces, then recover each new chunk's approximate timestamp by looking up the nearest checkpoint at or before its position in the merged text.

**Graceful failure handling for an unpredictable external source** — Unlike a user-uploaded PDF, a YouTube video may have captions disabled, be private, or be unavailable. All transcript fetching is wrapped in `try/except`, since there's no way to verify in advance whether a fetch will succeed.

**Caching per video ID** — `@st.cache_resource` keys the vector store build to the specific `video_id`, so asking multiple questions about the same video reuses the existing index instead of rebuilding it on every question.

**Deduplicated, sorted timestamp links** — Multiple retrieved chunks can map to the same or overlapping moments in the video. Timestamps are deduplicated and sorted chronologically before being displayed, so the "watch this moment" links are clean and easy to follow in order.

---

## Known Limitations

- **YouTube IP blocking on cloud platforms**: YouTube blocks transcript requests originating from known cloud provider IP ranges (AWS, GCP, Azure, and by extension Streamlit Cloud). This app works reliably when run locally. A production fix involves routing requests through a residential proxy service (e.g. Webshare) via `youtube-transcript-api`'s built-in proxy support.
- **Auto-generated captions only**: Videos without any captions (disabled or unavailable) cannot be processed; the app surfaces a clear error message in this case rather than crashing.
- **Single video at a time**: v1 supports Q&A on one video per session; cross-video comparison is a potential future addition.

---

## Future Improvements

- [ ] Add residential proxy support for reliable cloud deployment
- [ ] Support playlist/multi-video Q&A
- [ ] Add auto-generated video summary on load
- [ ] Add chat history for follow-up questions

---

## Author

**Geetanshu Singh Rajawat**  
ML Engineer | MSc AI & ML (ongoing)  
[LinkedIn](https://www.linkedin.com/in/your-linkedin) | [GitHub](https://github.com/your-username)
