import streamlit as st
import os
import pymupdf
from sentence_transformers import SentenceTransformer
import numpy as np

st.title("Computer Science Textbook Chatbot")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-V2')

model = load_model()

os.makedirs("books", exist_ok=True)

if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.sources = []
    st.session_state.chunk_emb = None

# Load each book separately and take chunks from each
book_files = [f for f in os.listdir("books") if f.endswith(".pdf")]
if book_files and st.session_state.chunk_emb is None:
    with st.spinner(f"Indexing {len(book_files)} books properly..."):
        all_chunks = []
        all_sources = []
        for book_name in book_files:
            doc = pymupdf.open(os.path.join("books", book_name))
            book_text = ""
            for i, page in enumerate(doc):
                txt = page.get_text()
                if i < 5 and "Copyright" in txt: continue
                book_text += txt + "\n"
            # Take 100 chunks from EACH book, not just first book
            book_text = book_text[:120000] # 120k per book
            chunks = [book_text[j:j+1000] for j in range(0, len(book_text), 800)]
            for c in chunks:
                if len(c) > 300 and "Copyright" not in c and "reproduced" not in c:
                    all_chunks.append(c)
                    all_sources.append(book_name)
        
        st.session_state.chunks = all_chunks
        st.session_state.sources = all_sources
        # IMPORTANT: normalize for real cosine similarity
        st.session_state.chunk_emb = model.encode(all_chunks, normalize_embeddings=True)
    st.success(f"Indexed {len(all_chunks)} chunks from {len(book_files)} books!")

question = st.text_input("Ask a question from your books:")
if question and st.session_state.chunk_emb is not None:
    q_emb = model.encode([question], normalize_embeddings=True)
    scores = (st.session_state.chunk_emb @ q_emb.T).flatten()
    
    # Get top 3 DIFFERENT chunks
    top_idx = np.argsort(scores)[-3:][::-1]
    
    st.subheader("Answer from your textbooks:")
    used_text = set()
    for idx in top_idx:
        txt = st.session_state.chunks[idx]
        if txt[:100] not in used_text: # avoid duplicate same answer
            st.markdown(f"**From {st.session_state.sources[idx]} | Score {scores[idx]:.2f}**")
            st.write(txt)
            st.divider()
            used_text.add(txt[:100])

    if max(scores) < 0.4:
        st.warning("Low relevance - this topic may not be in your 4 books. Try: web development, vue.js, cloud, etc.")