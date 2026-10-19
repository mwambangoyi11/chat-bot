import streamlit as st
import os
import pymupdf
from sentence_transformers import SentenceTransformer

st.title("Computer Science Textbook Chatbot")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-V2')

model = load_model()
st.success("AI Model is RUNNING!")

os.makedirs("books", exist_ok=True)

if "texts" not in st.session_state:
    st.session_state.texts = {}
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_emb" not in st.session_state:
    st.session_state.chunk_emb = None

# Auto-load PDFs already in books folder
book_files = [f for f in os.listdir("books") if f.endswith(".pdf")]
if book_files and not st.session_state.texts:
    for name in book_files:
        path = os.path.join("books", name)
        doc = pymupdf.open(path)
        text = "".join([p.get_text() for p in doc])
        st.session_state.texts[name] = text
    st.info(f"Loaded {len(st.session_state.texts)} books from folder")

# Index once
if st.session_state.texts and st.session_state.chunk_emb is None:
    with st.spinner("Indexing your books first time... please wait 30 sec"):
        all_text = " ".join(st.session_state.texts.values())
        # take first 400k chars to keep it fast
        all_text = all_text[:400000]
        st.session_state.chunks = [all_text[i:i+800] for i in range(0, len(all_text), 800)]
        st.session_state.chunk_emb = model.encode(st.session_state.chunks)
    st.success(f"Indexed {len(st.session_state.chunks)} pieces - Ready to answer!")

uploaded = st.file_uploader("Upload MORE PDFs (optional)", type="pdf", accept_multiple_files=True)
if uploaded:
    st.warning("New files uploaded - press R to rerun to index them")

question = st.text_input("Ask a question from your books:")

if question:
    if st.session_state.chunk_emb is None:
        st.warning("No books indexed yet")
    else:
        q_emb = model.encode([question])
        scores = (st.session_state.chunk_emb @ q_emb.T).flatten()
        best = scores.argmax()
        st.subheader("Answer from your textbooks:")
        st.write(st.session_state.chunks[best])
        st.caption(f"Score: {scores[best]:.2f}")