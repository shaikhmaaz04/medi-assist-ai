import json
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
CHROMA_DIR = "chroma_store"
COLLECTION = "fasting_research"
JSON_IN_PATH = "pubmed_articles_backup.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def run_ingestion():
    print(f"📂 Loading data from {JSON_IN_PATH}...")
    try:
        with open(JSON_IN_PATH, "r", encoding="utf-8") as f:
            full_articles = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {JSON_IN_PATH} not found.")
        return

    all_chunks = []
    
    # --- IMPROVEMENT: Larger chunk size for better medical context ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    print(f"🧩 Chunking {len(full_articles)} articles...")

    for art in full_articles:
        content_for_ai = f"STUDY: {art['title']}\nFINDINGS: {art['full_abstract']}"
        
        chunks = text_splitter.split_text(content_for_ai)
        for i, chunk in enumerate(chunks):
            all_chunks.append(Document(
                page_content=chunk,
                metadata={
                    "pmid": art['pmid'],
                    "title": art['title'],  # <-- NEW: Storing Title in Metadata
                    "citation": art['citation'],
                    "journal": art['journal'],
                    "category": art['category'],
                    "chunk_id": i
                }
            ))

    print(f"🧠 Storing {len(all_chunks)} chunks in Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    print(f"✨ Success! Database updated with {len(all_chunks)} chunks.")

if __name__ == "__main__":
    run_ingestion()