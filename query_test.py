import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
COLLECTION = "fasting_research"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def test_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION
    )

    test_queries = [
        "What does the evidence say about 16:8 time-restricted eating vs standard calorie restriction for T2DM remission?",
        "Impact of intermittent fasting on insulin sensitivity markers and HbA1c levels in obese adults.",
        "Are there reported risks of hypoglycemia or electrolyte imbalances for diabetic patients on fasting protocols?",
        "What are the primary barriers to patient adherence for long-term intermittent fasting?"
    ]

    print("\n" + "="*80)
    print("🏥 MEDI-ASSIST AI: CLINICAL KNOWLEDGE RETRIEVAL TEST")
    print("="*80)

    for query in test_queries:
        print(f"\n🔎 QUERY: '{query}'")
        results = vectorstore.similarity_search_with_relevance_scores(query, k=3)

        for i, (doc, score) in enumerate(results):
            print(f"\n   📍 [RANK {i+1}] | Score: {score:.4f}")
            # --- IMPROVEMENT: Printing Title from Metadata ---
            print(f"   📘 TITLE: {doc.metadata.get('title')}")
            print(f"   📖 CITATION: {doc.metadata.get('citation')}")
            print(f"   📚 JOURNAL: {doc.metadata.get('journal')}")
            content = doc.page_content.replace('\n', ' ').strip()
            print(f"   📝 EXCERPT: {content[:500]}...") 
            print("   " + "."*20)

if __name__ == "__main__":
    test_knowledge_base()