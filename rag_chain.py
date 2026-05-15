import os
from operator import itemgetter
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
load_dotenv()
CHROMA_DIR = "chroma_store"
COLLECTION = "fasting_research"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

def format_docs(docs):
    """Injects Title, Citation, AND PMID into the context block."""
    formatted_context = []
    for doc in docs:
        title = doc.metadata.get('title', 'Unknown Title')
        citation = doc.metadata.get('citation', 'Unknown Source')
        pmid = doc.metadata.get('pmid', 'N/A')
        
        formatted_context.append(
            f"PAPER TITLE: {title}\n"
            f"SOURCE: {citation}\n"
            f"PMID: {pmid}\n"
            f"CONTENT: {doc.page_content}"
        )
    return "\n\n".join(formatted_context)

def get_mediassist_chain():
    if not os.path.exists(CHROMA_DIR):
        print(f"⚠️ Warning: Chroma directory '{CHROMA_DIR}' not found. Ensure vector DB is initialized.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(model_name=LLM_MODEL, temperature=0.4, max_tokens=5000)

    # UPDATED SYSTEM PROMPT: 
    # 1. Changed inline citations to ONLY use the PMID to save space and look cleaner.
    # 2. Maintained the strict rule against carrying over previous references.
    system_prompt = (
        "You are MediAssist AI, a highly accurate clinical research assistant. "
        "Use the CURRENT CONTEXT and previous chat history to answer the user's question.\n\n"
        "RESPONSE GUIDELINES:\n"
        "1. SYNTHESIS: Merge findings into a cohesive, professional narrative.\n"
        "2. CITATIONS: Every clinical claim must have a concise inline citation using ONLY the ID, formatted exactly as: [PMID: XXXXXX]. Do not include the title or authors inline.\n"
        "3. DETAIL: Include specific metrics when available (e.g., HbA1c % changes, weight loss kg).\n"
        "4. CLINICAL SUMMARY: Conclude with a 2-3 sentence 'Executive Summary'.\n"
        "5. REFERENCES: Provide a numbered list of ONLY the papers cited in this specific response at the very end, including their title and authors. Do NOT copy, carry over, or include references from the previous chat history.\n"
        "6. GROUNDING: If the CURRENT CONTEXT lacks the information, explicitly state: 'The current database lacks this specific information.' Do NOT invent facts.\n\n"
        "CURRENT CONTEXT:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    setup_and_retrieval = RunnableParallel({
        "context": itemgetter("input") | retriever | format_docs, 
        "input": itemgetter("input"), 
        "chat_history": itemgetter("chat_history"),
        "raw_docs": itemgetter("input") | retriever
    })

    rag_chain = setup_and_retrieval | {
        "answer": prompt | llm | StrOutputParser(),
        "docs": itemgetter("raw_docs")
    }
    
    return rag_chain