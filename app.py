import streamlit as st
import os
import re # Added for Regex extraction
from dotenv import load_dotenv

from rag_chain import get_mediassist_chain
from router import get_route
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MediAssist AI | Clinical Research Assistant",
    page_icon="🏥",
    layout="wide"
)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    .source-box { 
        background-color: #f0f2f6; 
        padding: 10px; 
        border-radius: 5px; 
        border-left: 5px solid #007bff;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZATION & CACHING ---
@st.cache_resource
def load_rag_chain():
    return get_mediassist_chain()

@st.cache_resource
def load_chitchat_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏥 MediAssist AI")
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

    st.divider()
    st.write("**Cited Evidence PMIDs:**") # Renamed for clarity
    sources_placeholder = st.empty()

def update_sidebar_sources(pmids):
    sources_placeholder.empty()
    with sources_placeholder.container():
        if pmids:
            for pmid in pmids:
                if pmid and pmid != 'N/A':
                    st.markdown(f"🔗 [PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
        else:
            st.write("No sources cited yet.")

update_sidebar_sources(st.session_state.last_sources)

# --- MAIN INTERFACE ---
st.title("Clinical Research Assistant")

if not os.environ.get("GROQ_API_KEY"):
    st.warning("⚠️ `GROQ_API_KEY` is not set. Please add it to your environment or `.env` file.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a clinical question or just say hello..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history = []
    for m in st.session_state.messages[-7:-1]: 
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        else:
            chat_history.append(AIMessage(content=m["content"]))

    route = get_route(prompt)

    with st.chat_message("assistant"):
        
        # --- CHITCHAT ROUTE ---
        if route == "chitchat" or (route is None and len(prompt.split()) < 4):
            with st.spinner("MediAssist is typing..."):
                llm = load_chitchat_llm()
                
                system_instructions = (
                    "You are MediAssist AI, a professional, empathetic, and friendly clinical assistant. "
                    "For general greetings, identity questions, or casual talk, be warm and helpful. "
                    "CRITICAL: Do NOT use clinical formatting, do NOT mention 'References'."
                )
                
                context_messages = [SystemMessage(content=system_instructions)] + chat_history + [HumanMessage(content=prompt)]
                
                try:
                    response = llm.invoke(context_messages)
                    full_answer = response.content
                except Exception as e:
                    full_answer = f"Error communicating with LLM: {str(e)}"
                    
        # --- CLINICAL RAG ROUTE ---
        else:
            with st.spinner("Analyzing PubMed evidence"):
                chain = load_rag_chain()
                if chain:
                    try:
                        response = chain.invoke({"input": prompt, "chat_history": chat_history})
                        full_answer = response["answer"]
                        
                        # CRITICAL FIX: Extract PMIDs directly from the LLM's final answer text using Regex.
                        # This ensures the sidebar EXACTLY matches what the AI cited (3 vs 5).
                        cited_pmids = set(re.findall(r"PMID:\s*(\d+)", full_answer))
                        st.session_state.last_sources = sorted(list(filter(None, cited_pmids)))
                        
                        update_sidebar_sources(st.session_state.last_sources)
                        
                    except Exception as e:
                        full_answer = f"I encountered an error querying the research engine: {str(e)}"
                else:
                    full_answer = "I'm sorry, the clinical research engine is currently unavailable."

        st.markdown(full_answer)
        st.session_state.messages.append({"role": "assistant", "content": full_answer})

st.divider()
st.caption("Data powered by NCBI PubMed API.")