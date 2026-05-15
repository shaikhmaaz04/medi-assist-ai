from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.routers import SemanticRouter


chitchat = Route(
    name="chitchat",
    utterances=[
        "hi", "hello", "hey", "hi there", "hello assistant",
        "how are you?", "who are you?", "what is your name?",
        "what can you do?", "help me", "tell me a joke",
        "thank you", "thanks", "bye", "see ya", "good morning",
        "who created you?", "are you a doctor?", "how's it going?"
    ],
)

clinical = Route(
    name="clinical",
    utterances=[
        "hba1c levels in diabetics", "intermittent fasting weight loss",
        "side effects of 16:8", "hypoglycemia risks", "diabetes mellitus", 
        "research papers on fasting", "clinical trials for T2DM",
        "metabolic syndrome indicators", "is fasting safe?", 
        "what does the evidence say about time-restricted eating?",
        "clinical outcomes", "glucose levels", "patient safety"
    ],
)

try:
    _encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")
    _router = SemanticRouter(encoder=_encoder, routes=[chitchat, clinical], auto_sync="local")
except Exception as e:
    print(f"Failed to initialize router globally: {e}")
    _router = None

def get_route(text):
    """Executes the pre-loaded router safely."""
    if not _router:
        return None
        
    try:
        cleaned_text = text.strip().lower()
        
        # Fast heuristic bypass for simple greetings
        short_greetings = ["hi", "hello", "hey", "thanks", "bye", "ok", "okay"]
        if len(cleaned_text.split()) <= 2 and cleaned_text in short_greetings:
            return "chitchat"
            
        result = _router(text)
        return result.name if result else None
        
    except Exception as e:
        print(f"Routing Error: {e}")
        return None