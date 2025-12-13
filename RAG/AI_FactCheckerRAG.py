import streamlit as st
import chromadb
import requests
import re
import os
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langdetect import detect

# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="AI Fact-Checker",
    page_icon="🕵️‍♂️",
    layout="centered"
)

st.title("🕵️‍♂️ AI Fact-Checker")
st.markdown("Enter a claim and the AI will search the knowledge base to verify it.")

# =============================================================================
# 2. API CLIENT
# =============================================================================
class LLMClient:
    def __init__(self, model="qwen3:8b", api_key=None, api_url=None):
        self.model = model
        
        # URL Configuration: Use argument, environment variable, or default
        self.api_url = api_url or os.getenv("LLM_API_URL", "https://your-api-endpoint.com/api/generate")
        
        # Security: Priority -> Argument > Streamlit Secrets > Environment Variable
        if api_key:
            self.api_key = api_key
        elif "LLM_API_KEY" in st.secrets:
            self.api_key = st.secrets["LLM_API_KEY"]
        elif "LLM_API_KEY" in os.environ:
            self.api_key = os.environ["LLM_API_KEY"]
        else:
            # Default empty to avoid crashing immediately, but will fail on request
            self.api_key = ""
            st.warning("⚠️ API Key not found. Please configure 'LLM_API_KEY'.")

        self.options = {"temperature": 0.0, "num_ctx": 8192}

    def generate(self, prompt):
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self.options
        }
        try:
            # Note: verify=False is used here for self-signed certs; remove for production if possible
            response = requests.post(self.api_url, json=payload, headers=headers, verify=False)
            response.raise_for_status()
            response_text = response.json().get("response", "")
            
            # --- CLEAN <think> TAGS ---
            # Removes reasoning chains often output by reasoning models
            clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            
            return clean_text
            
        except Exception as e:
            return f"Error: {e}"

# Initialize Client
# Note: You can pass api_url here if needed, or set it via env vars
llm_client = LLMClient()

# =============================================================================
# 3. DATABASE LOADING
# =============================================================================
@st.cache_resource
def load_index():
    try:
        # print("🔌 Connecting to ChromaDB...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="intfloat/multilingual-e5-small"
        )
        # Ensure the path './chroma_db' exists or is correctly pointed to
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("rag")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return index
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

index = load_index()

# =============================================================================
# 4. VERIFICATION LOGIC (WITH HALLUCINATION GUARDRAILS)
# =============================================================================
def query_verifier(context, claim, target_language):
    # --- OPTIMIZED PROMPT ---
    prompt = f"""
You are an expert Polyglot Fact-Checker.
Your goal is to verify the CLAIM using the provided CONTEXT.

----------------
CONTEXT (Source material):
{context}
----------------

CLAIM: "{claim}"

### CRITICAL INSTRUCTIONS:
1. **TARGET LANGUAGE**: You MUST write the 'EXPLANATION' in **{target_language}**. 
   (Even if the Context is in English, the Explanation MUST be in {target_language}).
2. **EVIDENCE RULE**: 
   - If VERDICT is "NO_INFO", DO NOT generate any evidence. Write "None".
   - If VERDICT is "TRUE" or "FALSE", extract quotes **EXACTLY AS THEY APPEAR** (Verbatim) from the context.

### REQUIRED FORMAT:
VERDICT: [TRUE / FALSE / NO_INFO]
EXPLANATION: [Reasoning in {target_language}]
EVIDENCE:
- "Verbatim quote from text" || [SOURCE: Title (URL)]

### YOUR RESPONSE:
"""
    raw_response = llm_client.generate(prompt)
    
    # --- RESPONSE PARSING ---
    # 1. Default structure in case LLM fails
    result = {"verdict": "NO INFO", "explanation": "", "evidence": []}
    
    # 2. Parse line by line
    lines = raw_response.split('\n')
    mode = None
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # --- A) DETECT VERDICT ---
        if line.upper().startswith("VERDICT:") or line.upper().startswith("VEREDICTO:"):
            val = line.split(":")[1].strip().upper()

            if "TRUE" in val or "VERDADERO" in val: result["verdict"] = "TRUE"
            elif "FALSE" in val or "FALSO" in val: result["verdict"] = "FALSE"
            else: result["verdict"] = "NO INFO"
            
        # --- B) DETECT EXPLANATION ---
        elif line.upper().startswith("EXPLANATION:") or line.upper().startswith("EXPLICACIÓN:"):
            result["explanation"] = line.split(":", 1)[1].strip()
            mode = "explanation"

        # --- C) DETECT EVIDENCE ---    
        elif line.upper().startswith("EVIDENCE:") or line.upper().startswith("EVIDENCIAS:"):
            mode = "evidence"
        
        # --- D) PROCESS EVIDENCE LINES ---
        elif mode == "evidence" and "||" in line:
            parts = line.split("||")
            if len(parts) >= 2:
                result["evidence"].append({"quote": parts[0].strip(), "source": parts[1].strip()})
        
        # --- E) PROCESS MULTI-LINE EXPLANATION ---
        elif mode == "explanation" and not line.upper().startswith("EVIDENCE"):
            result["explanation"] += " " + line

    # --- GUARDRAIL: LOGICAL FILTER ---
    # If verdict is NO INFO, clear any potential hallucinations in evidence
    if result["verdict"] == "NO INFO":
        result["evidence"] = []
            
    return result

# =============================================================================
# AUXILIARY FUNCTION: CONTEXT SUMMARY
# =============================================================================

def generate_context_summary(context, claim, target_language):
    """
    Generates a narrative summary of available information regarding the claim.
    """
    prompt = f"""
    You are a helpful assistant.
    
    INPUTS:
    - TOPIC OF INTEREST: "{claim}"
    - TEXT TO SUMMARIZE: "{context}"
    
    TASK: 
    Write a clear and concise summary of the information found in the TEXT that is relevant to the TOPIC.
    
    GUIDELINES:
    1. **Focus**: Only include details related to the TOPIC. Discard unrelated information.
    2. **Style**: Write a smooth paragraph in your own words. Do not list sentences.
    3. **No Meta-talk**: Do not say "The text says" or "The article mentions". Just present the information.
    4. **Language**: Write the response strictly in **{target_language}**.
    
    SUMMARY:
    """
    
    return llm_client.generate(prompt)

# =============================================================================
# 5. USER INTERFACE (STATE MANAGEMENT & BUTTONS)
# =============================================================================

# Initialize Session State
if 'verification_data' not in st.session_state:
    st.session_state.verification_data = None
if 'current_context' not in st.session_state:
    st.session_state.current_context = None
if 'current_language' not in st.session_state:
    st.session_state.current_language = "ENGLISH"
if 'generated_summary' not in st.session_state:
    st.session_state.generated_summary = None

# User Input
claim_input = st.text_input("Enter a claim to verify (Any language):", 
                          placeholder="Ex: Climate change is not real")

# Main Button: Verify
if st.button("Verify Claim", type="primary"):
    if not index:
        st.error("Database is not loaded.")
    elif not claim_input:
        st.warning("Please enter a claim.")
    else:
        with st.spinner('🔍 Searching for evidence and analyzing...'):
            # Clear previous results
            st.session_state.verification_data = None
            st.session_state.generated_summary = None

            # 1. Language Detection
            try:
                lang_code = detect(claim_input)
            except:
                lang_code = "en"
            
            lang_map = {"es": "SPANISH", "en": "ENGLISH", "fr": "FRENCH", "pt": "PORTUGUESE", "de": "GERMAN", "it": "ITALIAN"}
            target_language = lang_map.get(lang_code, "ENGLISH")
            st.session_state.current_language = target_language

            # 2. Retrieval
            retriever = index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(claim_input)
            
            if not nodes:
                st.warning("⚠️ No relevant information found in the database.")
            else:
                # Prepare Context
                context_parts = []
                for n in nodes:
                    meta = n.metadata
                    title = meta.get('titulo') or meta.get('title') or "Doc"
                    url = meta.get('url') or "Local"
                    content = n.get_content().replace('\n', ' ')
                    context_parts.append(f"[SOURCE: {title} ({url})]\n{content}")
                
                context_str = "\n\n".join(context_parts)
                st.session_state.current_context = context_str
                
                # 3. Verification
                data = query_verifier(context_str, claim_input, target_language)
                st.session_state.verification_data = data

# --- RESULT VISUALIZATION ---

if st.session_state.verification_data:
    data = st.session_state.verification_data
    verdict = data["verdict"]
    
    st.divider()
    
    # Header with Color
    if "TRUE" in verdict:
        st.success(f"✅ VERDICT: {verdict}")
    elif "FALSE" in verdict:
        st.error(f"❌ VERDICT: {verdict}")
    else:
        st.info(f"⚪ VERDICT: {verdict}")
    
    # Explanation
    st.markdown(f"### 📝 Explanation\n{data['explanation']}")
    
    # Evidence
    if verdict == "NO INFO":
        st.warning("⚠️ Insufficient information found to verify this claim.")
    else:
        st.markdown("### 📂 Specific Evidence")
        if not data["evidence"]:
            st.write("*(The model used global context without citing exact phrases)*")
        
        for i, ev in enumerate(data["evidence"]):
            with st.expander(f"Evidence {i+1}"):
                st.markdown(f"**Quote:** *{ev['quote']}*")
                st.markdown(f"🔗 **Source:** {ev['source']}")

    st.divider()

    # --- SUMMARY BUTTON ---
    col_btn, col_info = st.columns([1, 2])
    
    with col_btn:
        if st.button("📝 Generate Context Summary"):
            with st.spinner("Generating summary..."):
                summary = generate_context_summary(
                    st.session_state.current_context, 
                    claim_input, 
                    st.session_state.current_language
                )
                st.session_state.generated_summary = summary

    # Show summary if generated
    if st.session_state.generated_summary:
        st.info(f"**📄 Summary of Retrieved Context:**\n\n{st.session_state.generated_summary}")
