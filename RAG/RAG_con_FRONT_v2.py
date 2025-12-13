import streamlit as st
import chromadb
import requests
import re  # <--- NUEVO: Para limpiar las etiquetas <think>
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langdetect import detect


# =============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Verificador de Hechos AI",
    page_icon="🕵️‍♂️",
    layout="centered"
)

st.title("🕵️‍♂️ Verificador de Hechos")
st.markdown("Introduce una afirmación y la IA buscará en la base de datos para verificarla.")

# =============================================================================
# 2. CLIENTE API
# =============================================================================
class ClienteUC3M:
    def __init__(self, model="qwen3:8b", api_key="sk-af55e7023913527f0d96c038eec2ef2d"):
        self.model = model
        self.api_url = "https://yiyuan.tsc.uc3m.es/api/generate"
        self.api_key = api_key
        self.options = {"temperature": 0.0, "num_ctx": 8192}

    def generar(self, prompt):
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self.options
        }
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, verify=False)
            response.raise_for_status()
            texto_respuesta = response.json().get("response", "")
            
            # --- LIMPIEZA DE ETIQUETAS <think> ---
            # Elimina todo lo que esté entre <think> y </think> (incluyendo saltos de línea)
            texto_limpio = re.sub(r'<think>.*?</think>', '', texto_respuesta, flags=re.DOTALL).strip()
            
            return texto_limpio
            
        except Exception as e:
            return f"Error: {e}"

llm_client = ClienteUC3M()

# =============================================================================
# 3. CARGA DE BASE DE DATOS
# =============================================================================
@st.cache_resource
def cargar_indice():
    try:
        print("🔌 Conectando a ChromaDB...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="intfloat/multilingual-e5-small"
        )
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("rag")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return index
    except Exception as e:
        st.error(f"Error cargando la base de datos: {e}")
        return None

index = cargar_indice()

# =============================================================================
# 4. LÓGICA DE VERIFICACIÓN (MEJORADA CON GUARDA RAILS)
# =============================================================================
def consultar_verificador(contexto, afirmacion, idioma_destino):
    # --- PROMPT OPTIMIZADO ---
    prompt = f"""
You are an expert Polyglot Fact-Checker.
Your goal is to verify the CLAIM using the provided CONTEXT.

----------------
CONTEXT (Source material):
{contexto}
----------------

CLAIM: "{afirmacion}"

### CRITICAL INSTRUCTIONS:
1. **TARGET LANGUAGE**: You MUST write the 'EXPLANATION' in **{idioma_destino}**. 
   (Even if the Context is in English, the Explanation MUST be in {idioma_destino}).
2. **EVIDENCE RULE**: 
   - If VERDICT is "NO_INFO", DO NOT generate any evidence. Write "None".
   - If VERDICT is "TRUE" or "FALSE", extract quotes **EXACTLY AS THEY APPEAR** (Verbatim) from the context.

### REQUIRED FORMAT:
VERDICT: [TRUE / FALSE / NO_INFO]
EXPLANATION: [Reasoning in {idioma_destino}]
EVIDENCE:
- "Verbatim quote from text" || [SOURCE: Title (URL)]

### YOUR RESPONSE:
"""
    respuesta_raw = llm_client.generar(prompt)
    
    # --- PARSEO DE RESPUESTA ---
    # 1. Inicializamos la estructura base por defecto.
    # Si el LLM falla o devuelve basura, al menos tendremos esto para no romper el programa.
    resultado = {"veredicto": "SIN INFO", "explicacion": "", "evidencias": []}
    
    # 2. Preparamos el bucle de lectura
    # Dividimos el texto gigante en líneas individuales para leerlas una a una.
    lines = respuesta_raw.split('\n')
    modo = None
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # --- A) DETECTAR VEREDICTO ---
        # Buscamos la línea que empieza por "VERDICT:" (o VEREDICTO en español)
        if line.upper().startswith("VERDICT:") or line.upper().startswith("VEREDICTO:"):
            # Cogemos lo que hay después de los dos puntos
            val = line.split(":")[1].strip().upper()

            # Normalizamos la respuesta a 3 valores estándar: VERDADERO, FALSO, SIN INFO
            if "TRUE" in val or "VERDADERO" in val: resultado["veredicto"] = "VERDADERO"
            elif "FALSE" in val or "FALSO" in val: resultado["veredicto"] = "FALSO"
            else: resultado["veredicto"] = "SIN INFO"
            
        # --- B) DETECTAR INICIO DE EXPLICACIÓN ---
        elif line.upper().startswith("EXPLANATION:") or line.upper().startswith("EXPLICACIÓN:"):
            resultado["explicacion"] = line.split(":", 1)[1].strip()
            modo = "explicacion"

        # --- C) DETECTAR INICIO DE EVIDENCIAS ---    
        elif line.upper().startswith("EVIDENCE:") or line.upper().startswith("EVIDENCIAS:"):
            modo = "evidencias"
        
        # --- D) PROCESAR LÍNEAS DE EVIDENCIAS ---
        # Si estamos en modo evidencias y la línea tiene el separador especial "||"
        elif modo == "evidencias" and "||" in line:
            parts = line.split("||")
            
            # Nos aseguramos de que haya dos partes (Cita || Fuente) antes de guardar
            if len(parts) >= 2:
                resultado["evidencias"].append({"cita": parts[0].strip(), "fuente": parts[1].strip()})
        
        # --- E) PROCESAR EXPLICACIÓN MULTI-LÍNEA ---
        # Si estamos en modo explicación, pero NO es una línea nueva de cabecera (como EVIDENCE)
        # significa que la explicación continúa en el párrafo siguiente. Concatenamos.
        elif modo == "explicacion" and not line.upper().startswith("EVIDENCE"):
            resultado["explicacion"] += " " + line

    # --- GUARDA RAIL 2: FILTRO LÓGICO DE ALUCINACIONES ---
    # Si el veredicto es "SIN INFO", borramos cualquier evidencia que se haya podido colar.
    # Esto evita alucinaciones donde el LLM dice "No sé nada" pero se inventa una fuente.
    if resultado["veredicto"] == "SIN INFO":
        resultado["evidencias"] = []
            
    return resultado

# =============================================================================
# FUNCIÓN AUXILIAR: RESUMEN DE CONTEXTO
# =============================================================================

def generar_resumen_contexto(contexto, claim, idioma_destino):
    """
    Genera un resumen narrativo de la información disponible sobre el claim.
    """
    prompt = f"""
    You are a helpful assistant.
    
    INPUTS:
    - TOPIC OF INTEREST: "{claim}"
    - TEXT TO SUMMARIZE: "{contexto}"
    
    TASK: 
    Write a clear and concise summary of the information found in the TEXT that is relevant to the TOPIC.
    
    GUIDELINES:
    1. **Focus**: Only include details related to the TOPIC. Discard unrelated information.
    2. **Style**: Write a smooth paragraph in your own words. Do not list sentences.
    3. **No Meta-talk**: Do not say "The text says" or "The article mentions". Just present the information.
    4. **Language**: Write the response strictly in **{idioma_destino}**.
    
    SUMMARY:
    """
    
    return llm_client.generar(prompt)

# =============================================================================
# 5. INTERFAZ DE USUARIO (GESTIÓN DE ESTADO Y BOTONES)
# =============================================================================

# Inicializar Session State para persistencia de datos
if 'datos_verificacion' not in st.session_state:
    st.session_state.datos_verificacion = None
if 'contexto_actual' not in st.session_state:
    st.session_state.contexto_actual = None
if 'idioma_actual' not in st.session_state:
    st.session_state.idioma_actual = "SPANISH"
if 'resumen_generado' not in st.session_state:
    st.session_state.resumen_generado = None

# Input del usuario
afirmacion = st.text_input("Escribe una frase para verificar (Cualquier idioma):", 
                          placeholder="Ej: El cambio climático no es real")

# Botón Principal: Verificar
if st.button("Verificar Hecho", type="primary"):
    if not index:
        st.error("La base de datos no está cargada.")
    elif not afirmacion:
        st.warning("Por favor escribe algo.")
    else:
        with st.spinner('🔍 Buscando evidencias y analizando...'):
            # Limpiar resultados anteriores
            st.session_state.datos_verificacion = None
            st.session_state.resumen_generado = None

            # 1. Detección de idioma
            try:
                lang_code = detect(afirmacion)
            except:
                lang_code = "es"
            
            mapa_idiomas = {"es": "SPANISH", "en": "ENGLISH", "fr": "FRENCH", "pt": "PORTUGUESE", "de": "GERMAN", "it": "ITALIAN"}
            idioma_destino = mapa_idiomas.get(lang_code, "SPANISH")
            st.session_state.idioma_actual = idioma_destino

            # 2. Retrieval
            retriever = index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(afirmacion)
            
            if not nodes:
                st.warning("⚠️ No se encontró información en la base de datos.")
            else:
                # Preparar contexto
                context_parts = []
                for n in nodes:
                    meta = n.metadata
                    titulo = meta.get('titulo') or meta.get('title') or "Doc"
                    url = meta.get('url') or "Local"
                    contenido = n.get_content().replace('\n', ' ')
                    context_parts.append(f"[SOURCE: {titulo} ({url})]\n{contenido}")
                
                context_str = "\n\n".join(context_parts)
                st.session_state.contexto_actual = context_str
                
                # 3. Verificación
                datos = consultar_verificador(context_str, afirmacion, idioma_destino)
                st.session_state.datos_verificacion = datos

# --- VISUALIZACIÓN DE RESULTADOS (Fuera del bloque del botón para persistencia) ---

if st.session_state.datos_verificacion:
    datos = st.session_state.datos_verificacion
    veredicto = datos["veredicto"]
    
    st.divider()
    
    # Encabezado con color
    if "VERDADERO" in veredicto:
        st.success(f"✅ VEREDICTO: {veredicto}")
    elif "FALSO" in veredicto:
        st.error(f"❌ VEREDICTO: {veredicto}")
    else:
        st.info(f"⚪ VEREDICTO: {veredicto}")
    
    # Explicación
    st.markdown(f"### 📝 Explicación\n{datos['explicacion']}")
    
    # Evidencias
    if veredicto == "SIN INFO":
        st.warning("⚠️ No se encontró información suficiente.")
    else:
        st.markdown("### 📂 Evidencias Específicas")
        if not datos["evidencias"]:
            st.write("*(El modelo usó el contexto global sin citar frases exactas)*")
        
        for i, ev in enumerate(datos["evidencias"]):
            with st.expander(f"Evidencia {i+1}"):
                st.markdown(f"**Cita:** *{ev['cita']}*")
                st.markdown(f"🔗 **Fuente:** {ev['fuente']}")

    st.divider()

    # --- BOTÓN DE RESUMEN ---
    col_btn, col_info = st.columns([1, 2])
    
    with col_btn:
        # Botón para generar resumen bajo demanda
        if st.button("📝 Generar Resumen del Contexto"):
            with st.spinner("Generando resumen..."):
                resumen = generar_resumen_contexto(st.session_state.contexto_actual, afirmacion, st.session_state.idioma_actual)
                st.session_state.resumen_generado = resumen

    # Mostrar resumen si ya fue generado
    if st.session_state.resumen_generado:
        st.info(f"**📄 Resumen del Contexto Recuperado:**\n\n{st.session_state.resumen_generado}")