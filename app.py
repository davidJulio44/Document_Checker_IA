import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import os
import json

# LLM local
llm = Ollama(model="gemma:2b-instruct-q4_0")

# Cargar índice FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("index/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Configuración de la interfaz
st.set_page_config(page_title="RAG - Competencias por Curso", layout="wide")
st.title("🔍 RAG - Competencias por Curso")
st.write("Carga el archivo RAE y luego busca el curso para validar competencias.")

# Entrada del curso
curso_input = st.text_input("📚 Código del curso (Ej: 22A10)", max_chars=10)

# Subida del archivo RAE
rae_file = st.file_uploader("📄 Carga el archivo PDF del RAE", type=["pdf"])

# Subida del archivo PDA
st.markdown("---")
st.header("📎 Análisis del PDA")
pda_file = st.file_uploader("📄 Carga el archivo PDF del PDA", type=["pdf"], key="pda")

# Función para cargar el JSON
def load_json_from_templates(filename):
    path = os.path.join("templates", filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# Función para verificar presencia
def verificar_presencia(lista_terminos, texto):
    encontrados = [t for t in lista_terminos if t.lower() in texto]
    return encontrados

# Si se carga el PDA
if pda_file:
    pda_reader = PdfReader(pda_file)
    pda_text = "\n".join(page.extract_text() or "" for page in pda_reader.pages).lower()

    st.subheader("🔍 Resultados del análisis del PDA")

    # Cargar los datos
    data = load_json_from_templates("competencias.json")

    competencias_generales = list(data.get("competencias genericas", {}).values())
    competencias_especificas = list(data.get("competencias especificas", {}).values())
    saberpro = list(data.get("SABER PRO", {}).values())
    dimensiones = list(data.get("dimensiones", {}).values())
    abet_items = list(data.get("ABET", {}).values())

    resultados = {
        "Competencias Genéricas": verificar_presencia(competencias_generales, pda_text),
        "Competencias Específicas": verificar_presencia(competencias_especificas, pda_text),
        "Competencias Saber Pro": verificar_presencia(saberpro, pda_text),
        "Dimensiones": verificar_presencia(dimensiones, pda_text),
        "ABET": verificar_presencia(abet_items, pda_text),
    }

    for categoria, encontrados in resultados.items():
        st.markdown(f"**🔸 {categoria}:**")
        if encontrados:
            for item in encontrados:
                st.success(f"✔️ {item}")
        else:
            st.warning("No se encontraron coincidencias.")

# Si se cargan ambos archivos y hay código de curso
if curso_input and rae_file:
    reader = PdfReader(rae_file)
    rae_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # Consultar con el agente RAG
    query = f"¿Cuáles son las competencias específicas, genéricas, saberpro, abet y dimensión del curso {curso_input}?"
    result = qa(query)

    st.markdown("### ✅ Competencias encontradas en el PDA:")
    st.success(result['result'])

    # Procesar el segundo archivo PDF: el PDA
if pda_file:
    pda_reader = PdfReader(pda_file)
    pda_text = "\n".join(page.extract_text() or "" for page in pda_reader.pages).lower()

    # Leer el archivo de competencias JSON
    with open("templates/competencias.json", "r", encoding="utf-8") as f:
        competencias_data = json.load(f)

    st.markdown("### 🧩 Comparación con competencias del JSON:")
    
    for categoria, competencias in competencias_data.items():
        st.markdown(f"**{categoria.upper()}**")
        for codigo, descripcion in competencias.items():
            if descripcion.lower() in pda_text:
                st.success(f"✔️ {codigo}: {descripcion}")
            else:
                st.warning(f"⚠️ {codigo} NO se encontró en el PDA")



    st.markdown("### 📄 Contenido del RAE relacionado:")
    if curso_input in rae_text:
        st.info("✔️ El código del curso está presente en el RAE.")
        if any(term in rae_text.lower() for term in ["especificas", "genericas", "dimension", "saberpro", "abet"]):
            st.success("Se encontraron términos clave en el RAE.")
        else:
            st.warning("No se encontraron términos clave en el RAE.")
    else:
        st.error("❌ El código del curso NO está presente en el RAE.")


