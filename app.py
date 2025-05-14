# Crear y cargar Vector Store
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

import os
import json
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Rutas
TEMPLATES_PATH = "templates"


# Función para cargar y dividir PDFs
def load_and_split_pdf(uploaded_file):
    # Guardar el archivo cargado en un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name
    
    # Usar la ruta del archivo temporal
    loader = PyPDFLoader(temp_path)
    pages = loader.load()
    
    # Eliminar el archivo temporal después de cargarlo
    os.unlink(temp_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(pages)

# Función para cargar definiciones desde JSON
def load_json_definitions(path, asignatura=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    flat_data = []
    for category, subdict in data.items():
        # Si se especifica una asignatura, solo procesar esa categoría
        if asignatura and category != asignatura:
            continue
        for key, value in subdict.items():
            metadata = {"category": category, "key": key}
            flat_data.append(Document(page_content=str(value), metadata=metadata))
    return flat_data

# Función para identificar automáticamente la asignatura desde el contenido del PDA
def identificar_asignatura(chunks, asignaturas):
    # Crear un contador para cada asignatura
    contador_asignaturas = {asignatura: 0 for asignatura in asignaturas}
    
    # Lista de palabras clave adicionales para cada asignatura (opcional)
    keywords = {}
    for asignatura in asignaturas:
        # Convertir el nombre de la asignatura en palabras clave
        keywords[asignatura] = [asignatura.lower()] + asignatura.lower().split()
    
    # Examinar cada chunk para buscar menciones a las asignaturas
    for chunk in chunks:
        texto = chunk.page_content.lower()
        for asignatura, palabras_clave in keywords.items():
            for palabra in palabras_clave:
                if palabra in texto and len(palabra) > 2:  # Reducido a 2 caracteres
                    contador_asignaturas[asignatura] += 1
    
    # Enfoque alternativo: usar el nombre del archivo
    if max(contador_asignaturas.values(), default=0) == 0:
        return list(asignaturas)[0] if asignaturas else None
    
    # Devolver la asignatura con más menciones
    asignatura_detectada = max(contador_asignaturas.items(), key=lambda x: x[1])[0]
    return asignatura_detectada

# Matching
def match_chunks_to_definitions(chunks, db, threshold=0.6):  # Umbral reducido para ser menos estricto
    matches = []
    for chunk in chunks:
        sims = db.similarity_search_with_score(chunk.page_content, k=3)
        for doc, score in sims:
            normalized_score = 1.0 - score  # Convertir distancia a similitud
            if normalized_score >= threshold:
                matches.append((chunk.page_content, doc.page_content, doc.metadata, normalized_score))
    return matches

# Interfaz Streamlit
st.title("Comparador RAE vs PDA")

# Selección de archivos JSON y PDF
json_files = [f for f in os.listdir(TEMPLATES_PATH) if f.endswith(".json")]

st.header("1️⃣ Seleccionar PDA y JSON de definiciones")
pda_selected = st.file_uploader("Selecciona el archivo PDF del PDA", type="pdf")
json_selected = st.selectbox("Selecciona el archivo JSON con definiciones", json_files)

if pda_selected and json_selected:
    # Cargar el JSON para obtener las asignaturas disponibles
    json_path = os.path.join(TEMPLATES_PATH, json_selected)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        asignaturas = list(data.keys())
    
    # Cargar los chunks del PDA
    pda_chunks = load_and_split_pdf(pda_selected)
    
    # Mostrar las primeras líneas para diagnóstico
    if st.checkbox("Mostrar diagnóstico de texto extraído"):
        st.text_area("Primeras líneas del documento:", 
                     "\n\n".join([chunk.page_content[:200] + "..." for chunk in pda_chunks[:3]]),
                     height=200)
    
    # Identificar automáticamente la asignatura
    asignatura_detectada = identificar_asignatura(pda_chunks, asignaturas)
    
    # Permitir sobrescribir la detección automática
    st.success(f"✅ Asignatura detectada: **{asignatura_detectada}**")
    
    if st.checkbox("¿Cambiar la asignatura detectada?"):
        asignatura_detectada = st.selectbox("Selecciona la asignatura correcta:", asignaturas)
        st.success(f"✅ Asignatura cambiada a: **{asignatura_detectada}**")
    
    # Filtrar definiciones por asignatura detectada
    definitions = load_json_definitions(json_path, asignatura_detectada)
    
    # Mostrar las definiciones cargadas para diagnóstico
    if st.checkbox("Mostrar definiciones cargadas"):
        st.write(f"Se cargaron {len(definitions)} definiciones para la asignatura '{asignatura_detectada}'")
        for i, doc in enumerate(definitions[:5]):
            st.write(f"Definición #{i+1}: {doc.metadata['key']}")
            st.text_area(f"Contenido #{i+1}", doc.page_content[:200] + "...", height=100)
    
    # Solo crear el vector store si hay definiciones
    if definitions:
        vector_store = create_vector_store(definitions)
        matches = match_chunks_to_definitions(pda_chunks, vector_store)

        st.subheader(f"Resultados para la asignatura: {asignatura_detectada}")
        
        if not matches:
            st.warning(f"No se encontraron coincidencias para la asignatura {asignatura_detectada}")
            st.info("Prueba ajustando la configuración: 👇")
            umbral = st.slider("Ajustar umbral de coincidencia:", min_value=0.1, max_value=0.9, value=0.6, step=0.05)
            
            # Reintentar con el nuevo umbral
            matches = match_chunks_to_definitions(pda_chunks, vector_store, threshold=umbral)
            
            if matches:
                st.success(f"¡Se encontraron {len(matches)} coincidencias con el nuevo umbral!")
            else:
                st.error("Aún no se encuentran coincidencias. Posibles problemas:")
                st.markdown("""
                1. El PDF no contiene texto extraíble (está escaneado)
                2. La asignatura seleccionada no es correcta
                3. El formato del PDA es muy diferente al esperado
                """)
        
        # Mostrar coincidencias
        for chunk, match_text, metadata, score in matches:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Coincidencia:** `{metadata['category']}` → `{metadata['key']}`")
                st.caption(f"🔍 Score: {score:.2f}")
            with col2:
                if st.checkbox(f"Ver detalles de coincidencia: {metadata['key'][:30]}...", key=f"match_{metadata['key']}"):
                    st.write(f"Texto PDA:\n{chunk[:500]}...")
                    st.success(f"✔ Coincide con:\n{match_text}")
    else:
        st.error(f"No se encontraron definiciones para la asignatura '{asignatura_detectada}' en el JSON")
        st.info("Revisa que el archivo JSON tenga la estructura correcta y contenga esta asignatura.")

# Comparar con informe RAE
st.header("2️⃣ Seleccionar informe RAE para comparar")
rae_selected = st.file_uploader("Selecciona el archivo PDF del informe RAE", type="pdf", key="rae")

if rae_selected and json_selected and 'asignatura_detectada' in locals() and 'vector_store' in locals():
    rae_chunks = load_and_split_pdf(rae_selected)
    
    matches_rae = match_chunks_to_definitions(rae_chunks, vector_store)
    
    st.subheader(f"Resultados RAE para la asignatura: {asignatura_detectada}")
    
    if not matches_rae:
        st.warning(f"No se encontraron coincidencias RAE para la asignatura {asignatura_detectada}")
        st.info("Prueba ajustando la configuración: 👇")
        umbral_rae = st.slider("Ajustar umbral de coincidencia RAE:", min_value=0.1, max_value=0.9, value=0.6, step=0.05)
        
        # Reintentar con el nuevo umbral
        matches_rae = match_chunks_to_definitions(rae_chunks, vector_store, threshold=umbral_rae)
        
        if matches_rae:
            st.success(f"¡Se encontraron {len(matches_rae)} coincidencias RAE con el nuevo umbral!")
    
    # Mostrar coincidencias RAE
    for chunk, match_text, metadata, score in matches_rae:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Coincidencia RAE:** `{metadata['category']}` → `{metadata['key']}`")
            st.caption(f"🔍 Score: {score:.2f}")
        with col2:
            if st.checkbox(f"Ver detalles RAE: {metadata['key'][:30]}...", key=f"rae_{metadata['key']}"):
                st.write(f"Texto RAE:\n{chunk[:500]}...")
                st.success(f"✔ Coincide con:\n{match_text}")