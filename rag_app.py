import streamlit as st
import os
from utils.document_loader import load_and_split_pdf
from utils.matcher import load_json_definitions, create_vector_store, match_chunks_to_definitions

TEMPLATES_PATH = "templates"
UPLOADS_PATH = "uploads"

st.title("Comparador RAE vs PDA")

# Mostrar lista de archivos JSON disponibles en templates/
json_files = [f for f in os.listdir(TEMPLATES_PATH) if f.endswith(".json")]
pdf_files = [f for f in os.listdir(UPLOADS_PATH) if f.endswith(".pdf")]

# Subir y analizar PDA
st.header("1️⃣ Seleccionar PDA y JSON de definiciones")
pda_selected = st.selectbox("Selecciona el archivo PDF del PDA", pdf_files, key="pda")
json_selected = st.selectbox("Selecciona el archivo JSON con definiciones", json_files)

if pda_selected and json_selected:
    pda_path = os.path.join(UPLOADS_PATH, pda_selected)
    json_path = os.path.join(TEMPLATES_PATH, json_selected)

    st.success(f"PDA seleccionado: {pda_selected}")
    st.success(f"Definiciones: {json_selected}")

    definitions = load_json_definitions(json_path)
    vector_store = create_vector_store(definitions)

    st.write("📄 Dividiendo el documento PDA...")
    pda_chunks = load_and_split_pdf(pda_path)

    st.write("🔍 Buscando coincidencias...")
    matches = match_chunks_to_definitions(pda_chunks, vector_store)

    for chunk, match_text, metadata, score in matches:
        st.markdown(f"**Coincidencia:** `{metadata['category']}` → `{metadata['key']}`")
        st.write(f"→ Texto: {chunk[:300]}...")
        st.success(f"✔ Coincide con: {match_text}")
        st.caption(f"🔍 Score: {score:.2f}")

# Comparar con informe RAE
st.header("2️⃣ Seleccionar informe RAE para comparar")
rae_selected = st.selectbox("Selecciona el archivo PDF del informe RAE", pdf_files, key="rae")

if rae_selected and json_selected:
    rae_path = os.path.join(UPLOADS_PATH, rae_selected)
    st.write("📄 Dividiendo el informe RAE...")
    rae_chunks = load_and_split_pdf(rae_path)

    st.write("🔍 Comparando con definiciones...")
    matches_rae = match_chunks_to_definitions(rae_chunks, vector_store)

    for chunk, match_text, metadata, score in matches_rae:
        st.markdown(f"**Coincidencia RAE:** `{metadata['category']}` → `{metadata['key']}`")
        st.write(f"→ Texto: {chunk[:300]}...")
        st.success(f"✔ Coincide con: {match_text}")
        st.caption(f"🔍 Score: {score:.2f}")

