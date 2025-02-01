import streamlit as st
import os
import json
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Función para cargar un PDF y extraer texto
def cargar_pdf(ruta_pdf):
    loader = PyPDFLoader(ruta_pdf)
    paginas = loader.load_and_split()
    contenido = "\n".join([p.page_content for p in paginas])
    return contenido

# Función para cargar estructura del JSON
def cargar_estructura_json():
    ruta_json = os.path.join("templates", "competencias.json")
    with open(ruta_json, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)
    return datos

# Configuración Streamlit
st.set_page_config(page_title="📄 Verificación de PDA y RAE", page_icon="📚")
st.title("📚 Verificador de Documentos PDA y RAE")

# Subir documentos
st.header("Sube tus documentos 📑")

pda_pdf = st.file_uploader("Sube el PDA (Planeación del curso) en PDF", type="pdf")
rae_pdf = st.file_uploader("Sube el RAE (Resultados de Aprendizaje Esperados) en PDF", type="pdf")

# Botón de analizar
if st.button("Analizar documentos 🚀"):
    if pda_pdf and rae_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pda:
            temp_pda.write(pda_pdf.read())
            pda_path = temp_pda.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_rae:
            temp_rae.write(rae_pdf.read())
            rae_path = temp_rae.name

        # Extraer textos
        texto_pda = cargar_pdf(pda_path)
        texto_rae = cargar_pdf(rae_path)
        estructura_json = cargar_estructura_json()

        st.success("✅ Documentos cargados exitosamente.")

        # Mostrar un resumen corto
        st.subheader("Resumen rápido de los documentos:")
        st.text_area("Contenido del PDA", texto_pda[:1000] + "...")
        st.text_area("Contenido del RAE", texto_rae[:1000] + "...")

        # Paso siguiente: enviar a Ollama
        st.info("Listo para comparar con la estructura. ¡Vamos!")

        # --- Aquí empieza la interacción con Ollama ---

        llm = ChatOllama(
            model="deepseek-coder:6.7b-instruct-q4_K_M",
            base_url="http://localhost:11434",
            temperature=0.2,
            streaming=False
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a university curriculum verification assistant. Your job is to check whether the provided RAE document complies with the expected structure according to the given JSON structure and is aligned with the PDA document."
            ),
            HumanMessagePromptTemplate.from_template(
                f"""
                PDA Document:
                {texto_pda}

                RAE Document:
                {texto_rae}

                JSON Structure (example of RAE requirements):
                {json.dumps(estructura_json, indent=2)}

                Instructions:
                1. First, identify if the RAE includes sections like Course Name, Code, Core, Credits, Weekly Intensity.
                2. Then, verify if the RAE learning outcomes match the expected outcomes mentioned in the PDA.
                3. Report any missing sections or inconsistencies.
                4. Conclude whether the RAE document is acceptable or needs corrections.

                Respond in a clear and structured report format.
                """
            )
        ])

        processing_pipeline = prompt | llm | StrOutputParser()

        resultado = processing_pipeline.invoke({})

        st.subheader("📋 Informe de verificación:")
        st.write(resultado)

    else:
        st.error("❗ Por favor, sube ambos documentos para proceder.")

