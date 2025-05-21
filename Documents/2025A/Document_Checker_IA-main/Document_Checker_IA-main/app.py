import os
import json
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.llms import Ollama


# Rutas
TEMPLATES_PATH = "templates"

llm = Ollama(model="llama3")

# Crear y cargar Vector Store
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)


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


# Función para cargar competencias de cursos
def load_competencias_cursos(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Función para identificar automáticamente la asignatura/curso desde el contenido del PDA
def identificar_asignatura(chunks, competencias_cursos):
    # Extraer códigos de curso del JSON
    codigos_curso = list(competencias_cursos.keys())

    # Crear un contador para cada código de curso
    contador_cursos = {codigo: 0 for codigo in codigos_curso}

    # Examinar cada chunk para buscar menciones a los códigos de curso
    for chunk in chunks:
        texto = chunk.page_content.lower()
        for codigo in codigos_curso:
            if codigo.lower() in texto:
                contador_cursos[codigo] += 3  # Dar más peso a coincidencias exactas de código
            # También buscar sin el prefijo (ej: "A52" si el código es "22A52")
            if codigo[2:].lower() in texto:
                contador_cursos[codigo] += 1

    # Si no hay coincidencias, intentar buscar otras palabras clave
    if max(contador_cursos.values(), default=0) == 0:
        return codigos_curso[0] if codigos_curso else None

    # Devolver el código con más menciones
    curso_detectado = max(contador_cursos.items(), key=lambda x: x[1])[0]
    return curso_detectado


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


# Extraer tipos de competencias de un documento
def extract_competencia_mentions(chunks):
    competencias_dict = {
        "especificas": [],
        "genericas": [],
        "saberpro": [],
        "abet": [],
        "dimension": []
    }

    # Mapeo de términos alternativos para buscar
    terminos_alternativos = {
        "especificas": ["especifica", "especificas", "específica", "específicas", "competencia específica",
                        "competencias específicas", "C1", "C2", "C3"],
        "genericas": ["generica", "genericas", "genérica", "genéricas", "competencia genérica",
                      "competencias genéricas", "1a", "1b", "1c", "1d", "1e", "1f", "1g", "1h", "1i", "1j", "1l"],
        "saberpro": ["saber pro", "saberpro", "SP1", "SP2", "SP3", "SP4", "SP5"],
        "abet": ["abet", "1.1", "1.2", "1.3", "2.1", "2.2", "2.3", "3.1", "3.2", "4.1", "4.2", "4.3", "5.1", "5.2",
                 "6.1", "6.2", "6.3", "7.1"],
        "dimension": ["dimension", "dimensión", "D1", "D2", "D3", "D4", "D5", "D6"]
    }

    patrones_especificos = {
        "especificas": ["C1", "C2", "C3"],
        "genericas": ["1a", "1b", "1c", "1d", "1e", "1f", "1g", "1h", "1i", "1j", "1l"],
        "saberpro": ["SP1", "SP2", "SP3", "SP4", "SP5"],
        "abet": ["1.1", "1.2", "1.3", "2.1", "2.2", "2.3", "3.1", "3.2", "4.1", "4.2", "4.3", "5.1", "5.2", "6.1",
                 "6.2", "6.3", "7.1"],
        "dimension": ["D1", "D2", "D3", "D4", "D5", "D6"]
    }

    for chunk in chunks:
        texto = chunk.page_content.lower()

        # Buscar menciones en el texto
        for tipo, terminos in terminos_alternativos.items():
            for termino in terminos:
                if termino.lower() in texto:
                    # Si encontramos un término genérico, buscar patrones específicos
                    for patron in patrones_especificos[tipo]:
                        if patron.lower() in texto:
                            if patron not in competencias_dict[tipo]:
                                competencias_dict[tipo].append(patron)

    return competencias_dict


# Analizar coincidencias entre competencias encontradas y las definidas en el JSON
def analyze_competencias_match(pda_competencias, rae_competencias, competencias_curso):
    # Inicializar resultados
    resultados = {
        "found_in_both": [],
        "only_in_pda": [],
        "only_in_rae": [],
        "missing_from_json": [],
        "coverage_percentage": 0
    }

    # Procesar cada tipo de competencia
    for tipo in competencias_curso.keys():
        # Competencias definidas en JSON para este curso
        competencias_definidas = set(competencias_curso.get(tipo, []))

        # Competencias encontradas en PDA y RAE
        competencias_pda = set(pda_competencias.get(tipo, []))
        competencias_rae = set(rae_competencias.get(tipo, []))

        # Encontrar coincidencias y diferencias
        both = competencias_pda.intersection(competencias_rae).intersection(competencias_definidas)
        only_pda = competencias_pda.intersection(competencias_definidas) - competencias_rae
        only_rae = competencias_rae.intersection(competencias_definidas) - competencias_pda

        # Competencias encontradas pero no definidas en JSON
        not_in_json = (competencias_pda.union(competencias_rae)) - competencias_definidas

        # Agregar al resultado con el tipo
        for comp in both:
            resultados["found_in_both"].append(f"{tipo}:{comp}")
        for comp in only_pda:
            resultados["only_in_pda"].append(f"{tipo}:{comp}")
        for comp in only_rae:
            resultados["only_in_rae"].append(f"{tipo}:{comp}")
        for comp in not_in_json:
            resultados["missing_from_json"].append(f"{tipo}:{comp}")

    # Calcular cobertura (competencias encontradas en ambos / total definidas en JSON)
    total_definidas = sum(len(comps) for comps in competencias_curso.values())
    if total_definidas > 0:
        resultados["coverage_percentage"] = (len(resultados["found_in_both"]) / total_definidas) * 100

    return resultados


# Generar informe de análisis simple
def generate_analysis_report(results, curso_id):
    # Construir el informe
    informe = f"""
    # Análisis de Competencias para el curso {curso_id}

    ## Resumen de cobertura
    * **Porcentaje de cobertura**: {results['coverage_percentage']:.1f}%
    * **Competencias encontradas en ambos documentos**: {len(results['found_in_both'])}
    * **Competencias solo en PDA**: {len(results['only_in_pda'])}
    * **Competencias solo en RAE**: {len(results['only_in_rae'])}
    * **Competencias no definidas en JSON pero mencionadas**: {len(results['missing_from_json'])}

    ## Competencias encontradas en ambos documentos
    {', '.join(results['found_in_both']) if results['found_in_both'] else 'Ninguna'}

    ## Competencias solo en PDA
    {', '.join(results['only_in_pda']) if results['only_in_pda'] else 'Ninguna'}

    ## Competencias solo en RAE
    {', '.join(results['only_in_rae']) if results['only_in_rae'] else 'Ninguna'}

    ## Recomendaciones
    """

    # Generar recomendaciones basadas en los resultados
    recomendaciones = []

    if results['only_in_pda']:
        recomendaciones.append(
            "- **Incorporar en RAE**: Las competencias que están solo en el PDA deberían incluirse en el RAE para mantener coherencia.")

    if results['only_in_rae']:
        recomendaciones.append(
            "- **Actualizar PDA**: Las competencias que están solo en el RAE deberían incluirse en el PDA.")

    if results['coverage_percentage'] < 50:
        recomendaciones.append(
            "- **Mejorar cobertura general**: La cobertura de competencias es baja. Se recomienda revisar ambos documentos para alinear mejor con las competencias definidas.")

    if results['missing_from_json']:
        recomendaciones.append(
            f"- **Revisar definiciones**: Hay competencias mencionadas que no están definidas en el JSON: {', '.join(results['missing_from_json'])}.")

    if not recomendaciones:
        recomendaciones.append("- Los documentos PDA y RAE están bien alineados con las competencias definidas.")

    informe += "\n".join(recomendaciones)
    return informe


# Interfaz Streamlit
st.title("Comparador RAE vs PDA con análisis de competencias")

# Cargar competencias de cursos
competencias_cursos_path = os.path.join(TEMPLATES_PATH, "competenciascursos.json")
competencias_cursos = load_competencias_cursos(competencias_cursos_path)
st.success(f"✅ Cargadas definiciones para {len(competencias_cursos)} cursos")

# Selección de archivos
st.header("1️⃣ Seleccionar PDA y competencias")
pda_selected = st.file_uploader("Selecciona el archivo PDF del PDA", type="pdf")

# Variables para almacenar resultados
pda_competencias = {}
rae_competencias = {}
curso_detectado = None

if pda_selected:
    # Cargar los chunks del PDA
    pda_chunks = load_and_split_pdf(pda_selected)

    # Mostrar las primeras líneas para diagnóstico
    if st.checkbox("Mostrar diagnóstico de texto extraído"):
        st.text_area("Primeras líneas del documento:",
                     "\n\n".join([chunk.page_content[:200] + "..." for chunk in pda_chunks[:3]]),
                     height=200)

    # Identificar automáticamente el curso
    curso_detectado = identificar_asignatura(pda_chunks, competencias_cursos)

    # Permitir sobrescribir la detección automática
    st.success(f"✅ Curso detectado: **{curso_detectado}**")

    if st.checkbox("¿Cambiar el curso detectado?"):
        curso_detectado = st.selectbox("Selecciona el curso correcto:",
                                       list(competencias_cursos.keys()))
        st.success(f"✅ Curso cambiado a: **{curso_detectado}**")

    # Extraer competencias mencionadas en PDA
    pda_competencias = extract_competencia_mentions(pda_chunks)

    # Mostrar competencias encontradas en PDA
    st.subheader(f"Competencias encontradas en PDA para el curso {curso_detectado}")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Competencias específicas")
        st.write(", ".join(pda_competencias["especificas"]) if pda_competencias["especificas"] else "Ninguna")

        st.write("Competencias genéricas")
        st.write(", ".join(pda_competencias["genericas"]) if pda_competencias["genericas"] else "Ninguna")

        st.write("SaberPRO")
        st.write(", ".join(pda_competencias["saberpro"]) if pda_competencias["saberpro"] else "Ninguna")

    with col2:
        st.write("ABET")
        st.write(", ".join(pda_competencias["abet"]) if pda_competencias["abet"] else "Ninguna")

        st.write("Dimensión")
        st.write(", ".join(pda_competencias["dimension"]) if pda_competencias["dimension"] else "Ninguna")

# Comparar con informe RAE
st.header("2️⃣ Seleccionar informe RAE para comparar")
rae_selected = st.file_uploader("Selecciona el archivo PDF del informe RAE", type="pdf", key="rae")

if rae_selected and curso_detectado:
    # Cargar los chunks del RAE
    rae_chunks = load_and_split_pdf(rae_selected)

    # Extraer competencias mencionadas en RAE
    rae_competencias = extract_competencia_mentions(rae_chunks)

    # Mostrar competencias encontradas en RAE
    st.subheader(f"Competencias encontradas en RAE para el curso {curso_detectado}")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Competencias específicas")
        st.write(", ".join(rae_competencias["especificas"]) if rae_competencias["especificas"] else "Ninguna")

        st.write("Competencias genéricas")
        st.write(", ".join(rae_competencias["genericas"]) if rae_competencias["genericas"] else "Ninguna")

        st.write("SaberPRO")
        st.write(", ".join(rae_competencias["saberpro"]) if rae_competencias["saberpro"] else "Ninguna")

    with col2:
        st.write("ABET")
        st.write(", ".join(rae_competencias["abet"]) if rae_competencias["abet"] else "Ninguna")

        st.write("Dimensión")
        st.write(", ".join(rae_competencias["dimension"]) if rae_competencias["dimension"] else "Ninguna")

    # Comparar con las competencias definidas en el JSON
    competencias_curso = competencias_cursos.get(curso_detectado, {})
    if competencias_curso:
        st.subheader("Comparación con competencias definidas")

        # Mostrar competencias definidas en JSON
        st.write("#### Competencias definidas para este curso:")
        for tipo, comps in competencias_curso.items():
            st.write(f"**{tipo}**: {', '.join(comps) if comps else 'Ninguna'}")

        # Analizar coincidencias
        results = analyze_competencias_match(pda_competencias, rae_competencias, competencias_curso)

        # Mostrar resumen de comparación
        st.subheader("3️⃣ Análisis de alineación PDA-RAE")

        # Mostrar métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Competencias en ambos", len(results["found_in_both"]))
        with col2:
            st.metric("Solo en PDA", len(results["only_in_pda"]))
        with col3:
            st.metric("Solo en RAE", len(results["only_in_rae"]))

        st.metric("Cobertura", f"{results['coverage_percentage']:.1f}%")

        # Generar y mostrar informe completo
        analysis_report = generate_analysis_report(results, curso_detectado)
        with st.expander("Ver informe completo"):
            st.markdown(analysis_report)
    else:
        st.error(f"No se encontraron definiciones de competencias para el curso {curso_detectado} en el JSON")

