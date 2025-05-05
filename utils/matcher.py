import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

def load_json_definitions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    flat_data = []
    for category, subdict in data.items():
        for key, value in subdict.items():
            metadata = {"category": category, "key": key}
            flat_data.append(Document(page_content=value, metadata=metadata))
    return flat_data

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db

def match_chunks_to_definitions(chunks, db, threshold=0.75):
    matches = []
    for chunk in chunks:
        sims = db.similarity_search_with_score(chunk.page_content, k=3)
        for doc, score in sims:
            if score >= threshold:
                matches.append((chunk.page_content, doc.page_content, doc.metadata, score))
    return matches
