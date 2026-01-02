import os
import faiss
import json
from sentence_transformers import SentenceTransformer
from app.config import settings

def init_empty_index():
    os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)

    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    dimension = model.get_sentence_embedding_dimension()

    index = faiss.IndexFlatL2(dimension)
    
    faiss.write_index(index, settings.FAISS_INDEX_PATH)
    with open(settings.FAISS_META_PATH, "w") as f:
        json.dump([], f)
    
    print(f" Created empty FAISS index at {settings.FAISS_INDEX_PATH}")

if __name__ == "__main__":
    init_empty_index()