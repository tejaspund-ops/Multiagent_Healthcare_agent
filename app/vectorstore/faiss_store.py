import faiss
import json
import numpy as np
import os 
from pathlib import Path
from app.core.config import settings

class FAISSStore:
    def __init__(self, dim: int):
        self.dim = dim 
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add(self, embeddings: list[list[float]], metadatas: list[dict]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.metadata.extend(metadatas)

    def search(self, query_embedding, top_k: int = 5):
        query = np.array([query_embedding]).astype("float32")
        scores, indices = self.index.search(query, top_k)

        results = []

        metric = getattr(self.index, 'metric_type', None)
        is_l2 = False
        if metric is not None:
            is_l2 = metric == getattr(faiss, 'METRIC_L2', None)
        else:
            cls_name = self.index.__class__.__name__
            is_l2 = 'L2' in cls_name or 'IndexFlatL2' in cls_name

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            if is_l2:
                sim = 1.0 - 0.5 * float(score)
            else:
                sim = float(score)

            results.append({
                "score": sim,
                "metadata": self.metadata[idx]
            })

        return results

    def save(self):
        Path(settings.FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, settings.FAISS_INDEX_PATH)
        with open(settings.FAISS_META_PATH, "w") as f:
            json.dump(self.metadata, f)

    def load(self):
        if os.path.exists(settings.FAISS_INDEX_PATH):
            self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
            try:
                self.dim = int(getattr(self.index, 'd', self.dim))
            except Exception:
                pass

            if os.path.exists(settings.FAISS_META_PATH):
                with open(settings.FAISS_META_PATH) as f:
                    self.metadata = json.load(f)
            print("Successfully loaded existing FAISS index.")
        else:
            print(f"No index found at {settings.FAISS_INDEX_PATH}. Starting with empty store.")
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []


            