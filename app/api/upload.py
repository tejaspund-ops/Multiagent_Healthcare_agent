from fastapi import APIRouter, UploadFile, File, Depends
from pathlib import Path
from app.ingestion.pdf_loader import load_pdf_text
from app.ingestion.chunker import chunk_text
from app.ingestion.embeddings import EmbeddingService
from app.vectorstore.faiss_store import FAISSStore
from app.api.auth import verify_token

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

embedding_service = EmbeddingService()


@router.post("/upload")
def upload_pdf(
    file: UploadFile = File(...),
    user=Depends(verify_token)
):
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    text = load_pdf_text(str(file_path))
    chunks = chunk_text(text)

    embeddings = embedding_service.embed(chunks)

    store = FAISSStore(dim=len(embeddings[0]))
    store.add(
        embeddings,
        [{"text": c, "source": file.filename} for c in chunks]
    )
    store.save()

    return {
        "status": "success",
        "chunks_indexed": len(chunks)
    }