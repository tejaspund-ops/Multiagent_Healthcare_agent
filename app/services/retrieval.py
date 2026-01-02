import logging
from app.vectorstore.faiss_store import FAISSStore
from app.ingestion.embeddings import EmbeddingService
from app.core.config import settings

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

class RetrievalService:
    def __init__(self, store: FAISSStore):
        self.store = store
        try:
            self.embedder = EmbeddingService()
            logger.info("RetrievalService initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingService: {str(e)}")
            raise

    def retrieve(self, query: str):
        """
        Retrieves relevant documents based on the query string.
        """
        try:
            logger.info(f"Processing retrieval for query: '{query[:50]}...'")

            try:
                query_embedding = self.embedder.embed([query])[0]
            except Exception as e:
                logger.error(f"Embedding generation failed: {str(e)}")
                return {"status": "ERROR", "message": "Failed to process query vector."}

            try:
                results = self.store.search(query_embedding)
            except Exception as e:
                logger.error(f"Vector store search failed: {str(e)}")
                return {"status": "ERROR", "message": "Search engine error."}

            if not results:
                logger.warning(f"No results found in vector store for query.")
                return {"status": "NO_DATA_FOUND"}

            top_score = results[0].get("score", 0.0)
            if top_score < settings.SIMILARITY_THRESHOLD:
                logger.info(f"Top result score ({top_score:.4f}) below threshold ({settings.SIMILARITY_THRESHOLD}).")
                return {"status": "NO_DATA_FOUND"}

            logger.info(f"Successfully retrieved {len(results)} results. Top score: {top_score:.4f}")
            return {
                "status": "FOUND",
                "results": results
            }

        except Exception as e:

            logger.critical(f"Unexpected error in RetrievalService.retrieve: {str(e)}", exc_info=True)
            return {"status": "ERROR", "message": "An internal retrieval error occurred."}
