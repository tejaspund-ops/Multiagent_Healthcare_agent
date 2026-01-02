from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = "data/upload/index.faiss"
    FAISS_META_PATH: str = "data/upload/meta.json"
    SIMILARITY_THRESHOLD: float = 0.35
    
    GROQ_API_KEY: str =None
    GEMINI_API_KEY: str =None
    LLM_PROVIDER: str = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore" 
    )

settings = Settings()