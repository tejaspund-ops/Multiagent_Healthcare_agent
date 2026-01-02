import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai 
from app.core.config import settings

class GeminiWrapper:
    def __init__(self, temperature: float):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_id = "gemini-2.0-flash"  
        self.temperature = temperature

    def invoke(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            generation_config={'temperature': self.temperature}
        )
        return response.text

def get_llm(temperature: float = 0.2):
    provider = settings.LLM_PROVIDER

    if provider == "groq":
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=temperature
        )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temperature,
            google_api_key=settings.GEMINI_API_KEY
        )

    raise ValueError("Unsupported LLM provider")
