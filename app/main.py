from fastapi import FastAPI
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router

app=FastAPI(
    title="Multiagent Healthcare System",
    description="A multiagent system for healthcare queries using specialized agents in various medical domains.",
    version="1.0.0"
)

app.include_router(upload_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.get("/")
def health_check():
    return {"status": "ok"}