from app.services.retrieval import RetrievalService


def paediatrics_agent(state: dict, retriever: RetrievalService) -> dict:
    result = retriever.retrieve(state["query"])
    return {**state, **result}
