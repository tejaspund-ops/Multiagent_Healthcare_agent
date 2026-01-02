from app.services.llm import get_llm
from app.services.web_search import web_search
import logging
logger=logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ANSWER_PROMPT = """
You are a healthcare assistant.
Answer ONLY using the provided context.
If context is insufficient, say so clearly.

Context:
{context}

User Question:
{query}

Answer:
"""

FOLLOWUP_PROMPT = """
Suggest up to 3 short follow-up questions
based on the answer.
"""


async def final_agent(state: dict) -> dict:
    llm = get_llm()

    def call_llm(prompt: str) -> str:
        try:
            result = llm.invoke(prompt)
            logger.info(f"LLM Response {result}.")
            return result.content if hasattr(result, "content") else result
        except Exception as e:
            logger.exception("LLM invocation failed")
            return "(LLM error: failed to generate response)"
    
    summery=state.get("memory_summary","")

    # Case 1: Vector DB success
    if state.get("status") == "FOUND":
        docs = state.get("results", [])
        context = "\n".join(d["metadata"].get("text", "") for d in docs)

        answer = call_llm(
            ANSWER_PROMPT.format(
                summery=summery,    
                context=context,
                query=state["query"]
            )
        )

        followups = call_llm(FOLLOWUP_PROMPT)

        return {
            **state,
            "answer": answer,
            "follow_up": followups,
            "source": "vector_db"
        }

    web_context = await web_search(state["query"])

    if not web_context:
        return {
            **state,
            "answer": "Reliable information could not be found.",
            "source": "none"
        }

    answer = call_llm(
        ANSWER_PROMPT.format(
            context=web_context,
            query=state["query"]
        )
    )

    return {
        **state,
        "answer": answer,
        "source": "web"
    }


