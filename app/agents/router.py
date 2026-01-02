import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.services.llm import get_llm
import logging
logger=logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ROUTER_PROMPT = """You are a medical query router.
Classify the query into ONE domain: cardiology, paediatrics, dermatology, or general.

Rules:
- Output MUST be valid JSON.
- Include a confidence score (0.0 to 1.0).

JSON Schema:
{{
  "domain": "string",
  "confidence": float,
  "reasoning": "string"
}}
User query: {query}
"""

def router_agent(state: dict) -> dict:
    """
    Routes medical queries using a structured JSON approach.
    Updates the graph state with the routing decision.
    """
    llm = get_llm(temperature=0)
    parser = JsonOutputParser()
    
    prompt_template = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    chain = prompt_template | llm | parser

    try:
        decision = chain.invoke({"query": state["query"]})
        logger.info(f"Routing decision: {decision}")
    except Exception as e:
        decision = {
            "domain": "general",
            "confidence": 0.0,
            "reasoning": f"Router failed: {str(e)}"
        }

    return {**state, "routing": decision}
