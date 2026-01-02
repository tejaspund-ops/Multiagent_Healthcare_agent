from typing import TypedDict, Optional, List, Literal


class RoutingDecision(TypedDict):
    domain: Literal["cardiology", "paediatrics", "dermatology", "general"]
    confidence: float
    reasoning: str


class GraphState(TypedDict):
    query: str
    routing: Optional[RoutingDecision]
    retrieved_docs: Optional[List[dict]]
    status: Optional[str]
    answer: Optional[str]
    memory_summary: Optional[str]
