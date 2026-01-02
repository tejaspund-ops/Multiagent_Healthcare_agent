from langgraph.graph import StateGraph,END
from app.graph.state import GraphState
from app.agents.router import router_agent
from app.agents.cardiology import cardiology_agent
from app.agents.paediatrics import paediatrics_agent
from app.agents.dermatology import dermatology_agent
from app.agents.final_agent import final_agent


ROUTING_THRESHOLD = 0.6

async def route_by_confidence(state: dict) -> str:
    routing = state["routing"]

    if not routing:
        return "final"

    if routing["confidence"] < ROUTING_THRESHOLD:
        return "final"

    return routing["domain"]


async def build_graph(retriever):
    graph = StateGraph(GraphState)

    graph.add_node("router", router_agent)
    graph.add_node("cardiology", lambda s: cardiology_agent(s, retriever))
    graph.add_node("paediatrics", lambda s: paediatrics_agent(s, retriever))
    graph.add_node("dermatology", lambda s: dermatology_agent(s, retriever))
    graph.add_node("final", final_agent)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_by_confidence,
        {
            "cardiology": "cardiology",
            "paediatrics": "paediatrics",
            "dermatology": "dermatology",
            "general":"final",
            "final": "final"
        }
    )

    graph.add_edge("cardiology", "final")
    graph.add_edge("paediatrics", "final")
    graph.add_edge("dermatology", "final")
    graph.add_edge("final", END)

    return graph.compile()
