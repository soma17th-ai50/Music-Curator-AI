"""
LangGraph workflow for the music recommendation agent.

Graph topology:
  START
    └─► analyze_input
          ├── (recommend) ──► load_profile ──► search_music ──► generate_response ──► END
          ├── (feedback)  ──► load_profile ──► update_profile ──► search_music ──► generate_response ──► END
          ├── (setup)     ──► load_profile ──► update_profile ──► generate_response ──► END
          └── (chat)      ──► generate_response ──► END
"""
from langgraph.graph import END, START, StateGraph

from .nodes import (
    analyze_input,
    generate_response,
    load_profile,
    route_after_load,
    route_after_update,
    route_intent,
    search_music,
    update_profile,
)
from .state import AgentState


def create_graph():
    builder = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    builder.add_node("analyze_input",    analyze_input)
    builder.add_node("load_profile",     load_profile)
    builder.add_node("update_profile",   update_profile)
    builder.add_node("search_music",     search_music)
    builder.add_node("generate_response", generate_response)

    # ── Edges ──────────────────────────────────────────────────────────────
    builder.add_edge(START, "analyze_input")

    # After intent classification: branch by intent
    builder.add_conditional_edges(
        "analyze_input",
        route_intent,
        {
            "recommend": "load_profile",
            "feedback":  "load_profile",
            "setup":     "load_profile",
            "chat":      "generate_response",
        },
    )

    # After loading profile: branch by intent
    builder.add_conditional_edges(
        "load_profile",
        route_after_load,
        {
            "search":  "search_music",
            "update":  "update_profile",
            "respond": "generate_response",
        },
    )

    # After updating profile: re-search (feedback) or respond (setup)
    builder.add_conditional_edges(
        "update_profile",
        route_after_update,
        {
            "search":  "search_music",
            "respond": "generate_response",
        },
    )

    builder.add_edge("search_music",     "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()


# Module-level singleton so Streamlit doesn't rebuild on every rerun
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = create_graph()
    return _graph
