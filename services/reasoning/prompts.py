"""
Prompt builders for LLM-based surveillance reasoning.
"""


# ---------------------------------------------------------------------------
# Scene Graph Prompt Integration
# ---------------------------------------------------------------------------

from services.reasoning.scene_graph import SceneGraph


def build_reasoning_prompt(event_description: str, scene_graph: SceneGraph) -> str:
    """
    Combine a scene graph snapshot with a natural-language event description
    into a single structured prompt for LLM reasoning.

    Keeps total context compact and well under model context limits.
    """
    graph_context = scene_graph.to_prompt_str()

    prompt = f"""{graph_context}

Event description:
{event_description}

Based on the scene graph and event above, analyze whether this activity is suspicious.
Consider spatial relationships, zone access, and object interactions.
Be concise and structured in your response."""

    return prompt
