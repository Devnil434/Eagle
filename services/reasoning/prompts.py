def build_reasoning_prompt(graph_text: str) -> str:

    prompt = f"""
You are an AI surveillance reasoning system.

Analyze the following scene graph and identify:

- suspicious behavior
- restricted zone violations
- unusual interactions
- possible threats

Scene Graph:
{graph_text}

Provide a short reasoning summary.
"""

    return prompt