"""
Scene Graph Manager for surveillance reasoning.
Builds a dynamic NetworkX MultiDiGraph from a TrackedFrame
and serializes it into an LLM-ready prompt snippet.
"""
from __future__ import annotations

import networkx as nx
from libs.schemas.graph import GraphNode, GraphEdge, NodeType, EdgeType


class SceneGraph:
    """Wraps a NetworkX MultiDiGraph with helper methods for surveillance scenes."""

    MAX_PROMPT_TOKENS = 300

    def __init__(self, timestamp: float = 0.0):
        self.timestamp = timestamp
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def add_node(self, node: GraphNode) -> None:
        self.graph.add_node(
            node.id,
            node_type=node.node_type.value,
            label=node.label or node.id,
        )

    def add_edge(self, edge: GraphEdge) -> None:
        attrs = {"edge_type": edge.edge_type.value}
        if edge.distance_px is not None:
            attrs["distance_px"] = edge.distance_px
        self.graph.add_edge(edge.source, edge.target, **attrs)

    @classmethod
    def from_tracked_frame(cls, frame) -> "SceneGraph":
        sg = cls(timestamp=getattr(frame, "timestamp", 0.0))

        # Add zones (skip if missing id)
        for zone in getattr(frame, "zones", []):
            zone_id = zone.get("id")
            if zone_id:
                sg.add_node(GraphNode(id=zone_id, node_type=NodeType.ZONE))

        # Add objects and their belongs_to relationships
        for obj in getattr(frame, "objects", []):
            obj_id = obj.get("id")
            if not obj_id:
                continue
            sg.add_node(GraphNode(id=obj_id, node_type=NodeType.OBJECT))
            belongs_to = obj.get("belongs_to")
            if belongs_to:
                sg.add_node(GraphNode(id=belongs_to, node_type=NodeType.ZONE))
                sg.add_edge(GraphEdge(
                    source=obj_id,
                    target=belongs_to,
                    edge_type=EdgeType.INSIDE,
                ))

        # Add persons and their relationships
        for person in getattr(frame, "persons", []):
            pid = person.get("id")
            if not pid:
                continue
            sg.add_node(GraphNode(id=pid, node_type=NodeType.PERSON))

            # INSIDE relationship
            zone = person.get("zone")
            if zone:
                sg.add_node(GraphNode(id=zone, node_type=NodeType.ZONE))
                sg.add_edge(GraphEdge(
                    source=pid,
                    target=zone,
                    edge_type=EdgeType.INSIDE,
                ))

            # NEAR relationships
            for nearby in person.get("nearby_objects", []):
                obj_id = nearby.get("id")
                if not obj_id:
                    continue
                dist = nearby.get("distance_px")
                sg.add_node(GraphNode(id=obj_id, node_type=NodeType.OBJECT))
                sg.add_edge(GraphEdge(
                    source=pid,
                    target=obj_id,
                    edge_type=EdgeType.NEAR,
                    distance_px=dist,
                ))

            # INTERACTING_WITH relationships
            for obj_id in person.get("interacting_with", []):
                if not obj_id:
                    continue
                sg.add_node(GraphNode(id=obj_id, node_type=NodeType.OBJECT))
                sg.add_edge(GraphEdge(
                    source=pid,
                    target=obj_id,
                    edge_type=EdgeType.INTERACTING_WITH,
                ))

        return sg

    def to_prompt_str(self) -> str:
        lines = [f"Scene graph at t={self.timestamp:.1f}s:", ""]

        for src, dst, data in self.graph.edges(data=True):
            edge_type = data.get("edge_type", "RELATED_TO")
            dist = data.get("distance_px")
            if edge_type == EdgeType.NEAR.value and dist is not None:
                edge_label = f"{edge_type}({int(dist)}px)"
            else:
                edge_label = edge_type
            lines.append(f"{src} \u2192 {edge_label} \u2192 {dst}")

        serialized = "\n".join(lines)
        # Rough trim if over 200 words (≈260 tokens, safe under 300)
        words = serialized.split()
        if len(words) > 200:
            serialized = " ".join(words[:200]) + "\n[...truncated]"
        return serialized

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()
