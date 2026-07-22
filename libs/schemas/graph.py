"""
Pydantic schemas for Scene Graph nodes and edges.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class NodeType(str, Enum):
    PERSON = "Person"
    ZONE = "Zone"
    OBJECT = "Object"


class EdgeType(str, Enum):
    INSIDE = "INSIDE"
    NEAR = "NEAR"
    INTERACTING_WITH = "INTERACTING_WITH"
    ENTERED_FROM = "ENTERED_FROM"


class GraphNode(BaseModel):
    id: str                        # e.g. "Person #3", "Keypad_01"
    node_type: NodeType
    label: Optional[str] = None    # human-readable label


class GraphEdge(BaseModel):
    source: str                    # node id
    target: str                    # node id
    edge_type: EdgeType
    distance_px: Optional[float] = None   # for NEAR edges
