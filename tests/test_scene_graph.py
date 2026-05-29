"""
Unit tests for SceneGraph builder and serializer.
"""
from types import SimpleNamespace
from services.reasoning.scene_graph import SceneGraph
from libs.schemas.graph import GraphNode, GraphEdge, NodeType, EdgeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frame(timestamp=22.4, persons=None, objects=None, zones=None):
    return SimpleNamespace(
        timestamp=timestamp,
        persons=persons or [],
        objects=objects  or [],
        zones=zones      or [],
    )


# ---------------------------------------------------------------------------
# Tests: manual graph construction
# ---------------------------------------------------------------------------

class TestSceneGraphManual:

    def test_add_person_node(self):
        sg = SceneGraph(timestamp=1.0)
        sg.add_node(GraphNode(id="Person #1", node_type=NodeType.PERSON))
        assert sg.node_count() == 1

    def test_add_edge(self):
        sg = SceneGraph()
        sg.add_node(GraphNode(id="Person #1", node_type=NodeType.PERSON))
        sg.add_node(GraphNode(id="Zone_A",    node_type=NodeType.ZONE))
        sg.add_edge(GraphEdge(source="Person #1", target="Zone_A", edge_type=EdgeType.INSIDE))
        assert sg.edge_count() == 1

    def test_near_edge_with_distance(self):
        sg = SceneGraph()
        sg.add_node(GraphNode(id="Person #3", node_type=NodeType.PERSON))
        sg.add_node(GraphNode(id="Keypad_01", node_type=NodeType.OBJECT))
        sg.add_edge(GraphEdge(
            source="Person #3", target="Keypad_01",
            edge_type=EdgeType.NEAR, distance_px=38.0,
        ))
        data = sg.graph.edges["Person #3", "Keypad_01", 0]
        assert data["distance_px"] == 38.0


# ---------------------------------------------------------------------------
# Tests: from_tracked_frame
# ---------------------------------------------------------------------------

class TestFromTrackedFrame:

    def test_empty_frame(self):
        frame = make_frame()
        sg = SceneGraph.from_tracked_frame(frame)
        assert sg.node_count() == 0
        assert sg.edge_count() == 0

    def test_person_inside_zone(self):
        frame = make_frame(persons=[
            {"id": "Person #3", "zone": "restricted_door",
             "nearby_objects": [], "interacting_with": []}
        ])
        sg = SceneGraph.from_tracked_frame(frame)
        assert sg.node_count() == 2          # person + zone
        assert sg.edge_count() == 1          # INSIDE

    def test_full_scene(self):
        frame = make_frame(
            timestamp=22.4,
            persons=[{
                "id": "Person #3",
                "zone": "restricted_door",
                "nearby_objects": [{"id": "Keypad_01", "distance_px": 38}],
                "interacting_with": ["Keypad_01"],
            }],
            objects=[{"id": "Keypad_01", "type": "keypad", "belongs_to": "restricted_door"}],
        )
        sg = SceneGraph.from_tracked_frame(frame)
        # Nodes: Person #3, restricted_door, Keypad_01  ? 3
        assert sg.node_count() == 3
        # Edges: INSIDE(person?zone), NEAR, INTERACTING_WITH, INSIDE(keypad?door) ? 4
        assert sg.edge_count() == 4

    def test_timestamp_stored(self):
        frame = make_frame(timestamp=99.9)
        sg = SceneGraph.from_tracked_frame(frame)
        assert sg.timestamp == 99.9


# ---------------------------------------------------------------------------
# Tests: serialization
# ---------------------------------------------------------------------------

class TestPromptSerialization:

    def _full_sg(self):
        frame = make_frame(
            timestamp=22.4,
            persons=[{
                "id": "Person #3",
                "zone": "restricted_door",
                "nearby_objects": [{"id": "Keypad_01", "distance_px": 38}],
                "interacting_with": ["Keypad_01"],
            }],
        )
        return SceneGraph.from_tracked_frame(frame)

    def test_header_present(self):
        sg = self._full_sg()
        prompt = sg.to_prompt_str()
        assert "t=22.4s" in prompt

    def test_near_distance_in_prompt(self):
        sg = self._full_sg()
        prompt = sg.to_prompt_str()
        assert "NEAR(38px)" in prompt

    def test_under_token_budget(self):
        sg = self._full_sg()
        prompt = sg.to_prompt_str()
        # Rough token estimate: words * 1.3
        estimated_tokens = len(prompt.split()) * 1.3
        assert estimated_tokens < 300, f"Prompt too long: ~{estimated_tokens:.0f} tokens"

    def test_empty_graph_serialization(self):
        sg = SceneGraph(timestamp=0.0)
        prompt = sg.to_prompt_str()
        assert "t=0.0s" in prompt
        assert "?" not in prompt      # no edges

