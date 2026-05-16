# services/reasoning/scene_graph.py

import math

import networkx as nx

from libs.observability.metrics import reasoning_triggers_total
from services.detection.zones import DEFAULT_ZONES

INTERACTION_OBJECTS = [
    "backpack",
    "handbag",
    "cell phone",
    "laptop"
]
class SceneGraphBuilder:
    def serialize_graph(self):

        serialized = []

        for source, target, data in self.graph.edges(data=True):

            relation = data.get("relation")

            edge_text = f"{source} -> [{relation}] -> {target}"

            if "distance" in data:

                edge_text += f" (distance={data['distance']})"

            serialized.append(edge_text)

        return "\n".join(serialized)

    def __init__(self, det_frame):

        self.det_frame = det_frame
        self.graph = nx.MultiDiGraph()

    def calculate_distance(self, center1, center2):

        x1, y1 = center1
        x2, y2 = center2

        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def build_graph(self):

        reasoning_triggers_total.inc()

        self.graph.clear()

        # Add zone nodes
        for zone in DEFAULT_ZONES:

            self.graph.add_node(
                zone.name,
                type="zone"
            )

        detections = self.det_frame.detections

        # Add detection nodes
        for idx, det in enumerate(detections):

            node_name = f"{det.label}_{idx}"

            self.graph.add_node(
                node_name,
                type=det.label,
                center=det.center
            )

            # Zone relationships
            for zone_name in det.zones_present:

                self.graph.add_edge(
                    node_name,
                    zone_name,
                    relation="INSIDE"
                )

        # Object relationships
        for i in range(len(detections)):

            for j in range(i + 1, len(detections)):

                det1 = detections[i]
                det2 = detections[j]

                node1 = f"{det1.label}_{i}"
                node2 = f"{det2.label}_{j}"

                dist = self.calculate_distance(
                    det1.center,
                    det2.center
                )

                # NEAR relationship
                if dist < 150:

                    self.graph.add_edge(
                        node1,
                        node2,
                        relation="NEAR",
                        distance=round(dist, 2)
                    )

                person_node = None
                object_node = None

                # Interaction detection
                if (
                    det1.label == "person"
                    and det2.label in INTERACTION_OBJECTS
                ):

                    person_node = node1
                    object_node = node2

                elif (
                    det2.label == "person"
                    and det1.label in INTERACTION_OBJECTS
                ):

                    person_node = node2
                    object_node = node1

                # INTERACTING_WITH
                if person_node and dist < 60:

                    self.graph.add_edge(
                        person_node,
                        object_node,
                        relation="INTERACTING_WITH"
                    )

                # HOLDING
                elif person_node and 60 <= dist < 80:

                    self.graph.add_edge(
                        person_node,
                        object_node,
                        relation="HOLDING"
                    )

        return self.graph