# tests/test_scene_graph.py





from services.detection.detection import Detection, DetectionFrame
from services.reasoning.scene_graph import SceneGraphBuilder


def test_scene_graph_build():

    
    person = Detection(
        label="person",
        bbox=[600, 250, 700, 430],
        confidence=0.96,
        center=(650, 340),
        zones_present=[
            "restricted_door",
            "keypad_area"
        ]
    )

    phone = Detection(
        label="cell phone",
        bbox=[660, 330, 690, 360],
        confidence=0.90,
        center=(675, 345),
        zones_present=[
            "restricted_door",
            "keypad_area"
        ]
    )

    # Backpack close to person

    backpack = Detection(
        label="backpack",
        bbox=[610, 320, 680, 410],
        confidence=0.88,
        center=(645, 365),
        zones_present=[
            "restricted_door",
            "keypad_area"
        ]
    )

    person2 = Detection(
        label="person",
        bbox=[100, 100, 220, 350],
        confidence=0.93,
        center=(160, 225),
        zones_present=[
            "safe_corridor"
        ]
    )

    det_frame = DetectionFrame(
        frame_id=25,
        detections=[
            person,
            phone,
            backpack,
            person2
        ],
        timestamp_ms=5000
    )

    builder = SceneGraphBuilder(det_frame)

    graph = builder.build_graph()

    assert graph.has_node("person_0")

    assert graph.has_node("cell phone_1")

    assert graph.has_node("backpack_2")

    assert graph.has_node("person_3")

    assert graph.has_node("restricted_door")

    assert graph.has_node("keypad_area")

    assert graph.has_node("safe_corridor")

    assert graph.has_edge(
        "person_0",
        "restricted_door"
    )

    assert graph.has_edge(
        "person_0",
        "keypad_area"
    )

    assert graph.has_edge(
        "person_3",
        "safe_corridor"
    )

    

    assert graph.has_edge(
        "person_0",
        "cell phone_1"
    )

    assert graph.has_edge(
        "person_0",
        "backpack_2"
    )

    # Print graph nodes

    print("\nGRAPH NODES:\n")

    for node, data in graph.nodes(data=True):
        print(node, data)


    # Print graph edges

    print("\nGRAPH EDGES:\n")

    for source, target, data in graph.edges(data=True):
        print(f"{source} ---> {target} | {data}")
