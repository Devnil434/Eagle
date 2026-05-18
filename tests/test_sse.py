from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_sse_headers():

    response = client.get("/alerts/stream")

    assert response.status_code == 200

    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )