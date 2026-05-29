"""
test_kafka_producer.py — Unit tests for KafkaProducer integration.
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from libs.schemas.memory import TrackEvent
from services.memory.kafka_producer import KafkaEventProducer

@pytest.fixture
def mock_producer():
    with patch('services.memory.kafka_producer.Producer') as mocked:
        yield mocked

def test_producer_initialization(mock_producer):
    producer = KafkaEventProducer(bootstrap_servers="localhost:9092", topic="test-topic")
    mock_producer.assert_called_once()
    assert producer.topic == "test-topic"

def test_produce_event_calls_kafka(mock_producer):
    # Setup
    producer = KafkaEventProducer(bootstrap_servers="localhost:9092", topic="test-topic")
    mock_kafka_instance = mock_producer.return_value
    
    event = TrackEvent(
        track_id=1,
        frame_id=10,
        timestamp_ms=123456789.0,
        camera_id="cam_01",
        confidence=0.99
    )
    
    # Execute
    producer.produce_event(event)
    
    # Verify
    mock_kafka_instance.produce.assert_called_once()
    args, kwargs = mock_kafka_instance.produce.call_args
    assert args[0] == "test-topic"
    assert kwargs['key'] == "1"
    assert "track_id\":1" in kwargs['value']

def test_producer_handles_none_instance():
    with patch('services.memory.kafka_producer.Producer', side_effect=Exception("Connection failed")):
        producer = KafkaEventProducer()
        assert producer.producer is None
        
        # Should not raise exception
        producer.produce_event(MagicMock())
