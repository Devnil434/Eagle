"""
kafka_producer.py — Kafka producer for track events.
"""
from __future__ import annotations
import logging
from confluent_kafka import Producer
from libs.config.settings import settings
from libs.schemas.memory import TrackEvent

logger = logging.getLogger(__name__)

class KafkaEventProducer:
    """
    Produces track events to a Kafka topic.
    """
    def __init__(self, bootstrap_servers: str = None, topic: str = None):
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.topic = topic or settings.kafka_topic
        
        conf = {
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': 'eagle-producer'
        }
        
        try:
            self.producer = Producer(conf)
            logger.info(f"Kafka Producer initialized for {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka Producer: {e}")
            self.producer = None

    def produce_event(self, event: TrackEvent):
        """
        Produce a TrackEvent to Kafka.
        """
        if self.producer is None:
            return

        try:
            # Serialize event to JSON
            payload = event.model_dump_json()
            
            # Use track_id as key for ordering within a partition
            self.producer.produce(
                self.topic,
                key=str(event.track_id),
                value=payload,
                callback=self._delivery_report
            )
            # Serve delivery reports (callbacks)
            self.producer.poll(0)
        except Exception as e:
            logger.error(f"Failed to produce event to Kafka: {e}")

    def flush(self):
        """Flush pending messages."""
        if self.producer:
            self.producer.flush()

    def _delivery_report(self, err, msg):
        """Called once for each message produced to indicate delivery result."""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            pass # Message delivered successfully
