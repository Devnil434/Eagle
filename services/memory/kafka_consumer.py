"""
kafka_consumer.py — Kafka consumer for track events.
"""
from __future__ import annotations
import json
import logging
from confluent_kafka import Consumer, KafkaError
from libs.config.settings import settings
from libs.schemas.memory import TrackEvent
from services.memory.memory import MemoryStore

logger = logging.getLogger(__name__)

class KafkaEventConsumer:
    """
    Consumes track events from Kafka and writes them to MemoryStore.
    """
    def __init__(self, bootstrap_servers: str = None, topic: str = None, group_id: str = "eagle-group"):
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.topic = topic or settings.kafka_topic
        self.group_id = group_id
        
        conf = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        }
        
        try:
            self.consumer = Consumer(conf)
            self.consumer.subscribe([self.topic])
            logger.info(f"Kafka Consumer subscribed to {self.topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka Consumer: {e}")
            self.consumer = None

    def run(self, store: MemoryStore):
        """
        Start the consumption loop.
        """
        if self.consumer is None:
            return

        logger.info("Starting Kafka consumption loop...")
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        break

                success = self._process_message(msg, store)
                if success:
                    try:
                        self.consumer.commit(message=msg, asynchronous=False)
                    except Exception as e:
                        logger.error(f"Failed to commit message offset: {e}")
        except KeyboardInterrupt:
            pass
        finally:
            self.consumer.close()

    def _process_message(self, msg, store: MemoryStore) -> bool:
        """Process a single Kafka message."""
        try:
            data = json.loads(msg.value().decode('utf-8'))
            event = TrackEvent(**data)
            
            # Deduplication logic using Redis
            # Key: dedupe:{camera_id}:{track_id}:{frame_id}
            dedupe_key = f"dedupe:{event.camera_id}:{event.track_id}:{event.frame_id}"
            
            # setnx returns True if the key was set (didn't exist)
            if not store._r.set(dedupe_key, "1", nx=True, ex=600): # 10 min TTL
                logger.debug(f"Duplicate event ignored: {dedupe_key}")
                return True

            # Store in Redis via MemoryStore
            store.store_event(event)
            return True

        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            return False
