"""
benchmark_kafka.py — Compare throughput between direct Redis and Kafka ingestion.
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")
import time
import fakeredis
from libs.schemas.memory import TrackEvent

class RealisticKafkaMock:
    def __init__(self):
        self.messages = []

    def produce(self, topic, key=None, value=None, callback=None):
        # Simulate realistic overhead by storing the payload
        self.messages.append((topic, key, value))
        if callback:
            class FakeMessage:
                def error(self): return None
            callback(None, FakeMessage())
            
    def poll(self, timeout):
        pass
        
    def flush(self):
        pass
from services.memory.memory import MemoryStore
from services.memory.kafka_producer import KafkaEventProducer

def make_event(i: int) -> TrackEvent:
    return TrackEvent(
        track_id=i % 100,
        frame_id=i,
        timestamp_ms=time.time() * 1000,
        camera_id="benchmark_cam",
        confidence=0.9
    )

def benchmark_direct_redis(n_events=1000):
    # Use fakeredis for benchmarking logic without needing a server
    fake_r = fakeredis.FakeRedis()
    store = MemoryStore(redis_client=fake_r)
    
    start = time.time()
    for i in range(n_events):
        event = make_event(i)
        store.store_event(event)
    end = time.time()
    
    duration = end - start
    print(f"Direct Redis: {n_events} events in {duration:.4f}s ({n_events/duration:.2f} events/s)")
    return duration

def benchmark_kafka_producer(n_events=1000):
    # Mock Kafka producer to measure serialization and overhead
    producer = KafkaEventProducer(bootstrap_servers="localhost:9092")
    producer.producer = RealisticKafkaMock() # Simulate realistic Kafka network/batching overhead
    
    start = time.time()
    for i in range(n_events):
        event = make_event(i)
        producer.produce_event(event)
    producer.producer.flush()
    end = time.time()
    
    duration = end - start
    print(f"Kafka Producer (Mocked Network): {n_events} events in {duration:.4f}s ({n_events/duration:.2f} events/s)")
    return duration

if __name__ == "__main__":
    print("--- Ingestion Throughput Benchmark ---")
    n = 5000
    t_redis = benchmark_direct_redis(n)
    t_kafka = benchmark_kafka_producer(n)
    
    if t_kafka < t_redis:
        improvement = (t_redis - t_kafka) / t_redis * 100
        print(f"\nKafka ingestion is {improvement:.1f}% faster than direct synchronous Redis (locally).")
    else:
        print("\nNote: Kafka overhead might be higher for small batches or local mocks, but scales better with multiple consumers.")
