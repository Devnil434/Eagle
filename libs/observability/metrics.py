from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.registry import REGISTRY


# Prevent duplicate metric registration
def metric_exists(name: str) -> bool:
    try:
        REGISTRY.get_sample_value(name)
        return True
    except Exception:
        return False


# Frames processed counter
if "agentic_frames_processed" not in REGISTRY._names_to_collectors:

    frames_processed_total = Counter(
        "agentic_frames_processed_total", "Total number of processed frames"
    )

else:

    frames_processed_total = REGISTRY._names_to_collectors["agentic_frames_processed"]


# Active tracks gauge
if "agentic_active_tracks" not in REGISTRY._names_to_collectors:

    active_tracks = Gauge("agentic_active_tracks", "Current active tracks")

else:

    active_tracks = REGISTRY._names_to_collectors["agentic_active_tracks"]


# Reasoning trigger counter
if "agentic_reasoning_triggers" not in REGISTRY._names_to_collectors:

    reasoning_triggers_total = Counter(
        "agentic_reasoning_triggers_total", "Total reasoning trigger executions"
    )

else:

    reasoning_triggers_total = REGISTRY._names_to_collectors["agentic_reasoning_triggers"]


# Redis latency histogram
if "agentic_redis_write_latency_seconds" not in REGISTRY._names_to_collectors:

    redis_write_latency = Histogram(
        "agentic_redis_write_latency_seconds",
        "Redis write latency in seconds",
        buckets=[5, 10, 20, 30, 60],
    )

else:

    redis_write_latency = REGISTRY._names_to_collectors["agentic_redis_write_latency_seconds"]


# Track dwell histogram
if "agentic_track_dwell_seconds" not in REGISTRY._names_to_collectors:

    track_dwell_seconds = Histogram(
        "agentic_track_dwell_seconds", "Track dwell duration", buckets=[5, 10, 20, 30, 60]
    )

else:
    
    track_dwell_seconds = REGISTRY._names_to_collectors["agentic_track_dwell_seconds"]
    
    # Workflow execution counter
    if "agentic_workflow_executions_total" not in REGISTRY._names_to_collectors:
        
        workflow_executions_total = Counter(
            "agentic_workflow_executions_total",
            "Total workflow executions"
        )

    else:
        
        workflow_executions_total = REGISTRY._names_to_collectors[
            "agentic_workflow_executions_total"
        ]
        
    # Workflow failures counter
    if "agentic_workflow_failures_total" not in REGISTRY._names_to_collectors:

        workflow_failures_total = Counter(
            "agentic_workflow_failures_total",
            "Total workflow failures"
        )

    else:

        workflow_failures_total = REGISTRY._names_to_collectors[
            "agentic_workflow_failures_total"
        ]


    # Workflow duration histogram
    if "agentic_workflow_duration_seconds" not in REGISTRY._names_to_collectors:

        workflow_duration_seconds = Histogram(
            "agentic_workflow_duration_seconds",
            "Workflow execution duration",
            buckets=[1, 5, 10, 30, 60]
        )

    else:

        workflow_duration_seconds = REGISTRY._names_to_collectors[
            "agentic_workflow_duration_seconds"
        ]
