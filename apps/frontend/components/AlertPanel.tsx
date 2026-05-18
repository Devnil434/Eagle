"use client";

import { useEffect, useState } from "react";

type Alert = {
  id?: string;
  label: string;
  reason: string;
  confidence: number;
};

export default function AlertPanel() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [status, setStatus] = useState("connecting");

  useEffect(() => {
    const eventSource = new EventSource(
      "http://localhost:8000/alerts/stream"
    );

    // ✅ connection opened
    eventSource.onopen = () => {
      console.log("SSE connected");
      setStatus("connected");
    };

    // ✅ receiving messages
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        setAlerts((prev) => [data, ...prev]);
      } catch (err) {
        console.log("Invalid SSE data:", event.data);
      }
    };

    // ❌ error handler (IMPORTANT FIXED)
    eventSource.onerror = () => {
      console.log("SSE error - retrying automatically...");
      setStatus("reconnecting");
    };

    return () => {
      eventSource.close();
    };
  }, []);

  return (
    <div style={{ padding: "20px" }}>
      <h2>🦅 Live Alerts</h2>

      <p>Status: {status}</p>

      {alerts.length === 0 ? (
        <p>No alerts yet...</p>
      ) : (
        alerts.map((alert, index) => (
          <div
            key={alert.id || index}
            style={{
              border: "1px solid #ddd",
              padding: "10px",
              marginBottom: "10px",
            }}
          >
            <h3>{alert.label}</h3>
            <p>{alert.reason}</p>
            <small>Confidence: {alert.confidence}</small>
          </div>
        ))
      )}
    </div>
  );
}