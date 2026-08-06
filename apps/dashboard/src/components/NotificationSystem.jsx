import { useState, useEffect } from "react";

export default function NotificationSystem({ operatorId = "operator-1", wsId }) {
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    if (!wsId) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/workspaces/${wsId}/activity?limit=5`);
        if (res.ok) {
          const data = await res.json();
          setNotifications(data.slice(0, 5));
        }
      } catch (e) {
        console.error("Failed to load notifications", e);
      }
    }, 30000);
    return () => clearInterval(interval);
  }, [wsId]);

  if (!wsId || notifications.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2">
      {notifications.map((n) => (
        <div key={n.id} className="bg-zinc-800 border border-zinc-700 rounded-lg p-3 shadow-lg max-w-sm">
          <div className="text-white text-sm font-medium">{n.action}</div>
          <div className="text-zinc-400 text-xs">{n.details}</div>
        </div>
      ))}
    </div>
  );
}
