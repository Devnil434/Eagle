import { useState, useEffect } from "react";

export default function TimelineView({ invId }) {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!invId) return;
    setLoading(true);
    fetch(`/bookmarks/investigations/${invId}/timeline`)
      .then((r) => r.ok ? r.json() : [])
      .then((data) => setEvents(data))
      .catch((e) => console.error("Failed to load timeline", e))
      .finally(() => setLoading(false));
  }, [invId]);

  if (loading) return <p className="text-zinc-500">Loading timeline...</p>;
  if (events.length === 0) return <p className="text-zinc-500">No events in timeline.</p>;

  return (
    <div className="space-y-4">
      <h3 className="text-white font-semibold">Timeline</h3>
      <div className="relative pl-6 border-l-2 border-zinc-700">
        {events.map((evt, idx) => (
          <div key={evt.id} className="mb-4 relative">
            <div className="absolute -left-[9px] top-1 w-4 h-4 rounded-full border-2 border-green-500 bg-zinc-900" />
            <div className="bg-zinc-800 rounded border border-zinc-700 p-3">
              <div className="flex justify-between">
                <span className="text-white font-medium">{evt.label}</span>
                <span className="text-xs text-zinc-400">{new Date(evt.timestamp * 1000).toLocaleString()}</span>
              </div>
              <div className="text-xs text-zinc-400 mt-1">Camera: {evt.camera_id} | Track: {evt.track_id}</div>
              {evt.notes && <div className="text-xs text-zinc-300 mt-1 italic">"{evt.notes}"</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
