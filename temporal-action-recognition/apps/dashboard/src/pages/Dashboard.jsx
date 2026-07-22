import { useEffect, useState } from "react";
import CameraCard from "../components/CameraCard";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const CAMERA_ID = "cam_01";

export default function Dashboard() {
  const [selectedTrack, setSelectedTrack] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [tracks, setTracks] = useState([]);
  const [apiError, setApiError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchTracks() {
      try {
        const res = await fetch(`${API_BASE}/cameras/${CAMERA_ID}/tracks`);
        if (!res.ok) {
          throw new Error(`API ${res.status}`);
        }
        const data = await res.json();
        if (!cancelled) {
          setTracks(data);
          setApiError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setApiError(err.message);
        }
      }
    }

    fetchTracks();
    const id = setInterval(fetchTracks, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const cameras = tracks.length
    ? tracks.map((t) => ({
        id: t.track_id,
        title: `Camera ${CAMERA_ID}`,
        trackId: `P-${t.track_id}`,
        action: t.current_action,
        actionConfidence: t.action_confidence,
        actionSource: t.action_source,
      }))
    : [{ id: 0, title: "Waiting for pipeline…", trackId: "—", action: null }];

  const filtered = cameras.filter((cam) =>
    cam.trackId.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="flex h-screen bg-black text-white">
      <div className="flex-1 p-4">
        <input
          aria-label="Search Track ID"
          type="text"
          placeholder="Search Track ID..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full mb-4 px-4 py-2 rounded bg-zinc-900 text-white"
        />

        {apiError && (
          <p className="text-amber-400 text-sm mb-2">
            Backend offline or pipeline not running: {apiError}. Start Redis, API, and{" "}
            <code className="text-zinc-300">python scripts/run_pipeline.py --source 0</code>
          </p>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filtered.map((cam) => (
            <div
              key={cam.id}
              onClick={() => setSelectedTrack(cam)}
              className={`cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-2xl ${
                selectedTrack?.id === cam.id
                  ? "border-2 border-green-500 scale-105 rounded-lg shadow-green-500/40 shadow-2xl"
                  : ""
              }`}
            >
              <CameraCard
                title={cam.title}
                trackId={cam.trackId}
                action={cam.action}
                actionConfidence={cam.actionConfidence}
                actionSource={cam.actionSource}
              />
            </div>
          ))}
        </div>
      </div>

      <div className="w-80 bg-zinc-950 border-l border-zinc-800 p-4">
        {selectedTrack ? (
          <>
            <h2 className="text-2xl font-bold mb-4">Identity Panel</h2>
            <p className="mb-2">
              <span className="font-semibold">Camera:</span> {selectedTrack.title}
            </p>
            <p className="mb-2">
              <span className="font-semibold">Track ID:</span> {selectedTrack.trackId}
            </p>
            {selectedTrack.action && (
              <p className="mb-2 text-orange-400">
                <span className="font-semibold">Action:</span> {selectedTrack.action}
                {selectedTrack.actionConfidence != null &&
                  ` (${(selectedTrack.actionConfidence * 100).toFixed(0)}%)`}
                {selectedTrack.actionSource && (
                  <span className="text-zinc-500 text-sm"> via {selectedTrack.actionSource}</span>
                )}
              </p>
            )}
            <p className="text-green-400 animate-pulse">ACTIVE TRACK</p>
          </>
        ) : (
          <p>Select a camera track</p>
        )}
      </div>
    </div>
  );
}
