import { useState, useEffect, useCallback } from "react";
import CameraCard from "./CameraCard";

export interface CameraInfo {
  camera_id: string;
  label: string;
  location: string;
  snapshot_url: string;
  status: string;
  last_seen_ms: number;
  fps: number;
  max_fps: number;
  alert_count: number;
}

type Layout = "1x1" | "2x2" | "3x3";

const LAYOUTS: { value: Layout; label: string; cols: number }[] = [
  { value: "1x1", label: "1x1", cols: 1 },
  { value: "2x2", label: "2x2", cols: 2 },
  { value: "3x3", label: "3x3", cols: 3 },
];

export default function CameraGrid() {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [layout, setLayout] = useState<Layout>("2x2");
  const [fullscreenId, setFullscreenId] = useState<string | null>(null);
  const [error, setError] = useState("");

  const loadCameras = useCallback(async () => {
    try {
      const res = await fetch("/cameras/registry");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: CameraInfo[] = await res.json();
      setCameras(data);
      setError("");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to load cameras";
      setError(msg);
    }
  }, []);

  useEffect(() => { loadCameras(); }, [loadCameras]);

  useEffect(() => {
    const interval = setInterval(loadCameras, 5000);
    return () => clearInterval(interval);
  }, [loadCameras]);

  const handleAddCamera = async () => {
    const id = prompt("Camera ID (e.g. cam_05):");
    if (!id) return;
    const label = prompt("Label:") || id;
    try {
      const res = await fetch("/cameras/registry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_id: id, label }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await loadCameras();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to add camera");
    }
  };

  const handleRemoveCamera = async (cameraId: string) => {
    try {
      const res = await fetch(`/cameras/registry/${encodeURIComponent(cameraId)}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await loadCameras();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to remove camera");
    }
  };

  const handleFpsUpdate = async (cameraId: string, fps: number) => {
    try {
      const res = await fetch(`/cameras/registry/${encodeURIComponent(cameraId)}/fps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fps }),
      });
      if (!res.ok) return;
      await loadCameras();
    } catch {
      // silent
    }
  };

  const currentLayout = LAYOUTS.find((l) => l.value === layout) ?? LAYOUTS[1];

  if (fullscreenId) {
    const cam = cameras.find((c) => c.camera_id === fullscreenId);
    if (!cam) {
      setFullscreenId(null);
      return null;
    }
    return (
      <div className="fixed inset-0 bg-black z-50 flex flex-col">
        <div className="flex items-center justify-between p-3 bg-zinc-900">
          <h2 className="text-white text-lg font-bold">{cam.label}</h2>
          <button
            onClick={() => setFullscreenId(null)}
            className="px-3 py-1 bg-red-600 text-white rounded"
          >
            Exit Fullscreen
          </button>
        </div>
        <div className="flex-1 p-4">
          <CameraCard
            title={cam.label}
            cameraId={cam.camera_id}
            status={cam.status}
            fps={cam.fps}
            maxFps={cam.max_fps}
            fullscreen
          />
        </div>
      </div>
    );
  }

  return (
    <div className="p-4">
      <div className="flex items-center gap-3 mb-4">
        <h2 className="text-white text-xl font-bold">Multi-Camera Dashboard</h2>
        <div className="flex gap-2">
          {LAYOUTS.map((l) => (
            <button
              key={l.value}
              onClick={() => setLayout(l.value)}
              className={`px-3 py-1 rounded text-sm font-semibold ${
                layout === l.value
                  ? "bg-sky-500 text-white"
                  : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
              }`}
            >
              {l.label}
            </button>
          ))}
        </div>
        <button
          onClick={handleAddCamera}
          className="ml-auto px-3 py-1 bg-green-600 text-white rounded text-sm font-semibold hover:bg-green-500"
        >
          + Add Camera
        </button>
      </div>

      {error && (
        <p role="alert" className="text-red-400 text-sm mb-3">
          {error}
        </p>
      )}

      {cameras.length === 0 && !error && (
        <p className="text-zinc-500 text-sm">No cameras registered. Click + Add Camera to begin.</p>
      )}

      <div
        className="grid gap-3"
        style={{
          gridTemplateColumns: `repeat(${currentLayout.cols}, minmax(0, 1fr))`,
        }}
      >
        {cameras.map((cam) => (
          <div key={cam.camera_id} className="relative">
            <div
              className="cursor-pointer"
              onClick={() => setFullscreenId(cam.camera_id)}
            >
              <CameraCard
                title={cam.label}
                cameraId={cam.camera_id}
                status={cam.status}
                fps={cam.fps}
                maxFps={cam.max_fps}
              />
            </div>
            <button
              onClick={() => handleRemoveCamera(cam.camera_id)}
              className="absolute top-2 right-2 bg-red-600/80 hover:bg-red-500 text-white text-xs px-2 py-1 rounded z-10"
              title={`Remove ${cam.label}`}
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
