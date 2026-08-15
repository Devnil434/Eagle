import { useState, useEffect, useMemo } from "react";
import CameraCard from "../components/CameraCard"

const CAMERA_VIEW = "__eagle_cameras__";

export default function Dashboard() {
  const [selectedTrack, setSelectedTrack] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [view, setView] = useState(CAMERA_VIEW);

  const featureModules = useMemo(() => {
    try {
      return import.meta.glob("./features/*.jsx", { eager: true });
    } catch {
      return {};
    }
  }, []);

  const featureTabs = useMemo(() => {
    const tabs = [{ id: CAMERA_VIEW, label: "Cameras", component: null }];
    Object.entries(featureModules).forEach(([path, mod]) => {
      const name = path.replace(/^\.\/features\//, "").replace(/\.jsx$/, "");
      const comp = mod.default || mod;
      if (comp && typeof comp === "function") {
        tabs.push({ id: name, label: mod.label || name, component: comp });
      }
    });
    return tabs;
  }, [featureModules]);

  const activeComponent = useMemo(() => {
    const tab = featureTabs.find((t) => t.id === view);
    return tab?.component || null;
  }, [view, featureTabs]);

  const cameras = [
    { id: 1, title: "Camera 1", trackId: "P-101" },
    { id: 2, title: "Camera 2", trackId: "P-102" },
    { id: 3, title: "Camera 3", trackId: "P-101" },
    { id: 4, title: "Camera 4", trackId: "P-103" },
  ];

  return (
    <div className="flex h-screen bg-black text-white">
      <div className="flex-1 p-4">
        <div className="flex gap-2 mb-4">
          {featureTabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setView(tab.id)}
              className={`px-4 py-2 rounded text-sm font-medium ${
                view === tab.id ? "bg-green-600 text-white" : "bg-zinc-800 text-zinc-300"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {view === CAMERA_VIEW && (
          <>
            <input
              aria-label="Search Track ID"
              type="text"
              placeholder="Search Track ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full mb-4 px-4 py-2 rounded bg-zinc-900 text-white"
            />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {cameras
                .filter((cam) =>
                  cam.trackId.toLowerCase().includes(searchQuery.toLowerCase())
                )
                .map((cam) => (
                  <div
                    key={cam.id}
                    onClick={() => setSelectedTrack(cam)}
                    className={`cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-2xl ${
                      selectedTrack?.id === cam.id
                        ? "border-2 border-green-500 scale-105 rounded-lg shadow-green-500/40 shadow-2xl"
                        : ""
                    }`}
                  >
                    <CameraCard title={cam.title} trackId={cam.trackId} />
                  </div>
                ))}
            </div>
          </>
        )}

        {activeComponent && <activeComponent />}
      </div>

      <div className="w-80 bg-zinc-950 border-l border-zinc-800 p-4">
        {selectedTrack !== null && view === CAMERA_VIEW ? (
          <>
            <h2 className="text-2xl font-bold mb-4">Identity Panel</h2>
            <p className="mb-2">
              <span className="font-semibold">Camera:</span> {selectedTrack.title}
            </p>
            <p className="mb-2">
              <span className="font-semibold">Track ID:</span> {selectedTrack.trackId}
            </p>
            <p className="text-green-400 animate-pulse">ACTIVE TRACK</p>
          </>
        ) : (
          <p>Select a camera track</p>
        )}
      </div>
    </div>
  );
}
