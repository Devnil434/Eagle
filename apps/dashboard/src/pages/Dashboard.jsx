import { useState } from "react";
import CameraCard from "../components/CameraCard"

export default function Dashboard() {

  const [selectedTrack, setSelectedTrack] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const cameras = [
  { id: 1, title: "Camera 1", trackId: "P-101" },
  { id: 2, title: "Camera 2", trackId: "P-102" },
  { id: 3, title: "Camera 3", trackId: "P-101" },
  { id: 4, title: "Camera 4", trackId: "P-103" },
];
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
    <CameraCard
      title={cam.title}
      trackId={cam.trackId}
    />
  </div>
))
}
</div>
</div>
<div className="w-80 bg-zinc-950 border-l border-zinc-800 p-4">

  {selectedTrack !== null ? (
    <>
      <h2 className="text-2xl font-bold mb-4">
        Identity Panel
      </h2>

      <p className="mb-2">
        <span className="font-semibold">Camera:</span>{" "}
        {selectedTrack.title}
      </p>

      <p className="mb-2">
        <span className="font-semibold">Track ID:</span>{" "}
       {selectedTrack.trackId}
      </p>

      <p className="text-green-400 animate-pulse">
        ACTIVE TRACK
      </p>
    </>
  ) : (
    <p>Select a camera track</p>
  )}

</div>
  </div>
  )
}