import { trackColors } from "../utils/colors"

export default function CameraCard({
  title = "Unknown Camera",
  cameraId = "N/A",
  status = "online",
  fps = 0,
  maxFps = 30,
  fullscreen = false,
}) {
  const color = trackColors[cameraId] || "#6b7280"
  const healthColor = status === "online" ? "#22c55e" : status === "degraded" ? "#f97316" : "#ef4444"

  return (
    <div className={`relative bg-gray-900 rounded-xl overflow-hidden ${fullscreen ? "h-full" : "h-[300px]"}`}>
      <div className="absolute top-2 left-2 bg-black/60 px-2 py-1 rounded text-white z-10 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: healthColor }} />
        {title}
      </div>

      <div className="absolute top-2 right-2 bg-black/60 px-2 py-1 rounded text-white z-10 text-xs font-mono">
        {fps.toFixed(1)} / {maxFps} FPS
      </div>

      <div
        className="absolute border-4"
        style={{
          borderColor: color,
          top: "20%",
          left: "30%",
          width: "25%",
          height: "40%",
        }}
      >
        <div
          className="text-white px-1"
          style={{
            backgroundColor: color
          }}
        >
          {cameraId}
        </div>
      </div>

    </div>
  )
}
