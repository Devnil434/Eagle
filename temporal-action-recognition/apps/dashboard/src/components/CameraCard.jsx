import { trackColors } from "../utils/colors";

export default function CameraCard({
  title = "Unknown Camera",
  trackId = "N/A",
  action = null,
  actionConfidence = null,
  actionSource = null,
}) {
  const color = trackColors[trackId] || "#6b7280";
  const isAlert =
    action &&
    ["fighting", "falling", "running", "loitering", "suspicious_stationary"].includes(
      action
    );

  return (
    <div className="relative bg-gray-900 rounded-xl overflow-hidden h-[300px]">
      <div className="absolute top-2 left-2 bg-black/60 px-2 py-1 rounded text-white z-10">
        {title}
      </div>

      {action && (
        <div
          className={`absolute top-2 right-2 px-2 py-1 rounded text-xs z-10 ${
            isAlert ? "bg-red-600 text-white" : "bg-zinc-800 text-zinc-200"
          }`}
        >
          {action}
          {actionConfidence != null && ` ${(actionConfidence * 100).toFixed(0)}%`}
          {actionSource && <span className="opacity-70"> · {actionSource}</span>}
        </div>
      )}

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
        <div className="text-white px-1" style={{ backgroundColor: color }}>
          {trackId}
        </div>
      </div>
    </div>
  );
}
