import { trackColors } from "../utils/colors"

export default function CameraCard({ 
  title = "Unknown Camera", 
  trackId = "N/A" 
}) {

  const color = trackColors[trackId] || "#6b7280"

  return (
    <div className="relative bg-gray-900 rounded-xl overflow-hidden h-[300px]">

      <div className="absolute top-2 left-2 bg-black/60 px-2 py-1 rounded text-white z-10">
        {title}
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
          {trackId}
        </div>
      </div>

    </div>
  )
}