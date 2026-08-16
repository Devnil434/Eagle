import { useState } from "react";

export default function ExportWorkspace({ invId, name }) {
  const [exportData, setExportData] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleExport() {
    setLoading(true);
    try {
      const res = await fetch(`/bookmarks/investigations/${invId}/export`);
      if (res.ok) {
        const data = await res.json();
        setExportData(data);
      }
    } catch (e) {
      console.error("Failed to export workspace", e);
    } finally {
      setLoading(false);
    }
  }

  function downloadExport() {
    if (!exportData) return;
    const blob = new Blob([exportData.data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name || "investigation"}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  return (
    <div>
      <h3 className="text-white font-semibold mb-3">Export Workspace</h3>
      <button
        onClick={handleExport}
        disabled={loading}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-500 disabled:opacity-50 mb-3"
      >
        {loading ? "Generating..." : "Generate Export"}
      </button>

      {exportData && (
        <div className="bg-zinc-800 p-3 rounded border border-zinc-700">
          <div className="flex justify-between items-center mb-2">
            <span className="text-white font-medium">Export Ready</span>
            <button onClick={downloadExport} className="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-500">
              Download JSON
            </button>
          </div>
          <pre className="text-xs text-zinc-300 overflow-auto max-h-64 bg-zinc-900 p-2 rounded">
            {exportData.data}
          </pre>
        </div>
      )}
    </div>
  );
}
