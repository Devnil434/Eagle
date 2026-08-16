import { useState, useEffect } from "react";

export default function BookmarkPanel({ operatorId = "operator-1" }) {
  const [bookmarks, setBookmarks] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchBookmarks();
  }, []);

  async function fetchBookmarks() {
    setLoading(true);
    try {
      const res = await fetch(`/bookmarks?operator_id=${operatorId}`);
      if (res.ok) {
        const data = await res.json();
        setBookmarks(data);
      }
    } catch (e) {
      console.error("Failed to load bookmarks", e);
    } finally {
      setLoading(false);
    }
  }

  async function removeBookmark(id) {
    try {
      const res = await fetch(`/bookmarks/${id}`, { method: "DELETE" });
      if (res.ok) {
        setBookmarks((prev) => prev.filter((b) => b.id !== id));
      }
    } catch (e) {
      console.error("Failed to remove bookmark", e);
    }
  }

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
      <h3 className="text-lg font-semibold text-white mb-3">Bookmarks</h3>
      {loading && <p className="text-zinc-500 text-sm">Loading...</p>}
      {bookmarks.length === 0 && !loading && <p className="text-zinc-500 text-sm">No bookmarks yet.</p>}
      <div className="space-y-2">
        {bookmarks.map((bm) => (
          <div key={bm.id} className="p-2 bg-zinc-800 rounded border border-zinc-700 flex justify-between items-center">
            <div>
              <div className="text-white text-sm font-medium">{bm.label}</div>
              <div className="text-xs text-zinc-400">Camera: {bm.camera_id} | Track: {bm.track_id}</div>
            </div>
            <button onClick={() => removeBookmark(bm.id)} className="text-red-400 hover:text-red-300 text-xs px-2 py-1">
              Remove
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
