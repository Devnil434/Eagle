import { useState } from "react";

export default function SearchBookmarks({ operatorId = "operator-1" }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [searching, setSearching] = useState(false);

  async function handleSearch(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    try {
      const res = await fetch(`/bookmarks/search?operator_id=${operatorId}&q=${encodeURIComponent(query)}`);
      if (res.ok) {
        const data = await res.json();
        setResults(data);
      }
    } catch (e) {
      console.error("Search failed", e);
    } finally {
      setSearching(false);
    }
  }

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
      <h3 className="text-lg font-semibold text-white mb-3">Search Bookmarks</h3>
      <form onSubmit={handleSearch} className="flex gap-2 mb-3">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search bookmarks..."
          className="flex-1 px-3 py-2 rounded bg-zinc-800 text-white border border-zinc-700 text-sm"
        />
        <button type="submit" disabled={searching} className="px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-500 disabled:opacity-50">
          {searching ? "..." : "Search"}
        </button>
      </form>
      {results.length === 0 && query && !searching && <p className="text-zinc-500 text-sm">No results found.</p>}
      <div className="space-y-2">
        {results.map((bm) => (
          <div key={bm.id} className="p-2 bg-zinc-800 rounded border border-zinc-700">
            <div className="text-white text-sm font-medium">{bm.label}</div>
            <div className="text-xs text-zinc-400">Camera: {bm.camera_id} | Track: {bm.track_id}</div>
            {bm.notes && <div className="text-xs text-zinc-300 mt-1 italic">"{bm.notes}"</div>}
          </div>
        ))}
      </div>
    </div>
  );
}
