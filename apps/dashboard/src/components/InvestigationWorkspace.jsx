import { useState, useEffect } from "react";

export default function InvestigationWorkspace({ operatorId = "operator-1" }) {
  const [investigations, setInvestigations] = useState([]);
  const [selectedInv, setSelectedInv] = useState(null);
  const [events, setEvents] = useState([]);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("events");

  useEffect(() => {
    fetchInvestigations();
  }, []);

  async function fetchInvestigations() {
    setLoading(true);
    try {
      const res = await fetch(`/bookmarks/investigations?operator_id=${operatorId}`);
      if (res.ok) {
        const data = await res.json();
        setInvestigations(data);
      }
    } catch (e) {
      console.error("Failed to load investigations", e);
    } finally {
      setLoading(false);
    }
  }

  async function createInvestigation() {
    if (!newName.trim()) return;
    setLoading(true);
    try {
      const res = await fetch("/bookmarks/investigations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newName, description: newDesc }),
      });
      if (res.ok) {
        const inv = await res.json();
        setInvestigations((prev) => [...prev, inv]);
        setNewName("");
        setNewDesc("");
      }
    } catch (e) {
      console.error("Failed to create investigation", e);
    } finally {
      setLoading(false);
    }
  }

  async function loadEvents(invId) {
    try {
      const res = await fetch(`/bookmarks/investigations/${invId}/events`);
      if (res.ok) {
        const data = await res.json();
        setEvents(data);
      }
    } catch (e) {
      console.error("Failed to load events", e);
    }
  }

  function handleSelectInv(inv) {
    setSelectedInv(inv);
    loadEvents(inv.id);
  }

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4 h-full flex flex-col">
      <h2 className="text-xl font-bold text-white mb-4">Investigation Workspace</h2>

      <div className="flex gap-2 mb-4">
        <input
          type="text"
          placeholder="New investigation name"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          className="flex-1 px-3 py-2 rounded bg-zinc-800 text-white border border-zinc-700"
        />
        <input
          type="text"
          placeholder="Description"
          value={newDesc}
          onChange={(e) => setNewDesc(e.target.value)}
          className="flex-1 px-3 py-2 rounded bg-zinc-800 text-white border border-zinc-700"
        />
        <button
          onClick={createInvestigation}
          disabled={loading}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-500 disabled:opacity-50"
        >
          Create
        </button>
      </div>

      <div className="flex flex-1 gap-4 overflow-hidden">
        <div className="w-1/3 overflow-y-auto border-r border-zinc-800 pr-2">
          {loading && investigations.length === 0 && <p className="text-zinc-500">Loading...</p>}
          {investigations.map((inv) => (
            <div
              key={inv.id}
              onClick={() => handleSelectInv(inv)}
              className={`p-3 rounded cursor-pointer mb-2 border ${
                selectedInv?.id === inv.id
                  ? "border-green-500 bg-green-500/10"
                  : "border-zinc-700 bg-zinc-800 hover:bg-zinc-700"
              }`}
            >
              <div className="font-semibold text-white">{inv.name}</div>
              <div className="text-xs text-zinc-400">{inv.description || "No description"}</div>
              <div className="text-xs text-zinc-500 mt-1">{inv.event_count || 0} events</div>
            </div>
          ))}
          {investigations.length === 0 && !loading && (
            <p className="text-zinc-500 text-sm">No investigations yet. Create one above.</p>
          )}
        </div>

        <div className="flex-1 overflow-y-auto">
          {selectedInv ? (
            <>
              <div className="flex gap-4 mb-4 border-b border-zinc-800">
                <button
                  onClick={() => setActiveTab("events")}
                  className={`pb-2 text-sm font-medium ${
                    activeTab === "events" ? "text-green-400 border-b-2 border-green-500" : "text-zinc-400"
                  }`}
                >
                  Events
                </button>
                <button
                  onClick={() => setActiveTab("timeline")}
                  className={`pb-2 text-sm font-medium ${
                    activeTab === "timeline" ? "text-green-400 border-b-2 border-green-500" : "text-zinc-400"
                  }`}
                >
                  Timeline
                </button>
                <button
                  onClick={() => setActiveTab("export")}
                  className={`pb-2 text-sm font-medium ${
                    activeTab === "export" ? "text-green-400 border-b-2 border-green-500" : "text-zinc-400"
                  }`}
                >
                  Export
                </button>
              </div>

              {activeTab === "events" && <EventList events={events} invId={selectedInv.id} operatorId={operatorId} onRefresh={loadEvents} />}
              {activeTab === "timeline" && <TimelineView invId={selectedInv.id} />}
              {activeTab === "export" && <ExportWorkspace invId={selectedInv.id} name={selectedInv.name} />}
            </>
          ) : (
            <p className="text-zinc-500">Select an investigation to view details.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function EventList({ events, invId, operatorId, onRefresh }) {
  const [selectedEvent, setSelectedEvent] = useState(null);

  async function addEvent() {
    const alertId = prompt("Enter alert ID:");
    if (!alertId) return;
    const cameraId = prompt("Enter camera ID:", "cam_01") || "cam_01";
    const trackId = parseInt(prompt("Enter track ID:", "1") || "1", 10);
    const label = prompt("Enter event label:", "suspicious") || "suspicious";
    try {
      const res = await fetch(`/bookmarks/investigations/${invId}/events`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ alert_id: alertId, camera_id: cameraId, track_id: trackId, label }),
      });
      if (res.ok) {
        onRefresh(invId);
      }
    } catch (e) {
      console.error("Failed to add event", e);
    }
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-white font-semibold">Events ({events.length})</h3>
        <button onClick={addEvent} className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-500">
          Add Event
        </button>
      </div>
      {events.length === 0 && <p className="text-zinc-500 text-sm">No events in this investigation.</p>}
      <div className="space-y-2">
        {events.map((evt) => (
          <div
            key={evt.id}
            onClick={() => setSelectedEvent(evt)}
            className={`p-3 rounded border cursor-pointer ${
              selectedEvent?.id === evt.id ? "border-green-500 bg-green-500/10" : "border-zinc-700 bg-zinc-800"
            }`}
          >
            <div className="flex justify-between">
              <span className="text-white font-medium">{evt.label}</span>
              <span className="text-xs text-zinc-400">{new Date(evt.created_at * 1000).toLocaleString()}</span>
            </div>
            <div className="text-xs text-zinc-400 mt-1">Camera: {evt.camera_id} | Track: {evt.track_id}</div>
            {evt.notes && <div className="text-xs text-zinc-300 mt-1 italic">"{evt.notes}"</div>}
          </div>
        ))}
      </div>

      {selectedEvent && (
        <div className="mt-4 p-3 bg-zinc-800 rounded border border-zinc-700">
          <h4 className="text-white font-semibold mb-2">Notes</h4>
          <AddNotes bookmarkId={selectedEvent.id} onAdded={() => {}} />
        </div>
      )}
    </div>
  );
}

function AddNotes({ bookmarkId, onAdded }) {
  const [note, setNote] = useState("");

  async function submitNote() {
    if (!note.trim()) return;
    try {
      const res = await fetch(`/bookmarks/${bookmarkId}/notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: note }),
      });
      if (res.ok) {
        setNote("");
        onAdded();
      }
    } catch (e) {
      console.error("Failed to add note", e);
    }
  }

  return (
    <div className="flex gap-2">
      <input
        type="text"
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="Add a note..."
        className="flex-1 px-3 py-2 rounded bg-zinc-900 text-white border border-zinc-700 text-sm"
      />
      <button onClick={submitNote} className="px-3 py-2 bg-green-600 text-white text-sm rounded hover:bg-green-500">
        Add
      </button>
    </div>
  );
}
