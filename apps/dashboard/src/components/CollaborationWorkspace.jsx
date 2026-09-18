import { useState, useEffect } from "react";

export default function CollaborationWorkspace({ operatorId = "operator-1" }) {
  const [workspaces, setWorkspaces] = useState([]);
  const [selectedWs, setSelectedWs] = useState(null);
  const [wsEvents, setWsEvents] = useState([]);
  const [assignments, setAssignments] = useState([]);
  const [activity, setActivity] = useState([]);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("events");

  useEffect(() => {
    fetchWorkspaces();
  }, []);

  async function fetchWorkspaces() {
    setLoading(true);
    try {
      const res = await fetch(`/workspaces?operator_id=${operatorId}`);
      if (res.ok) {
        const data = await res.json();
        setWorkspaces(data);
      }
    } catch (e) {
      console.error("Failed to load workspaces", e);
    } finally {
      setLoading(false);
    }
  }

  async function createWorkspace() {
    if (!newName.trim()) return;
    setLoading(true);
    try {
      const res = await fetch("/workspaces", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newName, description: newDesc }),
      });
      if (res.ok) {
        const ws = await res.json();
        setWorkspaces((prev) => [...prev, ws]);
        setNewName("");
        setNewDesc("");
      }
    } catch (e) {
      console.error("Failed to create workspace", e);
    } finally {
      setLoading(false);
    }
  }

  async function joinWorkspace(wsId) {
    try {
      await fetch(`/workspaces/${wsId}/join?operator_id=${operatorId}`, { method: "POST" });
      fetchWorkspaces();
    } catch (e) {
      console.error("Failed to join workspace", e);
    }
  }

  async function leaveWorkspace(wsId) {
    try {
      await fetch(`/workspaces/${wsId}/leave?operator_id=${operatorId}`, { method: "POST" });
      fetchWorkspaces();
      if (selectedWs?.id === wsId) setSelectedWs(null);
    } catch (e) {
      console.error("Failed to leave workspace", e);
    }
  }

  async function loadWorkspaceData(ws) {
    setSelectedWs(ws);
    try {
      const [eventsRes, assignmentsRes, activityRes] = await Promise.all([
        fetch(`/workspaces/${ws.id}/events`),
        fetch(`/workspaces/${ws.id}/assignments`),
        fetch(`/workspaces/${ws.id}/activity`),
      ]);
      if (eventsRes.ok) setWsEvents(await eventsRes.json());
      if (assignmentsRes.ok) setAssignments(await assignmentsRes.json());
      if (activityRes.ok) setActivity(await activityRes.json());
    } catch (e) {
      console.error("Failed to load workspace data", e);
    }
  }

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4 h-full flex flex-col">
      <h2 className="text-xl font-bold text-white mb-4">Collaboration Workspace</h2>

      <div className="flex gap-2 mb-4">
        <input
          type="text"
          placeholder="New workspace name"
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
        <button onClick={createWorkspace} disabled={loading} className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-500 disabled:opacity-50">
          Create
        </button>
      </div>

      <div className="flex flex-1 gap-4 overflow-hidden">
        <div className="w-1/3 overflow-y-auto border-r border-zinc-800 pr-2">
          {loading && workspaces.length === 0 && <p className="text-zinc-500">Loading...</p>}
          {workspaces.map((ws) => (
            <div
              key={ws.id}
              onClick={() => loadWorkspaceData(ws)}
              className={`p-3 rounded cursor-pointer mb-2 border ${
                selectedWs?.id === ws.id ? "border-green-500 bg-green-500/10" : "border-zinc-700 bg-zinc-800 hover:bg-zinc-700"
              }`}
            >
              <div className="font-semibold text-white">{ws.name}</div>
              <div className="text-xs text-zinc-400">{ws.description || "No description"}</div>
              <div className="text-xs text-zinc-500 mt-1">{ws.members.length} members</div>
            </div>
          ))}
          {workspaces.length === 0 && !loading && (
            <p className="text-zinc-500 text-sm">No workspaces yet. Create one above.</p>
          )}
        </div>

        <div className="flex-1 overflow-y-auto">
          {selectedWs ? (
            <>
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-white font-semibold">{selectedWs.name}</h3>
                <div className="flex gap-2">
                  <button onClick={() => joinWorkspace(selectedWs.id)} className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-500">
                    Join
                  </button>
                  <button onClick={() => leaveWorkspace(selectedWs.id)} className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-500">
                    Leave
                  </button>
                </div>
              </div>
              <div className="flex gap-4 mb-4 border-b border-zinc-800">
                <button onClick={() => setActiveTab("events")} className={`pb-2 text-sm font-medium ${activeTab === "events" ? "text-green-400 border-b-2 border-green-500" : "text-zinc-400"}>Events</button>
                <button onClick={() => setActiveTab("assignments")} className={`pb-2 text-sm font-medium ${activeTab === "assignments" ? "text-green-400 border-b-2 border-green-500" : "text-zinc-400"}>Assignments</button>
                <button onClick={() => setActiveTab("activity")} className={`pb-2 text-sm font-medium ${activeTab === "activity" ? "text-green-400 border-b-2 border-green-500" : "text-zinc-400"}>Activity</button>
              </div>
              {activeTab === "events" && <EventsList events={wsEvents} wsId={selectedWs.id} operatorId={operatorId} onRefresh={() => loadWorkspaceData(selectedWs)} />}
              {activeTab === "assignments" && <AssignmentsList assignments={assignments} wsId={selectedWs.id} operatorId={operatorId} onRefresh={() => loadWorkspaceData(selectedWs)} />}
              {activeTab === "activity" && <ActivityHistory activity={activity} />}
            </>
          ) : (
            <p className="text-zinc-500">Select a workspace to view details.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function EventsList({ events, wsId, operatorId, onRefresh }) {
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [comments, setComments] = useState([]);
  const [newComment, setNewComment] = useState("");

  async function selectEvent(evt) {
    setSelectedEvent(evt);
    try {
      const res = await fetch(`/workspaces/${wsId}/events/${evt.id}/comments`);
      if (res.ok) setComments(await res.json());
    } catch (e) {
      console.error("Failed to load comments", e);
    }
  }

  async function addComment() {
    if (!newComment.trim() || !selectedEvent) return;
    try {
      const res = await fetch(`/workspaces/${wsId}/events/${selectedEvent.id}/comments`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ operator_id: operatorId, content: newComment }),
      });
      if (res.ok) {
        setNewComment("");
        selectEvent(selectedEvent);
      }
    } catch (e) {
      console.error("Failed to add comment", e);
    }
  }

  return (
    <div>
      <h3 className="text-white font-semibold mb-3">Events ({events.length})</h3>
      {events.length === 0 && <p className="text-zinc-500 text-sm">No events in this workspace.</p>}
      <div className="space-y-2">
        {events.map((evt) => (
          <div
            key={evt.id}
            onClick={() => selectEvent(evt)}
            className={`p-3 rounded border cursor-pointer ${
              selectedEvent?.id === evt.id ? "border-green-500 bg-green-500/10" : "border-zinc-700 bg-zinc-800"
            }`}
          >
            <div className="flex justify-between">
              <span className="text-white font-medium">{evt.label}</span>
              <span className="text-xs text-zinc-400">{new Date(evt.timestamp * 1000).toLocaleString()}</span>
            </div>
            <div className="text-xs text-zinc-400 mt-1">Camera: {evt.camera_id} | Track: {evt.track_id} | By: {evt.added_by}</div>
            {evt.notes && <div className="text-xs text-zinc-300 mt-1 italic">"{evt.notes}"</div>}
          </div>
        ))}
      </div>
      {selectedEvent && (
        <div className="mt-4 p-3 bg-zinc-800 rounded border border-zinc-700">
          <h4 className="text-white font-semibold mb-2">Comments ({comments.length})</h4>
          <div className="space-y-2 mb-3">
            {comments.map((c) => (
              <div key={c.id} className="text-sm">
                <span className="text-zinc-400 font-medium">{c.operator_id}:</span> <span className="text-zinc-200">{c.content}</span>
              </div>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={newComment}
              onChange={(e) => setNewComment(e.target.value)}
              placeholder="Add a comment..."
              className="flex-1 px-3 py-2 rounded bg-zinc-900 text-white border border-zinc-700 text-sm"
            />
            <button onClick={addComment} className="px-3 py-2 bg-green-600 text-white text-sm rounded hover:bg-green-500">
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function AssignmentsList({ assignments, wsId, operatorId, onRefresh }) {
  const [incidentId, setIncidentId] = useState("");
  const [assignee, setAssignee] = useState("");
  const [desc, setDesc] = useState("");

  async function assign() {
    if (!incidentId.trim() || !assignee.trim()) return;
    try {
      const res = await fetch(`/workspaces/${wsId}/assignments?assigned_by=${operatorId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ operator_id: assignee, incident_id: incidentId, description: desc }),
      });
      if (res.ok) {
        setIncidentId("");
        setAssignee("");
        setDesc("");
        onRefresh();
      }
    } catch (e) {
      console.error("Failed to assign", e);
    }
  }

  return (
    <div>
      <h3 className="text-white font-semibold mb-3">Assignments ({assignments.length})</h3>
      <div className="flex gap-2 mb-3">
        <input type="text" value={incidentId} onChange={(e) => setIncidentId(e.target.value)} placeholder="Incident ID" className="flex-1 px-3 py-2 rounded bg-zinc-800 text-white border border-zinc-700 text-sm" />
        <input type="text" value={assignee} onChange={(e) => setAssignee(e.target.value)} placeholder="Assignee" className="flex-1 px-3 py-2 rounded bg-zinc-800 text-white border border-zinc-700 text-sm" />
        <input type="text" value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="Description" className="flex-1 px-3 py-2 rounded bg-zinc-800 text-white border border-zinc-700 text-sm" />
        <button onClick={assign} className="px-3 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-500">Assign</button>
      </div>
      {assignments.length === 0 && <p className="text-zinc-500 text-sm">No assignments yet.</p>}
      <div className="space-y-2">
        {assignments.map((a) => (
          <div key={a.id} className="p-2 bg-zinc-800 rounded border border-zinc-700">
            <div className="text-white text-sm font-medium">{a.incident_id} → {a.assigned_to}</div>
            <div className="text-xs text-zinc-400">{a.description || "No description"} | Status: {a.status}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ActivityHistory({ activity }) {
  return (
    <div>
      <h3 className="text-white font-semibold mb-3">Activity History</h3>
      {activity.length === 0 && <p className="text-zinc-500 text-sm">No activity yet.</p>}
      <div className="space-y-2">
        {activity.map((entry) => (
          <div key={entry.id} className="p-2 bg-zinc-800 rounded border border-zinc-700">
            <div className="text-white text-sm font-medium">{entry.action}</div>
            <div className="text-xs text-zinc-400">{entry.details} | By: {entry.operator_id}</div>
            <div className="text-xs text-zinc-500">{new Date(entry.timestamp * 1000).toLocaleString()}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
