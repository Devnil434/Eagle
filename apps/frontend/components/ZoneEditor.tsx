import { useState, useRef, useEffect, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

/** A 2-D point on the canvas in pixel coordinates. */
interface Point {
  x: number;
  y: number;
}

/** A completed restricted zone with metadata. */
interface Zone {
  id: string;
  name: string;
  color: string;
  points: Point[];
}

/** Possible states of the save operation. */
type SaveStatus = "idle" | "saving" | "saved" | "error";

// ─── Constants ────────────────────────────────────────────────────────────────

/** Pixel radius within which clicking the first point closes the polygon. */
const CLOSE_RADIUS = 10;

/** Minimum number of points required to form a valid polygon. */
const MIN_POLYGON_POINTS = 3;

/** Keyboard step sizes in canvas pixels. */
const KEY_STEP       = 10;
const KEY_STEP_LARGE = 50;

/** Canvas logical dimensions — must match the <canvas> width/height attrs. */
const CANVAS_W = 960;
const CANVAS_H = 540;

/** Preset colour palette for zone selection. */
const PALETTE = [
  "#ef4444", // red
  "#f97316", // orange
  "#eab308", // yellow
  "#22c55e", // green
  "#3b82f6", // blue
  "#a855f7", // purple
  "#ec4899", // pink
];

// ─── Pure helpers ─────────────────────────────────────────────────────────────

/**
 * Euclidean distance between two canvas points.
 * Used to decide whether the user is clicking near the first polygon vertex.
 */
function dist(a: Point, b: Point): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

/**
 * Converts a CSS hex colour string to an rgba() string.
 * @param hex   - Six-digit hex colour, e.g. "#ef4444"
 * @param alpha - Opacity in the range [0, 1]
 */
function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

/**
 * Generates a unique ID. Uses crypto.randomUUID when available,
 * falls back to a timestamp-based ID for older environments.
 */
function generateId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `zone-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/** Clamps a number between min and max (inclusive). */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// ─── Canvas drawing helpers ───────────────────────────────────────────────────

/**
 * Draws a completed zone polygon with a semi-transparent fill and a
 * centred name label onto the given 2D context.
 * Requires at least MIN_POLYGON_POINTS points to render.
 */
function drawZone(ctx: CanvasRenderingContext2D, zone: Zone): void {
  if (zone.points.length < MIN_POLYGON_POINTS) return;

  ctx.beginPath();
  ctx.moveTo(zone.points[0].x, zone.points[0].y);
  zone.points.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
  ctx.closePath();

  ctx.fillStyle = hexToRgba(zone.color, 0.25);
  ctx.fill();
  ctx.strokeStyle = zone.color;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Draw zone name at centroid
  const cx = zone.points.reduce((s, p) => s + p.x, 0) / zone.points.length;
  const cy = zone.points.reduce((s, p) => s + p.y, 0) / zone.points.length;
  ctx.font = "bold 13px monospace";
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(zone.name, cx, cy);
}

/**
 * Draws the in-progress (draft) polygon as a dashed line with vertex dots.
 * The first vertex is rendered larger to act as a close-target indicator.
 * A rubber-band line follows the mouse / keyboard cursor.
 */
function drawDraft(
  ctx: CanvasRenderingContext2D,
  points: Point[],
  mouse: Point | null,
  color: string
): void {
  if (points.length === 0) return;

  // Draw dashed outline
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  points.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
  if (mouse) ctx.lineTo(mouse.x, mouse.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 3]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw vertex dots — first is larger to hint it closes the polygon
  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, i === 0 ? CLOSE_RADIUS : 5, 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? hexToRgba(color, 0.4) : color;
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

// ─── Component ────────────────────────────────────────────────────────────────

/**
 * ZoneEditor — main exported component.
 *
 * Renders a canvas overlaid on the camera snapshot where operators can
 * draw, name, colour, and persist restricted zones.
 */
export default function ZoneEditor() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef    = useRef<HTMLImageElement | null>(null);

  // ── State ──────────────────────────────────────────────────────────────────

  const [zones,          setZones]          = useState<Zone[]>([]);
  const [draft,          setDraft]          = useState<Point[]>([]);
  const [mouse,          setMouse]          = useState<Point | null>(null);
  const [zoneName,       setZoneName]       = useState("");
  const [zoneColor,      setZoneColor]      = useState(PALETTE[0]);
  const [nameError,      setNameError]      = useState("");
  const [saveStatus,     setSaveStatus]     = useState<SaveStatus>("idle");
  const [errorMessage,   setErrorMessage]   = useState("");
  const [snapshotLoaded, setSnapshotLoaded] = useState(false);
  const [snapshotError,  setSnapshotError]  = useState(false);

  // [Fix 3] Hydration gate — block editing until /zones read completes.
  const [zonesHydrated,  setZonesHydrated]  = useState(false);
  const [zonesLoadError, setZonesLoadError] = useState(false);

  // [Fix 1] Keyboard virtual cursor — starts at canvas centre.
  const [keyCursor, setKeyCursor] = useState<Point>({ x: CANVAS_W / 2, y: CANVAS_H / 2 });

  // ── Refs ───────────────────────────────────────────────────────────────────

  /**
   * Always-current ref to zones array.
   * Prevents stale closure bugs — closePolygon reads this
   * instead of the captured zones snapshot from render time.
   */
  const zonesRef = useRef<Zone[]>([]);
  useEffect(() => { zonesRef.current = zones; }, [zones]);

  /**
   * [Fix 2] Save-nonce ref — incremented on every zone mutation.
   * saveZones() captures the value before the fetch; if it has changed
   * by the time the response arrives, the result is treated as stale and
   * we do NOT show "✓ Saved!".
   */
  const saveNonceRef = useRef<number>(0);
  useEffect(() => { saveNonceRef.current += 1; }, [zones]);

  // ── Load snapshot on mount ─────────────────────────────────────────────────
  useEffect(() => {
    const img   = new Image();
    img.onload  = () => { imgRef.current = img; setSnapshotLoaded(true); };
    img.onerror = () => setSnapshotError(true);
    img.src     = "/api/snapshot";
  }, []);

  // ── Restore persisted zones on mount ──────────────────────────────────────
  // [Fix 3] Sets zonesHydrated on success.
  // 404 / endpoint-not-found → treat as "no zones yet" (hydrated, empty list).
  // Only a genuine server error (5xx) or a network-level failure sets
  // zonesLoadError and blocks editing — those indicate the server is unhealthy
  // and we can't trust that our local list is complete.
  useEffect(() => {
    fetch("/zones")
      .then((r) => {
        // 404 means the endpoint exists but no zones have been saved yet —
        // safe to proceed with an empty list.
        if (r.status === 404) {
          setZonesHydrated(true);
          return null;
        }
        // Any other non-OK status (5xx, 401, …) is a real problem.
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: Zone[] | null) => {
        if (data === null) return; // 404 path already handled above
        if (!Array.isArray(data)) {
          // Server returned unexpected shape — treat as hydrated with nothing
          setZonesHydrated(true);
          return;
        }
        setZones((prev) => {
          // Server data wins on ID conflict — merge preserving both local and remote
          const merged = new Map<string, Zone>();
          prev.forEach((z) => merged.set(z.id, z));
          data.forEach((z) => merged.set(z.id, z));
          return Array.from(merged.values());
        });
        setZonesHydrated(true);
      })
      .catch(() => {
        // Network failure or 5xx — we genuinely don't know the server state,
        // so block editing to prevent overwriting unseen zones.
        setZonesLoadError(true);
      });
  }, []);

  // ── Canvas render loop ─────────────────────────────────────────────────────
  /**
   * Redraws the entire canvas: background image → saved zones → draft polygon.
   * Memoised so it only re-runs when relevant state changes.
   */
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (imgRef.current) {
      ctx.drawImage(imgRef.current, 0, 0, canvas.width, canvas.height);
    } else {
      // Placeholder background when snapshot is unavailable
      ctx.fillStyle = "#1a1a2e";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#64748b";
      ctx.font = "16px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(
        snapshotError ? "⚠ Snapshot unavailable" : "Loading snapshot…",
        canvas.width / 2,
        canvas.height / 2
      );
    }

    zones.forEach((z) => drawZone(ctx, z));
    drawDraft(ctx, draft, mouse, zoneColor);
  }, [zones, draft, mouse, zoneColor, snapshotError]);

  useEffect(() => { render(); }, [render, snapshotLoaded]);

  // ── Canvas coordinate helper ───────────────────────────────────────────────
  /**
   * Converts a mouse event position to canvas-space coordinates,
   * accounting for CSS scaling of the canvas element.
   */
  function getCanvasPoint(e: React.MouseEvent<HTMLCanvasElement>): Point {
    const rect   = canvasRef.current!.getBoundingClientRect();
    const scaleX = canvasRef.current!.width  / rect.width;
    const scaleY = canvasRef.current!.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top)  * scaleY,
    };
  }

  // ── Canvas event handlers ──────────────────────────────────────────────────

  /**
   * Handles canvas click: either closes the polygon (if near first point)
   * or appends a new vertex to the draft.
   */
  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>): void {
    const pt = getCanvasPoint(e);
    if (draft.length >= MIN_POLYGON_POINTS && dist(pt, draft[0]) <= CLOSE_RADIUS) {
      closePolygon();
      return;
    }
    setDraft((prev) => [...prev, pt]);
  }

  /** Tracks the mouse position for the live rubber-band preview line. */
  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>): void {
    setMouse(getCanvasPoint(e));
  }

  /** Clears the rubber-band endpoint when the cursor leaves the canvas. */
  function handleMouseLeave(): void {
    setMouse(null);
  }

  // [Fix 1] ── Keyboard handlers ─────────────────────────────────────────────

  /**
   * Moves the virtual keyboard cursor with Arrow keys and dispatches
   * vertex-placement / polygon-close / cancel actions via keyboard.
   *
   * Controls:
   *   Arrow keys        — move cursor by KEY_STEP px
   *   Shift + Arrow     — move cursor by KEY_STEP_LARGE px
   *   Enter / Space     — place vertex (or close polygon if near first point)
   *   C                 — close polygon (requires ≥ MIN_POLYGON_POINTS)
   *   Backspace         — cancel current draft
   */
  function handleKeyDown(e: React.KeyboardEvent<HTMLCanvasElement>): void {
    const step = e.shiftKey ? KEY_STEP_LARGE : KEY_STEP;

    switch (e.key) {
      case "ArrowLeft":
        e.preventDefault();
        setKeyCursor((prev) => {
          const next = { ...prev, x: clamp(prev.x - step, 0, CANVAS_W) };
          setMouse(next);
          return next;
        });
        break;
      case "ArrowRight":
        e.preventDefault();
        setKeyCursor((prev) => {
          const next = { ...prev, x: clamp(prev.x + step, 0, CANVAS_W) };
          setMouse(next);
          return next;
        });
        break;
      case "ArrowUp":
        e.preventDefault();
        setKeyCursor((prev) => {
          const next = { ...prev, y: clamp(prev.y - step, 0, CANVAS_H) };
          setMouse(next);
          return next;
        });
        break;
      case "ArrowDown":
        e.preventDefault();
        setKeyCursor((prev) => {
          const next = { ...prev, y: clamp(prev.y + step, 0, CANVAS_H) };
          setMouse(next);
          return next;
        });
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        handleKeyPlaceVertex();
        break;
      case "c":
      case "C":
        e.preventDefault();
        if (draft.length >= MIN_POLYGON_POINTS) closePolygon();
        break;
      case "Backspace":
        e.preventDefault();
        cancelDraft();
        break;
      default:
        break;
    }
  }

  /**
   * Places a vertex at the current keyboard cursor position,
   * or closes the polygon if the cursor is near the first point.
   */
  function handleKeyPlaceVertex(): void {
    const pt = keyCursor;
    if (draft.length >= MIN_POLYGON_POINTS && dist(pt, draft[0]) <= CLOSE_RADIUS) {
      closePolygon();
      return;
    }
    setDraft((prev) => [...prev, pt]);
    setMouse(pt); // keep rubber-band preview in sync
  }

  // ── Polygon lifecycle ──────────────────────────────────────────────────────

  /**
   * Validates the zone name and commits the draft polygon to the zone list.
   * Resets all drafting state on success.
   *
   * [Fix 3] Early-returns with an error message if server zones haven't
   * been confirmed yet, so the duplicate-name check always operates on
   * the full authoritative zone list.
   */
  function closePolygon(): void {
    // [Fix 3] Block until the initial /zones fetch has resolved
    if (!zonesHydrated || zonesLoadError) {
      setNameError(
        zonesLoadError
          ? "Cannot save: zone list failed to load from server."
          : "Please wait — loading existing zones…"
      );
      return;
    }

    if (draft.length < MIN_POLYGON_POINTS) return;

    const trimmed = zoneName.trim();
    if (!trimmed) {
      setNameError("Zone name is required.");
      return;
    }

    // Use zonesRef.current — always reflects latest zones, never stale
    const currentZones = zonesRef.current;
    if (currentZones.some((z) => z.name.toLowerCase() === trimmed.toLowerCase())) {
      setNameError("Zone name must be unique.");
      return;
    }

    const newZone: Zone = {
      id:     generateId(),
      name:   trimmed,
      color:  zoneColor,
      points: draft,
    };

    setZones((prev) => [...prev, newZone]);
    setDraft([]);
    setMouse(null);
    setZoneName("");
    setNameError("");
    setZoneColor(PALETTE[0]);
  }

  /** Discards the current in-progress polygon without saving. */
  function cancelDraft(): void {
    setDraft([]);
    setMouse(null);
    setNameError("");
  }

  /**
   * Removes a previously committed zone by its UUID.
   * @param id - UUID of the zone to remove
   */
  function deleteZone(id: string): void {
    setZones((prev) => prev.filter((z) => z.id !== id));
  }

  // ── Backend persistence ────────────────────────────────────────────────────

  /**
   * POSTs all current zones to /zones as JSON.
   *
   * [Fix 2] Captures saveNonceRef.current before the fetch. After the
   * response arrives, compares against the current nonce — only marks
   * "saved" when they match. If zones changed mid-flight, silently
   * returns to "idle" so the user knows they need to save again.
   *
   * [Fix 3] Blocked when zonesHydrated is false or zonesLoadError is set,
   * preventing a POST that would overwrite unseen server zones.
   *
   * Status transitions:
   *   idle → saving → saved  (auto-resets to idle after 2.5 s)
   *   idle → saving → idle   (zones mutated mid-flight — silent dirty state)
   *   idle → saving → error  (auto-resets to idle after 3 s; shows message)
   */
  async function saveZones(): Promise<void> {
    // [Fix 3] Block if we don't have a confirmed view of server zones
    if (!zonesHydrated || zonesLoadError) return;

    if (saveStatus === "saving" || zones.length === 0) return;

    setSaveStatus("saving");
    setErrorMessage("");

    // [Fix 2] Snapshot the nonce at the moment this save starts
    const nonceAtStart = saveNonceRef.current;

    try {
      const payload = zones.map(({ id, name, color, points }) => ({
        id,
        name,
        color,
        points,
      }));

      const res = await fetch("/zones", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "Unknown server error");
        throw new Error(text || `HTTP ${res.status}`);
      }

      // [Fix 2] Only report "saved" when zones haven't changed since we fired
      if (saveNonceRef.current === nonceAtStart) {
        setSaveStatus("saved");
        setTimeout(() => setSaveStatus("idle"), 2500);
      } else {
        // Zones mutated mid-flight — the sent payload is stale; go dirty
        setSaveStatus("idle");
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Save failed";
      setErrorMessage(msg);
      setSaveStatus("error");
      setTimeout(() => { setSaveStatus("idle"); setErrorMessage(""); }, 3000);
    }
  }

  // ── Derived UI values ──────────────────────────────────────────────────────

  /** Cursor becomes "cell" when hovering near the polygon close-target. */
  const cursorStyle: React.CSSProperties["cursor"] =
    draft.length >= MIN_POLYGON_POINTS && mouse && dist(mouse, draft[0]) <= CLOSE_RADIUS
      ? "cell"
      : "crosshair";

  /** Button label reflects the current save state. */
  const saveButtonLabel =
    saveStatus === "saving" ? "Saving…"  :
    saveStatus === "saved"  ? "✓ Saved!" :
    saveStatus === "error"  ? "⚠ Retry"  :
    "Save Zones";

  /** Button colour reflects the current save state. */
  const saveButtonColor =
    saveStatus === "saved"  ? "#16a34a" :
    saveStatus === "error"  ? "#dc2626" :
    "#22c55e";

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div style={styles.root}>
      <h2 style={styles.heading}>Restricted Zone Editor</h2>

      {/* ── Canvas ── */}
      <div style={styles.canvasWrapper}>
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          // [Fix 1] tabIndex + role make the canvas keyboard-focusable
          tabIndex={0}
          role="application"
          style={{ ...styles.canvas, cursor: cursorStyle }}
          onClick={handleCanvasClick}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          // [Fix 1] Keyboard vertex placement
          onKeyDown={handleKeyDown}
          onFocus={() => setMouse(keyCursor)}
          onBlur={() => setMouse(null)}
          aria-label={
            "Camera snapshot canvas for drawing restricted zones. " +
            "Use arrow keys to move cursor, Enter or Space to place a point, " +
            "C to close the polygon, Backspace to cancel."
          }
        />
        {draft.length > 0 && (
          <div style={styles.draftBadge}>
            Drawing… {draft.length} point{draft.length !== 1 ? "s" : ""}
            {draft.length >= MIN_POLYGON_POINTS
              ? " — click first point or press C to close"
              : " — click or press Enter/Space to add points"}
          </div>
        )}
      </div>

      {/* [Fix 3] Zone-load error banner — blocks editing when shown */}
      {zonesLoadError && (
        <p role="alert" style={styles.saveError}>
          ⚠ Could not load existing zones from server. Editing and saving are
          disabled to prevent overwriting unseen data. Please refresh to retry.
        </p>
      )}

      {/* ── Controls ── */}
      <div style={styles.controls}>

        {/* Zone name input */}
        <div style={styles.field}>
          <label htmlFor="zone-name-input" style={styles.label}>
            Zone name
          </label>
          <input
            id="zone-name-input"
            type="text"
            style={{
              ...styles.input,
              borderColor: nameError ? "#ef4444" : "#334155",
            }}
            value={zoneName}
            onChange={(e) => {
              setZoneName(e.target.value);
              setNameError("");
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && draft.length >= MIN_POLYGON_POINTS) closePolygon();
            }}
            placeholder="e.g. restricted_door_zone"
            aria-describedby={nameError ? "zone-name-error" : undefined}
            autoComplete="off"
            spellCheck={false}
          />
          {nameError && (
            <span id="zone-name-error" role="alert" style={styles.error}>
              {nameError}
            </span>
          )}
        </div>

        {/* Colour picker */}
        <div style={styles.field}>
          <span style={styles.label}>Color</span>
          <div style={styles.palette} role="radiogroup" aria-label="Zone colour">
            {PALETTE.map((c) => (
              <button
                key={c}
                type="button"
                title={c}
                aria-label={`Select colour ${c}`}
                aria-pressed={zoneColor === c}
                style={{
                  ...styles.swatch,
                  backgroundColor: c,
                  outline:       zoneColor === c ? "2px solid #ffffff" : "none",
                  outlineOffset: "2px",
                  transform:     zoneColor === c ? "scale(1.2)" : "scale(1)",
                }}
                onClick={() => setZoneColor(c)}
              />
            ))}
          </div>
        </div>

        {/* Action buttons */}
        <div style={styles.actions}>
          {draft.length >= MIN_POLYGON_POINTS && (
            <button type="button" style={styles.btnPrimary} onClick={closePolygon}>
              ✓ Close Polygon
            </button>
          )}
          {draft.length > 0 && (
            <button type="button" style={styles.btnGhost} onClick={cancelDraft}>
              ✕ Cancel
            </button>
          )}
          {zones.length > 0 && draft.length === 0 && (
            <button
              type="button"
              style={{
                ...styles.btnSave,
                backgroundColor: saveButtonColor,
                opacity: saveStatus === "saving" ? 0.7 : 1,
                cursor:  saveStatus === "saving" ? "not-allowed" : "pointer",
              }}
              onClick={saveZones}
              disabled={saveStatus === "saving"}
              aria-busy={saveStatus === "saving"}
            >
              {saveButtonLabel}
            </button>
          )}
        </div>
      </div>

      {/* Save error message */}
      {saveStatus === "error" && errorMessage && (
        <p role="alert" style={styles.saveError}>
          ⚠ Save failed: {errorMessage}
        </p>
      )}

      {/* ── Zone list ── */}
      {zones.length > 0 && (
        <div style={styles.zoneList}>
          <h3 style={styles.subheading}>Defined Zones ({zones.length})</h3>
          {zones.map((z) => (
            <div key={z.id} style={styles.zoneRow}>
              <span
                style={{ ...styles.colorDot, backgroundColor: z.color }}
                aria-hidden="true"
              />
              <span style={styles.zoneName}>{z.name}</span>
              <span style={styles.zonePoints}>{z.points.length} pts</span>
              <button
                type="button"
                style={styles.deleteBtn}
                onClick={() => deleteZone(z.id)}
                title={`Delete zone "${z.name}"`}
                aria-label={`Delete zone ${z.name}`}
              >
                🗑
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles: Record<string, React.CSSProperties> = {
  root: {
    fontFamily:  "'JetBrains Mono', 'Fira Code', monospace",
    background:  "#0f172a",
    color:       "#e2e8f0",
    minHeight:   "100vh",
    padding:     "24px",
    boxSizing:   "border-box",
  },
  heading: {
    fontSize:      "1.25rem",
    fontWeight:    700,
    color:         "#38bdf8",
    marginBottom:  "16px",
    letterSpacing: "0.05em",
    textTransform: "uppercase",
  },
  canvasWrapper: {
    position:     "relative",
    display:      "inline-block",
    borderRadius: "8px",
    overflow:     "hidden",
    border:       "1px solid #1e3a5f",
    maxWidth:     "100%",
  },
  canvas: {
    display:  "block",
    maxWidth: "100%",
    height:   "auto",
  },
  draftBadge: {
    position:      "absolute",
    bottom:        "10px",
    left:          "50%",
    transform:     "translateX(-50%)",
    background:    "rgba(15,23,42,0.85)",
    border:        "1px solid #38bdf8",
    color:         "#38bdf8",
    fontSize:      "0.75rem",
    padding:       "4px 12px",
    borderRadius:  "999px",
    pointerEvents: "none",
    whiteSpace:    "nowrap",
  },
  controls: {
    marginTop:  "20px",
    display:    "flex",
    flexWrap:   "wrap",
    gap:        "20px",
    alignItems: "flex-end",
  },
  field: {
    display:       "flex",
    flexDirection: "column",
    gap:           "6px",
  },
  label: {
    fontSize:      "0.7rem",
    textTransform: "uppercase",
    letterSpacing: "0.1em",
    color:         "#94a3b8",
  },
  input: {
    background:   "#1e293b",
    border:       "1px solid #334155",
    borderRadius: "6px",
    color:        "#e2e8f0",
    padding:      "8px 12px",
    fontSize:     "0.875rem",
    fontFamily:   "inherit",
    width:        "220px",
    outline:      "none",
  },
  error: {
    color:    "#ef4444",
    fontSize: "0.75rem",
  },
  saveError: {
    marginTop: "10px",
    color:     "#fca5a5",
    fontSize:  "0.8rem",
  },
  palette: {
    display: "flex",
    gap:     "6px",
  },
  swatch: {
    width:        "24px",
    height:       "24px",
    borderRadius: "50%",
    border:       "none",
    cursor:       "pointer",
    transition:   "transform 0.15s ease",
  },
  actions: {
    display:    "flex",
    gap:        "10px",
    alignItems: "center",
  },
  btnPrimary: {
    background:   "#0ea5e9",
    color:        "#fff",
    border:       "none",
    borderRadius: "6px",
    padding:      "8px 16px",
    fontSize:     "0.875rem",
    fontFamily:   "inherit",
    cursor:       "pointer",
    fontWeight:   600,
  },
  btnGhost: {
    background:   "transparent",
    color:        "#94a3b8",
    border:       "1px solid #334155",
    borderRadius: "6px",
    padding:      "8px 16px",
    fontSize:     "0.875rem",
    fontFamily:   "inherit",
    cursor:       "pointer",
  },
  btnSave: {
    color:        "#fff",
    border:       "none",
    borderRadius: "6px",
    padding:      "8px 20px",
    fontSize:     "0.875rem",
    fontFamily:   "inherit",
    fontWeight:   700,
    transition:   "opacity 0.2s, background-color 0.3s",
  },
  zoneList: {
    marginTop: "24px",
    maxWidth:  "480px",
  },
  subheading: {
    fontSize:      "0.7rem",
    textTransform: "uppercase",
    letterSpacing: "0.1em",
    color:         "#64748b",
    marginBottom:  "10px",
  },
  zoneRow: {
    display:      "flex",
    alignItems:   "center",
    gap:          "10px",
    padding:      "8px 12px",
    background:   "#1e293b",
    borderRadius: "6px",
    marginBottom: "6px",
    fontSize:     "0.875rem",
  },
  colorDot: {
    width:        "12px",
    height:       "12px",
    borderRadius: "50%",
    flexShrink:   0,
  },
  zoneName: {
    flex:       1,
    fontWeight: 600,
  },
  zonePoints: {
    color:    "#64748b",
    fontSize: "0.75rem",
  },
  deleteBtn: {
    background: "transparent",
    border:     "none",
    cursor:     "pointer",
    fontSize:   "0.875rem",
    opacity:    0.6,
    padding:    "2px 4px",
    lineHeight: 1,
  },
};
