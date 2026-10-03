import { useState, useEffect, useRef, useCallback } from "react";

type Mode = "idle" | "listening" | "processing" | "results";

interface QueryResult {
  query: string;
  language: string;
  total_matches: number;
  alerts: Array<Record<string, unknown>>;
  suggestions: string[];
}

const LANGUAGES = [
  { code: "en-US", label: "English" },
  { code: "es-ES", label: "Spanish" },
  { code: "fr-FR", label: "French" },
  { code: "de-DE", label: "German" },
  { code: "hi-IN", label: "Hindi" },
  { code: "zh-CN", label: "Chinese" },
];

export default function VoiceQuery() {
  const [mode, setMode] = useState<Mode>("idle");
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("en-US");
  const [results, setResults] = useState<QueryResult | null>(null);
  const [error, setError] = useState("");
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  const startListening = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setError("Speech recognition is not supported in this browser.");
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = language;
    recognition.interimResults = true;
    recognition.continuous = false;
    recognitionRef.current = recognition;

    recognition.onstart = () => {
      setMode("listening");
      setError("");
    };
    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = Array.from(event.results)
        .map((r) => r[0].transcript)
        .join("");
      setText(transcript);
    };
    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      setError(`Voice error: ${event.error}`);
      setMode("idle");
    };
    recognition.onend = () => {
      if (mode === "listening" && text.trim()) {
        submitQuery(text.trim());
      } else {
        setMode("idle");
      }
    };
    recognition.start();
  }, [language, text, mode]);

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
  }, []);

  const submitQuery = async (queryText: string) => {
    setMode("processing");
    setError("");
    try {
      const res = await fetch("/voice/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText, language, camera_id: "cam_01" }),
      });
      if (!res.ok) {
        const msg = await res.text().catch(() => `HTTP ${res.status}`);
        throw new Error(msg);
      }
      const data: QueryResult = await res.json();
      setResults(data);
      setMode("results");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Query failed";
      setError(msg);
      setMode("idle");
    }
  };

  const handleManualSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed) return;
    if (mode === "listening") stopListening();
    submitQuery(trimmed);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setText(suggestion);
    submitQuery(suggestion);
  };

  useEffect(() => {
    return () => {
      recognitionRef.current?.abort();
    };
  }, []);

  return (
    <div style={styles.root}>
      <h2 style={styles.heading}>Voice Query</h2>

      <div style={styles.controls}>
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          style={styles.select}
          aria-label="Query language"
        >
          {LANGUAGES.map((l) => (
            <option key={l.code} value={l.code}>{l.label}</option>
          ))}
        </select>

        {mode !== "listening" ? (
          <button
            type="button"
            onClick={startListening}
            disabled={mode === "processing"}
            style={{
              ...styles.micBtn,
              opacity: mode === "processing" ? 0.6 : 1,
              cursor: mode === "processing" ? "not-allowed" : "pointer",
            }}
            aria-label="Start voice query"
          >
            🎤 Speak
          </button>
        ) : (
          <button
            type="button"
            onClick={stopListening}
            style={{ ...styles.micBtn, backgroundColor: "#dc2626" }}
            aria-label="Stop listening"
          >
            ⏹ Stop
          </button>
        )}
      </div>

      <form onSubmit={handleManualSubmit} style={styles.form}>
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type or speak your query..."
          style={styles.input}
          aria-label="Query text"
        />
        <button
          type="submit"
          disabled={mode === "processing" || !text.trim()}
          style={{
            ...styles.submitBtn,
            opacity: (mode === "processing" || !text.trim()) ? 0.5 : 1,
            cursor: (mode === "processing" || !text.trim()) ? "not-allowed" : "pointer",
          }}
        >
          {mode === "processing" ? "Searching…" : "Search"}
        </button>
      </form>

      {mode === "listening" && (
        <p style={styles.status}>🎙 Listening… speak now</p>
      )}
      {mode === "processing" && (
        <p style={styles.status}>⏳ Processing query…</p>
      )}
      {error && (
        <p role="alert" style={styles.error}>{error}</p>
      )}

      {results && (
        <div style={styles.results}>
          <h3 style={styles.subheading}>
            {results.total_matches} match{results.total_matches !== 1 ? "es" : ""} for "{results.query}"
          </h3>
          {results.alerts.length === 0 && (
            <p style={styles.empty}>No matching events found.</p>
          )}
          {results.alerts.map((alert, idx) => (
            <div key={idx} style={styles.alertCard}>
              <div style={styles.alertHeader}>
                <span style={styles.alertLabel}>{String(alert.label ?? "Alert")}</span>
                <span style={styles.alertConf}>
                  {Math.round((Number(alert.confidence ?? 0) * 100))}%
                </span>
              </div>
              <p style={styles.alertReason}>{String(alert.reason ?? "")}</p>
              <p style={styles.alertMeta}>
                Track {alert.track_id} · Cam {alert.camera_id}
              </p>
            </div>
          ))}
          {results.suggestions.length > 0 && (
            <div style={styles.suggestions}>
              <span style={styles.suggestionsLabel}>Try:</span>
              {results.suggestions.map((s, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => handleSuggestionClick(s)}
                  style={styles.suggestionChip}
                >
                  {s}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    background: "#0f172a",
    color: "#e2e8f0",
    padding: "20px",
    borderRadius: "8px",
    maxWidth: "600px",
  },
  heading: {
    fontSize: "1.1rem",
    fontWeight: 700,
    color: "#38bdf8",
    marginBottom: "12px",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  controls: {
    display: "flex",
    gap: "10px",
    alignItems: "center",
    marginBottom: "12px",
  },
  select: {
    background: "#1e293b",
    color: "#e2e8f0",
    border: "1px solid #334155",
    borderRadius: "6px",
    padding: "6px 10px",
    fontFamily: "inherit",
  },
  micBtn: {
    background: "#22c55e",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    padding: "8px 14px",
    fontFamily: "inherit",
    fontWeight: 600,
    cursor: "pointer",
  },
  form: {
    display: "flex",
    gap: "8px",
    marginBottom: "10px",
  },
  input: {
    flex: 1,
    background: "#1e293b",
    border: "1px solid #334155",
    borderRadius: "6px",
    color: "#e2e8f0",
    padding: "8px 12px",
    fontFamily: "inherit",
    outline: "none",
  },
  submitBtn: {
    background: "#0ea5e9",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    padding: "8px 16px",
    fontFamily: "inherit",
    fontWeight: 600,
    cursor: "pointer",
  },
  status: {
    color: "#38bdf8",
    fontSize: "0.85rem",
    marginBottom: "8px",
  },
  error: {
    color: "#fca5a5",
    fontSize: "0.8rem",
    marginBottom: "8px",
  },
  results: {
    marginTop: "16px",
  },
  subheading: {
    fontSize: "0.85rem",
    color: "#94a3b8",
    marginBottom: "10px",
    textTransform: "uppercase",
    letterSpacing: "0.08em",
  },
  empty: {
    color: "#64748b",
    fontSize: "0.85rem",
  },
  alertCard: {
    background: "#1e293b",
    borderRadius: "6px",
    padding: "10px 12px",
    marginBottom: "8px",
  },
  alertHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "4px",
  },
  alertLabel: {
    fontWeight: 700,
    color: "#38bdf8",
    fontSize: "0.8rem",
    textTransform: "uppercase",
  },
  alertConf: {
    color: "#22c55e",
    fontSize: "0.75rem",
    fontWeight: 600,
  },
  alertReason: {
    color: "#e2e8f0",
    fontSize: "0.85rem",
    margin: "4px 0",
  },
  alertMeta: {
    color: "#64748b",
    fontSize: "0.75rem",
  },
  suggestions: {
    display: "flex",
    gap: "8px",
    flexWrap: "wrap",
    marginTop: "12px",
    alignItems: "center",
  },
  suggestionsLabel: {
    color: "#94a3b8",
    fontSize: "0.75rem",
    textTransform: "uppercase",
    letterSpacing: "0.08em",
  },
  suggestionChip: {
    background: "#1e293b",
    border: "1px solid #334155",
    color: "#38bdf8",
    borderRadius: "999px",
    padding: "4px 10px",
    fontSize: "0.75rem",
    cursor: "pointer",
    fontFamily: "inherit",
  },
};
