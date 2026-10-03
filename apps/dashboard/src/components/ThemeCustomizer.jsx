import { useState } from "react";
import { createCustomTheme, saveCustomTheme, getCustomThemes, themePresets } from "../utils/themes";

export default function ThemeCustomizer({ onThemeCreated }) {
  const [name, setName] = useState("");
  const [baseTheme, setBaseTheme] = useState("default");
  const [opacity, setOpacity] = useState(0.8);
  const [borderWidth, setBorderWidth] = useState(4);
  const [palette, setPalette] = useState(["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7"]);
  const [fontSize, setFontSize] = useState(12);
  const [trailOpacity, setTrailOpacity] = useState(0.5);
  const [trailWidth, setTrailWidth] = useState(2);
  const [highContrast, setHighContrast] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [largeText, setLargeText] = useState(false);

  function updatePalette(index, color) {
    setPalette((prev) => {
      const next = [...prev];
      next[index] = color;
      return next;
    });
  }

  function handleSave() {
    if (!name.trim()) return;
    const base = themePresets[baseTheme] || themePresets.default;
    const custom = createCustomTheme(name.trim(), {
      boundingBoxColors: {
        palette,
        opacity,
        borderWidth,
      },
      label: {
        ...base.label,
        fontSize,
      },
      confidence: base.confidence,
      trackingTrails: {
        ...base.trackingTrails,
        trailOpacity,
        trailWidth,
      },
      accessibility: {
        highContrast,
        reducedMotion,
        largeText,
      },
    });
    saveCustomTheme(custom);
    if (onThemeCreated) onThemeCreated(name.trim());
    setName("");
  }

  const customThemes = getCustomThemes();

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
      <h3 className="text-lg font-semibold text-white mb-3">Customize Theme</h3>

      <div className="space-y-3">
        <div>
          <label className="text-xs text-zinc-400 block mb-1">Theme Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="My Custom Theme"
            className="w-full px-3 py-2 rounded bg-zinc-800 text-white border border-zinc-700 text-sm"
          />
        </div>

        <div>
          <label className="text-xs text-zinc-400 block mb-1">Base Theme</label>
          <select
            value={baseTheme}
            onChange={(e) => setBaseTheme(e.target.value)}
            className="w-full px-3 py-2 rounded bg-zinc-800 text-white border border-zinc-700 text-sm"
          >
            {Object.entries(themePresets).map(([key, t]) => (
              <option key={key} value={key}>{t.name}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-xs text-zinc-400 block mb-1">Bounding Box Opacity: {opacity}</label>
          <input type="range" min="0" max="1" step="0.1" value={opacity} onChange={(e) => setOpacity(parseFloat(e.target.value))} className="w-full" />
        </div>

        <div>
          <label className="text-xs text-zinc-400 block mb-1">Border Width: {borderWidth}px</label>
          <input type="range" min="1" max="8" step="1" value={borderWidth} onChange={(e) => setBorderWidth(parseInt(e.target.value, 10))} className="w-full" />
        </div>

        <div>
          <label className="text-xs text-zinc-400 block mb-1">Label Font Size: {fontSize}px</label>
          <input type="range" min="8" max="20" step="1" value={fontSize} onChange={(e) => setFontSize(parseInt(e.target.value, 10))} className="w-full" />
        </div>

        <div>
          <label className="text-xs text-zinc-400 block mb-1">Trail Opacity: {trailOpacity}</label>
          <input type="range" min="0" max="1" step="0.1" value={trailOpacity} onChange={(e) => setTrailOpacity(parseFloat(e.target.value))} className="w-full" />
        </div>

        <div>
          <label className="text-xs text-zinc-400 block mb-1">Trail Width: {trailWidth}px</label>
          <input type="range" min="1" max="6" step="1" value={trailWidth} onChange={(e) => setTrailWidth(parseInt(e.target.value, 10))} className="w-full" />
        </div>

        <div>
          <label className="text-xs text-zinc-400 block mb-1">Bounding Box Colors</label>
          <div className="flex gap-2">
            {palette.map((color, i) => (
              <input
                key={i}
                type="color"
                value={color}
                onChange={(e) => updatePalette(i, e.target.value)}
                className="w-10 h-10 rounded cursor-pointer border-0"
              />
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input id="high-contrast" type="checkbox" checked={highContrast} onChange={(e) => setHighContrast(e.target.checked)} />
          <label htmlFor="high-contrast" className="text-sm text-zinc-300">High Contrast</label>
        </div>

        <div className="flex items-center gap-2">
          <input id="reduced-motion" type="checkbox" checked={reducedMotion} onChange={(e) => setReducedMotion(e.target.checked)} />
          <label htmlFor="reduced-motion" className="text-sm text-zinc-300">Reduced Motion</label>
        </div>

        <div className="flex items-center gap-2">
          <input id="large-text" type="checkbox" checked={largeText} onChange={(e) => setLargeText(e.target.checked)} />
          <label htmlFor="large-text" className="text-sm text-zinc-300">Large Text</label>
        </div>

        <button
          onClick={handleSave}
          disabled={!name.trim()}
          className="w-full px-4 py-2 bg-green-600 text-white rounded hover:bg-green-500 disabled:opacity-50"
        >
          Save Custom Theme
        </button>
      </div>

      {Object.keys(customThemes).length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-semibold text-zinc-300 mb-2">Saved Custom Themes</h4>
          <div className="space-y-1">
            {Object.entries(customThemes).map(([key, theme]) => (
              <div key={key} className="text-xs text-zinc-400 bg-zinc-800 p-2 rounded flex justify-between items-center">
                <span>{theme.name}</span>
                <div className="flex gap-1">
                  {theme.boundingBoxColors.palette.slice(0, 3).map((color, i) => (
                    <div key={i} className="w-3 h-3 rounded-sm" style={{ backgroundColor: color }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
