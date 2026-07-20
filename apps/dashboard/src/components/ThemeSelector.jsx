import { useState, useEffect } from 'react';
import { themePresets, THEME_STORAGE_KEY, getActiveTheme, setActiveTheme, getCustomThemes } from '../utils/themes';

export default function ThemeSelector({ onThemeChange }) {
  const [activeTheme, setActive] = useState(getActiveTheme());
  const [customThemes, setCustomThemes] = useState(getCustomThemes());
  const allThemes = { ...themePresets, ...customThemes };

  useEffect(() => {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved && allThemes[saved]) setActive(saved);
  }, [customThemes]);

  function handleSelect(themeName) {
    setActiveTheme(themeName);
    setActive(themeName);
    if (onThemeChange) onThemeChange(themeName);
  }

  return (
    <div className="theme-selector p-4 bg-zinc-900 rounded-lg">
      <h3 className="text-lg font-semibold mb-3 text-white">Detection Overlay Theme</h3>
      <div className="grid grid-cols-1 gap-2">
        {Object.entries(allThemes).map(([key, theme]) => (
          <button
            key={key}
            onClick={() => handleSelect(key)}
            className={`flex items-center justify-between px-4 py-3 rounded-lg border transition-all duration-200 ${
              activeTheme === key
                ? 'border-green-500 bg-green-500/10 text-green-400'
                : 'border-zinc-700 bg-zinc-800 text-zinc-300 hover:border-zinc-500'
            }`}
          >
            <div className="text-left">
              <div className="font-medium">{theme.name}</div>
              <div className="text-xs text-zinc-500 mt-1">{theme.description}</div>
            </div>
            <div className="flex gap-1">
              {theme.boundingBoxColors.palette.slice(0, 3).map((color, i) => (
                <div key={i} className="w-4 h-4 rounded-sm" style={{ backgroundColor: color }} />
              ))}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
