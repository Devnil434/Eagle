export const THEME_STORAGE_KEY = 'eagle_overlay_theme';

export const themePresets = {
  default: {
    name: 'Default',
    description: 'Standard surveillance overlay',
    boundingBoxColors: { palette: ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#a855f7'], opacity: 0.8, borderWidth: 4 },
    label: { visible: true, fontSize: 12, fontFamily: 'monospace', backgroundColor: 'rgba(0,0,0,0.6)', textColor: '#ffffff' },
    confidence: { visible: true, mode: 'percentage', color: '#ffffff' },
    trackingTrails: { enabled: true, maxLength: 20, trailOpacity: 0.5, trailWidth: 2 },
    accessibility: { highContrast: false, reducedMotion: false, largeText: false }
  },
  professional: {
    name: 'Professional',
    description: 'Clean, muted palette for long shifts',
    boundingBoxColors: { palette: ['#f87171', '#60a5fa', '#4ade80', '#fbbf24', '#c084fc'], opacity: 0.6, borderWidth: 3 },
    label: { visible: true, fontSize: 11, fontFamily: 'sans-serif', backgroundColor: 'rgba(30,30,30,0.7)', textColor: '#e0e0e0' },
    confidence: { visible: true, mode: 'percentage', color: '#d1d5db' },
    trackingTrails: { enabled: true, maxLength: 15, trailOpacity: 0.4, trailWidth: 2 },
    accessibility: { highContrast: false, reducedMotion: false, largeText: false }
  },
  highContrast: {
    name: 'High Contrast',
    description: 'Maximum visibility for accessibility',
    boundingBoxColors: { palette: ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff'], opacity: 1.0, borderWidth: 5 },
    label: { visible: true, fontSize: 14, fontFamily: 'sans-serif', backgroundColor: 'rgba(0,0,0,0.9)', textColor: '#ffffff', fontWeight: 'bold' },
    confidence: { visible: true, mode: 'bar', color: '#00ff00' },
    trackingTrails: { enabled: true, maxLength: 25, trailOpacity: 0.8, trailWidth: 3 },
    accessibility: { highContrast: true, reducedMotion: false, largeText: true }
  },
  minimal: {
    name: 'Minimal',
    description: 'Subtle overlay with reduced visual clutter',
    boundingBoxColors: { palette: ['#dc2626', '#2563eb', '#16a34a', '#d97706', '#9333ea'], opacity: 0.4, borderWidth: 2 },
    label: { visible: true, fontSize: 10, fontFamily: 'monospace', backgroundColor: 'transparent', textColor: '#ffffff' },
    confidence: { visible: false, mode: 'off', color: '#ffffff' },
    trackingTrails: { enabled: false, maxLength: 10, trailOpacity: 0.3, trailWidth: 1 },
    accessibility: { highContrast: false, reducedMotion: true, largeText: false }
  },
  nightMode: {
    name: 'Night Mode',
    description: 'Dimmed palette for dark environments',
    boundingBoxColors: { palette: ['#ff6b6b', '#74b9ff', '#55efc4', '#ffeaa7', '#a29bfe'], opacity: 0.5, borderWidth: 3 },
    label: { visible: true, fontSize: 12, fontFamily: 'monospace', backgroundColor: 'rgba(10,10,20,0.8)', textColor: '#a0a0b0' },
    confidence: { visible: true, mode: 'percentage', color: '#a0a0b0' },
    trackingTrails: { enabled: true, maxLength: 15, trailOpacity: 0.3, trailWidth: 2 },
    accessibility: { highContrast: false, reducedMotion: false, largeText: false }
  },
  colorBlind: {
    name: 'Color Blind Friendly',
    description: 'Pattern-based cues for color vision deficiency',
    boundingBoxColors: { palette: ['#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2'], opacity: 0.85, borderWidth: 4 },
    label: { visible: true, fontSize: 13, fontFamily: 'sans-serif', backgroundColor: 'rgba(0,0,0,0.7)', textColor: '#ffffff' },
    confidence: { visible: true, mode: 'bar', color: '#56b4e9' },
    trackingTrails: { enabled: true, maxLength: 20, trailOpacity: 0.6, trailWidth: 3 },
    accessibility: { highContrast: true, reducedMotion: false, largeText: true }
  }
};

export function getActiveTheme() {
  try { const saved = localStorage.getItem(THEME_STORAGE_KEY); if (saved && themePresets[saved]) return saved; } catch (e) {}
  return 'default';
}

export function setActiveTheme(themeName) {
  try { localStorage.setItem(THEME_STORAGE_KEY, themeName); } catch (e) {}
}

export function getThemeForTrack(trackIndex, themeName) {
  const name = themeName || getActiveTheme();
  const theme = themePresets[name] || themePresets.default;
  const palette = theme.boundingBoxColors.palette;
  return { borderColor: palette[trackIndex % palette.length], ...theme };
}

export function createCustomTheme(name, overrides = {}) {
  const base = { ...themePresets.default, ...overrides };
  return { name, ...base };
}

export function saveCustomTheme(theme) {
  try {
    const stored = JSON.parse(localStorage.getItem('eagle_custom_themes') || '{}');
    stored[theme.name] = theme;
    localStorage.setItem('eagle_custom_themes', JSON.stringify(stored));
  } catch (e) {}
}

export function getCustomThemes() {
  try {
    return JSON.parse(localStorage.getItem('eagle_custom_themes') || '{}');
  } catch (e) {
    return {};
  }
}
