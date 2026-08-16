"""
Output formats for incident summary reports.

Each renderer turns an `IncidentSummary` into bytes plus the media type to serve
it as.  Formats are looked up through `get_renderer`, so supporting a new one
means adding a class and a registry entry — no existing renderer or the route
itself needs to change.

The PDF renderer intentionally reuses the Markdown renderer's output rather than
owning a second template, keeping one definition of report content.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, get_args

from services.reporting.models import IncidentSummary

TEMPLATE_DIR = Path(__file__).parent / "templates"
SUMMARY_TEMPLATE = "summary.md.j2"

# Single source of truth for the formats on offer: the API route annotates its
# query parameter with this type, so validation and the registry cannot drift.
ReportFormat = Literal["markdown", "json", "pdf"]

# fpdf2's built-in fonts are latin-1 only, while VLM/LLM text and the template
# both use typographic characters.  Transliterate the common ones instead of
# dropping them, so a PDF stays readable.
_LATIN1_SUBSTITUTIONS = {
    "\u2192": "->",   # →
    "\u00d7": "x",    # ×
    "\u2014": "-",    # —
    "\u2013": "-",    # –
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u00a0": " ",
}


class ReportRenderError(RuntimeError):
    """Raised when a format cannot be produced in this environment."""


class ReportRenderer(Protocol):
    """Renders a summary into transferable bytes."""

    media_type: str
    extension: str

    def render(self, summary: IncidentSummary) -> bytes:
        ...


# ── Jinja filters ─────────────────────────────────────────────────────────────

def _format_timestamp(timestamp_ms: float) -> str:
    if not timestamp_ms:
        return "n/a"
    moment = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _oneline(text: str) -> str:
    """Make free text safe for a single Markdown table cell."""
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    return collapsed.replace("|", "\\|")


# ── Renderers ─────────────────────────────────────────────────────────────────

class JsonRenderer:
    """Machine-readable output, for dashboards and downstream tooling."""

    media_type = "application/json"
    extension = "json"

    def render(self, summary: IncidentSummary) -> bytes:
        return summary.model_dump_json(indent=2).encode("utf-8")


class MarkdownRenderer:
    """Human-readable report rendered from the Jinja2 template."""

    media_type = "text/markdown; charset=utf-8"
    extension = "md"

    def render(self, summary: IncidentSummary) -> bytes:
        return self.render_text(summary).encode("utf-8")

    def render_text(self, summary: IncidentSummary) -> str:
        template = _template_environment().get_template(SUMMARY_TEMPLATE)
        # Jinja keeps a trailing newline per block; collapse the runs of blank
        # lines that leaves so the Markdown reads cleanly.
        rendered = template.render(s=summary)
        return re.sub(r"\n{3,}", "\n\n", rendered).strip() + "\n"


@lru_cache(maxsize=1)
def _template_environment():
    """Build the Jinja environment once; templates are parsed per process."""
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ReportRenderError(
            "Jinja2 is required for Markdown reports. Install it with "
            "`pip install jinja2`."
        ) from exc

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        # Markdown is not markup, so HTML escaping would corrupt the output.
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["ts"] = _format_timestamp
    env.filters["pct"] = _format_percent
    env.filters["duration"] = _format_duration
    env.filters["oneline"] = _oneline
    return env


class PdfRenderer:
    """PDF export, converted from the Markdown report via HTML."""

    media_type = "application/pdf"
    extension = "pdf"

    def __init__(self, markdown_renderer: MarkdownRenderer | None = None) -> None:
        self._markdown = markdown_renderer or MarkdownRenderer()

    def render(self, summary: IncidentSummary) -> bytes:
        html = self._to_html(self._markdown.render_text(summary))
        return self._to_pdf(_to_latin1(html))

    @staticmethod
    def _to_html(markdown_text: str) -> str:
        try:
            import markdown
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ReportRenderError(
                "The `markdown` package is required for PDF reports. Install it "
                "with `pip install markdown`."
            ) from exc
        return markdown.markdown(markdown_text, extensions=["tables"])

    @staticmethod
    def _to_pdf(html: str) -> bytes:
        try:
            from fpdf import FPDF
        except ImportError as exc:
            raise ReportRenderError(
                "PDF export requires fpdf2. Install it with `pip install fpdf2`, "
                "or request the report as markdown or json."
            ) from exc

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=9)
        try:
            pdf.write_html(html)
        except Exception as exc:
            raise ReportRenderError(f"PDF generation failed: {exc}") from exc
        return bytes(pdf.output())


def _to_latin1(text: str) -> str:
    for source, replacement in _LATIN1_SUBSTITUTIONS.items():
        text = text.replace(source, replacement)
    # Anything still outside latin-1 becomes "?" rather than failing the export.
    return text.encode("latin-1", "replace").decode("latin-1")


_RENDERERS: dict[str, ReportRenderer] = {
    "markdown": MarkdownRenderer(),
    "json": JsonRenderer(),
    "pdf": PdfRenderer(),
}

SUPPORTED_FORMATS = get_args(ReportFormat)

assert set(_RENDERERS) == set(SUPPORTED_FORMATS), (
    "every ReportFormat needs a renderer"
)


def get_renderer(report_format: str) -> ReportRenderer:
    try:
        return _RENDERERS[report_format]
    except KeyError as exc:
        raise ReportRenderError(
            f"Unsupported report format {report_format!r}. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}."
        ) from exc
