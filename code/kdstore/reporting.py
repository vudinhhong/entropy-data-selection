"""Printing a run's output and writing it down.

Every experiment group ends the same way: a few tables on the terminal so you
can see what happened, and one JSON file plus one plain-text transcript in
`results/` so the numbers survive the session. `Report` does both from a single
set of calls, which is the only way the two stay in step.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from .paths import prepare


def _fmt(value, places=2):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:                       # nan
            return "-"
        if abs(value) < 1e-4 and value != 0.0:
            return f"{value:.2e}"
        return f"{value:.{places}f}"
    return str(value)


def _looks_numeric(text):
    try:
        float(str(text).replace("%", "").replace("x", ""))
        return True
    except ValueError:
        return str(text) == "-"


class Report:
    """Collects tables and values, prints them, writes them out."""

    def __init__(self, title, places=2):
        self.title = title
        self.places = places
        self.lines = []
        self.data = {"report": title,
                     "generated": datetime.now(timezone.utc)
                                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "tables": {}, "values": {}}
        self._emit("=" * 78)
        self._emit(title)
        self._emit("=" * 78)

    def _emit(self, line=""):
        print(line, flush=True)
        self.lines.append(line)

    def note(self, text=""):
        """A line of prose between tables."""
        self._emit(text)

    def value(self, key, value, label=None, unit=""):
        """One scalar worth recording."""
        self.data["values"][key] = value
        shown = _fmt(value, self.places)
        self._emit(f"  {label or key:<44s} {shown}{unit}")

    def table(self, key, headers, rows, title=None, align=None, places=None):
        """A table, printed with aligned columns and stored row by row."""
        places = self.places if places is None else places
        body = [[_fmt(c, places) for c in row] for row in rows]
        widths = [max(len(str(headers[i])), *(len(r[i]) for r in body))
                  if body else len(str(headers[i]))
                  for i in range(len(headers))]
        # Right-align a column only if everything in it reads as a number;
        # a column of prose right-aligned is unreadable.
        if align is None:
            align = []
            for i in range(len(headers)):
                values = [r[i] for r in body]
                numeric = all(_looks_numeric(v) for v in values) if values else False
                align.append("r" if numeric and i > 0 else "l")

        def line(cells):
            out = []
            for cell, width, how in zip(cells, widths, align):
                out.append(str(cell).ljust(width) if how == "l"
                           else str(cell).rjust(width))
            return "  " + "  ".join(out)

        self._emit()
        if title:
            self._emit(f"  {title}")
        self._emit(line(headers))
        self._emit("  " + "  ".join("-" * w for w in widths))
        for row in body:
            self._emit(line(row))

        self.data["tables"][key] = {
            "title": title or key,
            "headers": list(headers),
            "rows": [list(r) for r in rows],
        }

    def save(self, json_path, text_path=None):
        """Write the JSON record and the terminal transcript."""
        replaced = {}
        p = prepare(json_path)
        replaced[str(p)] = p.exists()
        p.write_text(json.dumps(self.data, indent=1, ensure_ascii=False,
                                default=str))
        written = [str(p)]
        if text_path:
            t = prepare(text_path)
            replaced[str(t)] = t.exists()
            t.write_text("\n".join(self.lines) + "\n")
            written.append(str(t))
        self._emit()
        for w in written:
            existed = " (REPLACED an existing file)" if replaced.get(w) else ""
            self._emit(f"  written: {w}{existed}")
        return written


def load_config(path):
    """Read a group's config.json."""
    return json.loads(Path(path).read_text())


def required(spec, key, where=""):
    """Fetch a config value that must not be guessed.

    A missing corpus or model silently falling back to a default is the kind
    of mistake that produces a plausible number from the wrong experiment, so
    these fail loudly instead.
    """
    if key not in spec or spec[key] is None:
        raise SystemExit(
            f"config error{' in ' + where if where else ''}: '{key}' is "
            f"required and has no safe default")
    return spec[key]
