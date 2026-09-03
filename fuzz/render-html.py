#!/usr/bin/env python3
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib


from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FUZZ_DIR = ROOT / "fuzz"
if str(FUZZ_DIR) not in sys.path:
    sys.path.insert(0, str(FUZZ_DIR))

try:
    from simjit_lisp_format import FormatError, format_lisp
except Exception:  # pragma: no cover - report generation should still work.
    FormatError = Exception

    def format_lisp(payload: str) -> str:
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Simjit fuzz outcome JSON as a grouped HTML report")
    parser.add_argument("input", type=Path, help="fuzz/simjit-fuzz outcome JSON")
    parser.add_argument("-o", "--output", type=Path, help="HTML output path")
    parser.add_argument("--include-passes", action="store_true", default=False, help="Render passing cases too")
    parser.add_argument("--max-message", type=int, default=220, help="Maximum message length in group headers")
    return parser.parse_args()


def html_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def json_block(value: Any) -> str:
    if value in ({}, [], None, ""):
        return ""
    return json.dumps(value, indent=2, sort_keys=True)


def normalized_message(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def error_metadata_fingerprint(item: dict[str, Any]) -> str:
    metadata = item.get("error_metadata")
    if not isinstance(metadata, dict) or not metadata:
        return ""

    parts: list[str] = []
    for key in sorted(metadata):
        info = metadata.get(key)
        if not isinstance(info, dict):
            parts.append(f"{key}:{info}")
            continue
        module = info.get("module_name", info.get("module", ""))
        kind = info.get("kind_name", info.get("kind", ""))
        subkind = info.get("subkind_name", info.get("subkind", ""))
        message = normalized_message(str(info.get("message", "")), 160)
        parts.append(f"{key}:{module}:{kind}:{subkind}:{message}")
    return " | ".join(parts)


def error_text(item: dict[str, Any]) -> str:
    errors = item.get("errors")
    if isinstance(errors, dict):
        if "comparison" in errors:
            return str(errors.get("comparison", ""))
        if "stage" in errors:
            return str(errors.get("stage", ""))
        if "syntax" in errors:
            return str(errors.get("syntax", ""))
        if errors:
            return json.dumps(errors, sort_keys=True)
    if errors:
        return str(errors)
    comparison = item.get("comparison")
    if comparison:
        return json.dumps(comparison, sort_keys=True)
    return ""


def fingerprint_item(item: dict[str, Any], *, max_message: int) -> str:
    metadata_fp = error_metadata_fingerprint(item)
    if metadata_fp:
        return normalized_message(metadata_fp, max_message)
    message = error_text(item)
    if message:
        return normalized_message(message, max_message)
    return "no error payload"


def pretty_serialized(payload: str) -> str:
    if not payload:
        return ""
    try:
        return format_lisp(payload)
    except FormatError:
        return payload
    except Exception:
        return payload


def load_outcome(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("outcome JSON must be an object")
    if payload.get("kind") != "outcome":
        raise RuntimeError("expected outcome JSON with kind='outcome'")
    items = payload.get("items")
    if not isinstance(items, list):
        raise RuntimeError("outcome JSON is missing list field 'items'")
    return payload


def item_sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(item.get("program_index", 0)),
        int(item.get("program_seed", 0)),
        str(item.get("id", "")),
    )


def group_items(items: list[dict[str, Any]], *, max_message: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        status = str(item.get("status", "unknown"))
        scalar_status = str(item.get("scalar_status", "unknown"))
        vector_status = str(item.get("vector_status", "unknown"))
        fingerprint = fingerprint_item(item, max_message=max_message)
        buckets[(status, scalar_status, vector_status, fingerprint)].append(item)

    groups = []
    for (status, scalar_status, vector_status, fingerprint), group in buckets.items():
        group.sort(key=item_sort_key)
        groups.append(
            {
                "status": status,
                "scalar_status": scalar_status,
                "vector_status": vector_status,
                "fingerprint": fingerprint,
                "items": group,
            }
        )
    groups.sort(key=lambda group: (-len(group["items"]), group["status"], group["fingerprint"]))
    return groups


def render_summary(summary: dict[str, Any], items: list[dict[str, Any]], rendered_count: int,
                   *, include_passes: bool) -> str:
    status_counts = Counter(str(item.get("status", "unknown")) for item in items)
    cards = []
    for key in sorted(summary):
        cards.append(
            f'<div class="stat"><span class="stat-label">{html_escape(key)}</span>'
            f'<span class="stat-value">{html_escape(summary[key])}</span></div>'
        )
    status_rows = "\n".join(
        f"<tr><td>{html_escape(status)}</td><td>{count}</td></tr>"
        for status, count in sorted(status_counts.items())
    )
    return f"""
<section class="summary">
  <div class="stats">{''.join(cards)}</div>
  <div class="summary-grid">
    <table>
      <caption>Status Counts</caption>
      <thead><tr><th>Status</th><th>Count</th></tr></thead>
      <tbody>{status_rows}</tbody>
    </table>
    <table>
      <caption>Render Scope</caption>
      <tbody>
        <tr><td>Total items</td><td>{len(items)}</td></tr>
        <tr><td>Rendered items</td><td>{rendered_count}</td></tr>
        <tr><td>Hidden passes</td><td>{0 if include_passes else status_counts.get("pass", 0)}</td></tr>
      </tbody>
    </table>
  </div>
</section>
"""


def render_code_panel(label: str, code: str, *, open_by_default: bool = False) -> str:
    if not code:
        return ""
    open_attr = " open" if open_by_default else ""
    escaped = html_escape(code)
    return f"""
<details class="code-panel"{open_attr}>
  <summary>{html_escape(label)}</summary>
  <button class="copy-btn" type="button">Copy</button>
  <pre>{escaped}</pre>
</details>
"""


def render_case(item: dict[str, Any]) -> str:
    status = str(item.get("status", "unknown"))
    case_id = item.get("id", "?")
    seed = item.get("program_seed", "?")
    index = item.get("program_index", "?")
    profile = item.get("profile", "?")
    preset = item.get("preset", "?")
    arch = item.get("arch", "?")
    scalar = item.get("scalar_status", "?")
    vector = item.get("vector_status", "?")
    serialized = pretty_serialized(str(item.get("serialized", "")))
    errors = json_block(item.get("errors"))
    error_metadata = json_block(item.get("error_metadata"))
    comparison = json_block(item.get("comparison"))

    panels = [
        render_code_panel("Serialized HIR", serialized, open_by_default=status != "pass"),
        render_code_panel("Errors", errors, open_by_default=bool(errors)),
        render_code_panel("Error Metadata", error_metadata),
        render_code_panel("Comparison", comparison),
        render_code_panel("Raw Outcome Item", json.dumps(item, indent=2, sort_keys=True)),
    ]
    return f"""
<details class="case" data-search="{html_escape(json.dumps(item, sort_keys=True))}">
  <summary>
    <span class="badge status-{html_escape(status)}">{html_escape(status)}</span>
    <span class="case-title">{html_escape(case_id)}</span>
    <span class="case-meta">seed={html_escape(seed)} idx={html_escape(index)} profile={html_escape(profile)}
      preset={html_escape(preset)} arch={html_escape(arch)}
      scalar={html_escape(scalar)} vector={html_escape(vector)}</span>
  </summary>
  <div class="case-body">
    {''.join(panels)}
  </div>
</details>
"""


def render_group(group: dict[str, Any], index: int) -> str:
    items = group["items"]
    first = items[0]
    cases = "\n".join(render_case(item) for item in items)
    return f"""
<details class="group" open data-search="{html_escape(group['fingerprint'])}">
  <summary>
    <span class="group-index">#{index}</span>
    <span class="badge status-{html_escape(group['status'])}">{html_escape(group['status'])}</span>
    <span class="group-count">{len(items)} case{'s' if len(items) != 1 else ''}</span>
    <span class="group-meta">scalar={html_escape(group['scalar_status'])} vector={html_escape(group['vector_status'])}
      first-seed={html_escape(first.get('program_seed', '?'))} first-idx={html_escape(first.get('program_index', '?'))}</span>
    <span class="fingerprint">{html_escape(group['fingerprint'])}</span>
  </summary>
  <div class="group-body">{cases}</div>
</details>
"""


def render_html(outcome: dict[str, Any], *, source: Path, include_passes: bool, max_message: int) -> str:
    items = list(outcome.get("items", []))
    rendered_items = items if include_passes else [item for item in items if item.get("status") != "pass"]
    groups = group_items(rendered_items, max_message=max_message)
    group_html = "\n".join(render_group(group, index + 1) for index, group in enumerate(groups))
    if not group_html:
        group_html = '<p class="empty">No rendered cases. Use --include-passes to inspect passing cases.</p>'

    title = "Simjit Fuzz Outcome"
    summary_html = render_summary(dict(outcome.get("summary", {})), items, len(rendered_items),
                                  include_passes=include_passes)
    include_text = "including passes" if include_passes else "non-pass cases only"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f7f7f4;
  --panel: #ffffff;
  --ink: #1e2329;
  --muted: #68707a;
  --line: #d8dce0;
  --accent: #245f9d;
  --pass: #20784f;
  --fail: #a13a32;
  --warn: #8a6200;
  --code-bg: #111821;
  --code-ink: #f3f5f7;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
header {{
  padding: 24px 32px 16px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}}
h1 {{ margin: 0 0 8px; font-size: 24px; }}
.subhead {{ color: var(--muted); font-size: 13px; }}
.toolbar {{
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 12px 32px;
  border-bottom: 1px solid var(--line);
  background: rgba(255,255,255,.96);
}}
input[type="search"] {{
  min-width: 280px;
  flex: 1;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  font: inherit;
}}
button {{
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  padding: 8px 10px;
  cursor: pointer;
}}
main {{ padding: 20px 32px 48px; }}
.summary {{
  margin-bottom: 16px;
}}
.stats {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 8px;
  margin-bottom: 12px;
}}
.stat {{
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  padding: 10px;
}}
.stat-label {{
  display: block;
  color: var(--muted);
  font-size: 12px;
}}
.stat-value {{
  display: block;
  margin-top: 4px;
  font: 600 18px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
.summary-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: hidden;
}}
caption {{
  text-align: left;
  padding: 8px 10px;
  font-weight: 600;
}}
th, td {{
  border-top: 1px solid var(--line);
  padding: 7px 10px;
  text-align: left;
  font-size: 13px;
}}
.group, .case {{
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  margin: 10px 0;
}}
.group > summary, .case > summary {{
  cursor: pointer;
  padding: 10px 12px;
}}
.group-body, .case-body {{
  border-top: 1px solid var(--line);
  padding: 10px 12px;
}}
.group-count, .group-meta, .case-meta, .fingerprint {{
  color: var(--muted);
  margin-left: 8px;
  font-size: 12px;
}}
.fingerprint {{
  display: block;
  margin: 6px 0 0 0;
  color: var(--ink);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
.case-title {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-weight: 600;
}}
.badge {{
  display: inline-block;
  min-width: 82px;
  border-radius: 999px;
  padding: 2px 8px;
  text-align: center;
  color: #fff;
  font-size: 12px;
  font-weight: 600;
}}
.status-pass {{ background: var(--pass); }}
.status-mismatch, .status-scalar-fail, .status-vector-fail, .status-builder-fail, .status-syntax-fail, .status-crash {{ background: var(--fail); }}
.status-unknown {{ background: var(--warn); }}
.code-panel {{
  position: relative;
  margin: 10px 0;
}}
.code-panel summary {{
  cursor: pointer;
  color: var(--accent);
  font-weight: 600;
}}
.copy-btn {{
  position: absolute;
  right: 8px;
  top: 28px;
  padding: 4px 7px;
  font-size: 12px;
}}
pre {{
  overflow: auto;
  max-height: 520px;
  margin: 8px 0 0;
  border-radius: 6px;
  background: var(--code-bg);
  color: var(--code-ink);
  padding: 12px;
  font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre;
}}
.empty {{
  border: 1px dashed var(--line);
  background: var(--panel);
  padding: 16px;
  color: var(--muted);
}}
.hidden {{ display: none; }}
@media (max-width: 700px) {{
  header, .toolbar, main {{ padding-left: 14px; padding-right: 14px; }}
  .toolbar {{ align-items: stretch; flex-wrap: wrap; }}
  input[type="search"] {{ min-width: 100%; }}
}}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="subhead">Source: {html_escape(source)} | Render scope: {html_escape(include_text)}</div>
</header>
<div class="toolbar">
  <input id="search" type="search" placeholder="Search status, seed, error text, or serialized HIR">
  <button type="button" id="expand">Expand groups</button>
  <button type="button" id="collapse">Collapse groups</button>
</div>
<main>
{summary_html}
<section id="groups">
{group_html}
</section>
</main>
<script>
function setGroups(open) {{
  document.querySelectorAll(".group").forEach(group => group.open = open);
}}
document.getElementById("expand").addEventListener("click", () => setGroups(true));
document.getElementById("collapse").addEventListener("click", () => setGroups(false));

document.querySelectorAll(".copy-btn").forEach(button => {{
  button.addEventListener("click", async event => {{
    event.preventDefault();
    const panel = button.closest(".code-panel");
    const text = panel ? panel.querySelector("pre").textContent : "";
    try {{
      await navigator.clipboard.writeText(text);
      const old = button.textContent;
      button.textContent = "Copied";
      setTimeout(() => button.textContent = old, 1200);
    }} catch (_err) {{
      button.textContent = "Copy failed";
      setTimeout(() => button.textContent = "Copy", 1200);
    }}
  }});
}});

document.getElementById("search").addEventListener("input", event => {{
  const query = event.target.value.trim().toLowerCase();
  document.querySelectorAll(".group").forEach(group => {{
    let groupMatch = !query || group.dataset.search.toLowerCase().includes(query);
    let visibleCases = 0;
    group.querySelectorAll(".case").forEach(caseEl => {{
      const match = !query || caseEl.dataset.search.toLowerCase().includes(query) || groupMatch;
      caseEl.classList.toggle("hidden", !match);
      if (match) visibleCases += 1;
    }});
    const visible = groupMatch || visibleCases > 0;
    group.classList.toggle("hidden", !visible);
    if (query && visible) group.open = true;
  }});
}});
</script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    outcome = load_outcome(args.input)
    output = args.output or args.input.with_suffix(".html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_html(outcome, source=args.input, include_passes=args.include_passes, max_message=args.max_message),
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
