"""Inject the payload and the vendored chart library into the page template."""
from __future__ import annotations

import json
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


def render(payload: dict) -> str:
    html = (WEB / "template.html").read_text()
    lib = (WEB / "vendor" / "echarts.min.js").read_text()
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return (html.replace("/*__ECHARTS__*/", lib)
                .replace("/*__DATA__*/null", data)
                .replace("__TITLE__", "Cycle & Rotation"))
