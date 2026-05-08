import csv
import json
import mimetypes
import os
import subprocess
import sys
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.dashboard_events import read_dashboard_events


BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BASE_DIR / "results" / "dashboard"
RESULTS_DIR = BASE_DIR / "results"
HOST = "127.0.0.1"
PORT = 8501

RUN_PROCESS = None
RUN_LOGS = deque(maxlen=500)
RUN_LOCK = threading.Lock()


def read_json(path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_packets(rows):
    if not rows:
        return {
            "packet_count": 0,
            "avg_payload_kb": 0,
            "privacy_risks": 0,
        }

    sizes = [float(row.get("payload_size_kb", 0) or 0) for row in rows]
    risks = [
        str(row.get("privacy_risk", "")).lower() in {"true", "1"}
        for row in rows
    ]
    return {
        "packet_count": len(rows),
        "avg_payload_kb": round(sum(sizes) / max(1, len(sizes)), 2),
        "privacy_risks": sum(1 for risk in risks if risk),
    }


def collect_state():
    events = read_dashboard_events(DASHBOARD_DIR)
    state = read_json(DASHBOARD_DIR / "state.json", default={}) or {}
    metadata = read_json(DASHBOARD_DIR / "metadata.json", default={}) or {}
    summary = read_json(DASHBOARD_DIR / "summary.json", default={}) or {}
    new_client = read_json(DASHBOARD_DIR / "new_client_result.json", default={}) or {}
    privacy_rows = read_json(RESULTS_DIR / "privacy_comparison_demo.json", default=[]) or []
    comparison_rows = read_csv_rows(RESULTS_DIR / "comparison_table.csv")
    packet_rows = read_csv_rows(RESULTS_DIR / "packet_analysis.csv")

    with RUN_LOCK:
        running = RUN_PROCESS is not None and RUN_PROCESS.poll() is None
        logs = list(RUN_LOGS)

    if running:
        state["status"] = "running"
    elif not state:
        state = {"status": "idle", "current_stage": "ready"}

    images = {
        "centralized_accuracy": "/results/centralized_accuracy.png",
        "federated_accuracy": "/results/federated_accuracy.png",
        "comparison_scores": "/results/comparison_scores.png",
        "comparison_privacy": "/results/comparison_privacy.png",
        "privacy_payload": "/results/privacy_payload_comparison.png",
        "privacy_raw_vs_weights": "/results/privacy_raw_vs_weights.png",
        "new_client_accuracy": "/results/new_client_accuracy.png",
        "federated_confusion": "/results/federated_confusion_matrix.png",
    }

    return {
        "state": state,
        "events": events[-300:],
        "metadata": metadata,
        "summary": summary,
        "new_client": new_client,
        "privacy_rows": privacy_rows,
        "comparison_rows": comparison_rows,
        "packet_summary": summarize_packets(packet_rows),
        "images": images,
        "logs": logs[-180:],
    }


def stream_process_output(process):
    for line in process.stdout:
        with RUN_LOCK:
            RUN_LOGS.append(line.rstrip())

    process.wait()
    with RUN_LOCK:
        RUN_LOGS.append(f"[dashboard] process exited with code {process.returncode}")


def start_run(rounds):
    global RUN_PROCESS

    with RUN_LOCK:
        if RUN_PROCESS is not None and RUN_PROCESS.poll() is None:
            return False, "A dashboard run is already in progress."

        RUN_LOGS.clear()
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        cmd = [
            sys.executable,
            "dashboard_runner.py",
            "--rounds",
            str(rounds),
            "--holdout-client",
            "30",
            "--output-dir",
            str(DASHBOARD_DIR),
        ]
        RUN_PROCESS = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        thread = threading.Thread(
            target=stream_process_output,
            args=(RUN_PROCESS,),
            daemon=True,
        )
        thread.start()

    return True, "Dashboard run started."


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.send_html(INDEX_HTML)
            return

        if path == "/api/state":
            self.send_json(collect_state())
            return

        if path.startswith("/results/"):
            self.serve_result_file(path.removeprefix("/results/"))
            return

        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}

        rounds = int(payload.get("rounds", 5))
        rounds = max(1, min(rounds, 15))
        started, message = start_run(rounds)
        self.send_json({"started": started, "message": message})

    def send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_result_file(self, relative_path):
        safe_relative = Path(unquote(relative_path))
        if safe_relative.is_absolute() or ".." in safe_relative.parts:
            self.send_error(403)
            return

        file_path = RESULTS_DIR / safe_relative
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Privacy Preserving HAR Dashboard</title>
  <style>
    :root {
      --bg: #f7fafc;
      --panel: #ffffff;
      --ink: #17212b;
      --muted: #607080;
      --line: #dbe4ec;
      --soft: #eef4f7;
      --blue: #215a96;
      --teal: #0f766e;
      --green: #15845b;
      --amber: #b7791f;
      --red: #b42318;
      --violet: #6957d5;
      --shadow: 0 18px 44px rgba(30, 45, 60, 0.09);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background:
        radial-gradient(circle at 18% 9%, rgba(33, 90, 150, 0.10), transparent 30%),
        radial-gradient(circle at 84% 18%, rgba(15, 118, 110, 0.10), transparent 31%),
        var(--bg);
      color: var(--ink);
      font-family: Inter, "Segoe UI", system-ui, sans-serif;
      letter-spacing: 0;
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 270px 1fr;
    }

    aside {
      border-right: 1px solid var(--line);
      background: #ffffff;
      padding: 26px 18px;
      position: sticky;
      top: 0;
      height: 100vh;
    }

    .brand {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-bottom: 26px;
    }

    .mark {
      width: 38px;
      height: 38px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: white;
      background: linear-gradient(135deg, var(--blue), var(--teal));
      font-weight: 900;
    }

    .brand strong {
      display: block;
      font-size: 16px;
      line-height: 1.2;
    }

    .brand span {
      color: var(--muted);
      font-size: 12px;
    }

    nav {
      display: grid;
      gap: 8px;
    }

    .tab {
      border: 0;
      border-radius: 8px;
      padding: 11px 12px;
      background: transparent;
      color: #405060;
      text-align: left;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    .tab.active {
      background: #e8f1f8;
      color: var(--blue);
    }

    .sidebox {
      margin-top: 28px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
    }

    .sidebox .label,
    .metric .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 800;
    }

    .status {
      margin-top: 8px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-weight: 900;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--muted);
    }

    .dot.running { background: var(--amber); animation: pulse 1s infinite; }
    .dot.completed { background: var(--green); }
    .dot.failed { background: var(--red); }

    @keyframes pulse {
      50% { transform: scale(1.35); opacity: 0.55; }
    }

    main {
      padding: 28px;
    }

    .topbar {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 22px;
    }

    h1 {
      margin: 0;
      font-size: 31px;
      line-height: 1.1;
    }

    .sub {
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 840px;
      line-height: 1.55;
    }

    .runbar {
      display: flex;
      gap: 10px;
      align-items: center;
      padding: 10px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    select,
    button.primary,
    button.secondary {
      height: 40px;
      border-radius: 8px;
      border: 1px solid var(--line);
      padding: 0 13px;
      font: inherit;
      font-weight: 800;
    }

    button.primary {
      color: white;
      background: var(--blue);
      border-color: var(--blue);
      cursor: pointer;
    }

    button.secondary {
      color: var(--blue);
      background: #eef6fb;
      cursor: pointer;
    }

    .page { display: none; }
    .page.active { display: block; }

    .overview-shell {
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,250,252,0.98)),
        var(--panel);
      box-shadow: var(--shadow);
      padding: 20px;
      margin-bottom: 16px;
      overflow: hidden;
    }

    .overview-head {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      margin-bottom: 18px;
    }

    .overview-head h2 {
      font-size: 22px;
      margin-bottom: 6px;
    }

    .live-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 32px;
      border-radius: 999px;
      border: 1px solid #cfe0ea;
      background: #f6fbff;
      color: var(--blue);
      padding: 0 12px;
      font-size: 13px;
      font-weight: 900;
      white-space: nowrap;
    }

    .system-map {
      display: grid;
      grid-template-columns: minmax(260px, 1.12fr) 84px minmax(250px, 0.95fr) 84px minmax(260px, 1fr);
      align-items: stretch;
      gap: 12px;
      min-height: 360px;
    }

    .map-column {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.84);
      padding: 15px;
      position: relative;
      overflow: hidden;
    }

    .map-column::before {
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 4px;
      background: linear-gradient(90deg, var(--blue), var(--teal));
    }

    .map-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
    }

    .map-title strong {
      font-size: 15px;
    }

    .map-title span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }

    .device-stack {
      display: grid;
      gap: 10px;
    }

    .device-card {
      display: grid;
      grid-template-columns: 38px 1fr;
      gap: 10px;
      align-items: center;
      border: 1px solid #d8e5ed;
      border-radius: 8px;
      padding: 10px;
      background: #fbfdff;
    }

    .device-card.active {
      border-color: rgba(15, 118, 110, 0.75);
      background: #effbf8;
    }

    .device-icon {
      width: 38px;
      height: 38px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: white;
      background: linear-gradient(135deg, #244f83, #0f766e);
      font-weight: 900;
    }

    .mini-label {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
    }

    .privacy-lock {
      display: flex;
      align-items: center;
      gap: 8px;
      border-radius: 8px;
      padding: 10px;
      margin-top: 12px;
      color: #075c3d;
      background: #e8f7ef;
      font-weight: 900;
      font-size: 13px;
    }

    .connector {
      display: grid;
      place-items: center;
      color: var(--blue);
      position: relative;
    }

    .connector::before {
      content: "";
      position: absolute;
      left: 50%;
      top: 54px;
      bottom: 54px;
      width: 2px;
      background: #d5e2ec;
      transform: translateX(-50%);
    }

    .arrow-chip {
      position: relative;
      z-index: 1;
      display: grid;
      place-items: center;
      width: 64px;
      height: 64px;
      border-radius: 999px;
      border: 1px solid #c8dae8;
      background: #ffffff;
      box-shadow: 0 10px 26px rgba(33, 90, 150, 0.12);
      font-size: 24px;
      font-weight: 900;
    }

    .arrow-chip.active {
      color: white;
      border-color: var(--teal);
      background: var(--teal);
      animation: pulse 1s infinite;
    }

    .server-core {
      display: grid;
      gap: 12px;
    }

    .server-module {
      border: 1px solid #d8e5ed;
      border-radius: 8px;
      padding: 13px;
      background: #f9fcff;
    }

    .server-module strong {
      display: block;
      font-size: 14px;
      margin-bottom: 5px;
    }

    .formula {
      margin-top: 8px;
      border-radius: 8px;
      background: #102032;
      color: #d8f8ff;
      padding: 10px;
      font: 13px Consolas, "Courier New", monospace;
    }

    .result-strip {
      display: grid;
      gap: 10px;
    }

    .result-card {
      border: 1px solid #d8e5ed;
      border-radius: 8px;
      background: #fbfdff;
      padding: 12px;
    }

    .result-card .big {
      font-size: 28px;
      font-weight: 900;
      margin-top: 4px;
    }

    .privacy-claim {
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .claim {
      border-radius: 8px;
      padding: 11px;
      font-size: 13px;
      font-weight: 900;
    }

    .claim.good { color: #075c3d; background: #e2f6eb; }
    .claim.info { color: #214d84; background: #e4effa; }

    .stage-rail {
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
    }

    .panel,
    .metric,
    .client,
    .stage,
    .image-card {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .panel { padding: 18px; }
    .metric { padding: 16px; }

    .span-3 { grid-column: span 3; }
    .span-4 { grid-column: span 4; }
    .span-5 { grid-column: span 5; }
    .span-6 { grid-column: span 6; }
    .span-7 { grid-column: span 7; }
    .span-8 { grid-column: span 8; }
    .span-12 { grid-column: span 12; }

    .metric .value {
      margin-top: 7px;
      font-size: 31px;
      font-weight: 900;
    }

    .metric .note {
      color: var(--muted);
      margin-top: 5px;
      font-size: 13px;
    }

    h2 {
      margin: 0 0 14px;
      font-size: 20px;
    }

    h3 {
      margin: 0;
      font-size: 15px;
    }

    .pipeline {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
    }

    .stage {
      padding: 13px;
      min-height: 104px;
      position: relative;
      overflow: hidden;
    }

    .stage::after {
      content: "";
      position: absolute;
      inset: auto 13px 13px 13px;
      height: 5px;
      border-radius: 999px;
      background: #e5edf3;
    }

    .stage.done::after { background: var(--green); }
    .stage.active::after { background: var(--amber); animation: widthpulse 1.2s infinite; }

    @keyframes widthpulse { 50% { opacity: 0.4; } }

    .stage span,
    .client span,
    .event span {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }

    .clients-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(132px, 1fr));
      gap: 10px;
      max-height: 440px;
      overflow: auto;
      padding-right: 4px;
    }

    .client {
      padding: 12px;
      min-height: 104px;
    }

    .client .chip {
      margin-top: 10px;
      display: inline-flex;
      padding: 4px 8px;
      border-radius: 999px;
      background: #e8f4ef;
      color: var(--green);
      font-size: 12px;
      font-weight: 900;
    }

    .client.selected {
      border-color: rgba(15, 118, 110, 0.7);
      background: #f1fbf8;
    }

    .server-flow {
      display: grid;
      grid-template-columns: minmax(230px, 1fr) 64px minmax(230px, 1fr) 64px minmax(230px, 1fr);
      align-items: stretch;
      gap: 10px;
    }

    .node {
      min-height: 170px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
      padding: 15px;
    }

    .flow-hero {
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(135deg, rgba(33,90,150,0.06), rgba(15,118,110,0.06)),
        #ffffff;
      padding: 18px;
      margin-bottom: 16px;
      box-shadow: var(--shadow);
    }

    .payload-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .inspect-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
      padding: 14px;
    }

    .inspect-card.blocked {
      background: #f2fbf6;
      border-color: #bfe5cf;
    }

    .arrow {
      display: grid;
      place-items: center;
      color: var(--blue);
      font-weight: 900;
      font-size: 28px;
    }

    .payload {
      margin-top: 12px;
      padding: 12px;
      border-radius: 8px;
      color: #dff8ff;
      background: #101820;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      overflow: auto;
      min-height: 138px;
      white-space: pre-wrap;
    }

    .events {
      display: grid;
      gap: 9px;
      max-height: 410px;
      overflow: auto;
      padding-right: 4px;
    }

    .event {
      padding: 11px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }

    th, td {
      padding: 11px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }

    th {
      color: #263849;
      background: #eef4f7;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .tag {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      font-weight: 900;
      white-space: nowrap;
    }

    .tag.good { color: #075c3d; background: #dff5ea; }
    .tag.bad { color: #8d1b12; background: #ffe4df; }
    .tag.warn { color: #7a3d00; background: #fff0d1; }
    .tag.info { color: #214d84; background: #e4effa; }

    .images {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }

    .image-card {
      padding: 12px;
    }

    .image-card img {
      width: 100%;
      display: block;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: white;
    }

    .image-card span {
      display: block;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }

    .new-flow {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
    }

    .new-step {
      padding: 16px;
      min-height: 150px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      position: relative;
    }

    .new-step::before {
      content: attr(data-step);
      display: grid;
      place-items: center;
      width: 30px;
      height: 30px;
      border-radius: 999px;
      color: white;
      background: var(--blue);
      font-size: 12px;
      font-weight: 900;
      margin-bottom: 12px;
    }

    .new-step.active {
      border-color: var(--teal);
      background: #eefbf8;
      box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12);
    }

    .new-step.done {
      border-color: #bfe5cf;
      background: #f2fbf6;
    }

    .new-step.done::before {
      background: var(--green);
    }

    .terminal {
      max-height: 220px;
      overflow: auto;
      color: #d9f5ff;
      background: #0c1720;
      border-radius: 8px;
      padding: 14px;
      font: 12px Consolas, "Courier New", monospace;
      white-space: pre-wrap;
    }

    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      aside { position: static; height: auto; }
      main { padding: 18px; }
      .topbar, .runbar { flex-direction: column; align-items: stretch; }
      .span-3, .span-4, .span-5, .span-6, .span-7, .span-8, .span-12 { grid-column: span 12; }
      .pipeline, .new-flow, .server-flow, .images, .system-map, .privacy-claim { grid-template-columns: 1fr; }
      .arrow, .connector { display: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">
        <div class="mark">FL</div>
        <div>
          <strong>Privacy HAR</strong>
          <span>Federated learning dashboard</span>
        </div>
      </div>
      <nav>
        <button class="tab active" data-page="run">Project Run</button>
        <button class="tab" data-page="flow">Federated Flow</button>
        <button class="tab" data-page="results">Results</button>
        <button class="tab" data-page="newclient">New Client</button>
        <button class="tab" data-page="privacy">Privacy Evidence</button>
      </nav>
      <div class="sidebox">
        <div class="label">Run Status</div>
        <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">Idle</span></div>
        <p id="stageText" class="sub">Ready to run the dashboard demo.</p>
      </div>
    </aside>

    <main>
      <div class="topbar">
        <div>
          <h1>Privacy-Preserving HAR Pipeline</h1>
          <p class="sub">A live examiner-facing view of the complete project: client data stays local, model updates move to the server, FedAvg creates the global model, and a reserved new client joins after training.</p>
        </div>
        <div class="runbar">
          <select id="rounds">
            <option value="5">5 FL rounds</option>
            <option value="10">10 FL rounds</option>
            <option value="15">15 FL rounds</option>
          </select>
          <button id="runButton" class="primary">Run Pipeline</button>
        </div>
      </div>

      <section id="page-run" class="page active">
        <div class="overview-shell">
          <div class="overview-head">
            <div>
              <h2>System Flow</h2>
              <p class="sub">Press run and watch the pipeline move from local client training to server aggregation, testing, privacy evidence, and new-client onboarding.</p>
            </div>
            <div class="live-pill"><span id="liveDot" class="dot"></span><span id="liveStage">Ready</span></div>
          </div>
          <div class="system-map">
            <div class="map-column">
              <div class="map-title"><strong>Client Devices</strong><span id="mapClientCount">29 clients</span></div>
              <div class="device-stack" id="mapDevices"></div>
              <div class="privacy-lock">Lock Raw HAR windows and labels stay on devices</div>
            </div>
            <div class="connector"><div id="toServerArrow" class="arrow-chip">-></div></div>
            <div class="map-column">
              <div class="map-title"><strong>Federated Server</strong><span>visible payload only</span></div>
              <div class="server-core">
                <div class="server-module">
                  <strong>Payload inspection</strong>
                  <div class="mini-label">Allowed keys: client_id, weights, num_samples</div>
                </div>
                <div class="server-module">
                  <strong>Secure aggregation view</strong>
                  <div class="mini-label">Individual raw records are not available to the server.</div>
                  <div class="formula">Global = Σ(client weights x sample ratio)</div>
                </div>
                <div class="server-module">
                  <strong>Global model</strong>
                  <div class="mini-label" id="globalModelStatus">Waiting for first aggregation</div>
                </div>
              </div>
            </div>
            <div class="connector"><div id="toResultsArrow" class="arrow-chip">-></div></div>
            <div class="map-column">
              <div class="map-title"><strong>Outputs</strong><span>model + proof</span></div>
              <div class="result-strip">
                <div class="result-card"><span class="mini-label">Federated Accuracy</span><div id="mapFedAcc" class="big">--</div></div>
                <div class="result-card"><span class="mini-label">Centralized Accuracy</span><div id="mapCentAcc" class="big">--</div></div>
                <div class="result-card"><span class="mini-label">Privacy-risk fields in FL packets</span><div id="mapRiskCount" class="big">0</div></div>
              </div>
              <div class="privacy-claim">
                <div class="claim good">Raw data not sent</div>
                <div class="claim info">New client joins later</div>
              </div>
            </div>
          </div>
          <div class="stage-rail">
            <div id="pipeline" class="pipeline"></div>
          </div>
        </div>
        <div class="grid">
          <div class="metric span-3"><div class="label">Training Clients</div><div id="mClients" class="value">29</div><div class="note">Reserved 1 subject as new client</div></div>
          <div class="metric span-3"><div class="label">New Client</div><div id="mNewClient" class="value">30</div><div class="note">Added after main model training</div></div>
          <div class="metric span-3"><div class="label">FL Packets</div><div id="mPackets" class="value">0</div><div class="note">Model-update payloads logged</div></div>
          <div class="metric span-3"><div class="label">Privacy Risks</div><div id="mRisks" class="value">0</div><div class="note">Raw-data fields found in FL packets</div></div>
          <div class="panel span-8">
            <h2>Live Event Stream</h2>
            <div id="events" class="events"></div>
          </div>
          <div class="panel span-4">
            <h2>Backend Log</h2>
            <div id="terminal" class="terminal"></div>
          </div>
        </div>
      </section>

      <section id="page-flow" class="page">
        <div class="flow-hero">
          <div class="server-flow">
            <div class="node"><h3>Client Device</h3><p class="sub">Each subject is represented as a separate client. Local training uses private sensor windows and labels inside the client boundary.</p><div class="tag good">Raw data local</div></div>
            <div class="arrow">-></div>
            <div class="node"><h3>Model Update Payload</h3><p class="sub">The server receives only parameters and sample count. The dashboard inspects this payload live.</p><div class="tag info">client_id, weights, num_samples</div></div>
            <div class="arrow">-></div>
            <div class="node"><h3>Global Model</h3><p class="sub">FedAvg merges updates and redistributes the improved HAR model to participating clients.</p><div class="tag good">Distributed back</div></div>
          </div>
        </div>
        <div class="grid">
          <div class="panel span-8">
            <h2>Federated Clients</h2>
            <div id="clientsGrid" class="clients-grid"></div>
          </div>
          <div class="panel span-4">
            <h2>Server View</h2>
            <div class="payload" id="serverPayload">{ "status": "Run the pipeline to inspect payloads" }</div>
          </div>
          <div class="panel span-12">
            <h2>Server Privacy Boundary</h2>
            <div class="payload-grid">
              <div class="inspect-card blocked">
                <h3>Stays on Client</h3>
                <p class="sub">Raw sensor windows, activity labels, subject records, and local validation data.</p>
              </div>
              <div class="inspect-card">
                <h3>Visible to Server</h3>
                <p class="sub">Client id, round number, model weights, update size, sample count, aggregate metrics.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="page-results" class="page">
        <div class="grid">
          <div class="panel span-7">
            <h2>Centralized vs Federated Metrics</h2>
            <table id="comparisonTable"></table>
          </div>
          <div class="panel span-5">
            <h2>Final Summary</h2>
            <div id="summaryBox"></div>
          </div>
          <div class="panel span-12">
            <h2>Generated Output Graphs</h2>
            <div class="images" id="resultImages"></div>
          </div>
        </div>
      </section>

      <section id="page-newclient" class="page">
        <div class="grid">
          <div class="panel span-12">
            <div class="topbar" style="margin:0 0 14px">
              <div>
                <h2>New Client Onboarding</h2>
                <p class="sub">The reserved client receives the latest global model, trains locally, sends one model update, and the server merges it back into the global model.</p>
              </div>
              <button id="replayButton" class="secondary">Replay New Client Flow</button>
            </div>
            <div class="new-flow" id="newFlow"></div>
          </div>
          <div class="panel span-5">
            <h2>New Client Result</h2>
            <div id="newClientBox"></div>
          </div>
          <div class="panel span-7">
            <h2>Accuracy Before and After</h2>
            <div class="image-card"><img id="newClientChart" alt="New client accuracy chart"><span>Generated after the new client update is merged.</span></div>
          </div>
        </div>
      </section>

      <section id="page-privacy" class="page">
        <div class="grid">
          <div class="panel span-6">
            <h2>Payload Privacy Check</h2>
            <table id="privacyTable"></table>
          </div>
          <div class="panel span-6">
            <h2>Server Visibility</h2>
            <table>
              <thead><tr><th>Data item</th><th>Server can view?</th><th>Reason</th></tr></thead>
              <tbody>
                <tr><td>Client ID</td><td><span class="tag info">Yes</span></td><td>Needed for round tracking.</td></tr>
                <tr><td>Model weights</td><td><span class="tag info">Yes</span></td><td>Needed for FedAvg aggregation.</td></tr>
                <tr><td>Raw sensor windows</td><td><span class="tag good">No</span></td><td>Remain inside the client dataset.</td></tr>
                <tr><td>Activity labels</td><td><span class="tag good">No</span></td><td>Used only during local training.</td></tr>
              </tbody>
            </table>
          </div>
          <div class="panel span-12">
            <h2>Privacy Visuals</h2>
            <div class="images" id="privacyImages"></div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const stages = [
      ["privacy_demo", "Privacy payload check"],
      ["load_dataset", "Load dataset"],
      ["centralized_training", "Centralized training"],
      ["federated_training", "Federated rounds"],
      ["new_client_onboarding", "New client update"],
      ["complete", "Final outputs"]
    ];
    const newSteps = [
      ["connect", "Client connects", "Reserved subject is introduced after the main run."],
      ["download", "Receive global model", "The latest global weights are copied to the client."],
      ["train", "Train locally", "Raw HAR windows and labels stay on the new client."],
      ["send", "Send update", "Only client_id, weights, and num_samples leave the device."],
      ["merge", "Merge global model", "Server validates and aggregates the update."]
    ];
    let latestState = {};
    let replayIndex = -1;

    document.querySelectorAll(".tab").forEach(button => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(tab => tab.classList.remove("active"));
        document.querySelectorAll(".page").forEach(page => page.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(`page-${button.dataset.page}`).classList.add("active");
      });
    });

    document.getElementById("runButton").addEventListener("click", async () => {
      const rounds = Number(document.getElementById("rounds").value);
      await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rounds })
      });
      poll();
    });

    document.getElementById("replayButton").addEventListener("click", () => {
      replayIndex = 0;
      renderNewFlow();
      const timer = setInterval(() => {
        replayIndex += 1;
        renderNewFlow();
        if (replayIndex >= newSteps.length - 1) clearInterval(timer);
      }, 900);
    });

    async function poll() {
      const response = await fetch("/api/state");
      latestState = await response.json();
      render(latestState);
    }

    function render(data) {
      const state = data.state || {};
      const metadata = data.metadata || {};
      const summary = data.summary || {};
      const packet = data.packet_summary || {};

      setText("statusText", titleCase(state.status || "idle"));
      setText("stageText", `Current stage: ${state.current_stage || "ready"}`);
      const dot = document.getElementById("statusDot");
      dot.className = `dot ${state.status || ""}`;

      setText("mClients", metadata.training_client_count || summary.training_clients || 29);
      setText("mNewClient", metadata.new_client_id || summary.holdout_client || 30);
      setText("mPackets", packet.packet_count || 0);
      setText("mRisks", packet.privacy_risks || 0);

      renderSystemMap(data);
      renderPipeline(state.current_stage || "ready", state.status || "idle");
      renderEvents(data.events || []);
      renderTerminal(data.logs || []);
      renderClients(metadata, data.events || []);
      renderServerPayload(data.events || []);
      renderComparison(data.comparison_rows || [], summary);
      renderImages(data.images || {});
      renderPrivacy(data.privacy_rows || [], data.images || {});
      renderNewClient(data.new_client || summary.new_client || {}, data.images || {});
    }

    function renderPipeline(currentStage, status) {
      const currentIndex = stages.findIndex(([key]) => key === currentStage);
      document.getElementById("pipeline").innerHTML = stages.map(([key, label], index) => {
        let cls = "stage";
        if (status === "completed" || index < currentIndex) cls += " done";
        if (index === currentIndex && status === "running") cls += " active";
        return `<div class="${cls}"><h3>${label}</h3><span>${stageCaption(key)}</span></div>`;
      }).join("");
    }

    function renderSystemMap(data) {
      const state = data.state || {};
      const metadata = data.metadata || {};
      const summary = data.summary || {};
      const packet = data.packet_summary || {};
      const events = data.events || [];
      const status = state.status || "idle";
      const stage = state.current_stage || "ready";
      const activeClientEvent = [...events].reverse().find(event => event.type === "round_started");
      const selected = new Set((activeClientEvent?.selected_clients || []).map(String));
      const counts = metadata.client_sample_counts || {};
      const ids = Object.keys(counts).sort((a, b) => Number(a) - Number(b));
      const shown = ids.slice(0, 5);
      const fallbackIds = ["1", "3", "5", "6", "7"];
      const deviceIds = shown.length ? shown : fallbackIds;

      setText("liveStage", titleCase(stage));
      setText("mapClientCount", `${metadata.training_client_count || summary.training_clients || 29} clients`);
      setText("mapRiskCount", packet.privacy_risks || 0);
      setText("mapFedAcc", summary.federated?.accuracy !== undefined ? pct(summary.federated.accuracy) : "--");
      setText("mapCentAcc", summary.centralized?.accuracy !== undefined ? pct(summary.centralized.accuracy) : "--");
      setText("globalModelStatus", stage === "complete" || summary.federated ? "Global model ready and distributed" : "Updated after each FedAvg round");

      const liveDot = document.getElementById("liveDot");
      liveDot.className = `dot ${status}`;

      document.getElementById("toServerArrow").className = `arrow-chip ${["federated_training", "centralized_training"].includes(stage) && status === "running" ? "active" : ""}`;
      document.getElementById("toResultsArrow").className = `arrow-chip ${["new_client_onboarding", "complete"].includes(stage) || summary.federated ? "active" : ""}`;

      document.getElementById("mapDevices").innerHTML = deviceIds.map((id, index) => {
        const active = selected.has(String(id)) || (!shown.length && index < 2 && stage === "federated_training");
        const samples = counts[id] ? `${counts[id]} samples` : "local HAR data";
        return `
          <div class="device-card ${active ? "active" : ""}">
            <div class="device-icon">${index + 1}</div>
            <div>
              <h3>Client ${id}</h3>
              <div class="mini-label">${samples} · ${active ? "training this round" : "data locked locally"}</div>
            </div>
          </div>
        `;
      }).join("") + `<div class="device-card"><div class="device-icon">+</div><div><h3>${Math.max(0, (metadata.training_client_count || 29) - deviceIds.length)} more clients</h3><div class="mini-label">Shown in the Federated Flow tab</div></div></div>`;
    }

    function renderEvents(events) {
      const visible = events.slice(-40).reverse();
      document.getElementById("events").innerHTML = visible.map(event => `
        <div class="event">
          <h3>${escapeHtml(event.title || event.type)}</h3>
          <span>${escapeHtml(event.timestamp || "")} · ${escapeHtml(event.type || "")}${event.round ? ` · round ${event.round}` : ""}</span>
        </div>
      `).join("") || `<div class="event"><h3>No events yet</h3><span>Click Run Pipeline to start.</span></div>`;
    }

    function renderTerminal(logs) {
      document.getElementById("terminal").textContent = logs.slice(-90).join("\n") || "Waiting for pipeline output...";
    }

    function renderClients(metadata, events) {
      const counts = metadata.client_sample_counts || {};
      const selected = new Set();
      [...events].reverse().find(event => {
        if (event.type === "round_started") {
          (event.selected_clients || []).forEach(id => selected.add(String(id)));
          return true;
        }
        return false;
      });
      const ids = Object.keys(counts).sort((a, b) => Number(a) - Number(b));
      document.getElementById("clientsGrid").innerHTML = ids.map(id => `
        <div class="client ${selected.has(id) ? "selected" : ""}">
          <h3>Client ${id}</h3>
          <span>${counts[id]} local samples</span>
          <div class="chip">${selected.has(id) ? "selected this round" : "local data locked"}</div>
        </div>
      `).join("") || `<div class="client"><h3>Clients waiting</h3><span>Run the pipeline to create the 29-client view.</span></div>`;
    }

    function renderServerPayload(events) {
      const update = [...events].reverse().find(event => event.type === "client_update" || event.type === "new_client_update");
      const payload = update ? {
        client_id: update.client_id,
        payload_keys: update.payload_keys,
        num_samples: update.num_samples,
        raw_data_sent: update.raw_data_sent
      } : { status: "No model-update payload received yet" };
      document.getElementById("serverPayload").textContent = JSON.stringify(payload, null, 2);
    }

    function renderComparison(rows, summary) {
      const table = document.getElementById("comparisonTable");
      if (!rows.length && summary.centralized) {
        rows = [
          { Model: "Centralized", Accuracy: summary.centralized.accuracy, "F1 Score": summary.centralized.f1, Loss: summary.centralized.loss, "Training Time (sec)": summary.centralized.training_time_sec, Privacy: "Low" },
          { Model: "Federated", Accuracy: summary.federated.accuracy, "F1 Score": summary.federated.f1, Loss: summary.federated.loss, "Training Time (sec)": summary.federated.training_time_sec, Privacy: "High" }
        ];
      }
      table.innerHTML = rows.length ? `
        <thead><tr><th>Model</th><th>Accuracy</th><th>F1</th><th>Loss</th><th>Time</th><th>Privacy</th></tr></thead>
        <tbody>${rows.map(row => `
          <tr>
            <td>${row.Model}</td>
            <td>${row.Accuracy}</td>
            <td>${row["F1 Score"]}</td>
            <td>${row.Loss}</td>
            <td>${row["Training Time (sec)"]} sec</td>
            <td><span class="tag ${row.Privacy === "High" ? "good" : "bad"}">${row.Privacy}</span></td>
          </tr>
        `).join("")}</tbody>` : `<tbody><tr><td>Run the pipeline to populate metrics.</td></tr></tbody>`;

      const box = document.getElementById("summaryBox");
      if (!summary.centralized) {
        box.innerHTML = `<p class="sub">Final training summary will appear here after the run completes.</p>`;
        return;
      }
      const gap = Math.abs((summary.centralized.accuracy || 0) - (summary.federated.accuracy || 0));
      box.innerHTML = `
        <div class="metric" style="box-shadow:none;margin-bottom:10px"><div class="label">Accuracy gap</div><div class="value">${pct(gap)}</div><div class="note">Centralized vs federated on reserved new-client test data.</div></div>
        <div class="metric" style="box-shadow:none"><div class="label">Elapsed</div><div class="value">${summary.elapsed_sec || 0}s</div><div class="note">Includes privacy demo, training, and new-client onboarding.</div></div>
      `;
    }

    function renderImages(images) {
      const resultImages = [
        ["comparison_scores", "Centralized vs federated score comparison"],
        ["federated_accuracy", "Federated accuracy by round"],
        ["comparison_privacy", "Privacy score comparison"],
        ["federated_confusion", "Federated confusion matrix"]
      ];
      document.getElementById("resultImages").innerHTML = resultImages.map(([key, label]) => imageCard(images[key], label)).join("");
    }

    function renderPrivacy(rows, images) {
      document.getElementById("privacyTable").innerHTML = rows.length ? `
        <thead><tr><th>Scenario</th><th>Payload keys</th><th>Risk</th><th>Meaning</th></tr></thead>
        <tbody>${rows.map(row => `
          <tr>
            <td>${row.scenario}</td>
            <td><code>${(row.payload_keys || []).join(", ")}</code></td>
            <td><span class="tag ${row.privacy_risk ? "bad" : "good"}">${row.privacy_risk}</span></td>
            <td>${row.server_interpretation}</td>
          </tr>
        `).join("")}</tbody>` : `<tbody><tr><td>Run the pipeline to inspect centralized and federated payloads.</td></tr></tbody>`;

      document.getElementById("privacyImages").innerHTML = [
        imageCard(images.privacy_payload, "Raw-data upload vs model-update payload"),
        imageCard(images.privacy_raw_vs_weights, "Raw sensor values vs CNN weights")
      ].join("");
    }

    function renderNewClient(result, images) {
      if (result.client_id && replayIndex < 0) replayIndex = newSteps.length;
      renderNewFlow();
      const box = document.getElementById("newClientBox");
      if (!result.client_id) {
        box.innerHTML = `<p class="sub">Run the pipeline to execute the reserved new-client update.</p>`;
      } else {
        box.innerHTML = `
          <table>
            <tbody>
              <tr><th>Client</th><td>${result.client_id}</td></tr>
              <tr><th>Train samples</th><td>${result.train_samples}</td></tr>
              <tr><th>Validation samples</th><td>${result.validation_samples}</td></tr>
              <tr><th>Payload keys</th><td><code>${(result.payload_keys || []).join(", ")}</code></td></tr>
              <tr><th>Raw data sent</th><td><span class="tag good">${result.raw_data_sent}</span></td></tr>
              <tr><th>Before accuracy</th><td>${pct(result.before?.accuracy || 0)}</td></tr>
              <tr><th>After accuracy</th><td>${pct(result.after?.accuracy || 0)}</td></tr>
            </tbody>
          </table>
        `;
      }
      document.getElementById("newClientChart").src = images.new_client_accuracy || "";
    }

    function renderNewFlow() {
      document.getElementById("newFlow").innerHTML = newSteps.map(([key, title, copy], index) => `
        <div data-step="${index + 1}" class="new-step ${index === replayIndex ? "active" : ""} ${replayIndex > index ? "done" : ""}">
          <h3>${title}</h3>
          <p class="sub">${copy}</p>
        </div>
      `).join("");
    }

    function imageCard(src, label) {
      return `<div class="image-card"><img src="${src || ""}" alt="${escapeHtml(label)}" onerror="this.style.display='none'"><span>${label}</span></div>`;
    }

    function stageCaption(key) {
      return {
        privacy_demo: "Compare centralized raw payload vs federated model update.",
        load_dataset: "Read all 30 UCI HAR subjects and reserve client 30.",
        centralized_training: "Train on combined data from 29 clients.",
        federated_training: "Train clients locally and aggregate model updates.",
        new_client_onboarding: "Add reserved client after the global model exists.",
        complete: "Write metrics, packet logs, charts, and dashboard artifacts."
      }[key] || "";
    }

    function pct(value) {
      return `${(Number(value || 0) * 100).toFixed(2)}%`;
    }

    function setText(id, value) {
      document.getElementById(id).textContent = value;
    }

    function titleCase(value) {
      return String(value).replace("_", " ").replace(/\b\w/g, c => c.toUpperCase());
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[char]));
    }

    renderNewFlow();
    poll();
    setInterval(poll, 1800);
  </script>
</body>
</html>"""


def main():
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Dashboard running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
