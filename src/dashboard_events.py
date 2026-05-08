import json
import os
import time
from pathlib import Path


class DashboardEventLogger:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.events_path = self.output_dir / "events.jsonl"
        self.state_path = self.output_dir / "state.json"
        self.event_count = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def reset(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("")
        self.event_count = 0
        self.write_state(status="idle", current_stage="ready")

    def log(self, event_type, title, **fields):
        self.event_count += 1
        event = {
            "index": self.event_count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": event_type,
            "title": title,
        }
        event.update(fields)

        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=_json_default) + "\n")

        return event

    def write_state(self, **fields):
        current_events = self._count_events()
        if current_events > self.event_count:
            self.event_count = current_events

        state = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event_count": self.event_count,
        }
        state.update(fields)

        tmp_path = self.state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, indent=2, default=_json_default))
        os.replace(tmp_path, self.state_path)
        return state

    def write_json(self, filename, payload):
        path = self.output_dir / filename
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=_json_default))
        os.replace(tmp_path, path)
        return path

    def _count_events(self):
        if not self.events_path.exists():
            return 0
        text = self.events_path.read_text(encoding="utf-8").strip()
        if not text:
            return 0
        return len(text.splitlines())


def read_dashboard_events(output_dir):
    events_path = Path(output_dir) / "events.jsonl"
    if not events_path.exists():
        return []

    events = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)
