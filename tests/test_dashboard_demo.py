import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.dashboard_demo import build_dashboard_demo_split
from src.dashboard_events import DashboardEventLogger


class DashboardDemoSplitTests(unittest.TestCase):
    def test_reserves_one_subject_and_keeps_remaining_clients(self):
        train_df = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4],
                "Activity": [1, 1, 2, 2],
                "Subject": [1, 2, 3, 4],
            }
        )
        test_df = pd.DataFrame(
            {
                "f1": [5, 6],
                "Activity": [1, 2],
                "Subject": [5, 6],
            }
        )

        demo = build_dashboard_demo_split(train_df, test_df, holdout_client=6)

        self.assertEqual(demo.new_client_id, 6)
        self.assertEqual(sorted(demo.training_clients.keys()), [1, 2, 3, 4, 5])
        self.assertEqual(demo.total_subjects, 6)
        self.assertEqual(demo.training_client_count, 5)
        self.assertNotIn(6, demo.client_sample_counts)

    def test_defaults_to_highest_subject_for_new_client(self):
        train_df = pd.DataFrame(
            {
                "f1": [1, 2],
                "Activity": [1, 2],
                "Subject": [10, 30],
            }
        )
        test_df = pd.DataFrame(
            {
                "f1": [3],
                "Activity": [1],
                "Subject": [20],
            }
        )

        demo = build_dashboard_demo_split(train_df, test_df)

        self.assertEqual(demo.new_client_id, 30)
        self.assertEqual(sorted(demo.training_clients.keys()), [10, 20])


class DashboardEventLoggerTests(unittest.TestCase):
    def test_logger_writes_jsonl_and_state_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DashboardEventLogger(tmpdir)

            logger.reset()
            logger.log("stage", "Started", step="load", value=1)
            logger.write_state(status="running", current_stage="load")

            events_path = Path(tmpdir) / "events.jsonl"
            state_path = Path(tmpdir) / "state.json"

            event = json.loads(events_path.read_text().strip())
            state = json.loads(state_path.read_text())

            self.assertEqual(event["type"], "stage")
            self.assertEqual(event["title"], "Started")
            self.assertEqual(event["step"], "load")
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["event_count"], 1)


if __name__ == "__main__":
    unittest.main()
