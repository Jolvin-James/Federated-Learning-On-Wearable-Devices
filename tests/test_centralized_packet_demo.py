import json
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import (
    build_centralized_client_payload,
    encode_json_payload,
    reconstruct_centralized_training_data,
)


class CentralizedPacketDemoTests(unittest.TestCase):
    def test_build_centralized_client_payload_exposes_raw_sensor_fields(self):
        client_splits = {
            1: {
                "X_train": pd.DataFrame(
                    {
                        "signal_0_0": [0.1, 0.2],
                        "signal_0_1": [0.3, 0.4],
                    }
                ),
                "y_train": pd.Series([1, 2]),
                "X_test": pd.DataFrame(),
                "y_test": pd.Series(dtype=int),
            }
        }

        payload = build_centralized_client_payload(1, client_splits[1])

        self.assertEqual(payload["client_id"], 1)
        self.assertEqual(payload["feature_columns"], ["signal_0_0", "signal_0_1"])
        self.assertEqual(payload["raw_sensor_data"], [[0.1, 0.3], [0.2, 0.4]])
        self.assertEqual(payload["activity_labels"], [1, 2])
        self.assertIn("raw HAR", payload["description"])

    def test_reconstruct_centralized_training_data_uses_received_payloads(self):
        payloads = [
            {
                "client_id": 2,
                "feature_columns": ["signal_0_0", "signal_0_1"],
                "raw_sensor_data": [[0.5, 0.6]],
                "activity_labels": [3],
                "num_samples": 1,
            },
            {
                "client_id": 1,
                "feature_columns": ["signal_0_0", "signal_0_1"],
                "raw_sensor_data": [[0.1, 0.2], [0.3, 0.4]],
                "activity_labels": [1, 2],
                "num_samples": 2,
            },
        ]

        X_train, y_train = reconstruct_centralized_training_data(
            payloads,
            shuffle=False,
        )

        self.assertEqual(list(X_train.columns), ["signal_0_0", "signal_0_1"])
        self.assertEqual(X_train.values.tolist(), [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        self.assertEqual(y_train.tolist(), [1, 2, 3])

    def test_encode_json_payload_is_human_readable_for_wireshark(self):
        raw = encode_json_payload({"client_id": 1, "raw_sensor_data": [[0.1]]})

        decoded = json.loads(raw.decode("utf-8"))

        self.assertEqual(decoded["client_id"], 1)
        self.assertEqual(decoded["raw_sensor_data"], [[0.1]])


if __name__ == "__main__":
    unittest.main()
