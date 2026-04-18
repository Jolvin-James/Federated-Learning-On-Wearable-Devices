# src/packet_analysis.py

"""
OPTIONAL PACKET ANALYSIS

This module logs federated communication traffic and payload sizes.

Goal:
1. Measure bytes exchanged between clients and server
2. Detect suspicious privacy-risk fields
3. Save communication logs
4. Generate packet analysis report

Works with your current FL project structure.

Usage:
    from src.packet_analysis import PacketAnalyzer
"""

import os
import json
import time
import pickle
import hashlib
import pandas as pd


class PacketAnalyzer:
    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        self.logs = []

        os.makedirs(self.save_dir, exist_ok=True)

    # Estimate Payload Size
    def get_payload_size(self, payload):
        """
        Convert payload into bytes
        """
        try:
            serialized = pickle.dumps(payload)
            return len(serialized)
        except Exception:
            return 0

    # Detect Raw Data Leakage
    def detect_privacy_risk(self, payload):

        blocked_words = [
            "raw_data",
            "sensor_data",
            "signals",
            "accelerometer",
            "gyro",
            "x_train",
            "x_test",
            "y_train",
            "y_test",
            "labels",
            "dataset",
            "activity"
        ]

        keys = payload.keys()

        for key in keys:
            lower_key = key.lower()

            for word in blocked_words:
                if word in lower_key:
                    return True, key

        return False, None

    # Hash Payload (for proof only)
    def payload_hash(self, payload):

        try:
            raw = pickle.dumps(payload)
            return hashlib.md5(raw).hexdigest()
        except Exception:
            return "NA"

    # Log Client -> Server Packet
    def log_packet(self, round_num, client_id, payload):

        size_bytes = self.get_payload_size(payload)

        risk, risky_key = self.detect_privacy_risk(payload)

        packet_hash = self.payload_hash(payload)

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": round_num,
            "client_id": client_id,
            "payload_keys": list(payload.keys()),
            "payload_size_bytes": size_bytes,
            "payload_size_kb": round(size_bytes / 1024, 2),
            "privacy_risk": risk,
            "risky_key": risky_key,
            "payload_hash": packet_hash
        }

        self.logs.append(row)

        print("\n[PACKET ANALYSIS]")
        print("Round          :", round_num)
        print("Client         :", client_id)
        print("Size (bytes)   :", size_bytes)
        print("Size (KB)      :", round(size_bytes / 1024, 2))
        print("Keys           :", list(payload.keys()))
        print("Privacy Risk   :", risk)

        if risk:
            print("Blocked Key    :", risky_key)

    # Save CSV Report
    def save_csv(self):

        if len(self.logs) == 0:
            return

        df = pd.DataFrame(self.logs)

        path = os.path.join(
            self.save_dir,
            "packet_analysis.csv"
        )

        df.to_csv(path, index=False)

        print("\nSaved:", path)

    # Summary
    def summary(self):

        if len(self.logs) == 0:
            print("No packets logged.")
            return

        df = pd.DataFrame(self.logs)

        print("\n========== PACKET SUMMARY ==========")
        print("Total Packets       :", len(df))
        print("Total Bytes         :", df["payload_size_bytes"].sum())
        print("Average Size (KB)   :", round(df["payload_size_kb"].mean(), 2))
        print("Max Size (KB)       :", round(df["payload_size_kb"].max(), 2))
        print("Privacy Risks Found :", df["privacy_risk"].sum())

    # Save JSON Report
    def save_json(self):

        path = os.path.join(
            self.save_dir,
            "packet_analysis.json"
        )

        with open(path, "w") as f:
            json.dump(self.logs, f, indent=4)

        print("Saved:", path)