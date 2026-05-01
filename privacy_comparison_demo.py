import argparse
import json
import os
import pickle
import socket
import struct
import threading
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data_loader import UCIHARDataLoader
from src.model import HAR_CNN


HOST = "127.0.0.1"
CENTRALIZED_PORT = 5001
FEDERATED_PORT = 5000
BUFFER = 4096

RISK_KEYWORDS = [
    "raw",
    "sensor",
    "signals",
    "accelerometer",
    "gyro",
    "x_train",
    "y_train",
    "labels",
    "subject_ids",
    "activity",
    "dataset",
]


def recvall(conn, n):
    data = b""
    while len(data) < n:
        packet = conn.recv(min(BUFFER, n - len(data)))
        if not packet:
            return None
        data += packet
    return data


def encode_payload(payload, encoding):
    if encoding == "json":
        return json.dumps(payload, indent=2).encode("utf-8")
    if encoding == "pickle":
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    raise ValueError(f"Unsupported encoding: {encoding}")


def decode_payload(raw, encoding):
    if encoding == "json":
        return json.loads(raw.decode("utf-8"))
    if encoding == "pickle":
        return pickle.loads(raw)
    raise ValueError(f"Unsupported encoding: {encoding}")


def send_payload(payload, port, encoding):
    raw = encode_payload(payload, encoding)
    header = struct.pack(">Q", len(raw))

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, port))
    client.sendall(header + raw)
    client.close()
    return len(raw)


def receive_payload(conn, encoding):
    header = recvall(conn, 8)
    if not header:
        return None

    msg_len = struct.unpack(">Q", header)[0]
    body = recvall(conn, msg_len)
    if not body:
        return None

    return decode_payload(body, encoding), msg_len


def receive_once(port, result_container, encoding):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, port))
    server.listen(1)

    conn, _ = server.accept()
    received = receive_payload(conn, encoding)
    conn.close()
    server.close()

    result_container["received"] = received


def detect_privacy_risk(payload):
    keys = list(payload.keys())
    risky_keys = []
    for key in keys:
        lower_key = key.lower()
        for word in RISK_KEYWORDS:
            if word in lower_key:
                risky_keys.append(key)
                break
    return len(risky_keys) > 0, risky_keys


def summarize_payload(name, port, payload, bytes_sent, encoding):
    privacy_risk, risky_keys = detect_privacy_risk(payload)
    summary = {
        "scenario": name,
        "port": port,
        "encoding": encoding,
        "payload_keys": list(payload.keys()),
        "bytes_sent": bytes_sent,
        "payload_size_kb": round(bytes_sent / 1024, 2),
        "privacy_risk": privacy_risk,
        "risky_keys": risky_keys,
        "server_interpretation": (
            "Server receives raw wearable data"
            if privacy_risk
            else "Server receives model update only"
        ),
    }
    return summary


def run_socket_round(name, port, payload, encoding):
    result_container = {}
    thread = threading.Thread(target=receive_once, args=(port, result_container, encoding))
    thread.start()
    time.sleep(0.2)

    bytes_sent = send_payload(payload, port, encoding)
    thread.join(timeout=10)

    if "received" not in result_container:
        raise RuntimeError(f"No payload received for scenario: {name}")

    received_payload, received_bytes = result_container["received"]
    if received_bytes != bytes_sent:
        raise RuntimeError(
            f"Byte mismatch for {name}: sent {bytes_sent}, received {received_bytes}"
        )

    return summarize_payload(name, port, received_payload, bytes_sent, encoding)


def build_payloads(dataset_dir, raw_rows):
    loader = UCIHARDataLoader(dataset_dir)
    train_df, _ = loader.load_full_dataset(use_inertial_signals=True)

    sample_df = train_df.head(raw_rows)
    raw_columns = [col for col in sample_df.columns if col not in ["Activity", "Subject"]]

    centralized_payload = {
        "client_id": int(sample_df["Subject"].iloc[0]),
        "raw_sensor_data": sample_df[raw_columns].values.tolist(),
        "activity_labels": sample_df["Activity"].astype(int).tolist(),
        "subject_ids": sample_df["Subject"].astype(int).tolist(),
        "description": "Centralized upload simulation: raw HAR windows and labels sent to server.",
    }

    model = HAR_CNN()
    federated_payload = {
        "client_id": int(sample_df["Subject"].iloc[0]),
        "weights": model.state_dict(),
        "num_samples": len(sample_df),
        "description": "Federated upload simulation: only model weights and sample count sent.",
    }

    return centralized_payload, federated_payload, sample_df[raw_columns].iloc[0], model


def save_payload_preview(payloads, output_dir):
    preview = {
        name: {
            "keys": list(payload.keys()),
            "value_types": {key: type(value).__name__ for key, value in payload.items()},
        }
        for name, payload in payloads.items()
    }
    with open(os.path.join(output_dir, "privacy_payload_preview.json"), "w") as f:
        json.dump(preview, f, indent=2)


def plot_payload_comparison(rows, output_dir):
    labels = [row["scenario"] for row in rows]
    sizes = [row["payload_size_kb"] for row in rows]
    colors = ["#b42318" if row["privacy_risk"] else "#15845b" for row in rows]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, sizes, color=colors)
    plt.ylabel("Payload Size (KB)")
    plt.title("Centralized Raw-Data Upload vs Federated Model-Update Upload")
    plt.grid(axis="y", alpha=0.25)

    for bar, row in zip(bars, rows):
        label = "Privacy Risk" if row["privacy_risk"] else "Raw Data Not Sent"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row['payload_size_kb']} KB\n{label}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "privacy_payload_comparison.png"), dpi=300)
    plt.close()


def plot_raw_vs_weights(raw_sample, model, output_dir):
    raw_values = raw_sample.values[:128]
    first_weight_tensor = next(iter(model.state_dict().values())).detach().cpu().flatten()
    weight_values = first_weight_tensor[:128].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(raw_values, color="#b42318", linewidth=1.7)
    axes[0].set_title("Raw HAR Sensor Values\n(Centralized risk)")
    axes[0].set_xlabel("Time/value index")
    axes[0].set_ylabel("Sensor value")
    axes[0].grid(alpha=0.25)

    axes[1].plot(weight_values, color="#15845b", linewidth=1.7)
    axes[1].set_title("CNN Weights Sent in FL\n(No raw signal directly visible)")
    axes[1].set_xlabel("Weight index")
    axes[1].set_ylabel("Weight value")
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "privacy_raw_vs_weights.png"), dpi=300)
    plt.close()


def print_summary(rows):
    print("\n========== PRIVACY COMPARISON DEMO ==========")
    for row in rows:
        print(f"\nScenario        : {row['scenario']}")
        print(f"TCP Port        : {row['port']}")
        print(f"Payload Keys    : {row['payload_keys']}")
        print(f"Payload Size    : {row['payload_size_kb']} KB")
        print(f"Privacy Risk    : {row['privacy_risk']}")
        print(f"Risky Keys      : {row['risky_keys']}")
        print(f"Interpretation  : {row['server_interpretation']}")

    print("\nWireshark filters:")
    print("  Centralized raw-data demo : tcp.port == 5001")
    print("  Federated model-update demo: tcp.port == 5000")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare centralized raw-data transmission with federated model-update transmission."
    )
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--raw-rows", type=int, default=5)
    return parser.parse_args()


def run_demo(dataset_dir="data", output_dir="results", raw_rows=5):
    os.makedirs(output_dir, exist_ok=True)

    centralized_payload, federated_payload, raw_sample, model = build_payloads(
        dataset_dir,
        raw_rows,
    )

    rows = [
        run_socket_round("Centralized Raw Data", CENTRALIZED_PORT, centralized_payload, "json"),
        run_socket_round("Federated Model Update", FEDERATED_PORT, federated_payload, "pickle"),
    ]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "privacy_comparison_demo.csv"), index=False)
    with open(os.path.join(output_dir, "privacy_comparison_demo.json"), "w") as f:
        json.dump(rows, f, indent=2)

    save_payload_preview(
        {
            "centralized_raw_data": centralized_payload,
            "federated_model_update": federated_payload,
        },
        output_dir,
    )
    plot_payload_comparison(rows, output_dir)
    plot_raw_vs_weights(raw_sample, model, output_dir)
    print_summary(rows)

    print(f"\nSaved CSV: {os.path.join(output_dir, 'privacy_comparison_demo.csv')}")
    print(f"Saved chart: {os.path.join(output_dir, 'privacy_payload_comparison.png')}")
    print(f"Saved chart: {os.path.join(output_dir, 'privacy_raw_vs_weights.png')}")


def main():
    args = parse_args()
    run_demo(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        raw_rows=args.raw_rows,
    )


if __name__ == "__main__":
    main()
