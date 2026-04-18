# main.py
# Wireshark-ready Federated Learning Demo
# Uses localhost socket communication so traffic can be captured in Wireshark
# Filter: tcp.port == 5000

import argparse
import copy
import os
import pickle
import random
import socket
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import UCIHARDataLoader
from src.partition import UserPartitioner
from src.centralized_data import CentralizedDataBuilder
from src.model import HAR_CNN
from src.model_utils import reshape_for_cnn
from src.client import FLClient
from src.server import FederatedServer
from src.comparison import ModelComparator


HOST = "127.0.0.1"
PORT = 5000
SOCKET_BUFFER = 4096


# SOCKET HELPERS
def recvall(conn, n):
    data = b""
    while len(data) < n:
        packet = conn.recv(min(SOCKET_BUFFER, n - len(data)))
        if not packet:
            return None
        data += packet
    return data


def send_payload(payload, host=HOST, port=PORT):
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack(">Q", len(raw))

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(header + raw)
    s.close()


def receive_payload(conn):
    header = recvall(conn, 8)
    if not header:
        return None

    msg_len = struct.unpack(">Q", header)[0]
    body = recvall(conn, msg_len)
    if not body:
        return None

    return pickle.loads(body)


# TRAIN / EVAL HELPERS
def compute_norm_stats(client_splits):
    all_train = [client_splits[cid]["X_train"] for cid in client_splits]
    X_all = pd.concat(all_train)
    mean = X_all.mean()
    std = X_all.std().replace(0, 1)
    return mean, std


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            out = model(X)
            loss = criterion(out, y)

            total_loss += loss.item()

            _, preds = torch.max(out, 1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="weighted")

    return avg_loss, acc, f1, preds_all, labels_all


def quick_global_test(model, partitioner, mean, std, device):
    X_test_raw, y_test_raw = partitioner.get_global_test()

    X_test = reshape_for_cnn((X_test_raw - mean) / std)

    y_test = torch.tensor(
        y_test_raw.values.squeeze() - 1,
        dtype=torch.long
    )

    loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=64,
        shuffle=False
    )

    _, acc, f1, _, _ = evaluate(model, loader, device)

    return acc, f1

# CENTRALIZED TRAINING
def run_centralized_training(client_splits, partitioner, device):
    print("\n========== CENTRALIZED TRAINING ==========")

    builder = CentralizedDataBuilder(client_splits)

    global_data = builder.combine_all_clients(
        add_client_id=False,
        shuffle=True
    )

    X_train_raw = global_data["X_train"]
    X_test_raw, y_test_raw = partitioner.get_global_test()

    mean = X_train_raw.mean()
    std = X_train_raw.std().replace(0, 1)

    X_train = reshape_for_cnn((X_train_raw - mean) / std)
    X_test = reshape_for_cnn((X_test_raw - mean) / std)

    y_train = torch.tensor(
        global_data["y_train"].values.squeeze() - 1,
        dtype=torch.long
    )

    y_test = torch.tensor(
        y_test_raw.values.squeeze() - 1,
        dtype=torch.long
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=64,
        shuffle=False
    )

    model = HAR_CNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(10):
        loss = train(model, train_loader, criterion, optimizer, device)
        _, acc, f1, _, _ = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch+1}/10 | "
            f"Loss={loss:.4f} | "
            f"Acc={acc:.4f} | "
            f"F1={f1:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            os.makedirs("results", exist_ok=True)
            torch.save(model.state_dict(), "results/centralized_best.pth")

    model.load_state_dict(torch.load("results/centralized_best.pth", weights_only=True))

    return model, mean, std


# FEDERATED TRAINING WITH SOCKETS
def run_federated_training(client_splits, partitioner, device, rounds=15):
    print("\n========== FEDERATED TRAINING ==========")
    print("Wireshark filter -> tcp.port == 5000")

    global_model = HAR_CNN().to(device)
    server = FederatedServer(model=global_model)

    clients = {}

    global_weights = copy.deepcopy(global_model.state_dict())

    for cid in client_splits:
        c = FLClient(cid, device)
        c.receive_global_model(global_weights)
        clients[cid] = c

    mean, std = compute_norm_stats(client_splits)

    # Normalize client splits
    for cid in client_splits:
        client_splits[cid]["X_train"] = (
            client_splits[cid]["X_train"] - mean
        ) / std
        client_splits[cid]["X_test"] = (
            client_splits[cid]["X_test"] - mean
        ) / std

    best_acc = 0.0

    for round_num in range(1, rounds + 1):
        print(f"\n========== ROUND {round_num} ==========")

        selected = random.sample(
            list(clients.keys()),
            max(2, int(len(clients) * 0.6))
        )

        # Start listening socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, PORT))
        sock.listen(len(selected))

        received_updates = []

        # Clients train + send
        for cid in selected:
            client = clients[cid]
            data = client_splits[cid]

            updated_weights = client.local_train(
                data["X_train"],
                data["y_train"],
                epochs=3,
                batch_size=32,
                lr=0.001
            )

            payload = {
                "client_id": cid,
                "weights": updated_weights,
                "num_samples": len(data["X_train"])
            }

            send_payload(payload)

        # Server receive all packets
        try:
            for _ in range(len(selected)):
                conn, addr = sock.accept()
                packet = receive_payload(conn)

                if packet is not None:
                    received_updates.append(packet)

                conn.close()
        finally:
            sock.close()

        # Existing validation + packet logging
        server.collect_updates(received_updates)

        global_weights = server.fedavg_aggregate()
        server.update_global_model(global_weights)
        server.broadcast_global_model(clients)

        acc, f1 = quick_global_test(
            server.get_global_model(),
            partitioner,
            mean,
            std,
            device
        )

        print(f"[ROUND {round_num}] Acc={acc:.4f} | F1={f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs("results", exist_ok=True)
            torch.save(
                server.get_global_model().state_dict(),
                "results/federated_best.pth"
            )

        server.clear_updates()

    global_model.load_state_dict(
        torch.load("results/federated_best.pth", weights_only=True)
    )

    return global_model, mean, std, server


# GLOBAL EVALUATION
def run_global_evaluation(model, partitioner, mean, std, device, model_name):
    print(f"\n========== GLOBAL EVALUATION : {model_name} ==========")

    X_test_raw, y_test_raw = partitioner.get_global_test()

    X_test = reshape_for_cnn((X_test_raw - mean) / std)

    y_test = torch.tensor(
        y_test_raw.values.squeeze() - 1,
        dtype=torch.long
    )

    loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=64,
        shuffle=False
    )

    loss, acc, f1, preds, labels = evaluate(model, loader, device)

    print(f"Loss     : {loss:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")

    cm = confusion_matrix(labels, preds)
    print(cm)

    report = classification_report(labels, preds)
    print(report)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["centralized", "federated", "both"]
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using Device:", device)

    loader = UCIHARDataLoader("data/UCI_HAR")

    train_df, test_df = loader.load_full_dataset(
        use_inertial_signals=True
    )

    partitioner = UserPartitioner(train_df, test_df)

    client_data = partitioner.create_clients()
    client_splits = partitioner.split_clients(client_data)

    central_model = None
    fed_model = None
    server = None

    # CENTRALIZED
    if args.mode in ["centralized", "both"]:
        central_model, c_mean, c_std = run_centralized_training(
            client_splits,
            partitioner,
            device
        )

        run_global_evaluation(
            central_model,
            partitioner,
            c_mean,
            c_std,
            device,
            "Centralized"
        )

    # FEDERATED
    if args.mode in ["federated", "both"]:
        fed_model, f_mean, f_std, server = run_federated_training(
            client_splits,
            partitioner,
            device
        )

        run_global_evaluation(
            fed_model,
            partitioner,
            f_mean,
            f_std,
            device,
            "Federated"
        )

        # Packet Analysis Output
        if server is not None:
            print("\n========== PACKET ANALYSIS ==========")

            server.packet_analyzer.summary()
            server.packet_analyzer.save_csv()
            server.packet_analyzer.save_json()


if __name__ == "__main__":
    main()