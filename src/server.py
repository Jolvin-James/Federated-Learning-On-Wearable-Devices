# src/server.py

import copy
import torch


class FederatedServer:
    def __init__(self, model=None):
        """
        Federated Server to manage:
        - Client updates
        - Aggregation
        - Global model state
        """
        self.client_updates = []
        self.total_samples = 0
        self.global_model = model

    # COLLECT CLIENT UPDATES
    def collect_updates(self, client_updates: list):
        print("\n[SERVER] ===== Collecting Client Updates =====")

        if not isinstance(client_updates, list):
            raise TypeError("[SERVER ERROR] client_updates must be a list")

        if len(client_updates) == 0:
            raise ValueError("[SERVER ERROR] No client updates received")

        self.client_updates = []
        self.total_samples = 0

        for update in client_updates:
            if not isinstance(update, dict):
                raise TypeError("[SERVER ERROR] Each update must be a dict")

            required_keys = ["client_id", "weights", "num_samples"]
            if not all(k in update for k in required_keys):
                raise ValueError(f"[SERVER ERROR] Missing keys in client update: {update}")

            client_id = update["client_id"]
            weights = update["weights"]
            num_samples = update["num_samples"]

            if not isinstance(weights, dict):
                raise TypeError(f"[SERVER ERROR] Invalid weights from Client {client_id}")

            if not isinstance(num_samples, int) or num_samples <= 0:
                raise ValueError(f"[SERVER ERROR] Invalid sample count from Client {client_id}")

            self.client_updates.append(update)
            self.total_samples += num_samples

            print(f"[SERVER] Client {client_id} | Samples: {num_samples}")

        if self.total_samples == 0:
            raise ValueError("[SERVER ERROR] Total samples is zero")

        print(f"[SERVER] Total Samples: {self.total_samples}")

        return self.client_updates

    # FEDAVG AGGREGATION
    def fedavg_aggregate(self):
        print("\n[SERVER] ===== FedAvg Aggregation =====")

        if not self.client_updates:
            raise ValueError("[SERVER ERROR] No updates to aggregate")

        total_samples = self.total_samples

        reference_weights = self.client_updates[0]["weights"]
        global_weights = {}

        # Initialize accumulator
        for key, val in reference_weights.items():
            global_weights[key] = torch.zeros_like(val, dtype=torch.float32)

        # Weighted aggregation
        for update in self.client_updates:
            weights = update["weights"]
            num_samples = update["num_samples"]

            weight_factor = num_samples / total_samples

            for key in global_weights:
                global_weights[key] += weights[key].float() * weight_factor

        # Restore original dtype
        for key in global_weights:
            global_weights[key] = global_weights[key].to(reference_weights[key].dtype)

        print("[SERVER] Aggregation Complete")

        return global_weights

    # UPDATE GLOBAL MODEL
    def update_global_model(self, global_weights: dict):
        print("\n[SERVER] ===== Updating Global Model =====")

        if self.global_model is None:
            raise ValueError("[SERVER ERROR] Global model not initialized")

        if not isinstance(global_weights, dict):
            raise TypeError("[SERVER ERROR] global_weights must be dict")

        # Validate keys
        model_keys = set(self.global_model.state_dict().keys())
        weight_keys = set(global_weights.keys())

        if model_keys != weight_keys:
            raise ValueError("[SERVER ERROR] Model & weight mismatch")

        # Debug norms
        old_norm = self._compute_model_norm(self.global_model.state_dict())

        # Update model safely
        self.global_model.load_state_dict(copy.deepcopy(global_weights))

        new_norm = self._compute_model_norm(self.global_model.state_dict())

        print("[SERVER] Model updated successfully")
        print(f"[DEBUG] Norm Before: {old_norm:.4f}")
        print(f"[DEBUG] Norm After : {new_norm:.4f}")

        return self.global_model

    # BROADCAST BACK
    def broadcast_global_model(self, clients: dict):
        print("\n[SERVER] ===== Broadcasting Global Model =====")

        if self.global_model is None:
            raise ValueError("[SERVER ERROR] Global model not initialized")

        if not isinstance(clients, dict) or len(clients) == 0:
            raise ValueError("[SERVER ERROR] No clients available for broadcast")

        global_weights = self.global_model.state_dict()

        for client_id, client in clients.items():
            client.receive_global_model(copy.deepcopy(global_weights))
            print(f"[SERVER] Global model sent to Client {client_id}")

        print("[SERVER] Broadcast Complete")

    # UTILITIES
    def _compute_model_norm(self, weights):
        return sum(torch.norm(v.float()).item() for v in weights.values())

    def get_global_model(self):
        return self.global_model

    def clear_updates(self):
        self.client_updates = []
        self.total_samples = 0