# src/server.py
import copy
import torch

class FederatedServer:
    def __init__(self):
        """
        Federated Server to manage client updates and aggregation
        """
        self.client_updates = []
        self.total_samples = 0

    def collect_updates(self, client_updates: list):
        """
        Collect updates from all clients

        Args:
            client_updates (list):
                [
                    {
                        "client_id": int,
                        "weights": state_dict,
                        "num_samples": int
                    },
                    ...
                ]
        """

        print("\n[SERVER] ===== Collecting Client Updates =====")

        if not isinstance(client_updates, list):
            raise TypeError("[SERVER ERROR] client_updates must be a list")

        if len(client_updates) == 0:
            raise ValueError("[SERVER ERROR] No client updates received")

        self.client_updates = []
        self.total_samples = 0

        for idx, update in enumerate(client_updates):

            required_keys = ["client_id", "weights", "num_samples"]
            if not all(key in update for key in required_keys):
                raise ValueError(
                    f"[SERVER ERROR] Missing keys in update {idx}. "
                    f"Required: {required_keys}"
                )

            client_id = update["client_id"]
            weights = update["weights"]
            num_samples = update["num_samples"]

            if not isinstance(client_id, int):
                raise TypeError("[SERVER ERROR] client_id must be int")

            if not isinstance(weights, dict):
                raise TypeError("[SERVER ERROR] weights must be dict")

            if not isinstance(num_samples, int) or num_samples <= 0:
                raise ValueError("[SERVER ERROR] num_samples must be positive int")

            # Validate tensor structure
            first_tensor = list(weights.values())[0]
            try:
                _ = first_tensor.shape
            except Exception:
                raise ValueError("[SERVER ERROR] Invalid tensor in weights")

            self.client_updates.append(update)
            self.total_samples += num_samples

            print(
                f"[SERVER] Client {client_id} | "
                f"Samples: {num_samples} | "
                f"Layers: {len(weights)}"
            )

        print("\n[SERVER] ===== Summary =====")
        print(f"[SERVER] Total Clients : {len(self.client_updates)}")
        print(f"[SERVER] Total Samples: {self.total_samples}")

        return self.client_updates

    def fedavg_aggregate(self):
        """
        Perform FedAvg aggregation

        Formula:
            W_global = Σ (Wi * ni / N)

        Returns:
            dict: aggregated global weights
        """

        print("\n[SERVER] ===== FedAvg Aggregation Started =====")

        if not self.client_updates:
            raise ValueError("[SERVER ERROR] No client updates to aggregate")

        if self.total_samples == 0:
            raise ValueError("[SERVER ERROR] Total samples cannot be zero")

        total_samples = self.total_samples

        # Use reference weights to ensure types match at the end
        reference_weights = self.client_updates[0]["weights"]
        global_weights = {}

        # --- INITIALIZE GLOBAL WEIGHTS AS FLOATS ---
        for key, val in reference_weights.items():
            # Use float32 to avoid RuntimeError when adding float tensors to Long tensors (like num_batches_tracked)
            global_weights[key] = torch.zeros_like(val, dtype=torch.float32)

        # --- WEIGHTED AGGREGATION ---
        for update in self.client_updates:
            client_id = update["client_id"]
            weights = update["weights"]
            num_samples = update["num_samples"]

            weight_factor = num_samples / total_samples

            print(
                f"[SERVER] Aggregating Client {client_id} | "
                f"Weight Factor: {weight_factor:.4f}"
            )

            for key in global_weights.keys():
                # Cast client weight to float and detach, effectively avoiding graph dependencies
                # and type casting errors.
                global_weights[key] += (weights[key].to(torch.float32).detach() * weight_factor)

        # --- RESTORE ORIGINAL DTYPES ---
        for key in global_weights.keys():
            global_weights[key] = global_weights[key].to(reference_weights[key].dtype)

        print("\n[SERVER] ===== Aggregation Complete =====")

        return global_weights

    def get_total_samples(self):
        return self.total_samples

    def get_client_updates(self):
        return self.client_updates

    def clear_updates(self):
        """
        Clear stored updates (important for next round)
        """
        self.client_updates = []
        self.total_samples = 0