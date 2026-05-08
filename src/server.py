# src/server.py

import copy
import torch
from src.packet_analysis import PacketAnalyzer

class FederatedServer:
    def __init__(self, model=None):
        """
        Federated Server to manage:
        - Client update collection
        - Payload inspection
        - Content validation
        - FedAvg aggregation
        - Global model update
        """
        self.client_updates = []
        self.total_samples = 0
        self.global_model = model
        self.packet_analyzer = PacketAnalyzer()
        self.current_round = 0

    # Inspect Communication Payload + Validate Content
    def collect_updates(self, client_updates: list):
        
        self.current_round += 1

        print("\n[SERVER] ===== Collecting Client Updates =====")

        if not isinstance(client_updates, list):
            raise TypeError(
                "[SERVER ERROR] client_updates must be a list"
            )

        if len(client_updates) == 0:
            raise ValueError(
                "[SERVER ERROR] No client updates received"
            )

        self.client_updates = []
        self.total_samples = 0

        # Forbidden privacy-risk fields
        blocked_words = [
            "raw_data",
            "sensor_data",
            "signals",
            "x_train",
            "x_test",
            "y_train",
            "y_test",
            "labels",
            "activity",
            "activities",
            "gyro",
            "accelerometer",
            "human_data",
            "dataset"
        ]

        for update in client_updates:

            # Validate Payload Container
            if not isinstance(update, dict):
                raise TypeError(
                    "[SERVER ERROR] Each payload must be dict"
                )

            required_keys = [
                "client_id",
                "weights",
                "num_samples"
            ]

            if not all(k in update for k in required_keys):
                raise ValueError(
                    f"[SERVER ERROR] Missing required keys: {update}"
                )

            client_id = update["client_id"]
            self.packet_analyzer.log_packet(
                round_num=self.current_round,
                client_id=client_id,
                payload=update
            )
            weights = update["weights"]
            num_samples = update["num_samples"]

            print(f"\n[PAYLOAD INSPECTION] Client {client_id}")
            print("-" * 50)

            payload_keys = list(update.keys())

            # Show Keys
            print("Payload Keys:")
            for key in payload_keys:
                print(" ->", key)

            # Detect Extra Keys
            allowed_keys = {
                "client_id",
                "weights",
                "num_samples"
            }

            extra_keys = set(payload_keys) - allowed_keys

            if len(extra_keys) > 0:
                print(
                    "[WARNING] Unexpected Keys Found:",
                    extra_keys
                )

            # Privacy Leak Detection
            for key in payload_keys:

                lower_key = key.lower()

                for word in blocked_words:
                    if word in lower_key:
                        raise ValueError(
                            f"[PRIVACY ALERT] Raw data key found: {key}"
                        )

            # Validate client_id
            if not isinstance(client_id, int):
                raise TypeError(
                    "[SERVER ERROR] client_id must be int"
                )

            # Validate num_samples
            if not isinstance(num_samples, int):
                raise TypeError(
                    "[SERVER ERROR] num_samples must be int"
                )

            if num_samples <= 0:
                raise ValueError(
                    "[SERVER ERROR] Invalid sample count"
                )

            # Validate weights content
            if not isinstance(weights, dict):
                raise TypeError(
                    "[SERVER ERROR] weights must be dict"
                )

            total_layers = 0
            total_params = 0

            suspicious_dtype = False

            for layer_name, tensor in weights.items():

                if not torch.is_tensor(tensor):
                    raise TypeError(
                        f"[SERVER ERROR] Non-tensor in layer {layer_name}"
                    )

                # Only numeric tensors allowed
                if tensor.dtype not in [
                    torch.float16,
                    torch.float32,
                    torch.float64,
                    torch.int32,
                    torch.int64
                ]:
                    suspicious_dtype = True

                total_layers += 1
                total_params += tensor.numel()

            print("Layers Sent     :", total_layers)
            print("Parameters Sent :", total_params)
            print("Samples Used    :", num_samples)

            if suspicious_dtype:
                print(
                    "[WARNING] Unusual tensor datatype detected"
                )

            # 8. Accept Payload
            self.client_updates.append(update)
            self.total_samples += num_samples

            print("[STATUS] Payload Safe")

        # Final Summary
        if self.total_samples == 0:
            raise ValueError(
                "[SERVER ERROR] Total samples is zero"
            )

        print("\n[SERVER] ===== Validation Summary =====")
        print("Clients Received :", len(self.client_updates))
        print("Total Samples    :", self.total_samples)
        print("[SERVER] All payloads validated successfully.")

        return self.client_updates

    # FedAvg Aggregation
    def fedavg_aggregate(self):

        print("\n[SERVER] ===== FedAvg Aggregation =====")

        if not self.client_updates:
            raise ValueError(
                "[SERVER ERROR] No updates available"
            )

        reference_weights = self.client_updates[0]["weights"]

        global_weights = {}

        for key, val in reference_weights.items():
            if torch.is_floating_point(val):
                global_weights[key] = torch.zeros_like(
                    val,
                    dtype=torch.float32
                )
            else:
                # BatchNorm counters are integer buffers, not learnable weights.
                # Averaging them can make diagnostics look unstable without
                # improving the model.
                global_weights[key] = val.clone()

        for update in self.client_updates:

            client_weights = update["weights"]
            num_samples = update["num_samples"]

            factor = num_samples / self.total_samples

            for key in global_weights:
                if torch.is_floating_point(client_weights[key]):
                    global_weights[key] += (
                        client_weights[key].float() * factor
                    )

        for key in global_weights:
            if torch.is_floating_point(reference_weights[key]):
                global_weights[key] = global_weights[key].to(
                    reference_weights[key].dtype
                )

        print("[SERVER] Aggregation Complete")

        return global_weights

    # Update Global Model
    def update_global_model(self, global_weights):

        print("\n[SERVER] ===== Updating Global Model =====")

        if self.global_model is None:
            raise ValueError(
                "[SERVER ERROR] Global model missing"
            )

        if not isinstance(global_weights, dict):
            raise TypeError(
                "[SERVER ERROR] global_weights must be dict"
            )

        model_keys = set(
            self.global_model.state_dict().keys()
        )

        weight_keys = set(global_weights.keys())

        if model_keys != weight_keys:
            raise ValueError(
                "[SERVER ERROR] Weight mismatch"
            )

        old_norm = self._compute_model_norm(
            self.global_model.state_dict()
        )

        self.global_model.load_state_dict(
            copy.deepcopy(global_weights)
        )

        new_norm = self._compute_model_norm(
            self.global_model.state_dict()
        )

        print("[SERVER] Model Updated Successfully")
        print(f"Norm Before : {old_norm:.4f}")
        print(f"Norm After  : {new_norm:.4f}")

        return self.global_model

    # Broadcast Global Model
    def broadcast_global_model(self, clients: dict):

        print("\n[SERVER] ===== Broadcasting Global Model =====")

        if self.global_model is None:
            raise ValueError(
                "[SERVER ERROR] Global model missing"
            )

        global_weights = self.global_model.state_dict()

        for client_id, client in clients.items():

            client.receive_global_model(
                copy.deepcopy(global_weights)
            )

            print(
                f"[SERVER] Model sent to Client {client_id}"
            )

        print("[SERVER] Broadcast Complete")

    # Utilities
    def _compute_model_norm(self, weights):

        total = 0.0

        for _, tensor in weights.items():
            if not torch.is_floating_point(tensor):
                continue

            total += torch.norm(
                tensor.float()
            ).item()

        return total

    def get_global_model(self):
        return self.global_model

    def clear_updates(self):
        self.client_updates = []
        self.total_samples = 0
