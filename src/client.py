# src/client.py

import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from src.model import HAR_CNN
from src.model_utils import reshape_for_cnn


class FLClient:
    def __init__(self, client_id, device):
        """
        Federated Learning Client

        Args:
            client_id (int): unique client id
            device (torch.device): cpu / cuda
        """
        self.client_id = client_id
        self.device = device

        # Each client has its own local model
        self.model = HAR_CNN().to(self.device)

    # Receive Global Model
    def receive_global_model(self, global_weights: dict):
        """
        Load global model weights sent by server
        """
        if not isinstance(global_weights, dict):
            raise TypeError("Global weights must be state_dict")

        safe_weights = copy.deepcopy(global_weights)
        self.model.load_state_dict(safe_weights)

    # Compute Weight Update
    def compute_weight_update(self, old_weights, new_weights):
        """
        ΔW = W_new - W_old
        """
        delta = {}

        for key in old_weights.keys():
            delta[key] = new_weights[key] - old_weights[key]

        return delta

    # Local Training
    def local_train(
        self,
        X_train,
        y_train,
        epochs=2,
        batch_size=32,
        lr=0.001,
        weight_decay=1e-4,
        max_grad_norm=5.0
    ):
        """
        Local training on client data

        Returns:
            updated model weights
        """

        # Save initial weights
        initial_weights = {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

        # Convert input
        X_tensor = reshape_for_cnn(X_train)

        y_tensor = torch.tensor(
            y_train.values.squeeze() - 1,
            dtype=torch.long
        )

        dataset = TensorDataset(X_tensor, y_tensor)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.model.train()

        for epoch in range(epochs):

            total_loss = 0.0

            for X, y in loader:

                X = X.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(X)

                loss = criterion(outputs, y)

                loss.backward()

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm
                    )

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)

            print(
                f"[Client {self.client_id}] "
                f"Epoch {epoch+1}/{epochs} "
                f"| Loss: {avg_loss:.4f}"
            )

        updated_weights = self.get_weights()

        # Optional debug
        delta = self.compute_weight_update(
            initial_weights,
            updated_weights
        )

        total_update_norm = sum(
            torch.norm(v.float()).item()
            for v in delta.values()
            if torch.is_floating_point(v)
        )

        print(
            f"[Client {self.client_id}] "
            f"Update Norm: {total_update_norm:.4f}"
        )

        return copy.deepcopy(updated_weights)

    # Local Evaluation
    def local_evaluate(
        self,
        X_test,
        y_test,
        batch_size=64
    ):
        """
        Evaluate local model on this client's test set

        Returns:
            {
                client_id,
                loss,
                accuracy,
                f1_score
            }
        """

        # Prepare tensors
        X_tensor = reshape_for_cnn(X_test)

        y_tensor = torch.tensor(
            y_test.values.squeeze() - 1,
            dtype=torch.long
        )

        dataset = TensorDataset(
            X_tensor,
            y_tensor
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.model.eval()

        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for X, y in loader:

                X = X.to(self.device)
                y = y.to(self.device)

                outputs = self.model(X)

                loss = criterion(outputs, y)

                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)

                all_preds.extend(
                    preds.cpu().numpy()
                )

                all_labels.extend(
                    y.cpu().numpy()
                )

        avg_loss = total_loss / len(loader)

        acc = accuracy_score(
            all_labels,
            all_preds
        )

        f1 = f1_score(
            all_labels,
            all_preds,
            average="weighted"
        )

        print(
            f"[Client {self.client_id}] "
            f"Local Eval -> "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"F1: {f1:.4f}"
        )

        return {
            "client_id": self.client_id,
            "loss": avg_loss,
            "accuracy": acc,
            "f1_score": f1
        }

    # Utilities
    def get_model(self):
        return self.model

    def get_weights(self):
        return self.model.state_dict()

    def set_model(self, model):
        self.model = model.to(self.device)

    def print_status(self):
        total_params = sum(
            p.numel()
            for p in self.model.parameters()
        )

        print(
            f"[Client {self.client_id}] "
            f"Model loaded with "
            f"{total_params} parameters"
        )
