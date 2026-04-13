# src/client.py

import copy
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import HAR_CNN
from src.model_utils import reshape_for_cnn


class FLClient:
    def __init__(self, client_id, device):
        """
        Initialize a federated learning client

        Args:
            client_id (int): Unique client identifier
            device (torch.device): CPU or GPU
        """
        self.client_id = client_id
        self.device = device

        # Each client maintains its own model instance
        self.model = HAR_CNN().to(self.device)

    # RECEIVE GLOBAL MODEL
    def receive_global_model(self, global_weights: dict):
        if not isinstance(global_weights, dict):
            raise TypeError("Global weights must be a state_dict (dict)")

        safe_weights = copy.deepcopy(global_weights)
        self.model.load_state_dict(safe_weights)

    # LOCAL TRAINING LOOP
    def local_train(
        self,
        X_train,
        y_train,
        epochs=2,
        batch_size=32,
        lr=0.001
    ):
        """
        Train the client model locally

        Args:
            X_train (DataFrame): Features
            y_train (Series): Labels
        Returns:
            dict: Updated model weights
        """

        # DATA PREPARATION
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

        # TRAINING SETUP
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        self.model.train()

        # LOCAL TRAINING LOOP
        for epoch in range(epochs):
            total_loss = 0

            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(X)
                loss = criterion(outputs, y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)

            print(
                f"[Client {self.client_id}] "
                f"Epoch {epoch+1}/{epochs} "
                f"| Loss: {avg_loss:.4f}"
            )

        # RETURN UPDATED WEIGHTS
        return self.get_weights()

    # UTILITIES
    def get_model(self):
        return self.model

    def get_weights(self):
        return self.model.state_dict()

    def set_model(self, model):
        self.model = model.to(self.device)

    def print_status(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Client {self.client_id}] Model loaded with {total_params} parameters")