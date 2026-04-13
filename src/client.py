# src/client.py
import copy
import torch
from src.model import HAR_CNN

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

    def receive_global_model(self, global_weights: dict):
        """
        Step 3.1: Receive global model from server

        Args:
            global_weights (dict): state_dict from global model
        """

        if not isinstance(global_weights, dict):
            raise TypeError("Global weights must be a state_dict (dict)")

        safe_weights = copy.deepcopy(global_weights)

        # Load weights into client model
        self.model.load_state_dict(safe_weights)

    def get_model(self):
        """
        Return client model (used in next steps)
        """
        return self.model

    def get_weights(self):
        """
        Return current client model weights
        (used later for sending updates to server)
        """
        return self.model.state_dict()

    def set_model(self, model):
        """
        Optional: replace model manually
        """
        self.model = model.to(self.device)

    def print_status(self):
        """
        Debug helper to verify model is loaded
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Client {self.client_id}] Model loaded with {total_params} parameters")