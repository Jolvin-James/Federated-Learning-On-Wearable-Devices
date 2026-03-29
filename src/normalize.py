import pandas as pd


class ClientNormalizer:
    def __init__(self):
        self.client_stats = {}

    def fit(self, X: pd.DataFrame, client_id: int):
        """
        Compute mean/std for a specific client
        """
        mean = X.mean()
        std = X.std()

        # Avoid division by zero
        std.replace(0, 1, inplace=True)

        self.client_stats[client_id] = {
            "mean": mean,
            "std": std
        }

    def transform(self, X: pd.DataFrame, client_id: int):
        """
        Normalize using that client's stats
        """
        stats = self.client_stats[client_id]

        X_norm = (X - stats["mean"]) / stats["std"]

        return X_norm

    def fit_transform(self, X: pd.DataFrame, client_id: int):
        self.fit(X, client_id)
        return self.transform(X, client_id)