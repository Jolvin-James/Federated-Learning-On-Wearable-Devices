# src/model_utils.py
import torch

def reshape_for_cnn(df):
    """
    Convert dataframe to CNN input tensor

    Input:
        X: pandas DataFrame of shape (samples, 1152)

    Output:
        Tensor: (samples, 9, 128)
    """
    X = df.values
    expected_features = 9 * 128

    if X.shape[1] != expected_features:
        raise ValueError(
            f"reshape_for_cnn expected {expected_features} features per sample "
            f"(9 channels x 128 timesteps), but got {X.shape[1]}. "
            "You are likely passing the 561 engineered features from X_train/X_test.txt "
            "instead of the raw inertial signals from the Inertial Signals folder."
        )

    X = X.reshape(-1, 9, 128)
    return torch.tensor(X, dtype=torch.float32)
