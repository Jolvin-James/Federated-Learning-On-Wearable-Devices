from src.data_loader import UCIHARDataLoader
from src.partition import UserPartitioner
from src.normalize import ClientNormalizer

def main():
    import pandas as pd

    # Load dataset
    data_path = "data/UCI_HAR"
    loader = UCIHARDataLoader(data_path)

    train_df, test_df = loader.load_full_dataset()

    print("\n--- DATASET LOADED ---")

    # Partition into clients
    partitioner = UserPartitioner(train_df, test_df)
    client_datasets = partitioner.create_clients()

    print("\n--- CLIENT DATASETS CREATED ---")

    # Split clients
    client_splits = partitioner.split_clients(client_datasets)

    print("\n--- CLIENT SPLITS DONE ---")

    # PER-CLIENT NORMALIZATION
    normalizer = ClientNormalizer()

    for client_id, data in client_splits.items():
        X_train = data["X_train"]
        X_test = data["X_test"]

        # Fit ONLY on train data (VERY IMPORTANT)
        X_train_norm = normalizer.fit_transform(X_train, client_id)

        # Apply same stats to test data
        X_test_norm = normalizer.transform(X_test, client_id)

        # Replace with normalized data
        client_splits[client_id]["X_train"] = X_train_norm
        client_splits[client_id]["X_test"] = X_test_norm

    print("\n--- PER-CLIENT NORMALIZATION DONE ---")

    # Debug check
    sample_clients = list(client_splits.keys())[:3]

    for cid in sample_clients:
        print(f"\nClient {cid}")
        print("Train mean:", client_splits[cid]["X_train"].mean().mean())
        print("Train std:", client_splits[cid]["X_train"].std().mean())


if __name__ == "__main__":
    main()