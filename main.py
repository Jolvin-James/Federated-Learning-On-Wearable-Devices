from src.data_loader import UCIHARDataLoader
from src.partition import UserPartitioner

def main():
    # Initialize loader
    data_path = "data/UCI_HAR"
    loader = UCIHARDataLoader(data_path)

    # Load dataset
    train_df, test_df = loader.load_full_dataset()

    print("\n--- DATASET LOADED SUCCESSFULLY ---\n")
    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)

    # User-wise Partitioning
    partitioner = UserPartitioner(train_df, test_df)

    # Create client datasets
    client_datasets = partitioner.create_clients()

    print("\n--- CLIENT DATASETS CREATED ---")
    print("Total Clients:", len(client_datasets))

    # Show sample clients
    sample_clients = list(client_datasets.keys())[:3]
    for cid in sample_clients:
        print(f"\nClient {cid} Shape:", client_datasets[cid].shape)

    # Split each client dataset
    client_splits = partitioner.split_clients(client_datasets)

    print("\n--- CLIENT TRAIN/TEST SPLIT DONE ---")

    for cid in sample_clients:
        if cid in client_splits:
            print(f"\nClient {cid}:")
            print("Train Shape:", client_splits[cid]["X_train"].shape)
            print("Test Shape:", client_splits[cid]["X_test"].shape)

    # Summary statistics
    summary = partitioner.get_client_summary(client_datasets)

    print("\n--- CLIENT SUMMARY ---")
    for cid in sample_clients:
        print(f"Client {cid} -> Samples: {summary[cid]['num_samples']}, Activities: {summary[cid]['num_activities']}")


if __name__ == "__main__":
    main()