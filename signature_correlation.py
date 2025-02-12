import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import numpy as np


@hydra.main(version_base="1.2", config_path="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.globals.seed)

    # Instantiate dataset and data module
    data_module = hydra.utils.instantiate(cfg.data_module)

    # Get the data loader
    data_loader = data_module.train_dataloader()

    # Initialize lists to store all values
    all_H = []
    all_K = []
    all_H1 = []
    all_K1 = []
    all_H2 = []
    all_K2 = []

    for i, batch in enumerate(data_loader):
        if i % 1000 == 0:
            print(f'Batch {i}')

        # Extract and store all quantities
        all_H.append(batch.H.detach().cpu().numpy())
        all_K.append(batch.K.detach().cpu().numpy())
        all_H1.append(batch.H1.detach().cpu().numpy())
        all_K1.append(batch.K1.detach().cpu().numpy())
        all_H2.append(batch.H2.detach().cpu().numpy())
        all_K2.append(batch.K2.detach().cpu().numpy())

    # Concatenate all batches
    H = np.concatenate(all_H)
    K = np.concatenate(all_K)
    H1 = np.concatenate(all_H1)
    K1 = np.concatenate(all_K1)
    H2 = np.concatenate(all_H2)
    K2 = np.concatenate(all_K2)

    # Stack all vectors as columns
    all_quantities = np.column_stack([H, K, H1, K1, H2, K2])

    # Compute correlation matrix
    corr_matrix = np.corrcoef(all_quantities.T)

    # Print the correlation matrix with aligned labels
    labels = ['H', 'K', 'H1', 'K1', 'H2', 'K2']
    print("\nCorrelation Matrix:")

    # Print header with proper spacing
    print(" " * 8, end="")  # Space for row labels
    for label in labels:
        print(f"{label:>10}", end="")
    print()

    # Print each row with proper alignment
    for i, label in enumerate(labels):
        print(f"{label:>8}", end="")
        for j in range(6):
            print(f"{corr_matrix[i, j]:10.3f}", end="")
        print()


if __name__ == "__main__":
    main()