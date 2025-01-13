# Standard library imports
from typing import Optional, List
from dataclasses import dataclass

# Third-party library imports
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data

# Local imports
from neural_signatures.datasets.synthetic_datasets import SyntheticSurfaceDataset
from neural_signatures.datasets.mesh_datasets import RemeshingDataset


# neural signatures
from neural_signatures.utils import utils


@dataclass
class DatasetSpecification:
    dataset: Dataset
    batch_size: int
    num_workers: int
    shuffle: bool


class SyntheticSurfaceDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_dataset_specification: DatasetSpecification,
            val_dataset_specifications: List[DatasetSpecification],
    ) -> None:
        super().__init__()
        self._train_dataset_specification = train_dataset_specification
        self._val_dataset_specifications = val_dataset_specifications

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._train_dataset_specification.dataset,
            batch_size=self._train_dataset_specification.batch_size,
            shuffle=self._train_dataset_specification.shuffle,
            num_workers=self._train_dataset_specification.num_workers,
            persistent_workers=self._train_dataset_specification.num_workers > 0
        )

    def val_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                dataset=val_dataset_specification.dataset,
                batch_size=val_dataset_specification.batch_size,
                shuffle=val_dataset_specification.shuffle,
                num_workers=val_dataset_specification.num_workers,
                persistent_workers=val_dataset_specification.num_workers > 0
            ) for val_dataset_specification in self._val_dataset_specifications
        ]

