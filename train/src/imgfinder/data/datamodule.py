from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pyrootutils
import torch
from lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import Compose, transforms

from .components.dataset import ImageDataset
from .utils.encoders import LabelEncoder


class DataModule(LightningDataModule):
    def __init__(
        self,
        images_dir: str,
        images_metadata_path: str,
        train_transforms: Compose,
        val_transforms: Compose,
        batch_size: int = 64,
        n_train_copies: int = 3,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.label_encoder = LabelEncoder()

        images_metadata = pd.read_csv(images_metadata_path)
        images_metadata["path"] = Path(images_dir) / images_metadata["filename"]
        images_metadata["label"] = self.label_encoder.fit_transform(
            images_metadata["name"]
        )

        self.train_metadata = images_metadata.query("isTemplate")
        self.val_metadata = images_metadata.query("~isTemplate")
        self.n_train_copies = n_train_copies

        self.data_train: Optional[ImageDataset] = None
        self.data_val: Optional[ImageDataset] = None
        # self.data_test: Optional[ImageDataset] = None

        self._num_classes = self.train_metadata["name"].nunique()

    @property
    def num_classes(self):
        return self._num_classes

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        return

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            self.data_train = ConcatDataset(
                [
                    ImageDataset(
                        self.train_metadata,
                        self.hparams.train_transforms,
                        "path",
                        "label",
                    )
                    for _ in range(self.n_train_copies)
                ]
            )

            self.data_val = ImageDataset(
                self.val_metadata, self.hparams.val_transforms, "path", "label"
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
