from typing import Literal, cast
from lightning import LightningDataModule
import numpy as np
import torch

from .utils import make_patched_ucr_transform

from .dataset import ExplainedItem, ConfoundingDataset
from torch.utils.data import DataLoader
from ..preprocessing import ScalerType, MinMaxScaler, StandardScaler

from .data_enums import Confounder
from darts import TimeSeries
from torchchronos.download import download_uea_ucr
from torchchronos.typing import DatasetSplit
from torchchronos.transforms import Transform
from torch.utils.data import random_split
import copy


class WrapperTransform(Transform):
    def __init__(self, transform: ScalerType) -> None:
        super().__init__()
        self.internal_transform = transform

    def fit(self, ts: torch.Tensor) -> None:
        self.internal_transform = self.internal_transform.fit(ts.view(-1, ts.shape[-1]))
        return self

    def transform(self, ts: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(
            self.internal_transform.transform(ts.view(-1, ts.shape[-1]))
        ).view(ts.shape)

    def __call__(self, ts: torch.Tensor) -> torch.Tensor:
        return self.transform(ts)

    def inverse_transform(self, ts: torch.Tensor) -> TimeSeries:
        # TODO Fix
        return torch.from_numpy(self.internal_transform.inverse_transform(ts.numpy()))

    def __repr__(self) -> str:
        return f"WrapperTransform({self.transform})"


class ClassificationData(LightningDataModule):
    def __init__(
        self,
        # split_ratio: float | tuple[float, float] = 0.8,
        confounder: Confounder = Confounder.NO_CONFOUNDER,
        batch_size: int = -1,
        val_confounder: Confounder = Confounder.NO_CONFOUNDER,
        scaler: ScalerType = MinMaxScaler(),
        confounder_len: int = 15,
        confounder_ampl: float = 0.1,
        sanity_padded_zeros: bool = True,
        num_workers: int = 0,
        num_classes: int = 3,
        name: str = "ArrowHead",
        lambda_time: float = 1.0,
        lambda_freq: float = 1.0,
        fast_dev_run: bool = False,
        train_val_split_seed: int = 1,
        feedback_percentage: float = 1.0,
    ) -> None:
        super().__init__()
        self.fast_dev_run = fast_dev_run

        # self.split_ratio = split_ratio
        self.confounder = confounder
        self.val_confounder = val_confounder
        self.batch_size = batch_size
        self.confounder_len = confounder_len
        self.confounder_ampl = confounder_ampl
        self.num_classes = num_classes

        self.data_transformer = WrapperTransform(scaler)

        self.num_workers = num_workers
        self.name = name
        self.sanity_padded = sanity_padded_zeros
        self.lambda_time = lambda_time
        self.lambda_freq = lambda_freq
        self.train_val_split_seed = train_val_split_seed
        self.feedback_percentage = feedback_percentage

    def prepare_data(self) -> None:
        download_uea_ucr(self.name)

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        self.train_dataset = ConfoundingDataset(
            self.confounder,
            self.confounder_len,
            self.confounder_ampl,
            self.name,
            split=DatasetSplit.TRAIN,
            padded_zeros=self.sanity_padded,
            scaler=None,
        )

        split_generator = torch.Generator().manual_seed(self.train_val_split_seed)
        train_subset, val_subset = random_split(self.train_dataset, [0.8, 0.2], generator=split_generator)

        train_feedback_idxs = train_subset.indices[: int(len(train_subset.indices) * self.feedback_percentage)]

        train_dataset = cast(ConfoundingDataset, copy.copy(train_subset.dataset))
        # only fit the training samples
        self.data_transformer = self.data_transformer.fit(
            train_dataset.xs[train_subset.indices]
        )
        train_dataset.transform = make_patched_ucr_transform(
            train_dataset.num_classes,
            self.confounder,
            self.confounder_ampl,
            train_dataset._confounder_width,
            self.data_transformer,
        )

        if self.confounder != Confounder.NO_CONFOUNDER:
            train_dataset.set_feedback_idxs(train_feedback_idxs)

        match stage:
            case "fit" | "validate":
                val_dataset = cast(
                    ConfoundingDataset, val_subset.dataset
                )  # .confounder = self.val_confounder
                val_dataset.confounder = self.val_confounder
                val_dataset.transform = make_patched_ucr_transform(
                    val_dataset.num_classes,
                    val_dataset.confounder,
                    val_dataset._confounder_ampl,
                    val_dataset._confounder_width,
                    self.data_transformer,
                )
                val_subset.dataset = val_dataset
                self.val_dataset = val_subset

                # set train as well
                train_subset.dataset = train_dataset
                self.train_dataset = train_subset
            case "test":
                self.test_dataset = ConfoundingDataset(
                    self.val_confounder,
                    self.confounder_len,
                    self.confounder_ampl,
                    self.name,
                    split=DatasetSplit.TEST,
                    padded_zeros=self.sanity_padded,
                    scaler=self.data_transformer,
                )
            case "predict":
                raise NotImplementedError("Predict not supported")

    def inverse_transform(self, data: torch.Tensor) -> np.ndarray:
        """
        Gets normalized data tensor of shape [Batch,Var,Seq] and returns a not normalized version of shape [Batch,Var,Seq]
        """
        return self.data_transformer.inverse_transform(data)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ExplainedItem.classification_collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.fast_dev_run,  # needed for plot callback
            collate_fn=ExplainedItem.classification_collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=ExplainedItem.classification_collate_fn,
            num_workers=self.num_workers,
        )
