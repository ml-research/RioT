from itertools import chain
from pathlib import Path
from typing import Any, Literal
import numpy as np

import pandas as pd
from torch.utils.data import DataLoader
from lib.data.data_enums import Confounder
from lib.preprocessing import MinMaxScaler, ScalerType
from .classification_data import ClassificationData
from .dataset import ExplainedItem, MechanicalClassificationDataset
from .utils import print_ratios


class MechanicalClassificationData(ClassificationData):
    def __init__(
        self,
        batch_size: int = -1,
        scaler: ScalerType = MinMaxScaler(),
        num_workers: int = 0,
        train_experiment_idxs: list[int] = [1, 2, 13, 23],
        test_experiment_idxs: list[int] = [3, 5, 6, 9, 10],
        lambda_time: float = 1.0,
        stage_type: str = "rondenschnitt",
        feedback_penalty_range: list[range] | None = None,
        fast_dev_run: bool = False,
    ) -> None:
        super().__init__(
            Confounder.NO_CONFOUNDER if feedback_penalty_range is None else Confounder.CLASSIFICATION_TIME,
            batch_size,
            Confounder.NO_CONFOUNDER,
            scaler,
            -1,
            0,
            False,
            num_workers,
            2,
            "Mechanical",
            lambda_time,
            1.0,
            fast_dev_run,
        )

        self.feedback_penalty_range = chain(*feedback_penalty_range) if feedback_penalty_range is not None else None
        self.kind = f"{stage_type}_segmented"
        self.train_experiment_idxs = train_experiment_idxs
        self.test_experiment_idxs = test_experiment_idxs

        self.root_dir = Path("./data/mechanical")
        self.cache_dir = Path(".cache/mechanical")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _process_experiment_data(self, experiment_idxs: list[int], meta: dict[str, Any], save_path: Path, print_prefix: str, shuffle: bool = True) -> None:
        speeds = []
        labels = []  # 0 = no defect, 1 = defect
        xs = []

        folder_prefix = "experiment_"

        # Serialize data into single file
        for idx in experiment_idxs:
            ds = self.root_dir / f"{folder_prefix}{idx}" / f"{self.kind}.npy"
            element = meta[meta["Versuchsnummer"] == idx]
            state = element["State"].values[0]
            speed = element["ProductionSpeed (Hub/min)"].values[0]
            has_defect = state == "Defect"
            x = np.load(ds)
            speeds.append(speed)
            labels.append(has_defect)
            xs.append(x)

        # Create numpy arrays for the concatenated attributes
        speed_np, label_np, x_np = self._concatenate_attributes(xs, speeds, labels)

        if shuffle:
            # Shuffle the data if required
            idxs = np.arange(x_np.shape[0])
            np.random.shuffle(idxs)
            x_np = x_np[idxs]
            label_np = label_np[idxs]
            speed_np = speed_np[idxs]

        print_ratios(print_prefix, label_np)

        # Save to cache as npz
        np.savez_compressed(save_path, x=x_np, y=label_np, speed=speed_np)

    def _concatenate_attributes(self, xs: list[np.ndarray], speeds: list[np.ndarray], labels: list[np.ndarray]):
        speed_np = np.concatenate([
            np.full((x.shape[0],), speed) for x, speed in zip(xs, speeds)
        ])

        label_np = np.concatenate([
            np.full((x.shape[0],), label, dtype=float) for x, label in zip(xs, labels)
        ])

        x_np = np.concatenate(xs).astype(float)

        return speed_np, label_np, x_np

    def prepare_data(self) -> None:
        # TODO only create numpy files if not present
        # Read metadata
        meta_path = self.root_dir / "meta.csv"
        meta = pd.read_csv(meta_path, sep=";")
        
        # Process and serialize train data
        self._process_experiment_data(
            experiment_idxs=self.train_experiment_idxs,
            meta=meta,
            save_path=self.cache_dir / "train.npz",
            print_prefix="\nMechanical Train"
        )
        
        # Process and serialize test data
        self._process_experiment_data(
            experiment_idxs=self.test_experiment_idxs,
            meta=meta,
            save_path=self.cache_dir / "test.npz",
            print_prefix="\nMechanical Test",
            shuffle=True  # Note that we pass an additional argument here for shuffling
        )

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        self.train_dataset = MechanicalClassificationDataset(
            self.cache_dir, self.data_transformer, self.feedback_penalty_range, "train"
        )

        self.test_dataset = MechanicalClassificationDataset(
            self.cache_dir, self.data_transformer, None, "test"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=ExplainedItem.classification_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=ExplainedItem.classification_collate_fn,
        )
