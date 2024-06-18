from typing import Any, Literal

from torch.utils.data import DataLoader
from lib.data.data_enums import Confounder, P2SFeedbackMode
from lib.preprocessing import MinMaxScaler, ScalerType
from .classification_data import ClassificationData
from .dataset import ExplainedItem, P2SDataset



class P2SData(ClassificationData):
    def __init__(
        self,
        batch_size: int = -1,
        scaler: ScalerType = MinMaxScaler(),
        num_workers: int = 0,
        mode: Literal["Decoy", "Normal"] = "Decoy",
        lambda_time: float = 1.0,
        feedback_mode: P2SFeedbackMode = P2SFeedbackMode.NO_FEEDBACK,
        fast_dev_run: bool = False,
    ) -> None:
        super().__init__(
            Confounder.NO_CONFOUNDER if mode == "Normal" else Confounder.CLASSIFICATION_TIME,
            batch_size,
            Confounder.NO_CONFOUNDER,
            scaler,
            -1,
            0,
            False,
            num_workers,
            2,
            "PS2",
            lambda_time,
            1.0,
            fast_dev_run,
        )

        self.feedback_mode = feedback_mode
        self.mode = mode

    def prepare_data(self) -> None:
        P2SDataset.download_p2s(self.mode)


    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        self.train_dataset = P2SDataset(self.mode, self.data_transformer, feedback_mode=self.feedback_mode,split="train")
        self.test_dataset = P2SDataset(self.mode, self.data_transformer, feedback_mode=P2SFeedbackMode.NO_FEEDBACK,split="test")

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
