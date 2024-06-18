from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing import Literal
from lightning.pytorch import LightningModule

from torch import nn
import torch
from torchmetrics import MetricCollection, Metric
from lib.data.dataset import ExplainedItem
from lib.loss import Right_Reason_Loss


class ModelSystem(LightningModule):
    def __init__(
        self,
        right_answer_loss: nn.Module,
        right_reason_loss: Right_Reason_Loss | None = None,
        metrics: list[Metric] = [],
        lambda_time: float = 1.0,
        lambda_freq: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.right_answer_loss = right_answer_loss
        self.rrr_loss = right_reason_loss
        self.lambda_time = lambda_time
        self.lambda_freq = lambda_freq

        _metrics = MetricCollection(metrics)
        self.train_metrics = _metrics.clone(prefix="train/")
        self.val_metrics = _metrics.clone(prefix="val/")
        self.test_metrics = _metrics.clone(prefix="test/")

    def log_optional_loss(self, kind: Literal["train", "val", "test"], metric_name: str, value: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        """
        Logs an optional loss value.

        Args:
            kind (Literal["train", "val", "test"]): The kind of loss (train, val, or test).
            metric_name (str): The name of the loss metric.
            value (torch.Tensor | None): The loss value to be logged. If None, nothing will be logged.
            batch_size (int): The batch size. Needed for self.log

        Returns:
            torch.Tensor: The logged loss value or 0.0 if value is None.
        """
        
        if value is not None:
            self.log(
                f"{kind}/{metric_name}",
                value,
                prog_bar=True,
                on_epoch=True,
                on_step=True,
                batch_size=batch_size,
            )
            return value
        
        return 0.0

    @property
    def applies_xil(self) -> bool:
        return self.rrr_loss is not None

    
    def validation_step(self, item: ExplainedItem, index: int) -> STEP_OUTPUT:
        return self._non_train_step(item, "val")

    def test_step(self, item: ExplainedItem, index: int) -> STEP_OUTPUT:
        return self._non_train_step(item, "test")


    def _non_train_step(self, item: ExplainedItem, step: Literal["val", "test"]) -> STEP_OUTPUT:
        x, y = item.x, item.y

        y_hat = self(x)

        loss = self.right_answer_loss(y_hat, y)
        self.log_dict(
            self.val_metrics(y_hat, y) if step == "val" else self.test_metrics(y_hat, y),
            on_epoch=True,
            on_step=True,
            batch_size=x.shape[0],
        )

        self.log(
            f"{step}/loss",
            loss,
            prog_bar=True,
            batch_size=x.shape[0],
            on_epoch=True,
            on_step=False,
        )
        return {"loss": loss, "y_hat": y_hat}
