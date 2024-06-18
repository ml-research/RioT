from typing import Any, Callable, Iterator, Literal, cast
from lightning.pytorch.utilities.types import STEP_OUTPUT

from torch import nn
from lib.loss import (
    Forecasting_Right_Reason_Loss,
    RRRIGLoss,
    RRRFIGLoss,
    RRRFTIGLoss,
    Classification_Right_Reason_Loss
)
from torchmetrics.classification import Accuracy, F1Score, Recall, Precision
from ..system import ModelSystem

from lib.data.dataset import ExplainedItem


class ClassificationModelSystem(ModelSystem):
    def __init__(
        self,
        right_answer_loss: nn.Module,
        right_reason_loss: Classification_Right_Reason_Loss | None = None,
        lambda_time: float = 1.0,
        lambda_freq: float = 1.0,
        num_classes: int = 2,
    ) -> None:
        if isinstance(right_reason_loss, Forecasting_Right_Reason_Loss):
            raise ValueError(
                "Forecasting right reason loss passed to classification system"
            )

        
        task = "multiclass"

        super().__init__(
            metrics=[
                Accuracy(num_classes=num_classes, task=task),
                F1Score(num_classes=num_classes, task=task),
                Recall(num_classes=num_classes, task=task),
                Precision(num_classes=num_classes, task=task),
            ],
            right_answer_loss=right_answer_loss,
            right_reason_loss=right_reason_loss,
            lambda_time=lambda_time,
            lambda_freq=lambda_freq,
        )
        self.num_classes = num_classes
        if right_reason_loss is not None:
            if isinstance(right_reason_loss, RRRFIGLoss):
                right_reason_loss.is_binary_classification = num_classes == 2
            elif isinstance(right_reason_loss, RRRFTIGLoss):
                right_reason_loss.freq.is_binary_classification = num_classes == 2
        

        self.rrr_loss = cast(Classification_Right_Reason_Loss, right_reason_loss)

    def training_step(self, item: ExplainedItem) -> STEP_OUTPUT:
        x, y, expl_p = item.x, item.y, item.expl_x
        x.requires_grad_()
        y_hat = self(x)

        loss = self.right_answer_loss(y_hat, y)
        right_reason_loss_time = None
        right_reason_loss_freq = None

        if self.applies_xil:
            if expl_p is None:
                raise ValueError("Right reason loss passed but no expl_p given")
            if isinstance(self.rrr_loss, RRRFTIGLoss):
                right_reason_loss_time, right_reason_loss_freq = self.rrr_loss(
                    self, x, y_hat, expl_p, y
                )
            elif isinstance(self.rrr_loss, RRRFIGLoss):
                right_reason_loss_freq = self.rrr_loss(self, x, y_hat, expl_p.freq, y)
            elif isinstance(self.rrr_loss, RRRIGLoss):
                right_reason_loss_time = self.rrr_loss(self, x, y_hat, expl_p.time, y)

        batch_size = x.shape[0]
        right_reason_loss_time = self.log_optional_loss("train", "rrr_loss_time", right_reason_loss_time, batch_size)
        right_reason_loss_freq = self.log_optional_loss("train", "rrr_loss_freq", right_reason_loss_freq, batch_size)
            
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            batch_size=x.shape[0],
            on_epoch=True,
            on_step=True,
        )

        total_loss = (
            loss + right_reason_loss_time * self.lambda_time + right_reason_loss_freq * self.lambda_freq
        )
        self.log("train/total_loss", total_loss, prog_bar=True, batch_size=x.shape[0])

        self.log_dict(
            self.train_metrics(y_hat, y),
            on_epoch=True,
            on_step=True,
            batch_size=x.shape[0],
        )

        return {"loss": total_loss, "y_hat": y_hat}

