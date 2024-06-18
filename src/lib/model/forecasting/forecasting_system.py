from typing import Any, Callable, Iterator, Literal, cast
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from torch import Tensor, nn
from lib.loss import (
    Classification_Right_Reason_Loss,
    RRRLoss,
    HorizonRRRIGLoss,
    Forecasting_Right_Reason_Loss,
)
from torchmetrics.regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    SymmetricMeanAbsolutePercentageError,
)

from lib.loss.rrr_loss import HorizonRRRFIGLoss, HorizonRRRFTIGLoss
from ..system import ModelSystem

from lib.data.dataset import ExplainedItem


class ForecastingModelSystem(ModelSystem):
    def __init__(
        self,
        right_answer_loss: nn.Module,
        right_reason_loss: Forecasting_Right_Reason_Loss | None = None,
        lambda_time: float = 1.0,
        lambda_freq: float = 1.0,
    ) -> None:
        if isinstance(right_reason_loss, Classification_Right_Reason_Loss):
            raise ValueError(
                "Classification right reason loss passed to forecasting system"
            )
        super().__init__(
            metrics=[
                MeanSquaredError(),
                MeanAbsoluteError(),
                SymmetricMeanAbsolutePercentageError(),
            ],
            right_answer_loss=right_answer_loss,
            right_reason_loss=right_reason_loss,
            lambda_time=lambda_time,
            lambda_freq=lambda_freq,
        )
        self.rrr_loss = cast(Forecasting_Right_Reason_Loss, right_reason_loss)

    def forward(self, x: Tensor):
        # this is not the model, this i called from the child class
        return x.unsqueeze(1)

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
            if isinstance(self.rrr_loss, HorizonRRRFTIGLoss):
                right_reason_loss_time, right_reason_loss_freq = self.rrr_loss(
                    self, x, y_hat, expl_p
                )
            elif isinstance(self.rrr_loss, HorizonRRRFIGLoss):
                right_reason_loss_freq = self.rrr_loss(self, x, y_hat, expl_p.freq)
            elif isinstance(self.rrr_loss, HorizonRRRIGLoss):
                right_reason_loss_time = self.rrr_loss(self, x, y_hat, expl_p.time)

        batch_size = x.shape[0]
        right_reason_loss_time = self.log_optional_loss(
            "train", "rrr_loss_time", right_reason_loss_time, batch_size
        )
        right_reason_loss_freq = self.log_optional_loss(
            "train", "rrr_loss_freq", right_reason_loss_freq, batch_size
        )

        self.log("train/loss", loss, prog_bar=True, batch_size=x.shape[0])

        total_loss = (
            loss
            + right_reason_loss_time * self.lambda_time
            + right_reason_loss_freq * self.lambda_freq
        )
        self.log("train/total_loss", total_loss, prog_bar=True, batch_size=x.shape[0])

        self.log_dict(
            self.train_metrics(y_hat, y),
            on_epoch=True,
            on_step=True,
            batch_size=x.shape[0],
        )

        return total_loss

    
