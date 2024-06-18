import torch.nn as nn
from .forecasting_system import ForecastingModelSystem
from lib.loss import Right_Reason_Loss


class LinearModel(ForecastingModelSystem):
    def __init__(
        self,
        right_answer_loss: nn.Module,
        lookback: int,
        prediction_horizon: int,
        lambda_time: float,
        lambda_freq: float,
        right_reason_loss: Right_Reason_Loss | None = None,
    ):
        super().__init__(
            right_answer_loss=right_answer_loss,
            right_reason_loss=right_reason_loss,
            lambda_time=lambda_time,
            lambda_freq=lambda_freq,
        )
        self.fc = nn.Linear(lookback, prediction_horizon)
        self.ft = nn.Flatten()

    def forward(self, x):
        x = self.ft(x)
        x = self.fc(x)
        return super().forward(x)


class MLPModel(ForecastingModelSystem):
    def __init__(
        self,
        right_answer_loss: nn.Module,
        lookback: int,
        prediction_horizon: int,
        lambda_time: float,
        lambda_freq: float,
        right_reason_loss: Right_Reason_Loss | None = None,
    ):
        super().__init__(
            right_answer_loss=right_answer_loss,
            right_reason_loss=right_reason_loss,
            lambda_time=lambda_time,
            lambda_freq=lambda_freq,
        )
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, prediction_horizon),
        )

    def forward(self, x):
        x = self.model(x)
        return super().forward(x)
