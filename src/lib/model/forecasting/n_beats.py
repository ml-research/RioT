import torch.nn as nn
from .forecasting_system import ForecastingModelSystem
from lib.loss import Right_Reason_Loss
from darts.models.forecasting.nbeats import _NBEATSModule as DartsNBEATS


class NBEATS(ForecastingModelSystem):
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

        num_stacks = 30

        layer_widths = [256] * num_stacks

        self.model = DartsNBEATS(
            input_dim=1,
            output_dim=1,
            nr_params=1,
            generic_architecture=True,
            num_stacks=num_stacks,
            num_blocks=1,
            num_layers=4,
            layer_widths=layer_widths,
            expansion_coefficient_dim=5,
            trend_polynomial_degree=2,
            dropout=0.0,
            activation="ReLU",
            batch_norm=False,
            input_chunk_length=lookback,
            output_chunk_length=prediction_horizon,
        )

    def forward(self, x):
        # darts samples are a tuple
        darts_x = x.transpose(1, 2), None  # batch, seq, var
        x = self.model(darts_x)
        # darts returns (batch,prediction_horizon_target,output_dim, nr_params)
        x = x.squeeze(-1)  # remove nr_params dimension
        return x.transpose(1, 2)
