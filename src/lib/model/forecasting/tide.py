import torch.nn as nn
from .forecasting_system import ForecastingModelSystem
from lib.loss import Right_Reason_Loss
from darts.models.forecasting.tide_model import _TideModule as DartsTiDE


class TiDE(ForecastingModelSystem):
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

        self.model = DartsTiDE(
            input_dim=1,
            output_dim=1,
            future_cov_dim=0,
            static_cov_dim=0,
            nr_params=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            decoder_output_dim=16,
            hidden_size=128,
            temporal_decoder_hidden=4,
            temporal_width_past=4,
            temporal_width_future=32,
            use_layer_norm=False,
            dropout=0.1,
            input_chunk_length=lookback,
            output_chunk_length=prediction_horizon,
        )

    def forward(self, x):
        # darts samples are made of (past_target, past_covariates, future_cov)
        darts_x = x.transpose(1, 2), None, None
        x = self.model(darts_x)
        # darts returns (batch,prediction_horizon_target,output_dim, nr_params)
        x = x.squeeze(-1) # remove nr_params dimension
        return x.transpose(1,2)
