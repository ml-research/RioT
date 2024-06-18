from .forecasting_system import ForecastingModelSystem
import torch.nn as nn
from lib.loss import Right_Reason_Loss
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel




def make_linear_layer(dim_in, dim_out):
    lin = nn.Linear(dim_in, dim_out)
    torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
    torch.nn.init.zeros_(lin.bias)
    return lin


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)  # type: ignore

    @staticmethod
    def _init_weight(out: torch.Tensor) -> torch.Tensor:
        """
        Features are not interleaved. The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        # set early to avoid an error in pytorch-1.8+
        out.requires_grad = False

        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(  # type: ignore
        self, input_ids_shape: torch.Size, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen x ...]."""
        _, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


# adapted from https://arxiv.org/pdf/2211.14730.pdf + gluonts implementation
class PatchTSTModel(nn.Module):
    """
    Module implementing a deterministic version of the PatchTST model for forecasting.

    Parameters
    ----------
    prediction_length : int
        Number of time points to predict.
    context_length : int
        Number of time steps prior to prediction time that the model uses as context.
    patch_len : int
        The length of each patch extracted from the input sequence.
    stride : int
        The stride with which patches are extracted.
    padding_patch : str
        Type of padding applied to patches. Options include 'end' for padding at the end.
    d_model : int
        The dimension of the input and output of the transformer model.
    nhead : int
        The number of heads in the multiheadattention models.
    dim_feedforward : int
        The dimension of the feedforward network model.
    dropout : float
        The dropout value.
    activation : str
        The activation function of the intermediate layer.
    norm_first : bool
        If True, encoder and decoder normalize before attention and feedforward operations. Otherwise, after.
    num_encoder_layers : int
        The number of sub-encoder-layers in the encoder.
    scaling : str
        The type of scaling to apply to the input data. Options include 'mean', 'std', and 'none'.


    Parameters:
    A.1.4 MODEL PARAMETERS
    By default, PatchTST contains 3 encoder layers with head number H = 16 and dimension of latent
    space D = 128. The feed forward network in Transformer encoder block consists of 2 linear
    layers with GELU (Hendrycks & Gimpel, 2016) activation function: one projecting the hidden
    representation D = 128 to a new dimension F = 256, and another layer that project it back to
    D = 128. For very small datasets (ILI, ETTh1, ETTh2), a reduced size of parameters is used
    (H = 4, D = 16 and F = 128) to mitigate the possible overfitting. Dropout with probability 0.2 is
    applied in the encoders for all experiments. The code will be publicly available.
    """

    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        patch_len: int,  # authors claim that patch length is pretty robust and values between 8 to 16 are a good choice
        stride: int = 8,  # stride is choosen to be the same length as the patch length to get non overlapping windows
        padding_patch: str = "other", #"end",
        d_model: int = 16,  # gluon 32,
        nhead: int = 4,  # gluon was 4,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        activation: nn.Module =  F.gelu,
        norm_first: bool = False,
        num_encoder_layers: int = 3,  # gluon default value was 2
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.padding_patch = padding_patch

        self.patch_num = int((context_length - patch_len) / stride + 1)
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            self.patch_num += 1

        self.patch_proj = make_linear_layer(patch_len, d_model)

        self.positional_encoding = SinusoidalPositionalEmbedding(
            self.patch_num, d_model
        )

        layer_norm_eps: float = 1e-5
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self.flatten = nn.Linear(d_model * self.patch_num, prediction_length * d_model)

        # Adjusted to directly predict the target values
        self.final_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Do patching
        if self.padding_patch == "end":
            x = self.padding_patch_layer(x)
        past_target_patches = x.unfold(
            dimension=1, size=self.patch_len, step=self.stride
        )

        # Project patches
        enc_in = self.patch_proj(past_target_patches)
        embed_pos = self.positional_encoding(enc_in.size())

        # https://github.com/pytorch/pytorch/issues/116350
        # Transformer encoder with positional encoding
        with sdpa_kernel(SDPBackend.MATH):
            enc_out = self.encoder(enc_in + embed_pos)

        # Flatten and project to prediction length
        flatten_out = self.flatten(enc_out.flatten(start_dim=1))

        # Final projection to output predictions
        predictions = self.final_proj(
            flatten_out.reshape(-1, self.prediction_length, self.d_model)
        ).squeeze(-1)

        return predictions


class PatchTST(ForecastingModelSystem):
    def __init__(
        self,
        right_answer_loss: nn.Module,
        lookback: int,
        prediction_horizon: int,
        lambda_time: float,
        lambda_freq: float,
        patch_len: int = 8,
        stride: int = 8,
        padding_patch: str = "other",
        d_model: int = 16,
        nhead: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        num_encoder_layers: int = 3,
        right_reason_loss: Right_Reason_Loss | None = None,
    ):
        super().__init__(
            right_answer_loss=right_answer_loss,
            right_reason_loss=right_reason_loss,
            lambda_time=lambda_time,
            lambda_freq=lambda_freq,
        )

        self.model = PatchTSTModel(
            prediction_length=prediction_horizon,
            context_length=lookback,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            # norm_first=False,
            num_encoder_layers=num_encoder_layers,
        )

    # def on_validation_epoch_end(self) -> None:
    #     if self.trainer.current_epoch == 70:
    #         self.lambda_time = 0
        
    #     return super().on_validation_epoch_end()

    def forward(self, x):
        x = self.model(x.squeeze(1))
        return x.unsqueeze(1)
