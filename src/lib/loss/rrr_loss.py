import torch
import torch.nn as nn

from ..explainers import (
    HorizonSailiencyExplainer,
    HorizonIntegratedGradientsExplainer,
    HorizonFrequencyIntegratedGradientsExplainer,
    IntegratedGradientsExplainer,
    FrequencyIntegratedGradientsExplainer,
)
from ..data.dataset import FeedbackPenalty


class RRRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.method = HorizonSailiencyExplainer(False)

    def forward(
        self, input: torch.Tensor, predictions: torch.Tensor, expl_p: torch.Tensor
    ):
        predictions.retain_grad()

        attribution = self.method.attribute(
            input, predictions, lambda A_gradX: (expl_p * A_gradX) ** 2
        )

        right_reason_loss = torch.sum(attribution)

        return right_reason_loss


class RRRIGLoss(RRRLoss):
    def __init__(self) -> None:
        super().__init__()
        self.method = IntegratedGradientsExplainer(False)

    def forward(
        self,
        forward_fn,
        input: torch.Tensor,
        predictions: torch.Tensor,
        expl_p: torch.Tensor,
        true_y: torch.Tensor,
    ):
        attribution = self.method.attribute(
            forward_fn,
            input,
            predictions,
            lambda sals: (expl_p * sals) ** 2,
            true_y=true_y,
        )

        right_reason_loss = torch.sum(attribution) / attribution.shape[0]

        return right_reason_loss


class HorizonRRRIGLoss(RRRLoss):
    def __init__(self) -> None:
        super().__init__()
        self.method = HorizonIntegratedGradientsExplainer(False)

    def forward(
        self,
        forward_fn,
        input: torch.Tensor,
        predictions: torch.Tensor,
        expl_p: torch.Tensor,
    ):
        predictions.retain_grad()
        attribution = self.method.attribute(
            forward_fn, input, predictions, lambda sals: (expl_p * sals) ** 2
        )

        right_reason_loss = torch.sum(attribution) / attribution.shape[0]

        return right_reason_loss


def set_diff_tensors(a: torch.Tensor, b: torch.Tensor):
    uniques, counts = torch.cat((a, b)).unique(return_counts=True)
    return uniques[counts == 1]

def handle_invalid_feedback(values, idxs: torch.Tensor, threshold: float):
    invalid_feedback_idxs = torch.where(values < threshold)[0].unique()
    if len(invalid_feedback_idxs) > 0:
        idxs_idxs = set_diff_tensors(torch.arange(len(idxs), device=invalid_feedback_idxs.device), invalid_feedback_idxs)
        idxs = idxs[idxs_idxs]
    return idxs, invalid_feedback_idxs

    

def compute_freq_loss(loss: nn.Module, fft_attrib: torch.Tensor, top_k_imag,top_k_imag_values: torch.Tensor, top_k_real:torch.Tensor, top_k_real_values:torch.Tensor, threshold:float, target_value: float, device: torch.device, dtype: torch.dtype):
    top_k_imag, imag_invalid_feedback_idxs = handle_invalid_feedback(top_k_imag_values, top_k_imag, threshold)
    top_k_real, real_invalid_feedback_idxs = handle_invalid_feedback(top_k_real_values, top_k_real, threshold)


    fft_attrib_idxs = torch.arange(len(fft_attrib), device=fft_attrib.device)        

    imag_loss = torch.tensor(0.0, dtype=dtype, device=device)
    real_loss = torch.tensor(0.0, dtype=dtype, device=device)
    if len(top_k_real) != 0:
        fft_attrib_idxs = set_diff_tensors(fft_attrib_idxs, real_invalid_feedback_idxs)
        fft_attrib_real = fft_attrib[fft_attrib_idxs]
        real = torch.gather(fft_attrib_real, 2, top_k_real).real.float()

        real_loss = loss(
            real,
            torch.ones_like(real) * target_value,
        )
        real_loss /= fft_attrib_real.shape[0]

    if len(top_k_imag) != 0:
        top_k_imag = top_k_imag.long()

        fft_attrib_idxs = set_diff_tensors(fft_attrib_idxs, imag_invalid_feedback_idxs)
        fft_attrib_imag = fft_attrib[fft_attrib_idxs]
        imag = torch.gather(fft_attrib_imag, 2, top_k_imag).imag.float()

        imag_loss = loss(
            imag,
            torch.ones_like(imag) * target_value,
        )

        imag_loss /= fft_attrib_imag.shape[0]

    
    return real_loss + imag_loss



class HorizonRRRFIGLoss(HorizonRRRIGLoss):
    # class HorizonRRRFIGLoss(RRRFIGLoss):
    def __init__(self, k: int = 1, target_value: float = 0.0) -> None:
        super().__init__()
        self.method = HorizonFrequencyIntegratedGradientsExplainer(False)
        self.k = k
        self.target_value = target_value
        self.loss = nn.MSELoss()
        # threshold which is used to filter out "invalid feedback",
        # necessary for the feedback percentage scaling experiment
        self.threshold = 1e-3

    def forward(
        self,
        forward_fn,
        input: torch.Tensor,
        predictions: torch.Tensor,
        expl_p: torch.Tensor,
    ):
        predictions.retain_grad()
        fft_attrib = self.method.attribute(forward_fn, input, predictions)

        top_imag_values, top_k_imag = torch.topk(
            torch.abs(expl_p.imag), k=self.k, dim=-1
        )
        top_real_values, top_k_real = torch.topk(
            torch.abs(expl_p.real), k=self.k, dim=-1
        )


        return compute_freq_loss(self.loss, fft_attrib, top_k_imag,top_imag_values, top_k_real, top_real_values, self.threshold, self.target_value, input.device, input.dtype)


class RRRFIGLoss(RRRIGLoss):
    def __init__(self, target_value: float = 0.0) -> None:
        super().__init__()
        self.method = FrequencyIntegratedGradientsExplainer(False)
        self.target_value = target_value
        self.loss = nn.MSELoss()

        self.is_binary_classification = False
        # threshold which is used to filter out "invalid feedback",
        # necessary for the feedback percentage scaling experiment
        self.threshold = 1e-3

    def _forward(
        self,
        input: torch.Tensor,
        fft_attrib: torch.Tensor,
        k: int,
        expl_p: torch.Tensor,
    ) -> torch.Tensor:
        
        top_imag_values, top_k_imag = torch.topk(
            torch.abs(expl_p.imag.transpose(1, 2)), k=k, dim=-1
        )
        top_real_values, top_k_real = torch.topk(
            torch.abs(expl_p.real.transpose(1, 2)), k=k, dim=-1
        )


        return compute_freq_loss(self.loss, fft_attrib, top_k_imag,top_imag_values, top_k_real, top_real_values, self.threshold, self.target_value, input.device, input.dtype)

        

    def forward(
        self,
        forward_fn,
        input: torch.Tensor,
        predictions: torch.Tensor,
        expl_p: torch.Tensor,
        true_y: torch.Tensor,
    ):
        predictions.retain_grad()
        fft_attribution = self.method.attribute(
            forward_fn, input, predictions, true_y=true_y
        )

        # find top k indices
        k = 2 if self.is_binary_classification else 1
        # for the binary case we apply feedback for two classes at once

        return self._forward(input, fft_attribution, k, expl_p)


class RRRFTIGLoss(RRRIGLoss):
    # TODO make method configurable
    def __init__(self, target_value: float = 0) -> None:
        super().__init__()

        self.freq = RRRFIGLoss(target_value=target_value)
        self.time = RRRIGLoss()

    def forward(
        self,
        forward_fn,
        input: torch.Tensor,
        predictions: torch.Tensor,
        expl_p: FeedbackPenalty,
        true_y: torch.Tensor,
    ):
        time = torch.tensor(0.0, dtype=input.dtype, device=input.device)
        freq = torch.tensor(0.0, dtype=input.dtype, device=input.device)
        if expl_p.time is not None:
            time = self.time(forward_fn, input, predictions, expl_p.time, true_y=true_y)
        if expl_p.freq is not None:
            freq = self.freq(forward_fn, input, predictions, expl_p.freq, true_y=true_y)

        return time, freq


class HorizonRRRFTIGLoss(RRRIGLoss):
    # TODO make method configurable
    def __init__(self, target_value: float = 0, k: int = 1) -> None:
        super().__init__()

        self.freq = HorizonRRRFIGLoss(target_value=target_value, k=k)
        self.time = HorizonRRRIGLoss()

    def forward(
        self,
        forward_fn,
        input: torch.Tensor,
        predictions: torch.Tensor,
        expl_p: FeedbackPenalty,
    ):
        time = torch.tensor(0.0, dtype=input.dtype, device=input.device)
        freq = torch.tensor(0.0, dtype=input.dtype, device=input.device)
        if expl_p.time is not None:
            time = self.time(forward_fn, input, predictions, expl_p.time)
        if expl_p.freq is not None:
            freq = self.freq(forward_fn, input, predictions, expl_p.freq)

        return time, freq
