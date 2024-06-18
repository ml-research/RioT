from typing import Callable
import torch
from captum.attr import IntegratedGradients
import torch
from torch.autograd import grad

from captum._utils.typing import (
    TargetType,
)
from captum._utils.common import _run_forward
from typing import Any, Callable, Union, Tuple
from torch import Tensor
import torch


# This is the same as the default compute_gradients
# function in captum._utils.gradient, except
# setting create_graph=True when calling
# torch.autograd.grad
def compute_gradients(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
) -> Tuple[Tensor, ...]:
    r"""
    Computes gradients of the output with respect to inputs for an
    arbitrary forward function.

    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        input:      Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        additional_forward_args: Additional input arguments that forward
                    function requires. It takes an empty tuple (no additional
                    arguments) if no additional arguments are required
    """
    with torch.backends.cudnn.flags(enabled=False):  # type: ignore
        with torch.autograd.set_grad_enabled(True):  # type: ignore
            # runs forward pass
            outputs = _run_forward(
                forward_fn, inputs, target_ind, additional_forward_args
            )
            assert outputs[0].numel() == 1, (
                "Target not provided when necessary, cannot"
                " take gradient with respect to multiple outputs."
            )
            # torch.unbind(forward_out) is a list of scalar tensor tuples and
            # contains batch_size * #steps elements
            grads = torch.autograd.grad(
                torch.unbind(outputs), inputs, create_graph=True
            )
    return grads


class HorizonIntegratedGradientsExplainer2:
    def __init__(self, normalize: bool = True, n_steps=50) -> None:
        self.normalize = normalize
        self.relu = torch.nn.ReLU()
        self.n_steps = n_steps

    def attribute(
        self,
        forward_fn,
        input: torch.Tensor,
        predictions: torch.Tensor,
        horizon_tf: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        baselines: torch.Tensor | None = None,
    ):
        if baselines == None:
            baselines = torch.zeros_like(input=input)

        # k/m in the formula
        alphas = torch.linspace(0, 1, self.n_steps).tolist()

        # direct path from baseline to input. shape : ([n_steps, n_features], )
        scaled_features = tuple(
            torch.stack(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(input, baselines)
        )
        preds = forward_fn(scaled_features[0])  # -> (50 x 5)
        u_preds = torch.unbind(preds.flatten())  # -> 250
        # u_preds = torch.unbind(torch.mean(preds, dim=-1,keepdim=True).unsqueeze(-1))
        grads = grad(
            outputs=u_preds,
            inputs=scaled_features,
            create_graph=True,
            allow_unused=True,
        )

        return horizon_tf(grads[0].mean(0) * self.relu(input)) * 1000


class HorizonIntegratedGradientsExplainer:
    def __init__(self, normalize: bool = True, n_steps=50) -> None:
        self.normalize = normalize
        self.relu = torch.nn.ReLU()
        self.n_steps = n_steps

    def attribute(
        self,
        forward_fn,
        input: torch.Tensor,
        predictions: torch.Tensor,
        horizon_tf: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        baselines: torch.Tensor | None = None,
    ):
        if baselines == None:
            baselines = torch.zeros_like(input=input)

        def wrapper(x):
            return forward_fn(x).mean(-1, keepdim=True).unsqueeze(-1)

        method = IntegratedGradients(wrapper, multiply_by_inputs=False)

        method.gradient_func = compute_gradients

        attr = method.attribute(
            input, baselines=baselines, target=0, n_steps=self.n_steps
        )

        attr = horizon_tf(attr * torch.abs(input))

        if self.normalize:
            attr_min = torch.min(attr, dim=-1, keepdim=True).values
            attr_max = torch.max(attr, dim=-1, keepdim=True).values
            attr_range = attr_max - attr_min
            # check if A_range (b, var, 1) is close to 0
            if torch.allclose(attr_range, torch.zeros_like(attr_range)):
                raise ValueError("A_range is close to 0")

            attr = (attr - attr_min) / attr_range

        return attr


class HorizonFrequencyIntegratedGradientsExplainer(HorizonIntegratedGradientsExplainer):
    def __init__(self, normalize: bool = True, n_steps=50) -> None:
        super().__init__(normalize, n_steps)

    def attribute(
        self,
        forward_fn,
        input: torch.Tensor,
        predictions: torch.Tensor,
        baselines: torch.Tensor | None = None,
    ):
        attr = super().attribute(forward_fn, input, predictions, lambda x: x, baselines)

        attr_fft = torch.fft.rfft(attr, dim=-1, norm="backward")

        return attr_fft


class IntegratedGradientsExplainer:
    def __init__(self, normalize: bool = True, n_steps=5) -> None:
        self.normalize = normalize
        self.relu = torch.nn.ReLU()
        self.n_steps = n_steps

    def attribute(
        self,
        forward_fn,
        input: torch.Tensor,
        predictions: torch.Tensor,
        tf: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        baselines: torch.Tensor | None = None,
        true_y: torch.Tensor | None = None,
    ):
        if baselines == None:
            baselines = torch.zeros_like(input=input)

        method = IntegratedGradients(forward_fn, multiply_by_inputs=False)

        method.gradient_func = compute_gradients

        if true_y is None:
            attr_lst = [
                method.attribute(
                    input, baselines=baselines, target=index, n_steps=self.n_steps
                )
                for index in range(predictions.shape[-1])
            ]  # optimize

            attr = torch.mean(torch.relu(torch.stack(attr_lst)), dim=0)
        else:
            attr = method.attribute(
                input,
                baselines=baselines,
                target=true_y,
                n_steps=self.n_steps,
            )

        attr_tf = tf(attr * torch.abs(input))

        if self.normalize:
            attr_min = torch.min(attr_tf, dim=-1, keepdim=True).values
            attr_max = torch.max(attr_tf, dim=-1, keepdim=True).values 
            attr_range = attr_max - attr_min 
            # check if A_range (b, var, 1) is close to 0
            if torch.allclose(attr_range, torch.zeros_like(attr_range)):
                raise ValueError("A_range is close to 0")

            # normalization [0, 1]
            # attr_tf_orig = (attr_tf - attr_min) / attr_range
            # # normalization [-1, 1]
            # attr_tf_v2 = 2 * (attr_tf - attr_min) / attr_range - torch.ones_like(attr_tf)

            # normalization [-1, 1], keeping positive and negative values and 0 = 0
            attr_max_abs = torch.max(torch.abs(attr_tf), dim=-1, keepdim=True).values 
            attr_tf = attr_tf / attr_max_abs

        return attr_tf


class FrequencyIntegratedGradientsExplainer(IntegratedGradientsExplainer):
    def __init__(self, normalize: bool = True, n_steps=5) -> None:
        super().__init__(normalize, n_steps)

    def attribute(
        self,
        forward_fn,
        input: torch.Tensor,
        predictions: torch.Tensor,
        baselines: torch.Tensor | None = None,
        true_y: torch.Tensor | None = None,
    ):
        attr = super().attribute(forward_fn, input, predictions, lambda x: x, baselines, true_y=true_y)

        attr_fft = torch.fft.rfft(attr, dim=-1, norm="backward")

        return attr_fft
