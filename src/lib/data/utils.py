from typing import Any
import torch
import numpy as np

from .data_enums import Confounder
from torchchronos.transforms import Transform

def print_ratios(dataset_name: str, y: np.ndarray):
    print(f"Problem: {dataset_name}\n")
    ratios = np.unique(y, return_counts=True)[1] / len(y)
    final_str = "Num data points: " + str(len(y)) + "\n"
    for i, ratio in enumerate(ratios):
        final_str += f"{', ' if i != 0 else ''}{i}: {ratio:.3f}"
    print(final_str)

def create_mask(
    data_shape: tuple[int, int, int],
    len_y: int,
    start_index: int,
    stride: int,
    num_elements_per_part: int | None = None,
) -> np.ndarray[Any, bool]:
    """
    Creates mask for a given confounder y.
    This confounder may be split into multiple parts.

    Args:
        data_shape: The shape [1 x Seq x Var] to create the mask for.
        len_y: The length of the confounder.
        start_index: The start index of the confounder.
        stride: The stride between parts of the confounder, if num_elements_per_part is None this will be ignored.
        num_elements_per_part: The number of elements per part of the confounder.

    Returns:
        The mask for the given confounder.
        [Seq x Var]
    """
    if num_elements_per_part is None:
        assert start_index + len_y <= data_shape[1], "The confounder is too long or the start index is too high"
        mask = np.zeros(data_shape)
        mask[:, start_index:start_index + len_y] = 1
    else:
        comp_stride = stride + num_elements_per_part
        assert (
            len_y % num_elements_per_part == 0
        ), "The number of ys must be divisible by the number of parts"
        mask = np.zeros(data_shape)
        for i in range(num_elements_per_part):
            mask[:, start_index + i :: comp_stride][
                :, : len_y // num_elements_per_part
            ] = 1
    mask = mask.astype(int)
    return mask.astype(bool)[0]


def make_patched_ucr_transform(num_classes: int, confounder: Confounder, confounder_ampl: float, confounder_width: int, scaler: Transform):
    """
    Creates a patched transform function that adds potential confounders and generates feedback w.r.t the input data based on the given parameters.

    Args:
        num_classes (int): The number of classes in the dataset.
        confounder (Confounder): The type of confounder to apply.
        confounder_ampl (float): The amplitude of the confounder.
        confounder_width (int): The width of the confounder.
        scaler (Transform): The scaler to apply to the input data.

    Returns:
        patched_transform (function): The patched transform function that applies modifications to the input data.

    """
    def patched_transform(label: int, x: torch.Tensor):
        pre = scaler(x).T

        expl_penalty_time = None
        expl_penalty_freq = None

        if Confounder.SANITY in confounder:
            idx_freq = np.random.randint(0, num_classes)

            if Confounder.CLASSIFICATION_FREQ in confounder:
                t = torch.arange(0, 1.0, 1 / pre.shape[-1])
                expl_penalty_ts = (
                    torch.sin(2.0 * torch.pi * 2 * (idx_freq + 1 + 0) * t)
                    * confounder_ampl
                ).unsqueeze(0)
                pre += expl_penalty_ts

            if Confounder.CLASSIFICATION_TIME in confounder:
                pre[:, 0:confounder_width] = (
                    torch.sin(
                        torch.linspace(0, 1.0, confounder_width)
                        * (2.0 + idx_freq)
                        * np.pi
                    )
                    + 1
                ) / 4

            if (
                Confounder.CLASSIFICATION_TIME not in confounder
                and Confounder.CLASSIFICATION_FREQ not in Confounder
            ):
                pre[:, 0:confounder_width] = 0

        else:
            if Confounder.CLASSIFICATION_TIME in confounder and Confounder.CLASSIFICATION_FREQ in confounder:
                
                t = torch.arange(0, 1.0, 1 / pre.shape[-1])
                # expl_penalty_ts = (
                #     (label / 8 + 0.1) * torch.sin(2 * torch.pi * 10 * t) / 8
                # ).unsqueeze(0)
                expl_penalty_ts = (
                    torch.sin(2.0 * torch.pi * 2 * (label + 1+0) * t)
                    * confounder_ampl
                ).unsqueeze(0)

                pre += expl_penalty_ts

                if num_classes == 2:
                    expl_penalty_ts += (
                        torch.sin(2.0 * torch.pi * 2  * (((label + 1) % 2) + 1+0) * t)
                        * confounder_ampl
                    ).unsqueeze(0)

                expl_penalty_freq = torch.fft.rfft(
                    expl_penalty_ts  # , norm="backward"
                )

                expl_penalty_time = torch.zeros_like(pre)
                pre[:, 0:confounder_width] = (
                    torch.sin(
                        torch.linspace(0, 1.0, confounder_width)
                        * (2.0 + label)
                        * np.pi
                    )
                    + 1
                ) / 4
                expl_penalty_time[:, 0:confounder_width] = 1
                expl_penalty_time = expl_penalty_time.T

            elif Confounder.CLASSIFICATION_TIME in confounder:
                expl_penalty_time = torch.zeros_like(pre)
                pre[:, 0:confounder_width] = (
                    torch.sin(
                        torch.linspace(0, 1.0, confounder_width)
                        * (2.0 + label)
                        * np.pi
                    )
                    + 1
                ) / 4
                expl_penalty_time[:, 0:confounder_width] = 1
                expl_penalty_time = expl_penalty_time.T

            elif Confounder.CLASSIFICATION_FREQ in confounder:
                t = torch.arange(0, 1.0, 1 / pre.shape[-1])
                # expl_penalty_ts = (
                #     (label / 8 + 0.1) * torch.sin(2 * torch.pi * 10 * t) / 8
                # ).unsqueeze(0)
                expl_penalty_ts = (
                    torch.sin(2.0 * torch.pi * 2 * (label + 1+0) * t)
                    * confounder_ampl
                ).unsqueeze(0)

                pre += expl_penalty_ts

                if num_classes == 2:
                    expl_penalty_ts += (
                        torch.sin(2.0 * torch.pi * 2  * (((label + 1) % 2) + 1+0) * t)
                        * confounder_ampl
                    ).unsqueeze(0)

                expl_penalty_freq = torch.fft.rfft(
                    expl_penalty_ts  # , norm="backward"
                )  # TODO only unvaiariate

        return pre, expl_penalty_time, expl_penalty_freq
    return patched_transform