from typing import Literal
from lightning import Callback, Trainer
from matplotlib import axes
import numpy as np
import torch.nn as nn
import torch

from lib.explainers import IntegratedGradientsExplainer

Kind = Literal["train", "val", "test", "sanity"]


class ConfounderExplanationPlotCallback(Callback):
    def __init__(self, plot_known_start: bool = False) -> None:
        super().__init__()

        self.plot_known_start = plot_known_start

    def compute_ig_attribution(
        self, xs: torch.Tensor, model: nn.Module, label: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        model.train()
        xs = xs.requires_grad_().to(xs.device)
        pred = model(xs)
        explainer = IntegratedGradientsExplainer(normalize=True)
        norm_saliencies = explainer.attribute(model, xs, pred, true_y=label)
        attrib = norm_saliencies.detach().cpu().numpy()
        # pred = self.inverse_normalization(pred)
        model.eval()
        return pred.detach().cpu().numpy(), attrib

    def compute_conf_idxs(self, expl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        diffs = np.diff((expl.astype(int) == 1).astype(int), axis=-1)
        starts = np.argwhere(diffs == 1) + 1
        if expl[0]:
            starts = np.concatenate([np.array([[0]]), starts])

        stops = np.argwhere(diffs == -1)

        if isinstance(starts, int):
            starts = np.array([starts])
            stops = np.array([stops])

        starts = starts.flatten()
        stops = stops.flatten()

        return starts, stops

    def draw_confounder_region(
        self,
        x: np.ndarray,
        ax: axes.Axes,
        expl: np.ndarray,
        row_idx: int = 0,
        is_train: bool = True,
    ):
        starts, stops = self.compute_conf_idxs(expl)

        split_mode = len(starts) > 1

        for cnt, (start, stop) in enumerate(zip(starts, stops)):
            title = "Confounder"
            if not is_train:
                title += " (Train)"
            if cnt != 1 and split_mode:
                title = None

            if row_idx != 0:
                title = None

            self.draw_region(title, start, stop, ax, facecolor="gray")

        # check if x has a inital value region of 3 identical values
        # Check if the first three numbers are non-zero and identical
        # This is the KNOWN_START case
        if x[0] != 0 and x[1] != 0 and x[2] != 0 and self.plot_known_start:
            if x[0] == x[1] == x[2]:
                self.draw_region("$Hint_{pos}$", 0, 2, ax, facecolor="orange")

    def draw_region(
        self,
        region_title: str | None,
        roi_start: int,
        roi_end: int,
        ax: axes.Axes,
        facecolor: str = "blue",
        title_top_padding: float = 0.30,
    ) -> None:
        ax.axvspan(roi_start, roi_end, facecolor=facecolor, alpha=0.3)
        label_x = roi_start + (roi_end - roi_start) / 2
        min_y, max_y = ax.get_ylim()
        label_y = min_y  # + #(max_y - min_y) / 4
        # ax.text(label_x, label_y, 'Forecast Region', ha='center', va='center', fontsize=12)
        if region_title is not None:
            ax.annotate(
                region_title,
                xy=(label_x, label_y),
                xycoords="data",
                xytext=(label_x, max_y + title_top_padding),
                textcoords="data",
                # arrowprops=dict(arrowstyle="->", lw=1.5),
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    edgecolor="gray",
                    lw=1.5,
                ),
                ha="center",
                fontsize=9,
            )

    def plot_piecewise_attribution(
        self,
        t: np.ndarray,
        x: np.ndarray,
        cmap,
        attrib,
        ax: axes,
        label: str = "Attribution",
    ):
        scaled_attrib = (attrib.copy() + 1) / 2  

        for k in range(len(t) - 1):
            ax.plot(
                [t[k], t[k + 1]],
                [x[k], x[k + 1]],
                c=cmap(scaled_attrib[k]),
                linewidth=2,
                label=label if k == 0 else None,
            )


    def should_log(self, trainer: Trainer) -> bool:
        return (
            trainer.current_epoch % self.log_interval == 0
            or ((trainer.max_epochs or 1000) - 1) == trainer.current_epoch
        )
