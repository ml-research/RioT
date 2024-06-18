from typing import Any, Callable, Literal, cast
import warnings
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from matplotlib.axes import Axes
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize as mplNormalize, Colormap

from captum.attr import GuidedGradCam, GradientShap, Occlusion
from pathlib import Path


from ..aim_logger import AimLogger
from ..explainers import (
    ExplanationMethod,
    HorizonSailiencyExplainer,
    HorizonIntegratedGradientsExplainer,
    HorizonFrequencyIntegratedGradientsExplainer,
)
from lib.model.forecasting import ConvModel
from lib.model import ModelSystem
from torch.utils.data import DataLoader
from ..data.dataset import ExplainedItem
from ..data import ForecastingMode
from .expl_conf_plot_callback import ConfounderExplanationPlotCallback, Kind


class ExplanationtPlot(ConfounderExplanationPlotCallback):
    """
    Callback to plot the explanation of the model.

    Args:
        inverse_normalization: Function to inverse the normalization of the data.
        explanation_method: The explanation method to use.
        target_features: The target feature(s) to explain.
        forecasting_mode: The forecasting mode.
        num_samples: The number of samples used in the subplot. The first value is the number of rows, the second value is the number of columns.
        log_interval: The interval in which the explanation is plotted.
    """

    def __init__(
        self,
        inverse_normalization: Callable[[torch.Tensor], np.ndarray],
        explanation_method: ExplanationMethod,
        target_features: str | list[str],
        source_features: list[str] | None = None,
        forecasting_mode: ForecastingMode = ForecastingMode.UNIVARIATE,
        num_samples: tuple[int, int] = (2, 2),
        log_interval: int = 15,
    ) -> None:
        super().__init__()
        self.results = []
        self.inverse_normalization = inverse_normalization
        self.num_samples_row = num_samples[0]
        self.num_samples_col = num_samples[1] 
        self.num_samples = self.num_samples_row * self.num_samples_col
        self.explanation_method = explanation_method
        self.target_feature_names = target_features
        self.forcasting_mode = forecasting_mode
        self.log_interval = log_interval
        self.source_features = source_features

    def compute_explanation(
        self,
        trainer: Trainer,
        pl_module: ModelSystem,
        data_loader: DataLoader,
        kind: Kind = "train",
    ):
        item: ExplainedItem = next(iter(data_loader))
        xs = item.x[: self.num_samples]
        xs = xs.to(pl_module.device)

        target = item.y[: self.num_samples]
        expl = item.expl_x[: self.num_samples] if item.expl_x is not None else None

        horizon_length = target.shape[-1]
        history_length = xs.shape[-1]

        target = self.inverse_normalization(target)
        match self.explanation_method:
            case ExplanationMethod.GRAD_CAM:
                pred, attrib = self.compute_gc_attribution(
                    xs, horizon_length, pl_module
                )
            case ExplanationMethod.FREQ_INTEGRATED_GRADIENTS:
                pred, attrib = self.compute_ig_attribution(xs, pl_module)
                attrib_fft = self.compute_freq_ig_attribution(xs, pl_module)
                num_freqs = xs.shape[-1]  # TODO: Why is num_freqs == sample_rate
                self.ampl_plot(
                    trainer,
                    attrib_fft,
                    self.target_feature_names,
                    xs.shape[-1],
                    num_freqs,
                    lim=3,
                    kind=kind,
                    applies_xil=pl_module.applies_xil,
                    feature_name=self.target_feature_names[0],
                )
            case ExplanationMethod.INTEGRATED_GRADIENTS:
                pred, attrib = self.compute_ig_attribution(xs, pl_module)
            case ExplanationMethod.SAILIENCY:
                pred, attrib = self.compute_saliency_attribution(xs, pl_module)
            case ExplanationMethod.GRADIENT_SHAP:

                def mean_agg(x):
                    return torch.mean(pl_module(x), dim=-1, keepdim=True)

                method = GradientShap(mean_agg)
                baselines = torch.randn(20, *xs.shape[1:], device=pl_module.device)
                attrib = method.attribute(xs, baselines=baselines, target=0)
                sal_min = torch.min(attrib, dim=0, keepdim=True).values
                sal_max = torch.max(attrib, dim=0, keepdim=True).values

                norm_saliencies = (attrib - sal_min) / (sal_max - sal_min + 1e-6)
                # attrib = (attrib - attrib.min()) / (attrib.max() - attrib.min())
                attrib = norm_saliencies.detach().cpu()
                pred = self.inverse_normalization(pl_module(xs))
            case ExplanationMethod.OCCLUSION:

                def mean_agg(x):
                    return torch.mean(pl_module(x), dim=-1, keepdim=True)

                method = Occlusion(mean_agg)
                baselines = torch.randn(20, *xs.shape[1:], device=pl_module.device)
                attrib = method.attribute(xs, target=0, sliding_window_shapes=(1, 3))
                sal_min = torch.min(attrib, dim=0, keepdim=True).values
                sal_max = torch.max(attrib, dim=0, keepdim=True).values

                norm_saliencies = (attrib - sal_min) / (sal_max - sal_min + 1e-6)
                # attrib = (attrib - attrib.min()) / (attrib.max() - attrib.min())
                attrib = norm_saliencies.detach().cpu()
                pred = self.inverse_normalization(pl_module(xs))
            case _:
                raise ValueError("Invalid explanation method")

        # Create a colormap
        colormap = "viridis"
        cmap = cm.get_cmap(colormap)

        xs = self.inverse_normalization(xs)

        t = np.arange(history_length + horizon_length)

        match self.forcasting_mode:
            case ForecastingMode.UNIVARIATE | ForecastingMode.MULTIVARIATE_MULTIVARIATE:
                concat_ys = np.concatenate((xs[..., -1:], target), axis=-1)
                concat_ys_hat = np.concatenate((xs[..., -1:], pred), axis=-1)

                for var_idx in range(xs.shape[1]):
                    if expl is not None:
                        if expl.freq is not None:
                            expl = expl.freq[:, var_idx].numpy()
                        else:
                            expl = expl.time[:, var_idx].numpy()
                    else:
                        expl = None
                    self.generate_and_upload_figure(
                        trainer,
                        pl_module.applies_xil,
                        kind,
                        xs[:, var_idx],
                        expl,
                        history_length,
                        attrib[:, var_idx],
                        colormap,
                        cmap,
                        t,
                        concat_ys[:, var_idx],
                        concat_ys_hat[:, var_idx],
                        self.target_feature_names[var_idx],
                    )
            case ForecastingMode.MULTIVARIATE_UNIVARIATE:
                raise NotImplementedError("Explanation is malforemd")
                pass  # TODO into single plot
                concat_ys = np.concatenate((xs[:, -1:, -1:], target), axis=-1)
                concat_ys_hat = np.concatenate((xs[:, -1:, -1:], pred), axis=-1)
                xs = xs[:2]
                for sample_idx in range(xs.shape[0]):
                    self.generate_and_upload_multivariate_figure(
                        trainer,
                        pl_module.applies_xil,
                        kind,
                        xs[sample_idx],
                        expl[sample_idx].numpy() if expl is not None else None,
                        history_length,
                        attrib[sample_idx],
                        colormap,
                        cmap,
                        t,
                        concat_ys[sample_idx],
                        concat_ys_hat[sample_idx],
                        cast(
                            list[str], self.source_features
                        ),  # we know that it is not none
                        self.target_feature_names[0],
                    )

    def generate_and_upload_multivariate_figure(
        self,
        trainer: Trainer,
        applies_xil: bool,
        kind: Kind,
        xs: np.ndarray,
        expls: np.ndarray | None,
        history_length: int,
        attrib: np.ndarray,
        colormap: str,
        cmap: Colormap,
        t: np.ndarray,
        concat_ys: np.ndarray,
        concat_ys_hat: np.ndarray,
        source_names: list[str],
        feature_names: str,
    ):
        fig = plt.figure(constrained_layout=True, dpi=250, figsize=(10, 8))
        fig.suptitle(f"Epoch: {trainer.current_epoch} - {kind}")

        subfigs = fig.subfigures(2, 1, height_ratios=[1, 2])
        target_ax = subfigs[0].subplots(1, 1)
        target_ax.set_title("Target", pad=20)
        target_ax.set_xlabel("Time")
        target_ax.set_ylabel(feature_names.upper())
        if feature_names == "Dayaheadprices":
            # work_around = True
            data_idx = 6
        else:
            data_idx = source_names.index(feature_names.upper())
        self.plot_piecewise_attribution(
            t[: history_length - 1], xs[data_idx], cmap, attrib[data_idx], target_ax
        )

        concat_y = concat_ys[0]
        concat_y_hat = concat_ys_hat[0]

        forecast_timesteps = t[history_length - 1 :]

        forecast_roi_start = forecast_timesteps[0]
        forecast_roi_end = forecast_timesteps[-1]

        target_ax.plot(forecast_timesteps, concat_y, c=cmap(attrib.min()), linewidth=2)

        target_ax.plot(
            forecast_timesteps,
            concat_y_hat,
            c="red",
            linewidth=2,
            linestyle="dashed",
        )
        self.draw_region(
            "Forecast",
            forecast_roi_start,
            forecast_roi_end,
            target_ax,
        )
        if expls is not None:
            self.draw_confounder_region(xs[data_idx], target_ax, expls[data_idx])

        axs = subfigs[1].subplots(
            self.num_samples_row,
            self.num_samples_col,
            sharex=True,
            # sharey=True,
        )

        subfigs[1].suptitle("Covariates")
        for i, source_feat in enumerate(source_names[:-1]):
            if feature_names == "Dayaheadprices" and i == 3:
                break
            if i == data_idx:
                raise ValueError("Invalid covariate index")
            ax = axs[i // self.num_samples_col, i % self.num_samples_col]
            self.plot_piecewise_attribution(
                t[: history_length - 1], xs[i], cmap, attrib[i], ax
            )
            if i == len(source_names) - 1:
                ax.set_xlabel("Time")
            ax.set_ylabel(source_feat)
            if expls is not None:
                self.draw_confounder_region(xs[i], ax, expls[i])
            if i == self.num_samples_row - 1:
                ax.set_xlabel("Time")

        axs[1, -1].axis("off")

        # plt.suptitle(
        # )
        # plt.tight_layout()
        # fig.tight_layout()

        cax = fig.add_axes([1.05, 0.1, 0.05, 1])

        # Create the colorbar using the ScalarMappable
        sm = cm.ScalarMappable(
            cmap=colormap,
            norm=mplNormalize(vmin=attrib.min(), vmax=attrib.max()),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, location="right")
        cbar.set_label("Attribution")

        # Show the plot or log it if trainer.logger is available
        if trainer.fast_dev_run:  # type: ignore
            Path("dev_run").mkdir(exist_ok=True)

            plt.savefig(
                f"dev_run/{kind}_expl_xil={applies_xil}_{trainer.current_epoch}.png"
            )
        elif trainer.logger:
            plt.close()

            logger: AimLogger = cast(AimLogger, trainer.logger)
            logger.log_image(
                f"{kind}/expl",
                fig,
                epoch=trainer.current_epoch,
                context={
                    "xil": applies_xil,
                    # "explanation_method": self.explanation_method.pretty_name,
                    # "confounded": expl is not None,
                    # "sample": ,
                },
            )
        else:
            plt.show()

    def generate_and_upload_figure(
        self,
        trainer: Trainer,
        applies_xil: bool,
        kind: Kind,
        xs: np.ndarray,
        expls: np.ndarray | None,
        history_length: int,
        attrib: np.ndarray,
        colormap: str,
        cmap: Colormap,
        t: np.ndarray,
        concat_ys: np.ndarray,
        concat_ys_hat: np.ndarray,
        feature_name: str,
    ):
        fig, axs = plt.subplots(
            self.num_samples_row,
            self.num_samples_col,
            dpi=250,
            sharex=True,
            sharey=True,
            figsize=(10, 8),
        )

        axs = cast(Axes, axs)

        for i in range(self.num_samples_row):
            for j in range(self.num_samples_col):
                idx = i * self.num_samples_row + j
                if xs.shape[0] <= idx:
                    warnings.warn(f"not enough samples {xs.shape[0]}{idx}")
                    continue
                x = xs[idx]
                expl = expls[idx] if expls is not None else None
                for k in range(history_length - 1):
                    # Plot the data with colored line segments
                    # TODO only univariate
                    axs[i, j].plot(
                        [t[k], t[k + 1]],
                        [x[k], x[k + 1]],
                        c=cmap(attrib[idx, k]),
                        linewidth=2,
                    )
                ax = cast(Axes, axs[i, j])

                concat_y = concat_ys[idx]
                concat_y_hat = concat_ys_hat[idx]

                forecast_timesteps = t[history_length - 1 :]

                forecast_roi_start = forecast_timesteps[0]
                forecast_roi_end = forecast_timesteps[-1]

                ax.plot(forecast_timesteps, concat_y, c=cmap(attrib.min()), linewidth=2)

                ax.plot(
                    forecast_timesteps,
                    concat_y_hat,
                    c="red",
                    linewidth=2,
                    linestyle="dashed",
                )
                self.draw_region(
                    "Forecast" if i == 0 else None,
                    forecast_roi_start,
                    forecast_roi_end,
                    ax,
                )

                if expl is not None:
                    self.draw_confounder_region(x, ax, expl, row_idx=i)

                ax.set_title(f"Sample {idx}")
                if j == 0:
                    ax.set_ylabel(feature_name)
                if i == self.num_samples_row - 1:
                    ax.set_xlabel("Time")
        plt.suptitle(
            f"Epoch: {trainer.current_epoch} - {kind}: Attribution for each time step"
        )
        plt.tight_layout()

        # Add space on the right side of the plots for the colorbar
        fig.subplots_adjust(right=0.85)

        # Create an additional axes object for the colorbar
        cax = fig.add_axes(
            [0.9, 0.15, 0.05, 0.7]
        )  # Adjust the values as per your requirement

        # Create the colorbar using the ScalarMappable
        sm = cm.ScalarMappable(
            cmap=colormap,
            norm=mplNormalize(vmin=attrib.min(), vmax=attrib.max()),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("Attribution")

        # Show the plot or log it if trainer.logger is available
        if trainer.fast_dev_run:  # type: ignore
            Path("dev_run").mkdir(exist_ok=True)

            plt.savefig(
                f"dev_run/{kind}_expl_xil={applies_xil}_{trainer.current_epoch}.png"
            )
        if trainer.logger:
            plt.close()

            logger: AimLogger = cast(AimLogger, trainer.logger)
            logger.log_image(
                f"{kind}/expl",
                fig,
                epoch=trainer.current_epoch,
                context={
                    "xil": applies_xil,
                    # "explanation_method": self.explanation_method.pretty_name,
                    # "confounded": expl is not None,
                    "feature_name": feature_name,
                },
            )
        else:
            plt.show()

    def compute_gc_attribution(self, xs, horizon_length, model):
        if not isinstance(model, ConvModel):
            raise ValueError("Model must be a ConvModel")
        model.train()  # necessary?
        exp = GuidedGradCam(model, model.last_conv)
        attribs = []
        xs = xs.requires_grad_()
        for i in range(horizon_length):
            attrib = exp.attribute(xs, target=i, interpolate_mode="area")
            # zero? could also be 1
            attribs.append(attrib)

        model.eval()

        attrib = torch.maximum(
            torch.stack(attribs, dim=-1), torch.zeros_like(torch.stack(attribs, dim=-1))
        )
        attrib = torch.sum(attrib, dim=-1)
        # .max(dim=-1)[0]

        sal_min = torch.min(attrib, dim=0, keepdim=True).values
        sal_max = torch.max(attrib, dim=0, keepdim=True).values

        attrib = (attrib - sal_min) / (sal_max - sal_min + 1e-6)

        attrib = attrib.detach().cpu()
        return self.inverse_normalization(model(xs)), attrib

    def compute_saliency_attribution(self, xs, model):
        with torch.set_grad_enabled(True):
            model.eval()
            xs = xs.requires_grad_().to(xs.device)
            pred = model(xs)
            pred.retain_grad()
            explainer = HorizonSailiencyExplainer()
            norm_saliencies = explainer.attribute(xs, pred)
            attrib = norm_saliencies.detach().cpu().numpy()
            pred = self.inverse_normalization(pred)
        return pred, attrib

    def compute_ig_attribution(self, xs, model) -> tuple[np.ndarray, np.ndarray]:
        model.train()
        xs = xs.requires_grad_().to(xs.device)
        pred = model(xs)
        explainer = HorizonIntegratedGradientsExplainer(normalize=True)
        norm_saliencies = explainer.attribute(model, xs, pred)
        attrib = norm_saliencies.detach().cpu().numpy()
        pred = self.inverse_normalization(pred)
        model.eval()
        return pred, attrib

    def compute_freq_ig_attribution(self, xs, model) -> tuple[np.ndarray, np.ndarray]:
        model.train()
        xs = xs.requires_grad_().to(xs.device)
        pred = model(xs)
        explainer = HorizonFrequencyIntegratedGradientsExplainer(normalize=True)
        norm_saliencies = explainer.attribute(model, xs, pred)
        attrib = norm_saliencies.detach().cpu().numpy()
        model.eval()
        return attrib

    def ampl_plot(
        self,
        trainer: Trainer,
        fft: torch.Tensor,
        title: str,
        sampling_rate: int,
        num_total_freqs: int,
        lim: int = -1,
        kind: Kind = "train",
        applies_xil: bool = False,
        feature_name: str = "Target",
    ):
        fig, axs = plt.subplots(2, 2)
        fr = sampling_rate / 2 * np.linspace(0, 1, num_total_freqs // 2 + 1)
        for i in range(2):
            for j in range(2):
                real_part = fft[i * 2 + j, 0].real
                imag_part = fft[i * 2 + j, 0].imag
                axs[i, j].stem(
                    fr, 2 / num_total_freqs * real_part, linefmt="C0-", markerfmt="C0o"
                )
                axs[i, j].stem(
                    fr, 2 / num_total_freqs * imag_part, linefmt="C1-", markerfmt="C1o"
                )
                if lim != -1:
                    axs[i, j].set_ylim([-lim, lim])
                # add legend to show real and imaginary parts
                axs[i, j].legend(["real", "imag"])
        fig.suptitle(title)
        if trainer.fast_dev_run:  # type: ignore
            Path("dev_run").mkdir(exist_ok=True)

            plt.savefig(
                f"dev_run/{kind}_freq_expl_xil={applies_xil}_{trainer.current_epoch}.png"
            )
        if trainer.logger:
            plt.close()

            logger: AimLogger = cast(AimLogger, trainer.logger)
            logger.log_image(
                f"{kind}/freq_expl",
                fig,
                epoch=trainer.current_epoch,
                context={
                    "xil": applies_xil,
                    # "explanation_method": self.explanation_method.pretty_name,
                    # "confounded": expl is not None,
                    "feature_name": feature_name,
                },
            )
        else:
            plt.show()

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        mod = cast(ModelSystem, pl_module)
        if self.should_log(trainer) and trainer.train_dataloader is not None:
            self.compute_explanation(trainer, mod, trainer.train_dataloader, "train")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        mod = cast(ModelSystem, pl_module)
        if trainer.val_dataloaders is None:
            raise ValueError("No validation dataloader")

        if trainer.train_dataloader is None:
            # sanity check should happen
            self.compute_explanation(
                trainer=trainer,
                pl_module=mod,
                data_loader=trainer.val_dataloaders,
                kind="sanity",
            )
        elif self.should_log(trainer):
            self.compute_explanation(trainer, mod, trainer.val_dataloaders, "val")
