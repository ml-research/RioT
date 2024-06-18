from pathlib import Path
from typing import Any, cast
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np
import torch
from matplotlib import cm, pyplot as plt
from lib.aim_logger import AimLogger
from lib.explainers import ExplanationMethod
from lib.model.system import ModelSystem
from ..data.dataset import ExplainedItem, FeedbackPenalty
from matplotlib.colors import LinearSegmentedColormap, Normalize as mplNormalize
import seaborn as sns
from .expl_conf_plot_callback import ConfounderExplanationPlotCallback, Kind


class ClassificationExplanationPlot(ConfounderExplanationPlotCallback):
    def __init__(
        self,
        # inverse_normalization: Callable[[torch.Tensor], np.ndarray],
        explanation_method: ExplanationMethod,
        # num_samples: tuple[int, int] = (2, 2),
        log_interval: int = 5,
        freq_lim: float = 0.5,
    ) -> None:
        super().__init__()
        self.results = []
        # self.inverse_normalization = inverse_normalization
        # self.num_samples_row = num_samples[0]
        # self.num_samples_col = num_samples[1]
        # self.num_samples = self.num_samples_row * self.num_samples_col
        self.explanation_method = explanation_method
        self.log_interval = log_interval
        self.num_samples = 4
        self.freq_lim = freq_lim
        self.train_samples: ExplainedItem | None = None
        self.val_samples: ExplainedItem | None = None

    def compute_explanation(
        self,
        trainer: Trainer,
        pl_module: ModelSystem,
        item: ExplainedItem,
        kind: Kind = "train",
    ):
        xs = item.x
        label = item.y
        expl = item.expl_x if item.expl_x is not None else None

        match self.explanation_method:
            case ExplanationMethod.INTEGRATED_GRADIENTS:
                pred, attrib = self.compute_ig_attribution(xs, pl_module, label)
            case _:
                raise NotImplementedError

        # Create a colormap
        colormap = LinearSegmentedColormap.from_list('Divergent time series', (
            # Edit this gradient at https://eltos.github.io/gradient/#Divergent%20time%20series=0:0036F8-15:1D42DA-35:3493C4-50:5D5757-65:F81900-85:F8002F-100:F8002D
            (0.000, (0.000, 0.212, 0.973)),
            (0.150, (0.114, 0.259, 0.855)),
            (0.350, (0.204, 0.576, 0.769)),
            (0.500, (0.365, 0.341, 0.341)),
            (0.650, (0.973, 0.098, 0.000)),
            (0.850, (0.973, 0.000, 0.184)),
            (1.000, (0.973, 0.000, 0.176))))
        cmap = cm.get_cmap(colormap)
        cmap = cm.get_cmap("coolwarm")

        # xs = self.inverse_normalization(xs)

        t = np.arange(len(xs[0, 0]))

        sns.set_style("dark")
        # set dark sns style and add border around
        sns.despine()
        # sns.set_style("dark", {"axes.facecolor": ".9"})
        sns.set_context("paper")
        # sns.set_style("ticks")

        if expl is not None and expl.freq is not None:
            freq_samples = self.num_samples - 2
            fig, axs = plt.subplots(freq_samples, 3, figsize=(12, 9))
            fig.suptitle(f"Epoch: {trainer.current_epoch} - {kind}")
            num_total_freqs = sampling_rate = xs.shape[-1]
            for i in range(freq_samples):
                title = f"Label {label[i].item()} | Pred {np.argmax(pred[i])}"

                variate = 0
                self.plot_piecewise_attribution(
                    t,
                    xs[i, variate].detach().cpu(),
                    cmap,
                    attrib[i, variate],
                    axs[i, 0],
                    label="Sample" if kind != "train" else "Confounded Sample",
                )

                expl_freq_time = np.fft.irfft(
                    expl.freq[i, :, variate].cpu().numpy(), n=num_total_freqs
                )
                if kind == "train":
                    x_data = xs[i, variate].detach().cpu().numpy() - expl_freq_time
                else:
                    x_data = xs[i, variate].detach().cpu().numpy()

                if kind == "train":
                    axs[i, 0].plot(
                        t,
                        x_data,
                        linestyle="dotted",
                        color="red",
                        label="Original Sample",
                    )
                axs[i, 0].set_ylim([0, 1.5])

                axs[i, 1].plot(t, expl_freq_time)

                fft = np.fft.rfft(attrib[i], norm="backward")

                fr = sampling_rate / 2 * np.linspace(0, 1, num_total_freqs // 2 + 1)
                real_part = fft[0].real
                imag_part = fft[0].imag
                axs[i, 2].stem(
                    fr, 2 / num_total_freqs * real_part, linefmt="C0-", markerfmt="C0o"
                )
                axs[i, 2].stem(
                    fr, 2 / num_total_freqs * imag_part, linefmt="C1-", markerfmt="C1o"
                )
                # if self.freq_lim != -1:
                axs[i, 1].set_ylim([-self.freq_lim, self.freq_lim])
                axs[i, 2].set_ylim([-self.freq_lim, self.freq_lim])

                top_imag_values, top_k_imag = torch.topk(
                    expl.freq[i, :, variate].imag.abs(), k=1
                )
                top_real_values, top_k_real = torch.topk(
                    expl.freq[i, :, variate].real.abs(), k=1
                )
                if (torch.abs(top_imag_values) < 1e-1).any().item():
                    top_k_imag = None

                if (torch.abs(top_real_values) < 1e-1).any().item():
                    top_k_real = None

                freq_conf = None
                if top_k_imag is not None:
                    self.draw_region(
                        None,
                        top_k_imag.item() - 2,
                        top_k_imag.item() + 2,
                        axs[i, 2],
                        facecolor="orange",
                    )
                    freq_conf = "Imag Confounder"
                    if kind == "val":
                        freq_conf += " (Train)"
                elif top_k_real is not None:
                    self.draw_region(
                        None,
                        top_k_real.item() - 2,
                        top_k_real.item() + 2,
                        axs[i, 2],
                        facecolor="orange",
                    )
                    freq_conf = "Real Confounder"

                    if kind == "val":
                        freq_conf += " (Train)"

                if i == 0:
                    title += "\nConfounder Frequency"
                    if kind == "val":
                        title += " (Train)"
                    axs[0, 0].set_title("Time Sample")
                    axs[0, 2].set_title("Frequency Explanation")
                    axs[0, 0].legend(
                        loc="upper right",
                    )
                    if freq_conf is not None:
                        # add legend to show real and imaginary parts
                        axs[0, 2].legend([freq_conf, "Real", "Imag"])
                axs[i, 1].set_title(title)  # pad=20)

                if expl.time is not None:
                    self.draw_confounder_region(
                        xs[i, variate].cpu().numpy(),
                        axs[i, 0],
                        expl.time[i, variate].cpu().numpy(),
                        is_train=kind == "train",
                    )

            # Create the colorbar using the ScalarMappable
            sm = cm.ScalarMappable(
                cmap=cmap,
                norm=mplNormalize(vmin=0, vmax=1),
            )
            sm.set_array([])
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)

            cbar.set_label("Attribution")
            fig.subplots_adjust(hspace=0.5)
            # fig.suptitle(title)
            if trainer.fast_dev_run:  # type: ignore
                Path("dev_run").mkdir(exist_ok=True)
                plt.savefig(
                    f"dev_run/{kind}_freq_expl_xil={pl_module.applies_xil}_{trainer.current_epoch}.png"
                )
            if trainer.logger:
                plt.close()

                logger: AimLogger = cast(AimLogger, trainer.logger)
                logger.log_image(
                    f"{kind}/freq_expl",
                    fig,
                    epoch=trainer.current_epoch,
                    context={
                        "xil": pl_module.applies_xil,
                        # "explanation_method": self.explanation_method.pretty_name,
                        # "confounded": expl is not None,
                        # "feature_name": feature_name,
                    },
                )
            else:
                plt.show()

        else:
            # fig = plt.figure(constrained_layout=True, dpi=250, figsize=(10, 8))
            fig, axs = plt.subplots(self.num_samples, 1, sharex=True, figsize=(10, 8), dpi=250)
            # fig.set

            fig.suptitle(f"Epoch: {trainer.current_epoch} - {kind}")
            for i in range(self.num_samples):
                axs[i].set_title(
                    f"Label {label[i].item()} | Pred {np.argmax(pred[i])}"
                )  # pad=20)

                variate = 0
                self.plot_piecewise_attribution(
                    t, xs[i, variate].detach().cpu(), cmap, attrib[i, variate], axs[i]
                )

                # expl = item.expl_x[: self.num_samples] if item.expl_x is not None else None
                if expl is not None:
                    if expl.freq is not None:
                        print("Not implemented TODO:")

                    if expl.time is not None:
                        time_expl = expl.time[: self.num_samples]
                        self.draw_confounder_region(
                            xs[i, variate].cpu().numpy(),
                            axs[i],
                            time_expl[i, variate].cpu().numpy(),
                            row_idx=i,
                            is_train=kind == "train",
                        )

            # Create the colorbar using the ScalarMappable
            sm = cm.ScalarMappable(
                cmap=cmap,
                norm=mplNormalize(vmin=0, vmax=1),
            )
            sm.set_array([])
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            # cbar_ax.collections[0].set_clim(-1,1)

            cbar.set_label("Attribution")
            fig.subplots_adjust(hspace=0.5)

            # # Show the plot or log it if trainer.logger is available
            if trainer.fast_dev_run:  # type: ignore
                Path("dev_run").mkdir(exist_ok=True)
                plt.savefig(
                    f"dev_run/{kind}_expl_xil={pl_module.applies_xil}_{trainer.current_epoch}.png"
                )
            elif trainer.logger:
                plt.close()

                logger: AimLogger = cast(AimLogger, trainer.logger)
                logger.log_image(
                    f"{kind}/expl",
                    fig,
                    epoch=trainer.current_epoch,
                    context={
                        "xil": pl_module.applies_xil,
                        # "explanation_method": self.explanation_method.pretty_name,
                        # "confounded": expl is not None,
                        # "sample": ,
                    },
                )
            else:
                plt.show()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.train_samples is None:
            self.train_samples = batch

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        mod = cast(ModelSystem, pl_module)
        if self.should_log(trainer) and trainer.train_dataloader is not None:
            samples = self.train_samples[: self.num_samples]
            self.compute_explanation(trainer, mod, samples, "train")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: ExplainedItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.val_samples is None:
            self.val_samples = batch
        else:
            self.val_samples.append(batch)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        mod = cast(ModelSystem, pl_module)
        if (
            self.should_log(trainer)
            and trainer.val_dataloaders is not None
            and self.train_samples is not None
        ):
            # shuffle self.val_samples["item"].{x,y} and y_hat
            idxs = torch.randperm(self.val_samples.x.shape[0])
            self.val_samples = self.val_samples[idxs]

            train_samples = self.train_samples[: self.num_samples]
            val_samples: ExplainedItem | None = None
            for i_t in range(len(train_samples)):
                t_item = train_samples[i_t]
                for i_v in range(self.val_samples.y.shape[0]):
                    v_item = self.val_samples[i_v]
                    if t_item.y == v_item.y:
                        if val_samples is None:
                            val_samples = v_item
                        else:
                            val_samples.append(v_item)
                        if val_samples.expl_x is None:
                            val_samples.expl_x = t_item.expl_x
                        else:
                            val_samples.expl_x.append(t_item.expl_x)
                        break

            self.compute_explanation(trainer, mod, val_samples, "val")

            self.val_samples = self.train_samples = None
