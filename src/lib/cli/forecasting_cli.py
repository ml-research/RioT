from lib.aim_logger import AimLogger
from lib.data import ForecastingData, Confounder
from lib.explainers import ExplanationMethod
from lib.model import ModelSystem
import lib.callbacks as cb

from lightning.pytorch.cli import LightningArgumentParser
from jsonargparse import lazy_instance
import torch.optim as optim
from lib.cli.base_cli import BaseCLI, make_cli_factory


class ForecastingCLI(BaseCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)

        # parser.add_lightning_class_args(cb.ExplanationtPlot, "explanation_plot")

        # General Defaults
        parser.set_defaults(
            {
                "model": lazy_instance(
                    ModelSystem,
                    right_answer_loss="torch.nn.MSELoss",
                ),
                "data": lazy_instance(
                    ForecastingData,
                    split_ratio=0.8,
                    sanity_padded_zeros=True,
                    val_confounder=Confounder.NO_CONFOUNDER,
                ),
                # "explanation_plot.explanation_method": ExplanationMethod.FREQ_INTEGRATED_GRADIENTS,
                # "explanation_plot.num_samples": (2, 2),
                # "explanation_plot.log_interval": 15,
            }
        )

        # Argument Linking
        parser.link_arguments(
            "data.init_args.prediction_horizon", "model.init_args.prediction_horizon"
        )
        parser.link_arguments("data.init_args.lookback", "model.init_args.lookback")

        # parser.link_arguments(
        #     "data.inverse_transform",
        #     target="explanation_plot.inverse_normalization",
        #     apply_on="instantiate",
        # )

        # parser.link_arguments(
        #     "data.pretty_target_feature_name",
        #     target="explanation_plot.target_features",
        #     apply_on="instantiate",
        # )
        # parser.link_arguments(
        #     "data.pretty_source_feature_names",
        #     target="explanation_plot.source_features",
        #     apply_on="instantiate",
        # )

        # parser.link_arguments(
        #     "data.forecasting_mode",
        #     target="explanation_plot.forecasting_mode",
        #     apply_on="instantiate",
        # )

        parser.link_arguments(
            "data.init_args.source_feature_names",
            compute_fn=lambda x: len(x),
            target="model.init_args.num_channels",
            apply_on="instantiate",
        )



make_cli = make_cli_factory(ForecastingCLI, ForecastingData)


if __name__ == "__main__":
    make_cli()
