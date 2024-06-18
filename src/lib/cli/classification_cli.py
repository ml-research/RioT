from lib.aim_logger import AimLogger
from lib.data import ClassificationData, Confounder
from lib.explainers import ExplanationMethod
from lib.model import ModelSystem
import lib.callbacks as cb

from lightning.pytorch.cli import LightningArgumentParser
from jsonargparse import lazy_instance
import torch.optim as optim
from lib.cli.base_cli import BaseCLI, make_cli_factory

class ClassificationCLI(BaseCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)

        # parser.add_lightning_class_args(cb.ClassificationExplanationPlot, "classification_explanation_plot")

        parser.set_defaults(
            {
                "model": lazy_instance(
                    ModelSystem,
                    right_answer_loss="torch.nn.CrossEntropyLoss",
                ),
                "data": lazy_instance(
                    ClassificationData,
                    # split_ratio=0.8,
                    # sanity_padded_zeros=True,
                    val_confounder=Confounder.NO_CONFOUNDER,
                ),
                # "classification_explanation_plot.explanation_method": ExplanationMethod.INTEGRATED_GRADIENTS,
                # "explanation_plot.num_samples": (2, 2),
                # "explanation_plot.log_interval": 15,
            }
        )


        parser.link_arguments(
            "data.init_args.num_classes",
            target="model.init_args.num_classes",
            apply_on="instantiate",
        )
        


make_cli = make_cli_factory(ClassificationCLI, ClassificationData)


if __name__ == "__main__":
    make_cli()
