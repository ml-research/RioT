import os
from typing import Any, Literal
from jsonargparse import ActionConfigFile, lazy_instance
from lightning import LightningDataModule

import torch
import torch.optim as optim

from lib.aim_logger import AimLogger
from lib.cli.custom_yaml_parser import CustomArgumentParser
import lightning as L
from lib.data import Confounder
from lib.model import ModelSystem
import lib.callbacks as cb
from lib.util import parse_string_ranges
from pathlib import Path
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from jsonargparse.typing import register_type  # type: ignore
from dotenv import load_dotenv
import os
import sys
from aim import Run

load_dotenv(Path(sys.path[1]) / ".executor.env")


def register_confounder_type():
    def serializer(confounder: Confounder) -> str:
        confounder_str = str(confounder)
        return confounder_str.split(".")[-1]  # Remove the enum flag class name

    def deserializer(confounder: str) -> Confounder:
        if "Confounder." not in confounder:
            return Confounder[confounder]

        conf_wo_name: str = confounder.split(".")[-1]
        if "|" in conf_wo_name:
            confs: list[str] = conf_wo_name.split("|")

            return Confounder(
                sum([int(Confounder[conf_str].value) for conf_str in confs])
            )
        else:
            return Confounder[conf_wo_name]

    register_type(Confounder, serializer, deserializer, fail_already_registered=False)


def register_list_of_ranges_type():
    def serializer(feedback_penalty_range: list[range]) -> list[str]:
        return [
            f"{fb_range.start}:{fb_range.stop}"
            + (f":{fb_range.step}" if fb_range.step > 1 else "")
            for fb_range in feedback_penalty_range
        ]

    def deserializer(feedback_penalty_range: list[str] | None) -> list[range] | None:
        if feedback_penalty_range is None:
            return None

        if isinstance(
            feedback_penalty_range[0], range
        ):  # needed due to the limitation of only allowing simple types
            return feedback_penalty_range

        return parse_string_ranges(feedback_penalty_range)

    register_type(list[range], serializer, deserializer, fail_already_registered=False)


def disable_user_warnings(disable: bool = True):
    if disable:
        import warnings
        from lightning.fabric.utilities.warnings import PossibleUserWarning

        # filter lightning warning
        warnings.filterwarnings("ignore", category=PossibleUserWarning)
        # filter pydantic warning -> UserWarning: `pydantic.utils:Representation`
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


def prepare_tags(
    experiment: Run,
    has_right_reason_loss: bool,
    confounder: Confounder,
    task: Literal["forecasting", "classification"],
):
    tags = experiment.tags  # clear tags
    for tag in tags:
        experiment.remove_tag(tag)

    if Confounder.NO_CONFOUNDER in confounder:
        # No tags
        print("No tags needed")
    else:
        if (
            Confounder.CLASSIFICATION_TIME in confounder
            or Confounder.FORECASTING_TIME in confounder
        ):
            experiment.add_tag("CSP")

        if (
            Confounder.CLASSIFICATION_FREQ in confounder
            or Confounder.FORECASTING_NOISE in confounder
            or Confounder.FORECASTING_DIRAC in confounder
        ):
            experiment.add_tag("CSF")

        if Confounder.SANITY in confounder:
            experiment.add_tag("SAN")

        if has_right_reason_loss:
            experiment.add_tag("XIL")

    experiment.add_tag(task)


def prepare_train_confounder(cfg):
    if cfg.data.class_path == "lib.data.MechanicalClassificationData":
        if cfg.data.init_args.feedback_penalty_range is not None:
            train_confounder = Confounder.CLASSIFICATION_TIME
        else:
            train_confounder = Confounder.NO_CONFOUNDER
    elif cfg.data.class_path == "lib.data.P2SData":
        if cfg.data.init_args.mode == "Decoy":
            train_confounder = Confounder.CLASSIFICATION_TIME
        else:
            train_confounder = Confounder.NO_CONFOUNDER
    else:
        train_confounder = cfg.data.init_args.confounder
    return train_confounder


class BaseCLI(LightningCLI):

    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        """Method that instantiates the argument parser."""
        kwargs.setdefault("dump_header", [f"lightning.pytorch=={L.__version__}+merge"])
        parser = CustomArgumentParser(**kwargs)
        parser.add_argument(
            "-c",
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )
        return parser

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # Optimizer
        # parser.add_optimizer_args(optim.Adam)
        parser.add_optimizer_args((optim.Adam, optim.RAdam))

        executor_initals = os.environ.get("EXECUTOR", None)
        if executor_initals is None:
            raise ValueError("Executor initials not set in .executor.env")

        repo = os.environ.get("AIM_REPO", None)
        # parse string none to python None
        if repo is not None and repo.capitalize() in ["NULL", "NONE"]:
            repo = None

        # General Defaults
        parser.set_defaults(
            {
                "trainer.max_epochs": 40,
                "trainer.logger": lazy_instance(AimLogger, repo=repo),
                "seed_everything": 34234,
                "optimizer": {"class_path": "torch.optim.RAdam", "init_args": {"lr": 1e-3},}
                # "explanation_plot.num_samples": (2, 2),
                # "explanation_plot.log_interval": 15,
            }
        )

        parser.link_arguments(
            "trainer.fast_dev_run", target="data.init_args.fast_dev_run"
        )

        parser.link_arguments(
            "data.init_args.lambda_time",
            target="model.init_args.lambda_time",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.init_args.lambda_freq",
            target="model.init_args.lambda_freq",
            apply_on="instantiate",
        )

    def before_fit(self) -> None:
        if isinstance(self.trainer.logger, AimLogger):
            train_confounder = prepare_train_confounder(self.config.fit)
            prepare_tags(
                self.trainer.logger.experiment,
                self.config.fit.model.init_args.right_reason_loss is not None,
                train_confounder,
                task=self.__class__.__name__.split("CLI")[0].lower(),
            )


def configure_args(
    subcommand: Literal["fit", "validate", "test", "predict"] | None = None,
    model_config: str | None = None,
    data_config: str | None = None,
    trainer_config: str | None = None,
    exp_config: str | None = None,
    confounder: Confounder | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    dev_run: bool = False,
    extra_args: list[str] = [],
):
    root_dir = os.getcwd()
    root_dir = Path("configs") if "src" in root_dir else Path("src") / "configs"

    args: list[str] | None = []

    if exp_config is not None:
        config = root_dir / "exp" / exp_config
        args.append("-c")
        args.append(f"{config}")

    if model_config is not None:
        config = root_dir / "model" / model_config
        args.append("--model")
        args.append(f"{config}")

    if data_config is not None:
        config = root_dir / "data" / data_config
        args.append("--data")
        args.append(f"{config}")

    if trainer_config is not None:
        config = root_dir / "trainer" / trainer_config
        args.append("--trainer")
        args.append(f"{config}")

    if confounder is not None:
        args.append(f"--data.init_args.confounder={str(confounder)}")

    args += extra_args

    if experiment_name is not None:
        # workaround due to change in argparse
        args.append(
            f"--trainer.logger.init_args.experiment_name={experiment_name.replace(' ','_w_') if ' ' in experiment_name else experiment_name}"
        )
        # args.append(f"--trainer.logger.init_args.experiment_name={experiment_name}")

    if run_name is not None:
        # workaround due to change in argparse
        # args.append(f"--trainer.logger.init_args.run_name={run_name}")
        args.append(
            f"--trainer.logger.init_args.run_name={run_name.replace(' ','_w_') if ' ' in run_name else run_name}"
        )

    if dev_run:
        args.append("--trainer.fast_dev_run=true")

    if subcommand is not None:
        args.insert(0, subcommand)

    return args if len(args) > 0 else None


def make_cli_factory(
    cli_cls: type[BaseCLI] = BaseCLI,
    data_cls: type[LightningDataModule] = LightningDataModule,
):
    def cli(
        subcommand: Literal["fit", "validate", "test", "predict"] | None = None,
        model_config: str | None = None,
        data_config: str | None = None,
        trainer_config: str | None = None,
        exp_config: str | None = None,
        confounder: Confounder | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
        dev_run: bool = False,
        extra_args: list[str] = [],
    ):
        if not dev_run:
            disable_user_warnings()
        register_confounder_type()
        register_list_of_ranges_type()

        # check if cli_cls is type of ClassificationCLI or ForecastingCLI
        if cli_cls.__name__ == "ClassificationCLI":
            mode = "classification/"
        elif cli_cls.__name__ == "ForecastingCLI":
            mode = "forecasting/"
        else:
            mode = ""

        # TODO missing optimizer parameter
        args = configure_args(
            subcommand=subcommand,
            model_config=mode + model_config if model_config is not None else None,
            data_config=mode + data_config if data_config is not None else None,
            trainer_config=trainer_config,
            exp_config=mode + exp_config if exp_config is not None else None,
            confounder=confounder,
            experiment_name=experiment_name,
            run_name=run_name,
            dev_run=dev_run,
            extra_args=extra_args,
        )

        torch.set_float32_matmul_precision("high")

        match args:
            case ("tune", *cfgs):
                # remove first element
                # args = args[1:]
                cli = cli_cls(
                    ModelSystem,
                    data_cls,
                    subclass_mode_data=True,
                    subclass_mode_model=True,
                    save_config_callback=cb.LoggerSaveConfig,
                    args=cfgs,
                    run=False,
                )
                # cli.model.init_args

                train_confounder = prepare_train_confounder(cli.config)
                prepare_tags(
                    cli.trainer.logger.experiment,
                    cli.config.model.init_args.right_reason_loss is not None,
                    train_confounder,
                    task=cli_cls.__name__.split("CLI")[0].lower(),
                )

                cli.trainer.fit(cli.model, cli.datamodule)
                cli.trainer.limit_val_batches = 1.0
                cli.trainer.validate(
                    cli.model,
                    cli.datamodule,
                    ckpt_path=cli.trainer.checkpoint_callback._last_checkpoint_saved,
                )  # "last does not seem to work, therefore we access the protected variable"

                if "val/MeanSquaredError_epoch" in cli.trainer.logged_metrics:
                    mse = cli.trainer.logged_metrics[
                        "val/MeanSquaredError_epoch"
                    ].item()
                    mae = cli.trainer.logged_metrics[
                        "val/MeanAbsoluteError_epoch"
                    ].item()
                    return {"mse": mse, "mae": mae}
                else:
                    metric = cli.trainer.logged_metrics[
                        "val/MulticlassAccuracy_epoch"
                    ].item()
                return metric
            case ("fit+test", *cfgs):
                # if args[0] == "fit+test":
                # remove first element
                # args = args[1:]
                cli = cli_cls(
                    ModelSystem,
                    data_cls,
                    subclass_mode_data=True,
                    subclass_mode_model=True,
                    save_config_callback=cb.LoggerSaveConfig,
                    args=cfgs,
                    run=False,
                )

                train_confounder = prepare_train_confounder(cli.config)
                prepare_tags(
                    cli.trainer.logger.experiment,
                    cli.config.model.init_args.right_reason_loss is not None,
                    train_confounder,
                    task=cli_cls.__name__.split("CLI")[0].lower(),
                )

                cli.trainer.fit(cli.model, cli.datamodule)
                cli.trainer.test(
                    cli.model,
                    cli.datamodule,
                    ckpt_path=cli.trainer.checkpoint_callback._last_checkpoint_saved,
                )  # "last does not seem to work, therefore we access the protected variable"
            # else:
            case _:
                cli_cls(
                    ModelSystem,
                    data_cls,
                    subclass_mode_data=True,
                    subclass_mode_model=True,
                    save_config_callback=cb.LoggerSaveConfig,
                    args=args,
                )

    return cli
