import os
from typing import Any, Dict, Optional, Union
from argparse import Namespace

import packaging.version

try:
    import lightning as L

    from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
    from lightning.pytorch.utilities.rank_zero import rank_zero_only

except ImportError:
    try:
        import pytorch_lightning as pl

        if packaging.version.parse(pl.__version__) < packaging.version.parse("1.7"):
            from pytorch_lightning.loggers.base import (
                LightningLoggerBase as Logger,
                rank_zero_experiment,
            )
        else:
            from pytorch_lightning.loggers.logger import (
                Logger,
                rank_zero_experiment,
            )

        from pytorch_lightning.utilities import rank_zero_only
    except ImportError:
        raise RuntimeError(
            "This contrib module requires Lightning to be installed. "
            "Please install version prior 2.0 via: \n pip install pytorch-lightning"
            "Or for the newest version: \n pip install lightning"
        )
# _
from aim.sdk.run import Run
from aim.sdk.repo import Repo
from aim.sdk.utils import clean_repo_path, get_aim_repo_name
from aim import Figure, Image

# from aim._ext.system_info import DEFAULT_SYSTEM_TRACKING_INT
DEFAULT_SYSTEM_TRACKING_INT = 10


class AimLogger(Logger):
    """
    AimLogger logger class.

    Args:
        repo (:obj:`str`, optional): Aim repository path or Repo object to which Run object is bound.
            If skipped, default Repo is used.
        experiment_name (:obj:`str`, optional): Sets Run's `experiment` property. 'default' if not specified.
            Can be used later to query runs/sequences.
        system_tracking_interval (:obj:`int`, optional): Sets the tracking interval in seconds for system usage
            metrics (CPU, Memory, etc.). Set to `None` to disable system metrics tracking.
        log_system_params (:obj:`bool`, optional): Enable/Disable logging of system params such as installed packages,
            git info, environment variables, etc.
        capture_terminal_logs (:obj:`bool`, optional): Enable/Disable terminal stdout logging.
        train_metric_prefix (:obj:`str`, optional): Training metric prefix.
        val_metric_prefix (:obj:`str`, optional): validation metric prefix.
        test_metric_prefix (:obj:`str`, optional): testing metric prefix.
        run_name (:obj:`str`, optional): Aim run name, for reusing the specified run.
        run_hash (:obj:`str`, optional): Aim run hash, for reusing the specified run.
    """

    def __init__(
        self,
        repo: Optional[str] = None,
        experiment_name: Optional[str] = None,
        system_tracking_interval: Optional[int] = DEFAULT_SYSTEM_TRACKING_INT,
        log_system_params: Optional[bool] = True,
        capture_terminal_logs: Optional[bool] = True,
        train_metric_prefix: Optional[str] = "train/",
        val_metric_prefix: Optional[str] = "val/",
        test_metric_prefix: Optional[str] = "test/",
        step_postfix: Optional[str] = "_step",
        epoch_postfix: Optional[str] = "_epoch",
        run_name: Optional[str] = None,
        run_hash: Optional[str] = None,
    ):
        super().__init__()

        if run_name is not None:
            run_name = run_name.replace("_w_", " ")
        if experiment_name is not None:
            experiment_name = experiment_name.replace("_w_", " ")

        self._repo_path = repo
        self._experiment_name = experiment_name
        self._system_tracking_interval = system_tracking_interval
        self._log_system_params = log_system_params
        self._capture_terminal_logs = capture_terminal_logs

        self._train_metric_prefix = train_metric_prefix
        self._val_metric_prefix = val_metric_prefix
        self._test_metric_prefix = test_metric_prefix
        self._step_postfix = step_postfix
        self._epoch_postfix = epoch_postfix

        self._run_name = run_name
        self._run_hash = run_hash

        self._run = None

    @staticmethod
    def _convert_params(params: Union[Dict[str, Any], Namespace]) -> Dict[str, Any]:
        # in case converting from namespace
        if isinstance(params, Namespace):
            params = vars(params)

        if params is None:
            params = {}

        return params

    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        if self._run is None:
            if self._run_hash:
                self._run = Run(
                    self._run_hash,
                    repo=self._repo_path,
                    system_tracking_interval=self._system_tracking_interval,
                    capture_terminal_logs=self._capture_terminal_logs,
                )
                    
            else:
                self._run = Run(
                    repo=self._repo_path,
                    experiment=self._experiment_name,
                    system_tracking_interval=self._system_tracking_interval,
                    log_system_params=self._log_system_params,
                    capture_terminal_logs=self._capture_terminal_logs,
                )
                self._run_hash = self._run.hash
            
            if self._run_name is not None:
                self._run.name = self._run_name

        return self._run

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]):
        params = self._convert_params(params)

        # Handle OmegaConf object
        try:
            from omegaconf import OmegaConf
        except ModuleNotFoundError:
            pass
        else:
            # Convert to primitives
            if OmegaConf.is_config(params):
                params = OmegaConf.to_container(params, resolve=True)

        for key, value in params.items():
            self.experiment.set(("hparams", key), value, strict=False)

    @rank_zero_only
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        context: dict[str, Any] = {},
    ):
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metric_items: dict[str, Any] = {k: v for k, v in metrics.items()}

        if "epoch" in metric_items:
            epoch: int | None = metric_items.pop("epoch")
        else:
            epoch = None

        for k, v in metric_items.items():
            name = k
            _context = context.copy()
            if self._step_postfix and name.endswith(self._step_postfix):
                name = name[: -len(self._step_postfix)]
                _context["interval"] = "step"
            elif self._epoch_postfix and name.endswith(self._epoch_postfix):
                name = name[: -len(self._epoch_postfix)]
                _context["interval"] = "epoch"

            if self._train_metric_prefix and name.startswith(self._train_metric_prefix):
                name = name[len(self._train_metric_prefix) :]
                _context["subset"] = "train"
            elif self._test_metric_prefix and name.startswith(self._test_metric_prefix):
                name = name[len(self._test_metric_prefix) :]
                _context["subset"] = "test"
            elif self._val_metric_prefix and name.startswith(self._val_metric_prefix):
                name = name[len(self._val_metric_prefix) :]
                _context["subset"] = "val"
            self.experiment.track(
                v, name=name, step=step, epoch=epoch, context=_context
            )

    @rank_zero_only
    def log_figure(
        self,
        key: str,
        figure: Any,
        step: Optional[int] = None,
        epoch: int | None = None,
        context: dict[str, Any] = {},
        **kwargs: Any
    ) -> None:
        aim_figure = Figure(figure)
        self.log_metrics({key: aim_figure, "epoch": epoch}, step=step, context=context)

    @rank_zero_only
    def log_image(
        self,
        key: str,
        image: Any,
        step: Optional[int] = None,
        context: dict[str, Any] = {},
        **kwargs: Any
    ) -> None:
        aim_image = Image(image)
        self.log_metrics({key: aim_image}, step=step, context=context)

    @rank_zero_only
    def finalize(self, status: str = "") -> None:
        super().finalize(status)
        if self._run:
            self._run.close()
            del self._run
            self._run = None

    def __del__(self):
        self.finalize()

    @property
    def save_dir(self) -> str:
        repo_path = clean_repo_path(self._repo_path) or Repo.default_repo_path()
        return os.path.join(repo_path, get_aim_repo_name())

    @property
    def name(self) -> str:
        return self._experiment_name

    @property
    def version(self) -> str:
        return self.experiment.hash
