from lightning import LightningModule, Trainer
from lib.aim_logger import AimLogger
from pathlib import Path
from jsonargparse import namespace_to_dict

from lightning.pytorch.cli import SaveConfigCallback
from lightning.fabric.utilities.cloud_io import get_filesystem


class LoggerSaveConfig(SaveConfigCallback):
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        if not isinstance(trainer.logger, AimLogger):
            raise RuntimeError("Currently only aim is supported as logger")

        log_dir = trainer.log_dir  # this broadcasts the directory
        assert log_dir is not None
        assert trainer.logger._run_hash is not None
        log_dir = Path(log_dir) / trainer.logger.name / trainer.logger._run_hash
        config_path = log_dir / self.config_filename
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        # save the file on rank 0
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )
            config = namespace_to_dict(
                self.config
            )  # self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility

            # traverse the config and delete all __path__ entries and return the cleaned dict
            def clean_dict(d):
                if isinstance(d, dict):
                    return {k: clean_dict(v) for k, v in d.items() if k != "__path__"}
                return d

            config = clean_dict(config)
            del config["config"]
            trainer.logger.log_hyperparams(config)
            self.already_saved = True

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)
