from typing import Literal
from lightning import LightningDataModule
import numpy as np
import torch

from .dataset import ExplainedItem, SequentialDataset
from torch.utils.data import DataLoader
from ..preprocessing import ScalerType, MinMaxScaler

from .data_enums import Confounder, ForecastingMode
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries


class ForecastingData(LightningDataModule):
    def __init__(
        self,
        sanity_padded_zeros: bool = True,
        split_ratio: float | tuple[float, float] = 0.8,
        confounder: Confounder = Confounder.NO_CONFOUNDER,
        confounder_freq_len: int = 0,
        confounder_freq_strength: float = 0,
        prediction_horizon: int = -1,
        lookback: int = -1,
        batch_size: int = -1,
        val_confounder: Confounder = Confounder.NO_CONFOUNDER,
        scaler: ScalerType = MinMaxScaler(),
        target_feature_names: str | list[str] = "target",
        forecasting_mode: ForecastingMode = ForecastingMode.UNIVARIATE,
        num_workers: int = 0,
        lambda_freq: float = 1.0,
        lambda_time: float = 1.0,
        feedback_percentage: float = 1.0
    ) -> None:
        super().__init__()
        if (
            confounder.is_multivariate
            and forecasting_mode == ForecastingMode.UNIVARIATE
        ):
            raise ValueError(
                "Cannot use univariate forecasting mode with multivariate confounder"
            )

        self.target_feature_names = target_feature_names
        self.sanity_padded = sanity_padded_zeros
        self.split_ratio = split_ratio
        self.confounder = confounder
        self.val_confounder = val_confounder
        self.prediction_horizon = prediction_horizon
        self.lookback = lookback
        self.batch_size = batch_size
        self.forecasting_mode = forecasting_mode
        self.data_transformer = Scaler(scaler)
        self.num_workers = num_workers
        self.lambda_freq = lambda_freq
        self.lambda_time = lambda_time
        self.confounder_freq_len = confounder_freq_len
        self.confounder_freq_strength = confounder_freq_strength
        self.feedback_percentage = feedback_percentage

    @property
    def num_target_features(self) -> int:
        if isinstance(self.target_feature_names, str):
            return 1
        else:
            return len(self.target_feature_names)

    def inverse_transform(self, data: torch.Tensor) -> np.ndarray:
        """
        Gets normalized data tensor of shape [Batch,Var,Seq] and returns a not normalized version of shape [Batch,Var,Seq]
        """
        fitted_params: list[ScalerType] = self.data_transformer._fitted_params  # type: ignore

        if len(fitted_params) == 1:
            # do normal stuff
            res: TimeSeries = self.data_transformer.inverse_transform(TimeSeries.from_values(data.detach().cpu().numpy()))  # type: ignore
            return res.all_values()
        elif len(fitted_params) == 2:
            cov_scaler_n_features_in = fitted_params[0].n_features_in_
            target_scaler_n_features_in = fitted_params[1].n_features_in_
            if data.shape[1] == cov_scaler_n_features_in:
                # covariate mode -> inverse transform only covariates
                inversion_result: list[TimeSeries] = self.data_transformer.inverse_transform([TimeSeries.from_values(data.detach().cpu().numpy())])  # type: ignore
                return inversion_result[0].all_values()
            elif data.shape[1] == target_scaler_n_features_in:
                # target mode -> inverse transform only target
                dummy_covariate = TimeSeries.from_values(
                    np.ones((1, cov_scaler_n_features_in, 1))
                )
                inversion_result: list[TimeSeries] = self.data_transformer.inverse_transform([dummy_covariate, TimeSeries.from_values(data.detach().cpu().numpy())])  # type: ignore
                return inversion_result[1].all_values()
            else:
                # All data mode -> inverse transform all data
                cov, target = torch.split(
                    data, [cov_scaler_n_features_in, target_scaler_n_features_in], dim=1
                )
                inversion_result: list[TimeSeries] = self.data_transformer.inverse_transform([TimeSeries.from_values(cov.detach().cpu().numpy()), TimeSeries.from_values(target.detach().cpu().numpy())])  # type: ignore
                return inversion_result[0].stack(inversion_result[1]).all_values()

        else:
            raise ValueError("Invalid number of fitted params")

    @property
    def pretty_target_feature_name(self) -> str | list[str]:
        if isinstance(self.target_feature_names, str):
            target_names = [self.target_feature_names]
        else:
            target_names = self.target_feature_names
        for i, name in enumerate(target_names):
            target_names[i] = name.replace("_", " ").title()

        return target_names

    def post_setup(
        self,
        series_dataset: TimeSeries,
        stage: Literal["fit", "validate", "test", "predict"],
    ) -> None:
        train_ds, val_ds, test_ds = self.split_series_dataset(series_dataset)

        # Assign train/val datasets for use in dataloaders
        if self.forecasting_mode == ForecastingMode.MULTIVARIATE_UNIVARIATE:
            train_covs = train_ds.drop_columns(self.target_feature_names)
            val_covs = val_ds.drop_columns(self.target_feature_names)
            train_targets = train_ds[self.target_feature_names]
            val_targets = val_ds[self.target_feature_names]
            train_ds = self.data_transformer.fit_transform([train_covs, train_targets])
            val_ds = self.data_transformer.transform([val_covs, val_targets])

            if test_ds is not None and stage == "test":
                test_covs = test_ds.drop_columns(self.target_feature_names)
                test_targets = test_ds[self.target_feature_names]
                test_ds = self.data_transformer.transform([test_covs, test_targets])

        else:
            train_ds = self.data_transformer.fit_transform(train_ds)

            val_ds = self.data_transformer.transform(val_ds)

            if test_ds is not None and stage == "test":
                test_ds = self.data_transformer.transform(test_ds)
                self.test_dataset = SequentialDataset(
                    self.data_transformer.transform(test_ds),
                    lookback=self.lookback,
                    prediction_horizon=self.prediction_horizon,
                    confounder=self.confounder,
                    padded_zeros=self.sanity_padded,
                    target_feature_names=self.target_feature_names,
                )

        self.train_dataset = SequentialDataset(
            train_ds,
            lookback=self.lookback,
            prediction_horizon=self.prediction_horizon,
            confounder=self.confounder,
            padded_zeros=self.sanity_padded,
            target_feature_names=self.target_feature_names,
        )
        self.val_dataset = SequentialDataset(
            val_ds,
            lookback=self.lookback,
            prediction_horizon=self.prediction_horizon,
            confounder=self.val_confounder,
            padded_zeros=self.sanity_padded,
            target_feature_names=self.target_feature_names,
        )

    def split_series_dataset(self, series_dataset: TimeSeries):
        if (
            self.forecasting_mode == ForecastingMode.UNIVARIATE
            and series_dataset.n_components > 1
        ):
            series_dataset = series_dataset[self.target_feature_names]
        test_ds: TimeSeries | list[TimeSeries] | None = None
        if isinstance(self.split_ratio, tuple):
            # train/val/test split
            train_ds, eval_ds = series_dataset.split_before(self.split_ratio[0])
            split = round(1 - self.split_ratio[1], 2)
            val_ds, test_ds = eval_ds.split_before(split)
        else:
            # train/val split
            train_ds, val_ds = series_dataset.split_before(self.split_ratio)
        return train_ds, val_ds, test_ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ExplainedItem.forecasting_collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=ExplainedItem.forecasting_collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=ExplainedItem.forecasting_collate_fn,
            num_workers=self.num_workers,
        )
