import math
from darts import TimeSeries
import torch
from torch.utils.data import DataLoader
import numpy as np

from lib.data.data_enums import Confounder, ForecastingMode
from .forecasting_data import ForecastingData
from .dataset import ExplainedItem, WindowedSequentialDataset
from lib.util import get_debug_dataset_creation
from darts.utils.missing_values import fill_missing_values
from einops import repeat

from darts.datasets.dataset_loaders import DatasetLoader


class ConfoundedSourceData(ForecastingData):
    def __init__(self, darts_loader: DatasetLoader, **kwargs):
        super().__init__(**kwargs)
        self.darts_loader = darts_loader

        self.stride = int((self.lookback + self.prediction_horizon) / 2)
        suffix = f"_confounded_h{self.prediction_horizon}_l{self.lookback}_f{self.confounder_freq_len}x{self.confounder_freq_strength}"
        self.data_path = self.darts_loader._get_path_dataset()
        self.data_path = self.data_path.with_stem(self.data_path.stem + suffix)
        self.data_path = self.data_path.with_suffix(".npz")
        self.train_path = self.data_path.with_stem(
            f"{self.data_path.stem}_{self.confounder}_TRAIN"
        )
        if self.val_confounder != Confounder.NO_CONFOUNDER:
            self.val_path = self.data_path.with_stem(
                f"{self.data_path.stem}_{self.val_confounder}_VAL"
            )
        else:
            self.val_path = self.data_path.with_stem(self.data_path.stem + "_VAL")

        self.test_path = self.data_path.with_stem(
                f"{self.data_path.stem}_{self.val_confounder}_TEST"
            )

        self.has_test = isinstance(self.split_ratio, tuple)

    def confound_timeseries(
        self,
        ts: np.ndarray,
        horizon: int = 1,
        lookback: int = 5,
        stride: int = 2,
        zeros: bool = False,
    ):
        mask = np.zeros_like(ts)
        for i in range(0, len(ts) // (stride * 2)):
            idx = i * stride * 2
            hor_pos = idx + lookback
            ts[idx : horizon + idx] = (
                np.zeros_like(ts[hor_pos : hor_pos + horizon])
                if zeros
                else ts[hor_pos : hor_pos + horizon]
            )
            if not zeros:
                mask[idx : horizon + idx] = 1

        return ts, mask
    
    def confound_freq_and_spatial_timeseries(
        self,
        ts: np.ndarray,
        horizon: int = 1,
        lookback: int = 5,
        stride: int = 2,
        zeros: bool = False,
    ):
        mask = np.zeros_like(ts)
        t = torch.arange(0, 1.0, 1 /self.lookback)
        freq_mask = np.zeros_like(t.numpy())
        if Confounder.FORECASTING_DIRAC in self.confounder:
            
            expl_penalty_ts = torch.zeros_like(t)
            for l in range(self.confounder_freq_len):
                expl_penalty_ts[l::8] = self.confounder_freq_strength
            expl_penalty_freq = torch.fft.rfft(
                expl_penalty_ts  # , norm="backward"
            )  # TODO only unvaiariate
            freq_mask = expl_penalty_freq.numpy()

        for i in range(0, len(ts) // (stride * 2)):
            idx = i * stride * 2
            hor_pos = idx + lookback
            ts[idx : horizon + idx] = ts[hor_pos : hor_pos + horizon]
            
            if not zeros:
                mask[idx : horizon + idx] = 1

            ts[horizon + idx: hor_pos] += expl_penalty_ts[horizon: hor_pos].numpy()  

        return ts, mask, freq_mask

    def prepare_data(self) -> None:
        if not self.darts_loader._is_already_downloaded():
            self.darts_loader._download_dataset()
            self.train_path.unlink(missing_ok=True)
            self.val_path.unlink(missing_ok=True)
            self.test_path.unlink(missing_ok=True)

        if (
            not get_debug_dataset_creation()
            and self.train_path.exists()
            and self.val_path.exists()
            and (self.test_path.exists() or not isinstance(self.split_ratio, tuple) )
        ):
            return
        else:
            self.train_path.unlink(missing_ok=True)
            self.val_path.unlink(missing_ok=True)
            self.test_path.unlink(missing_ok=True)

            print(f"Creating confounded dataset at {self.data_path}")
            dts = self.darts_loader.load()

            train_dts, val_dts, test_dts = self.split_series_dataset(dts)

            ts = self.process_forecasting_mode(
                train_dts, val_dts, test_dts, self.target_feature_names
            )
            ts_darts = TimeSeries.from_values(ts)
            ts_darts = fill_missing_values(ts_darts)
            ts = ts_darts.values().squeeze()

            res = {"original_ts": ts, "confounded_ts": ts, "mask": np.zeros_like(ts), "freq": None}

            if (Confounder.FORECASTING_NOISE in self.confounder or Confounder.FORECASTING_DIRAC in self.confounder) and Confounder.FORECASTING_TIME in self.confounder:
                    ts_new, mask, freq_mask = self.confound_freq_and_spatial_timeseries(
                        res["confounded_ts"], self.prediction_horizon, self.lookback, stride=self.stride
                    )
                    res["confounded_ts"] = ts_new
                    res["mask"] = mask
                    res["freq"] = freq_mask
            else:
                if Confounder.FORECASTING_NOISE in self.confounder:
                    t = torch.arange(0, 1.0, 1 /self.lookback)
                    tiles_num = math.ceil(ts.shape[0] / self.lookback)
                    # scaler = self.confounder_freq # 20 # int(np.quantile(np.arange(len(t)),0.65))
                    scaler = 60 #int(np.quantile(np.arange(len(t)),0.65))
                    # amp = 10000 #int(torch.quantile(torch.from_numpy(ts).float(),torch.tensor([0.95])).item())
                    amp = self.confounder_freq_strength
                    # expl_penalty_ts = torch.normal(t) * amp #torch.cos(2.0 * torch.pi * scaler * t) * amp + torch.sin(2.0 * torch.pi * scaler * t) * amp

                    expl_penalty_ts = torch.sin(2.0 * torch.pi * 2 * scaler * t) * amp

                    expl_penalty_freq = torch.fft.rfft(
                        expl_penalty_ts  # , norm="backward"
                    )  # TODO only unvaiariate
                    res["freq"] = expl_penalty_freq.numpy()

                    expl_penalty_ts_full = repeat(expl_penalty_ts, 't -> (tiles t)', tiles=tiles_num)[:ts.shape[0]].numpy()

                    res["confounded_ts"] += expl_penalty_ts_full
            
                if Confounder.FORECASTING_DIRAC in self.confounder:
                    t = torch.arange(0, 1.0, 1 /self.lookback)
                    tiles_num = math.ceil(ts.shape[0] / self.lookback)
                    
                    expl_penalty_ts = torch.zeros_like(t)
                    for l in range(self.confounder_freq_len):
                        expl_penalty_ts[l::8] = self.confounder_freq_strength

                    expl_penalty_ts_full = repeat(expl_penalty_ts, 't -> (tiles t)', tiles=tiles_num)[:ts.shape[0]].numpy()
                    res["confounded_ts"] += expl_penalty_ts_full

                    expl_penalty_freq = torch.fft.rfft(
                        expl_penalty_ts  # , norm="backward"
                    )  # TODO only unvaiariate
                    res["freq"] = expl_penalty_freq.numpy()
            
                if Confounder.FORECASTING_TIME in self.confounder:
                    ts_new, mask = self.confound_timeseries(
                        res["confounded_ts"], self.prediction_horizon, self.lookback, stride=self.stride
                    )
                    res["confounded_ts"] = ts_new
                    res["mask"] = mask
            

            if Confounder.SANITY in self.confounder:
                ts_new, mask = self.confound_timeseries(
                    ts,
                    self.prediction_horizon,
                    self.lookback,
                    stride=self.stride,
                    zeros=True,
                )
                res["confounded_ts"] = ts_new
                res["mask"] = mask

            if Confounder.NO_CONFOUNDER in self.confounder:
                mask = np.zeros_like(ts)
                res["confounded_ts"] = ts
                mask = np.zeros_like(ts)

            if not any(
                confounder in self.confounder
                for confounder in [
                    Confounder.FORECASTING_TIME,
                    Confounder.SANITY,
                    Confounder.FORECASTING_NOISE,
                    Confounder.FORECASTING_DIRAC,
                    Confounder.NO_CONFOUNDER,
                ]
            ):
                raise ValueError("No supported confounder", self.confounder)

            
            np.savez_compressed(self.train_path, **res)

            ts_new = val_dts.values().squeeze()
            mask = np.zeros_like(ts_new)
            
            np.savez_compressed(
                self.val_path,
                confounded_ts=ts_new,
                mask=np.zeros_like(ts_new),
                freq=None,
            )

            if test_dts is not None:
                ts_new = test_dts.values().squeeze()
                mask = np.zeros_like(ts_new)
                
                np.savez_compressed(
                    self.test_path,
                    original_ts=ts_new,
                    mask=np.zeros_like(ts_new),
                    freq=None,
                )

    def setup(self, stage: str = None) -> None:
        # load pkl into dataframe
        np_dict = np.load(self.train_path, allow_pickle=True)
        
        train_ds = TimeSeries.from_values(np_dict["confounded_ts"])
        orig_train_ds = TimeSeries.from_values(np_dict["original_ts"])
        # train_mask = df["mask"].values
        train_mask = np_dict["mask"]
        freq_mask = None
        if not np.all(np.equal(np_dict["freq"], None)):
            freq_mask = np_dict["freq"]
        # train_ds = fill_missing_values(train_ds)

        np_dict = np.load(self.val_path, allow_pickle=True)
        val_ds = TimeSeries.from_values(np_dict["confounded_ts"])
        val_mask = np_dict["mask"]
        val_ds = fill_missing_values(val_ds)


        # self.data_transformer.fit(orig_train_ds)# TODO should we change that back?
        self.data_transformer.fit(train_ds)# TODO should we change that back?
        train_ds = self.data_transformer.transform(train_ds) 

        val_ds = self.data_transformer.transform(val_ds)

        if self.has_test:
            np_dict = np.load(self.test_path, allow_pickle=True)
            test_ds = TimeSeries.from_values(np_dict["original_ts"])
            test_mask = np_dict["mask"]
            test_ds = fill_missing_values(test_ds)

            test_ds = self.data_transformer.transform(test_ds)



        self.train_dataset = WindowedSequentialDataset(
            train_ds,
            lookback=self.lookback,
            prediction_horizon=self.prediction_horizon,
            confounder=self.confounder,
            mask=train_mask,
            freq_mask=freq_mask,
            stride=self.stride,
            padded_zeros=self.sanity_padded and self.confounder == Confounder.SANITY,
            feedback_percentage=self.feedback_percentage,
        )

        self.val_dataset = WindowedSequentialDataset(
            val_ds,
            lookback=self.lookback,
            prediction_horizon=self.prediction_horizon,
            confounder=self.val_confounder,
            mask=val_mask,
            freq_mask=None,
            stride=self.stride,
            padded_zeros=False,

        )

        if self.has_test:
            self.test_dataset = WindowedSequentialDataset(
                test_ds,
                lookback=self.lookback,
                prediction_horizon=self.prediction_horizon,
                confounder=Confounder.NO_CONFOUNDER,
                mask=test_mask,
                freq_mask=None,
                stride=self.stride,
                padded_zeros=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=ExplainedItem.forecasting_collate_fn,
            num_workers=self.num_workers,
        )

    def process_forecasting_mode(
        self, train_dts, val_dts, test_dts, target_feature_names
    ) -> np.ndarray:
        if self.forecasting_mode == ForecastingMode.MULTIVARIATE_UNIVARIATE:
            train_covs = train_dts.drop_columns(target_feature_names)
            val_covs = val_dts.drop_columns(target_feature_names)
            train_targets = train_dts[target_feature_names]
            val_targets = val_dts[target_feature_names]

            if test_dts is not None:
                test_covs = test_dts.drop_columns(target_feature_names)
                test_targets = test_dts[target_feature_names]
            raise NotImplementedError("Not implemented yet")
        else:
            ts = train_dts.values().squeeze()
        return ts
