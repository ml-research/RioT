from dataclasses import dataclass
from datasets import load_dataset
from functools import reduce
from itertools import chain
from pathlib import Path
from typing import Callable, Literal
import numpy as np
import torch
from torch.utils.data import Dataset
from darts.timeseries import TimeSeries
from darts.utils.data.sequential_dataset import PastCovariatesSequentialDataset
from torchvision import transforms
from einops import rearrange, repeat
from .data_enums import Confounder, P2SFeedbackMode
from .utils import create_mask, make_patched_ucr_transform
import warnings
from torchchronos.datasets import UCRUEADataset
from torchchronos.typing import DatasetSplit
from torchchronos.transforms import Transform
import os


@dataclass
class FeedbackPenalty:
    freq: torch.Tensor | None
    time: torch.Tensor | None

    def append(self, val: "FeedbackPenalty"):
        if self.freq is not None and val.freq is not None:
            self.freq = torch.cat([self.freq, val.freq], dim=0)
        elif val.freq is not None:
            self.freq = val.freq

        if self.time is not None and val.time is not None:
            self.time = torch.cat([self.time, val.time], dim=0)
        elif val.time is not None:
            self.time = val.time

    def __getitem__(self, key):
        if isinstance(key, slice) or not isinstance(key, int):
            return FeedbackPenalty(
                self.freq[key] if self.freq is not None else None,
                self.time[key] if self.time is not None else None,
            )
        else:
            return FeedbackPenalty(
                self.freq[key : key + 1] if self.freq is not None else None,
                self.time[key : key + 1] if self.time is not None else None,
            )


@dataclass
class ExplainedItem:
    x: torch.Tensor
    y: torch.Tensor
    expl_x: FeedbackPenalty | None

    def append(self, val: "ExplainedItem"):
        if self.expl_x is not None and val.expl_x is not None:
            self.expl_x.append(val.expl_x)
        elif val.expl_x is not None:
            self.expl_x = val.expl_x

        self.x = torch.cat([self.x, val.x], dim=0)
        self.y = torch.cat([self.y, val.y], dim=0)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, key):
        # if isinstance(key,slice):
        penalty = None
        if self.expl_x is not None:
            penalty = self.expl_x[key]
        if isinstance(key, slice) or not isinstance(key, int):
            return ExplainedItem(x=self.x[key], y=self.y[key], expl_x=penalty)
        else:
            return ExplainedItem(
                x=self.x[key : key + 1], y=self.y[key : key + 1], expl_x=penalty
            )

    @classmethod
    def __base_collate_fn(cls, batch: list["ExplainedItem"]):
        # TODO: Use custom __getitem and append
        tf: Callable[[torch.Tensor], torch.Tensor] = lambda x: rearrange(
            x, "batch ts var -> batch var ts"
        )

        if batch[0].expl_x is None:
            freq = None
            time = None
        else:
            freq = (
                tf(
                    torch.stack(
                        reduce(lambda acc, elem: acc + [elem.expl_x.freq], batch, [])
                    )
                )
                if batch[0].expl_x.freq is not None
                else None
            )
            time = (
                tf(
                    torch.stack(
                        reduce(lambda acc, elem: acc + [elem.expl_x.time], batch, [])
                    )
                )
                if batch[0].expl_x.time is not None
                else None
            )

        return tf, cls(
            tf(
                torch.stack(reduce(lambda acc, elem: acc + [elem.x], batch, []))
            ).float(),
            torch.tensor([]),
            (
                FeedbackPenalty(freq, time)
                if freq is not None or time is not None
                else None
            ),
        )

    @classmethod
    def forecasting_collate_fn(cls, batch: list["ExplainedItem"]) -> "ExplainedItem":
        tf, item = cls.__base_collate_fn(batch)
        item.y = tf(
            torch.stack(reduce(lambda acc, elem: acc + [elem.y], batch, []))
        ).float()

        return item

    @classmethod
    def classification_collate_fn(cls, batch: list["ExplainedItem"]) -> "ExplainedItem":
        tf, item = cls.__base_collate_fn(batch)

        item.y = torch.stack(reduce(lambda acc, elem: acc + [elem.y], batch, [])).long()

        return item


class P2SDataset(Dataset):
    """
    TODO
    """

    def __init__(
        self,
        mode: Literal["Decoy", "Normal"],
        scaler: Transform,
        feedback_mode: P2SFeedbackMode = P2SFeedbackMode.NO_FEEDBACK,
        split: Literal["train", "test"] = "train",
    ) -> None:
        super().__init__()
        self.transform = scaler

        self.hg_dataset = load_dataset(
            "json", data_files={split: self.make_url(split, mode.lower())}
        )
        self.x_key = "dowel_deep_drawing_ow"
        self.feedback_mode = feedback_mode

        if split == "train":  # we base our transform parameters on the train set
            self.transform = self.transform.fit(
                self.hg_dataset.with_format("torch")[self.x_key]
            )

        def _transform(data, limited_feedback: bool = False):
            res = {}
            res["mask"] = torch.tensor(data["mask"], dtype=torch.float64).unsqueeze(-1)
            res["label"] = torch.tensor(data["label"], dtype=torch.float64)
            res["speed"] = torch.tensor(data["speed"])

            res[self.x_key] = self.transform(
                torch.tensor(data[self.x_key], dtype=torch.float64)
            ).unsqueeze(-1)

            if limited_feedback:
                res["mask"][..., 800:950] = res["mask"][..., 3250:3550] = 0

            return res

        self.hg_dataset.set_transform(
            lambda data: _transform(
                data, feedback_mode == P2SFeedbackMode.LIMITED_FEEDBACK
            )
        )

    @classmethod
    def make_url(cls, kind: str, config: str):
        return f"https://anonymous.4open.science/r/p2s/datasets/p2s-{config}/{kind}/dataset_info.json"

    @classmethod
    def download_p2s(cls, mode: Literal["Decoy", "Normal"]):
        _ = load_dataset(
            "json",
            data_files={
                "train": cls.make_url("train", mode.lower()),
                "test": cls.make_url("test", mode.lower()),
            },
        )

    def __len__(self) -> int:
        return self.hg_dataset.num_rows

    def __getitem__(self, index: int) -> ExplainedItem:
        item = self.hg_dataset[index]
        feedback = None
        if self.feedback_mode != P2SFeedbackMode.NO_FEEDBACK:
            feedback = FeedbackPenalty(freq=None, time=item["mask"])
        return ExplainedItem(item[self.x_key], item["label"], feedback)


class MechanicalClassificationDataset(Dataset):
    """
    This dataset resembles a wrapper around data which is in itself confounded. True explanations are provided via a lookup table
    """

    def __init__(
        self,
        cache_dir: Path,
        scaler: Transform,
        feedback_range: chain | None = None,
        split: Literal["train", "test"] = "train",
    ) -> None:
        super().__init__()
        is_train = split == "train"
        self.path = cache_dir / "train.npz" if is_train else cache_dir / "test.npz"
        self.transform = scaler

        data = np.load(self.path)
        self.xs = torch.from_numpy(data["x"])
        self.ys = torch.from_numpy(data["y"])
        self.speeds = torch.from_numpy(data["speed"])
        self.expl_p = None

        if is_train:  # we base our transform parameters on the train set
            self.transform = self.transform.fit(self.xs)
            if feedback_range is not None:
                self.expl_p = torch.zeros_like(self.xs)
                self.expl_p[:, list(feedback_range)] = 1

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, index: int) -> ExplainedItem:
        x = self.transform(self.xs[index]).unsqueeze(-1)
        feedback = None
        if self.expl_p is not None:
            feedback = FeedbackPenalty(freq=None, time=self.expl_p[index].unsqueeze(-1))
        return ExplainedItem(x, self.ys[index], feedback)


class ConfoundingDataset(UCRUEADataset):
    def __init__(
        self,
        confounder: Confounder,
        confounder_len: int,
        confounder_ampl: float,
        ds_name: str,
        split: DatasetSplit,
        padded_zeros: bool,
        scaler: Transform,
    ):
        super().__init__(
            ds_name,
            path=Path(os.getcwd()) / ".cache" / "data",
            split=split,
            transform=scaler,
        )

        if confounder not in [
            Confounder.CLASSIFICATION_TIME,
            Confounder.CLASSIFICATION_FREQ,
            Confounder.CLASSIFICATION_TIME | Confounder.CLASSIFICATION_FREQ,
            Confounder.CLASSIFICATION_TIME
            | Confounder.CLASSIFICATION_FREQ
            | Confounder.SANITY,
            Confounder.CLASSIFICATION_TIME | Confounder.SANITY,
            Confounder.CLASSIFICATION_FREQ | Confounder.SANITY,
            Confounder.SANITY,
            Confounder.NO_CONFOUNDER,
        ]:
            raise ValueError("Confounder for classification not supported")

        max_confounder_width = self.series_length // self.num_classes

        self._confounder_width = min(max_confounder_width, confounder_len)
        self._confounder_ampl = confounder_ampl

        self.feedback_idxs = None

        self.transform = make_patched_ucr_transform(
            self.num_classes,
            confounder=confounder,
            confounder_ampl=confounder_ampl,
            confounder_width=self._confounder_width,
            scaler=scaler,
        )

        self.confounder = confounder

    def set_feedback_idxs(self, feedback_idxs: list[int]):
        self.feedback_idxs = feedback_idxs

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.LongTensor]:
        x, expl_p_time, expl_p_freq = self.transform(self.ys[index], self.xs[index])

        if self.feedback_idxs is not None and index not in self.feedback_idxs:
            if expl_p_time is not None:
                expl_p_time = torch.zeros_like(expl_p_time)
            if expl_p_freq is not None:
                expl_p_freq = torch.zeros_like(expl_p_freq)

        return ExplainedItem(
            x.T,
            self.ys[index],
            (
                FeedbackPenalty(freq=expl_p_freq, time=expl_p_time)
                if expl_p_freq is not None or expl_p_time is not None
                else None
            ),
        )


class WindowedSequentialDataset(Dataset):
    """
    Dataset for custom windowing of any sequential data. The dataset is a list of tuples (x, y, expl_x) where x is the input sequence, y is the target sequence and expl_x is the explanation sequence.

    Args:
        series: TimeSeries or list of covariates and targets (in this order)
        lookback: Number of time steps to look back
        prediction_horizon: Number of time steps to predict
        stride: Window stride
        confounder: Confounder to use
        mask: Mask of counfounder
        freq_mask: FFT mask of confounder
        padded_zeros: Whether to pad zeros to the beginning of the sequence (this is needed for the sanity check)
        transform: Transform to apply to the data

    """

    def __init__(
        self,
        series: TimeSeries | list[TimeSeries],
        lookback: int,
        prediction_horizon: int,
        stride: int,
        confounder: Confounder,
        mask: np.ndarray,
        freq_mask: np.ndarray | None,
        padded_zeros: bool,
        transform: Callable[
            [np.ndarray | torch.Tensor], torch.Tensor
        ] = transforms.ToTensor(),
        feedback_percentage: float = 1.0,
    ):
        super().__init__()
        self.transform = transform
        self.lookback = lookback
        self.prediciton_horizon = prediction_horizon
        self.stride = stride
        self.confounder = confounder
        self.padded_zeros = padded_zeros

        x, y, mask, _, freq_mask, _ = self._split_windows(
            series.values(), mask, freq_mask
        )
        self.x = x
        self.y = y

        feedback = mask[::2]
        if feedback_percentage < 1:
            feedback[-int(len(feedback) * (1 - feedback_percentage)) :] = 0

        self.mask = mask[..., None]
        if freq_mask is not None:
            if feedback_percentage < 1:
                freq_mask[-int(len(freq_mask) * (1 - feedback_percentage)) :] = 0
            self.freq_mask = freq_mask[..., None]
        else:
            self.freq_mask = None

    def _split_windows(
        self, ts: np.ndarray, mask: np.ndarray, freq_mask: np.ndarray | None
    ):
        lookback_times = []
        horizon_times = []
        lookback_masks = []
        horizon_masks = []

        num_conf = 0
        for i in range(
            0, len(ts) - self.lookback - self.prediciton_horizon, self.stride
        ):
            lookback_time = ts[i : i + self.lookback]
            horizon_time = ts[
                i + self.lookback : i + self.lookback + self.prediciton_horizon
            ]
            lookback_mask = mask[i : i + self.lookback]

            horizon_mask = mask[
                i + self.lookback : i + self.lookback + self.prediciton_horizon
            ]
            """Copilot says thats faster:
            
            def split_windows(self, ts: np.ndarray, mask: np.ndarray):
                end_index = len(ts) - self.lookback - self.horizon
                ranges = [(i, i + self.lookback, i + self.lookback, i + self.lookback + self.horizon) for i in range(0, end_index, self.stride)]
                
                lookback_times = [ts[start:stop] for start, stop, _, _ in ranges]
                horizon_times = [ts[start:stop] for _, _, start, stop in ranges]
                lookback_masks = [mask[start:stop] for start, stop, _, _ in ranges]
                horizon_masks = [mask[start:stop] for _, _, start, stop in ranges]

                # Do something with the created lists

            """
            if np.sum(lookback_mask) > 0 and lookback_mask[0] != 1:
                lookback_mask = np.zeros_like(lookback_mask)
                # continue
            # if Confounder.NO_CONFOUNDER in self.confounder or (np.sum(horizon_mask) ==0 and (np.sum(lookback_mask) == 0 or (np.sum(lookback_mask) > 0 and lookback_mask[0] == 1))):
            if lookback_mask[0] == 1:
                num_conf += 1
            lookback_times.append(lookback_time)
            horizon_times.append(horizon_time)
            lookback_masks.append(lookback_mask)
            horizon_masks.append(horizon_mask)

        if freq_mask is not None:
            freq_mask_samples = np.stack([freq_mask] * len(lookback_times))
        else:
            freq_mask_samples = None

        return (
            np.stack(lookback_times),
            np.stack(horizon_times),
            np.stack(lookback_masks),
            np.stack(horizon_masks),
            freq_mask_samples,
            freq_mask_samples,
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int) -> ExplainedItem:
        x = self.transform(self.x[idx]).squeeze(0)
        y = self.transform(self.y[idx]).squeeze(0)
        if self.freq_mask is None:
            freq_mask = None
        elif (
            Confounder.FORECASTING_NOISE in self.confounder
            or Confounder.FORECASTING_DIRAC in self.confounder
        ):
            freq_mask = torch.tensor(self.freq_mask[idx])

        if Confounder.NO_CONFOUNDER not in self.confounder:
            mask = torch.tensor(self.mask[idx]).float()

            # if self.padded_zeros: TODO we do that directly in the dataset
            #     x[mask] = 0
        else:
            mask = None
            freq_mask = None

        return ExplainedItem(x, y, FeedbackPenalty(freq=freq_mask, time=mask))


class SequentialDataset(Dataset):
    """
    Dataset for sequential data. The dataset is a list of tuples (x, y, expl_x) where x is the input sequence, y is the target sequence and expl_x is the explanation sequence.

    Args:
        series: TimeSeries or list of covariates and targets (in this order)
        lookback: Number of time steps to look back
        prediction_horizon: Number of time steps to predict
        confounder: Confounder to use
        padded_zeros: Whether to pad zeros to the beginning of the sequence (this is needed for the sanity check)
        target_feature_names: Name of the target feature(s)
        transform: Transform to apply to the data

    """

    def __init__(
        self,
        series: TimeSeries | list[TimeSeries],
        lookback: int,
        prediction_horizon: int,
        confounder: Confounder,
        padded_zeros: bool,
        target_feature_names: str | list[str],
        transform: Callable[
            [np.ndarray | torch.Tensor], torch.Tensor
        ] = transforms.ToTensor(),
    ):
        self.transform = transform

        idx = self._compute_confounder_position(
            confounder, past_len=lookback, future_len=prediction_horizon
        )

        if isinstance(series, list):
            if isinstance(idx, list):
                raise NotImplementedError("Not implemented for multiple confounders")

            covariates = series[0]
            targets = series[1]

            self.dart_dataset = PastCovariatesSequentialDataset(
                targets,
                covariates=covariates,
                input_chunk_length=lookback,
                output_chunk_length=prediction_horizon,
                use_static_covariates=False,
            )

            out_target_shape = (1, lookback, targets.n_components)
            out_cov_shape = (1, lookback, covariates.n_components)
            self.expl_penalty = self._compute_explanation_mask(
                idx, confounder, prediction_horizon, out_cov_shape, out_target_shape
            )
        else:
            self.dart_dataset = PastCovariatesSequentialDataset(
                series,
                input_chunk_length=lookback,
                output_chunk_length=prediction_horizon,
                use_static_covariates=False,
            )
            out_shape = (1, lookback, series.n_components)
            self.expl_penalty = self._compute_explanation_mask(
                idx, confounder, prediction_horizon, None, out_shape
            )

        self.confounder = confounder
        self.padded_zeros = padded_zeros
        self.target_feature_names = target_feature_names

    def __len__(self):
        return len(self.dart_dataset)

    def _compute_explanation_mask(
        self,
        idx: int | list[int] | None,
        confounder: Confounder,
        prediction_horizon: int,
        covariates_shape: tuple[int, int, int] | None,
        targets_shape: tuple[int, int, int],
    ) -> np.ndarray | None:
        if idx is None:
            # we don't need to compute an explanation/penalty mask
            return None
        if isinstance(idx, list):
            target_expl_penalty = np.zeros_like(targets_shape).astype(bool)[0]
            for pos in idx:
                target_expl_penalty ^= create_mask(
                    targets_shape, prediction_horizon // 4, pos, 0
                )
        else:
            target_expl_penalty = create_mask(targets_shape, prediction_horizon, idx, 0)

        target_expl_penalty = target_expl_penalty.reshape(targets_shape)

        if Confounder.COVARIATE in confounder and covariates_shape is not None:
            cov_expl_penalty = create_mask(covariates_shape, prediction_horizon, idx, 0)
            if Confounder.SINGLE in confounder:
                cov_expl_penalty[:, 1:] = False

            cov_expl_penalty = cov_expl_penalty.reshape(covariates_shape)
            if Confounder.TARGET not in confounder:
                target_expl_penalty = np.zeros_like(target_expl_penalty).astype(bool)
            expl_penalty = np.concatenate(
                [cov_expl_penalty, target_expl_penalty], axis=-1
            )
        elif covariates_shape is not None:
            expl_penalty = np.concatenate(
                [np.zeros(covariates_shape).astype(bool), target_expl_penalty], axis=-1
            )
        else:
            expl_penalty = target_expl_penalty

        expl_penalty = repeat(
            expl_penalty, "n ts var -> (n ds) var ts", ds=self.__len__()
        )

        # TODO: Multivariate confounder version
        if Confounder.MOVING_START in confounder:
            start_idx = 3 if Confounder.KNOWN_START in confounder else 0
            end_idx = 6 if Confounder.KNOWN_START in confounder else 4
            # TODO make the std variable
            shifts = np.random.randint(start_idx, end_idx, (expl_penalty.shape[0],))
            for idx, shift in enumerate(shifts):
                expl_penalty[idx][0] = np.roll(expl_penalty[idx][0], shift.item())

        return expl_penalty.transpose(0, 2, 1)

    def _compute_confounder_position(
        self, confounder: Confounder, past_len: int, future_len: int
    ) -> int | list[int] | None:
        """Computes the confounder starting position in the past sequence

        In the case that INSERT_SPLIT_4 is in confounder, then we need to compute the indices of the 4 confounder parts e.g. future_len // 4 per confounder part
        assert (conf_idx_3 + future_len // 4) < past_len
        if INSERT_START is in confounder, then conf_idx_0 = 0 with spacing 3 to next confounder part
        if INSERT_MIDDLE is in confounder, then conf_idx_0 = past_len // 2 with spacing 3 to next confounder part

        """
        if Confounder.INSERT_START in confounder:
            indices = [0]
            if Confounder.INSERT_SPLIT_4 in confounder:
                spacing = future_len // 4 + 3
                for i in range(3):
                    indices.append(indices[-1] + spacing)
                    assert indices[-1] < past_len, "Index out of bounds"
                return indices

            return indices[0]
        elif Confounder.INSERT_MIDDLE in confounder:
            start_idx = past_len // 2

            if Confounder.INSERT_SPLIT_4 in confounder:
                warnings.warn("INSERT_SPLIT_4 not really tested")
                spacing = future_len // 4 + 3
                indices = [start_idx]
                for i in range(3):
                    indices.append(indices[-1] + spacing)
                    assert indices[-1] < past_len, "Index out of bounds"
                return indices
            else:
                while start_idx + future_len > (past_len - future_len // 2):
                    start_idx -= 1
                return start_idx
        else:
            return None

    def __getitem__(self, idx: int) -> ExplainedItem:
        past_target, past_covariates, _, future = self.dart_dataset[idx]
        if past_covariates is not None:
            past = np.concatenate([past_covariates, past_target], axis=-1)
        else:
            past = past_target

        # we need to copy the array to avoid issues with overlapping memory
        past = self.transform(past.copy())
        future = transforms.ToTensor()(future.copy())

        expl_penalty: torch.Tensor | None = None

        if self.confounder != Confounder.NO_CONFOUNDER:
            if self.expl_penalty is None:
                raise ValueError("Sanity check requires an explanation mask")

            if self.confounder.is_sanity_check:
                if self.padded_zeros:
                    past[:, self.expl_penalty[idx]] = 0
                else:
                    past = past[:, ~self.expl_penalty[idx]]
            else:
                future_len = future.shape[1]

                target_vars = past_target.shape[1]
                covariates_vars = (
                    past_covariates.shape[1] if past_covariates is not None else 0
                )

                if self.confounder.is_only_covariate:
                    future_rep = repeat(
                        future,
                        "batch ts var -> batch ts (var rep)",
                        ts=future_len,
                        rep=covariates_vars,
                    )
                elif (
                    Confounder.SINGLE in self.confounder
                    and Confounder.COVARIATE in self.confounder
                    and Confounder.TARGET in self.confounder
                ):
                    future_rep = repeat(
                        future,
                        "batch ts var -> batch ts (var rep)",
                        ts=future_len,
                        rep=target_vars + 1,
                    )
                elif self.confounder.is_only_target:
                    future_rep = future
                elif self.confounder.is_multivariate:
                    future_rep = repeat(
                        future,
                        "batch ts var -> batch ts (var rep)",
                        ts=future_len,
                        rep=covariates_vars + target_vars,
                    )
                else:
                    future_rep = future

                if False:  # idx % 2 == 0:
                    # mimick confounding every second idx
                    expl_penalty = torch.zeros_like(
                        torch.tensor(self.expl_penalty[idx])
                    ).bool()
                else:
                    expl_penalty = torch.tensor(self.expl_penalty[idx])
                    past[:, expl_penalty] = future_rep.flatten()

                if Confounder.KNOWN_START in self.confounder:
                    past[:, 0:3] = (
                        torch.ones_like(past[:, 0:3])
                        * torch.argwhere(expl_penalty)[0][0].item()
                    )

                if Confounder.INSERT_SPLIT_4 in self.confounder:
                    expl_penalty = FeedbackPenalty(freq=expl_penalty, time=None)
                elif (
                    Confounder.SANITY in self.confounder
                    or Confounder.NO_CONFOUNDER in self.confounder
                ):
                    expl_penalty = None
                else:
                    expl_penalty = FeedbackPenalty(freq=None, time=expl_penalty)

        # TODO forecasting is broken
        return ExplainedItem(
            past.squeeze(0),
            future.squeeze(0),
            expl_penalty,
        )
