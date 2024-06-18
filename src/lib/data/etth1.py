from typing import Literal
from .forecasting_data import ForecastingData
from darts.datasets import ETTh1Dataset


class ETTh1Data(ForecastingData):
    def prepare_data(self) -> None:
        # Download dataset
        _ = ETTh1Dataset().load()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        series_dataset = ETTh1Dataset().load()
        self.post_setup(series_dataset, stage)
