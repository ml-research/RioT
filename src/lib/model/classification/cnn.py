from collections import OrderedDict
from typing import cast
import torch
import torch.nn as nn
from .classification_system import ClassificationModelSystem
from lib.loss import Right_Reason_Loss
from lib.util import execute_activation_hook
from abc import ABC, abstractmethod


# class ConvModel(ABC, ClassificationModelSystem):
#     @property
#     @abstractmethod
#     def last_conv(self) -> nn.Module:
#         pass

#     @abstractmethod
#     def last_conv_shape(self, ts_length: int) -> torch.Size:
#         pass


class SimpleUniConvForecast(ClassificationModelSystem):
    def __init__(
        self,
        num_classes: int,
        right_answer_loss: nn.Module,
        lambda_time: float,
        lambda_freq: float,
        right_reason_loss: Right_Reason_Loss | None = None,
    ):
        super().__init__(
            num_classes=num_classes,
            right_answer_loss=right_answer_loss,
            right_reason_loss=right_reason_loss,
            lambda_time=lambda_time,
            lambda_freq=lambda_freq,
        )
        self.conv_encoder = nn.Sequential(
            OrderedDict(
                [
                    ("input_conv", nn.Conv1d(1, 20, 3, 1)),
                    ("relu1", nn.ReLU()),
                    ("max_pool1", nn.MaxPool1d(2, 2)),
                    ("last_conv", nn.Conv1d(20, 50, 3, 1)),
                    ("relu2", nn.ReLU()),
                    ("max_pool2", nn.MaxPool1d(2, 2)),
                    ("flatten", nn.Flatten()),
                ]
            )
        )
        

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("input_fc", nn.LazyLinear(256)),
                    ("relu", nn.ReLU()),
                    ("out_fc", nn.Linear(256, num_classes)),
                ]
            )
        )

    # @property
    # def last_conv(self) -> nn.Module:
    #     return cast(nn.Module, self.conv_encoder.last_conv)

    # def last_conv_shape(self, ts_length: int):
    #     x = torch.rand((1, 1, ts_length))
    #     x_shape = execute_activation_hook(
    #         self.conv_encoder, self.last_conv, x, lambda x: x.shape
    #     )
    #     return x_shape

    def forward(self, x):
        x = self.conv_encoder(x)
        x = self.fc(x)
        return x
