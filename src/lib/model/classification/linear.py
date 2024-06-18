from .classification_system import ClassificationModelSystem
from torch import nn
from lib.loss import Right_Reason_Loss


class MLPModel(ClassificationModelSystem):
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
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x
