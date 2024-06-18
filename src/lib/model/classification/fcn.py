import torch.nn as nn
from .classification_system import ClassificationModelSystem
from lib.loss import Right_Reason_Loss
from ..base_fcn import FCN as Backbone

# not used
class Classifier(nn.Module):
    def __init__(self, input_dims, output_dims) -> None:
        super(Classifier, self).__init__()

        self.dense = nn.Linear(input_dims, output_dims)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.dense(x))



class FCN(ClassificationModelSystem):
    def __init__(self, num_classes: int, right_answer_loss: nn.Module, right_reason_loss: Right_Reason_Loss | None = None, lambda_time: float = 1.0, lambda_freq: float = 1.0):
        super().__init__(
            num_classes=num_classes,
            right_answer_loss=right_answer_loss,
            right_reason_loss=right_reason_loss,
            lambda_time=lambda_time,
            lambda_freq=lambda_freq,
        )

        self.backbone = Backbone(input_size=1)
        self.classifier = nn.LazyLinear(num_classes)


    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

