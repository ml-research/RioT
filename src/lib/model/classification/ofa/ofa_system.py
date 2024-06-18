from typing import Any
from .gpt4ts import GPT4ts
from ..classification_system import ClassificationModelSystem
import torch.nn as nn
from lib.loss import Right_Reason_Loss


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


class OFA(ClassificationModelSystem):
    def __init__(
        self,
        num_classes: int,
        right_answer_loss: nn.Module,
        max_seq_len: int,
        patch_size: int,
        stride: int,
        dropout: float,
        d_model: int = 768,
        feat_dim: int = 1,
        right_reason_loss: Right_Reason_Loss | None = None,
        lambda_time: float = 1.0,
        lambda_freq: float = 1.0,
    ):
        super().__init__(
            num_classes=num_classes,
            right_answer_loss=right_answer_loss,
            right_reason_loss=right_reason_loss,
            lambda_time=lambda_time,
            lambda_freq=lambda_freq,
        )

        self.model = GPT4ts(
            max_seq_len, patch_size, stride, dropout, num_classes, d_model, feat_dim
        )
        # self.backbone = Backbone(input_size=1)
        # self.classifier = nn.LazyLinear(num_classes)

        # for name, param in self.model.named_parameters():
        #     if name.startswith("output_layer"):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        print("Model:\n{}".format(self.model))
        print("Total number of parameters: {}".format(count_parameters(self.model)))
        print(
            "Trainable parameters: {}".format(
                count_parameters(self.model, trainable=True)
            )
        )

    def forward(self, x):
        out = self.model(x.transpose(1, 2))
        return out
    
    # only needed for a single freq run and seed, due to current gpu issues in our cluster
    # def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
    #     # only needed for freq
    #     # Custom hyperparameter scheduler logic
    #     if self.current_epoch == 0 and batch_idx == 200:
    #         if int(self.hparams.get("seed_everything", 34234)) == 34234:
    #                 self.lambda_freq *= 10  # Change the parameter value after the first epoch due to a bug
    #                 print(f"changed_seed to {self.lambda_freq}")
    #                 # self.log('my_param', self.my_param, prog_bar=True)
    #         # if int(self.hparams.get("seed_everything", 34237)) == 34237:
    #         #         self.lambda_freq *= 10  # Change the parameter value after the first epoch due to a bug
    #         #         print(f"changed_seed to {self.lambda_freq}")
    #         #         # self.log('my_param', self.my_param, prog_bar=True)
            


    #     return super().on_train_batch_end(outputs, batch, batch_idx)

    # def on_train_epoch_end(self):
    #     # Custom hyperparameter scheduler logic
    #     if int(self.hparams.get("seed_everything", 34234)) == 34234:
    #         if self.current_epoch == 1:
    #             self.lambda_freq *= 10  # Change the parameter value after the first epoch due to a bug
    #             print(f"changed_seed to {self.lambda_freq}")
    #             # self.log('my_param', self.my_param, prog_bar=True)
