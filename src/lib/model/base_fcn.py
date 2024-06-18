# Taken from https://github.com/qianlima-lab/time-series-ptms/blob/master/model/tsm_model.py
import torch.nn as nn
import torch


class SqueezeChannels(nn.Module):
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)



class FCN(nn.Module):
    def __init__(self, input_size=1):
        super(FCN, self).__init__()
        trs = False
        # self.expl_computation = False
        # self.bn1_running_mean = None
        # self.bn1_running_var = None
        # def hook_in(module: nn.BatchNorm1d, input: torch.Tensor):
        #     self.bn1_running_mean = module.running_mean
        #     self.bn1_running_var = module.running_var

        # def hook_out(module: nn.BatchNorm1d, args, output: torch.Tensor):
        #     if self.expl_computation:
        #         module.running_mean = self.bn1_running_mean
        #         module.running_var = self.bn1_running_var

        bn1 = nn.BatchNorm1d(128, track_running_stats=trs)
        # bn1.register_forward_pre_hook(hook_in)
        # bn1.register_forward_hook(hook_out)

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=128,
                      kernel_size=8, padding='same'),
            bn1, #, track_running_stats=False / model hooks (vor RRR los mean/std soeichern und dancah überschreiben)
            nn.ReLU()
        )


        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=5, padding='same'),
            nn.BatchNorm1d(256, track_running_stats=trs),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128,
                      kernel_size=3, padding='same'),
            nn.BatchNorm1d(128, track_running_stats=trs),
            nn.ReLU()
        )

        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            SqueezeChannels(),
        )


    def forward(self, x, expl_computation=False):
        if expl_computation:
            self.expl_computation = True
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        self.expl_computation = False
        return x
    
