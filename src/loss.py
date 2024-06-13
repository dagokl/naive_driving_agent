import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialL1Loss(nn.Module):
    def __init__(self, start_index, stop_index, reduction='mean'):
        super().__init__()
        self.start_index = start_index
        self.stop_index = stop_index
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return F.l1_loss(
            output[:, self.start_index : self.stop_index],
            target[:, self.start_index : self.stop_index],
            reduction=self.reduction,
        )


class PartialL2Loss(nn.Module):
    def __init__(self, start_index, stop_index, reduction='mean'):
        super().__init__()
        self.start_index = start_index
        self.stop_index = stop_index
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return F.mse_loss(
            output[:, self.start_index : self.stop_index],
            target[:, self.start_index : self.stop_index],
            reduction=self.reduction,
        )
