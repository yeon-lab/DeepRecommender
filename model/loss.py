import torch
import torch.nn as nn

def MSELoss(output, target):
    cr = nn.MSELoss(reduction="mean")
    return cr(output, target)
