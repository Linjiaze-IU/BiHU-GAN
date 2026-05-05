import torch

def cycle_l1_loss(x_src, x_cycle):
    return torch.mean(torch.abs(x_src - x_cycle))