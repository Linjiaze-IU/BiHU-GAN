import torch

def lsgan_d_loss(d_real, d_fake):
    real_loss = torch.mean((d_real - 1.0) ** 2)
    fake_loss = torch.mean((d_fake - 0.0) ** 2)
    return 0.5 * (real_loss + fake_loss)

def lsgan_g_loss(d_fake):
    return torch.mean((d_fake - 1.0) ** 2)