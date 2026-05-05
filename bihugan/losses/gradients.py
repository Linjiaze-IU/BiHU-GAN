import torch
import torch.nn.functional as F

class SobelGrad2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        gx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]]).float().view(1, 1, 3, 3)
        gy = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]]).float().view(1, 1, 3, 3)
        self.register_buffer("gx", gx)
        self.register_buffer("gy", gy)

    def forward(self, x):
        gx = F.conv2d(x, self.gx, padding=1)
        gy = F.conv2d(x, self.gy, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

def gradient_consistency_loss(x_fake, x_real, grad_op=None):
    if grad_op is None:
        grad_op = SobelGrad2D()
    gf = grad_op(x_fake)
    gr = grad_op(x_real)
    return torch.mean(torch.abs(gf - gr))