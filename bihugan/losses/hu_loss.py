import torch
import torch.nn as nn

def inv_normalize_hu(x_norm, hu_min=-1000.0, hu_max=2047.0):
    return (x_norm + 1.0) * 0.5 * (hu_max - hu_min) + hu_min

class HUConsistencyLoss(nn.Module):
    def __init__(self, hu_min=-1000.0, hu_max=2047.0,
                 body_hu_thresh=-600.0, bone_hu_thresh=250.0,
                 w_soft=1.0, w_bone=1.5, eps=1e-8):
        super().__init__()
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        self.body_hu_thresh = float(body_hu_thresh)
        self.bone_hu_thresh = float(bone_hu_thresh)
        self.w_soft = float(w_soft)
        self.w_bone = float(w_bone)
        self.eps = float(eps)

    def forward(self, x_norm_ref, x_norm_pred, mask_body=None):
        hu_ref = inv_normalize_hu(x_norm_ref, self.hu_min, self.hu_max)
        hu_pred = inv_normalize_hu(x_norm_pred, self.hu_min, self.hu_max)

        if mask_body is None:
            mask_body = (hu_ref > self.body_hu_thresh).float()
        else:
            mask_body = mask_body.float()

        mask_bone = ((hu_ref >= self.bone_hu_thresh) & (mask_body > 0.5)).float()
        mask_soft = ((hu_ref < self.bone_hu_thresh) & (mask_body > 0.5)).float()

        w = mask_soft * self.w_soft + mask_bone * self.w_bone
        diff = torch.abs(hu_pred - hu_ref)

        numer = (mask_body * w * diff).sum()
        denom = mask_body.sum().clamp_min(1.0)
        return numer / (denom + self.eps)