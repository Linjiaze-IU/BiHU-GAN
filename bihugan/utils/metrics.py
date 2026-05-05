import numpy as np
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

def inv_normalize_hu_np(x_norm, hu_min=-1000.0, hu_max=2047.0):
    return (x_norm + 1.0) * 0.5 * (hu_max - hu_min) + hu_min

def _to_numpy_and_squeeze(img, mask=None):
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0, 0]
    elif img.ndim == 3:
        img = img[0]

    if mask is not None:
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        if mask.ndim == 4:
            mask = mask[0, 0]
        elif mask.ndim == 3:
            mask = mask[0]
        mask = (mask > 0.5).astype(np.float32)
        return img.astype(np.float32), mask
    return img.astype(np.float32), None

def compute_ssim_psnr(x_ref, x_pred, data_range=2.0):
    ref, _ = _to_numpy_and_squeeze(x_ref)
    pred, _ = _to_numpy_and_squeeze(x_pred)
    ssim = ssim_metric(ref, pred, data_range=data_range)
    psnr = psnr_metric(ref, pred, data_range=data_range)
    return float(ssim), float(psnr)

def compute_ssim_psnr_masked(x_ref, x_pred, mask, data_range=2.0):
    ref, m = _to_numpy_and_squeeze(x_ref, mask)
    pred, _ = _to_numpy_and_squeeze(x_pred, None)
    if m is None or m.sum() < 10:
        return 0.0, 0.0
    ref_fill = ref * m + (1 - m) * ref[m > 0].mean()
    pred_fill = pred * m + (1 - m) * pred[m > 0].mean()
    ssim = ssim_metric(ref_fill, pred_fill, data_range=data_range)
    psnr = psnr_metric(ref_fill, pred_fill, data_range=data_range)
    return float(ssim), float(psnr)

def compute_nmae_masked(x_fake, x_real, mask):
    fake, m = _to_numpy_and_squeeze(x_fake, mask)
    real, _ = _to_numpy_and_squeeze(x_real, None)
    if m is None or not m.any():
        return 0.0
    mae = np.mean(np.abs(fake[m.astype(bool)] - real[m.astype(bool)]))
    return float(mae / 2.0)

def compute_hu_mad(xA_norm_ref, xA_norm_pred, mask_body=None,
                   hu_min=-1000.0, hu_max=2047.0, empty_mask_fallback=0.0):
    ref, m = _to_numpy_and_squeeze(xA_norm_ref, mask_body)
    pred, _ = _to_numpy_and_squeeze(xA_norm_pred, None)
    if m is None:
        return float(np.mean(np.abs(pred - ref)))
    m = m.astype(bool)
    if not m.any():
        return empty_mask_fallback
    hu_ref = inv_normalize_hu_np(ref, hu_min, hu_max)
    hu_pred = inv_normalize_hu_np(pred, hu_min, hu_max)
    mad = np.mean(np.abs(hu_pred[m] - hu_ref[m]))
    return float(mad)