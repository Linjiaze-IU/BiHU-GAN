import os
import re
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
from pydicom.errors import InvalidDicomError

try:
    import cv2
except Exception:
    cv2 = None  # OpenCV not available

try:
    import scipy.ndimage as ndi
except Exception:
    ndi = None  # scipy.ndimage not available

from skimage.transform import resize as skimage_resize

def _cv2_or_ski_resize(img2d: np.ndarray, out_h: int, out_w: int,
                       interpolation=None) -> np.ndarray:
    """
    Resize a 2D image using OpenCV if available, otherwise fall back to skimage.
    """
    if img2d.shape[0] == out_h and img2d.shape[1] == out_w:
        return img2d
    if cv2 is not None:
        # Choose interpolation: area for downsampling, bilinear for upsampling
        interp = interpolation if interpolation is not None else (
            cv2.INTER_AREA if img2d.shape[0] > out_h or img2d.shape[1] > out_w else cv2.INTER_LINEAR
        )
        return cv2.resize(img2d, (out_w, out_h), interpolation=interp)
    # Fallback to skimage
    return skimage_resize(img2d, (out_h, out_w), order=1,
                          preserve_range=True, anti_aliasing=True).astype(img2d.dtype)

def _binary_closing(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply 3D binary closing to fill small holes in a binary mask."""
    if ndi is None:
        raise ImportError("scipy.ndimage is required for mask processing.")
    structure = np.ones((ksize, ksize, ksize), dtype=bool)
    return ndi.binary_closing(mask, structure=structure)


def _largest_connected_component_3d(mask: np.ndarray) -> np.ndarray:
    """Extract the largest connected component in a 3D binary mask."""
    if ndi is None:
        raise ImportError("scipy.ndimage is required for mask processing.")
    labeled, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if n <= 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # ignore background
    return labeled == counts.argmax()


def _fill_holes_3d(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside a 3D binary mask."""
    if ndi is None:
        raise ImportError("scipy.ndimage is required for mask processing.")
    return ndi.binary_fill_holes(mask)


def _numeric_sort_key(path: str) -> int:
    """
    Extract an integer sorting key from a filename.
    Used to order DICOM slices by the last number found in the file name.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"\d+", stem)
    return int(nums[-1]) if nums else 0


@lru_cache(maxsize=None)
def _get_series_root_from_slice_path(slice_path: str) -> str:
    """Return the parent directory of a slice path (cached)."""
    return os.path.dirname(slice_path)


def _read_series_sorted(series_root: str) -> List[str]:
    """
    Collect and sort all DICOM files in a directory.
    Files are identified by extension or by attempting pydicom read.
    """
    candidates = [os.path.join(series_root, f) for f in os.listdir(series_root)]
    dcm_files = []
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        if fpath.lower().endswith(('.dcm', '.dicom', '.ima')):
            dcm_files.append(fpath)
        else:
            try:
                pydicom.dcmread(fpath, stop_before_pixels=True)
                dcm_files.append(fpath)
            except InvalidDicomError:
                pass
    return sorted(dcm_files, key=_numeric_sort_key)


def _dcm_pixel_to_hu(ds: pydicom.dataset.FileDataset,
                     slope_override: Optional[float] = None) -> np.ndarray:
    """
    Convert DICOM pixel array to Hounsfield Units (HU) using RescaleSlope and RescaleIntercept.
    Allows overriding the slope if provided.
    """
    img = ds.pixel_array.astype(np.float32)
    intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    if slope_override is not None:
        slope = float(slope_override)
    if slope == 0.0:
        slope = 1.0
    return img * slope + intercept


def _build_paired_root(rootA: str, pair_from: str, pair_to: str) -> str:
    """
    Derive the paired series directory (B) from the A series directory
    by replacing the pair_from token with pair_to in the path.
    """
    parts = rootA.split(os.sep)
    new_parts = [pair_to if part == pair_from else part for part in parts]
    new_parts[-1] = new_parts[-1].replace(pair_from, pair_to)
    return os.sep.join(new_parts)


@dataclass
class VolumeCacheItem:
    """Container for a cached paired series."""
    A_norm: np.ndarray       # Normalized A volume (D x H x W)
    B_norm: np.ndarray       # Normalized B volume
    mask_body_A: np.ndarray  # Body mask for A
    mask_body_B: np.ndarray  # Body mask for B
    rows: int
    cols: int
    A_paths: List[str]       # File paths for A slices
    B_paths: List[str]       # File paths for B slices


class PairedVolumeDataset(Dataset):
    """
    PyTorch Dataset for paired 3D DICOM volumes (A and B).
    Each item is a 2D slice with its paired counterpart and body masks.
    Masks are generated on-the-fly and cached per series.
    """
    def __init__(
        self,
        list_file: str,
        pair_from: str,
        pair_to: str,
        hu_min: float,
        hu_max: float,
        body_hu_thresh: float,
        bone_hu_thresh: float,
        model_input_resolution: int,
        upsample_to_model: bool,
        enable_augmentation: bool,
        slope_for_hu: Optional[float] = None,
        max_cached_series: int = 4,
        rng_seed: int = 0,
    ):
        if ndi is None:
            raise ImportError("scipy.ndimage is required. Install scipy first.")
        if not os.path.exists(list_file):
            raise FileNotFoundError(list_file)

        self.pair_from = pair_from
        self.pair_to = pair_to
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        self.body_hu_thresh = float(body_hu_thresh)
        self.bone_hu_thresh = float(bone_hu_thresh)
        self.model_input_resolution = int(model_input_resolution)
        self.upsample_to_model = bool(upsample_to_model)
        self.enable_augmentation = bool(enable_augmentation)
        self.slope_for_hu = slope_for_hu
        self.max_cached_series = int(max_cached_series)
        self.rng_seed = int(rng_seed)

        # Read all A slice paths from list file
        with open(list_file, "r", encoding="utf-8") as f:
            all_A_slices = [line.strip() for line in f if line.strip()]

        # Deduce unique series directories from slice paths
        series_roots_candidates = sorted({
            _get_series_root_from_slice_path(p) for p in all_A_slices
        })

        valid_roots = []
        valid_num_slices = []
        min_mask_ratio = 0.01  # Minimum fraction of body pixels required

        # Validate each series pair
        for rootA in series_roots_candidates:
            rootB = _build_paired_root(rootA, pair_from, pair_to)
            if not os.path.isdir(rootA) or not os.path.isdir(rootB):
                continue
            A_paths = _read_series_sorted(rootA)
            B_paths = _read_series_sorted(rootB)
            if not A_paths or not B_paths:
                continue
            min_slices = min(len(A_paths), len(B_paths))
            if min_slices <= 0:
                continue

            # Fast shape check using first slice metadata only
            try:
                ds = pydicom.dcmread(A_paths[0], force=True, stop_before_pixels=True)
                rows, cols = int(ds.Rows), int(ds.Columns)
            except Exception as e:
                print(f"Warning: series {rootA} failed header read: {e}, skipped.")
                continue

            # Quick body check on first and middle slice to discard empty series
            try:
                ds_first = pydicom.dcmread(A_paths[0], force=True)
                test_hu = np.clip(_dcm_pixel_to_hu(ds_first, slope_override=self.slope_for_hu),
                                  self.hu_min, self.hu_max)
                if (test_hu > self.body_hu_thresh).sum() / (rows * cols) < min_mask_ratio:
                    mid = A_paths[min_slices // 2]
                    ds_mid = pydicom.dcmread(mid, force=True)
                    test_hu_mid = np.clip(_dcm_pixel_to_hu(ds_mid, slope_override=self.slope_for_hu),
                                         self.hu_min, self.hu_max)
                    if (test_hu_mid > self.body_hu_thresh).sum() / (rows * cols) < min_mask_ratio:
                        print(f"Warning: series {rootA} has empty body mask, skipped.")
                        continue
            except Exception as e:
                print(f"Warning: series {rootA} failed pre-check: {e}, skipped.")
                continue

            valid_roots.append(rootA)
            valid_num_slices.append(min_slices)

        self.series_roots_A = valid_roots
        self._series_num_slices = valid_num_slices
        # Build index map: (series_id, slice_z)
        self._index_map = [
            (sid, z)
            for sid, n_slices in enumerate(valid_num_slices)
            for z in range(n_slices)
        ]

        self._cache: Dict[int, VolumeCacheItem] = {}
        # Independent random generators for worker reproducibility
        self._rng_np = np.random.RandomState(self.rng_seed)
        self._rng_random = random.Random(self.rng_seed)

    def __len__(self):
        """Total number of paired slices."""
        return len(self._index_map)

    def _generate_mask(self, hu_volume: np.ndarray) -> np.ndarray:
        """
        Generate a body mask from a HU volume: threshold, keep largest component,
        fill holes, and apply binary closing.
        """
        mask = (hu_volume > self.body_hu_thresh).astype(bool)
        mask = _largest_connected_component_3d(mask)
        mask = _fill_holes_3d(mask)
        mask = _binary_closing(mask, ksize=3)
        return mask.astype(np.float32)

    def _maybe_load_series(self, sid: int) -> VolumeCacheItem:
        """Load a paired series into memory (with caching)."""
        if sid in self._cache:
            return self._cache[sid]

        rootA = self.series_roots_A[sid]
        rootB = _build_paired_root(rootA, self.pair_from, self.pair_to)

        A_paths = _read_series_sorted(rootA)
        B_paths = _read_series_sorted(rootB)
        min_slices = min(len(A_paths), len(B_paths))
        A_paths = A_paths[:min_slices]
        B_paths = B_paths[:min_slices]

        # Determine actual rows/cols from a valid slice header
        rows, cols = None, None
        for p in A_paths:
            try:
                ds_tmp = pydicom.dcmread(p, force=True, stop_before_pixels=True)
                rows, cols = int(ds_tmp.Rows), int(ds_tmp.Columns)
                break
            except Exception:
                continue
        if rows is None:
            rows, cols = 512, 512   # fallback

        # Read all slices into HU volumes
        huA, huB = [], []
        for a_p, b_p in zip(A_paths, B_paths):
            try:
                dsA = pydicom.dcmread(a_p, force=True)
                dsB = pydicom.dcmread(b_p, force=True)
                huA.append(_dcm_pixel_to_hu(dsA, slope_override=self.slope_for_hu)[None, ...])
                huB.append(_dcm_pixel_to_hu(dsB, slope_override=self.slope_for_hu)[None, ...])
            except Exception as e:
                print(f"Warning: corrupt slice pair {a_p} / {b_p}, using placeholder. Error: {e}")
                place = np.full((rows, cols), self.hu_min, dtype=np.float32)
                huA.append(place[None, ...])
                huB.append(place[None, ...])

        huA = np.concatenate(huA, axis=0)
        huB = np.concatenate(huB, axis=0)

        # Clip HU values
        huA_clip = np.clip(huA, self.hu_min, self.hu_max)
        huB_clip = np.clip(huB, self.hu_min, self.hu_max)

        # Generate body masks
        mask_A = self._generate_mask(huA_clip)
        mask_B = self._generate_mask(huB_clip)

        # Normalize to [-1, 1]
        denom = self.hu_max - self.hu_min
        A_norm = (huA_clip - self.hu_min) / denom * 2.0 - 1.0
        B_norm = (huB_clip - self.hu_min) / denom * 2.0 - 1.0

        item = VolumeCacheItem(
            A_norm=A_norm.astype(np.float32),
            B_norm=B_norm.astype(np.float32),
            mask_body_A=mask_A,
            mask_body_B=mask_B,
            rows=rows, cols=cols,
            A_paths=A_paths, B_paths=B_paths
        )

        # Simple cache eviction (remove oldest)
        while len(self._cache) >= self.max_cached_series:
            self._cache.pop(next(iter(self._cache)), None)
        self._cache[sid] = item
        return item

    def _add_noise(self, x: np.ndarray, snr_db: float = 40.0) -> np.ndarray:
        """Add Gaussian noise at a given signal-to-noise ratio (SNR in dB)."""
        sig_power = float(np.mean(x ** 2)) + 1e-12
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise_std = math.sqrt(noise_power)
        noise = self._rng_np.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        return (x + noise).astype(np.float32)

    def _augment_pair_mask(
        self, A2d: np.ndarray, B2d: np.ndarray,
        MA2d: np.ndarray, MB2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply data augmentation to a paired 2D slice and its masks:
        random horizontal flip, affine transform, brightness jitter, and noise.
        """
        if not self.enable_augmentation:
            return A2d, B2d, MA2d, MB2d

        # Random horizontal flip
        if self._rng_random.random() < 0.5:
            A2d = np.flip(A2d, axis=1).copy()
            B2d = np.flip(B2d, axis=1).copy()
            MA2d = np.flip(MA2d, axis=1).copy()
            MB2d = np.flip(MB2d, axis=1).copy()

        # Random affine parameters: rotation ±15°, translation ±5 pixels
        angle = self._rng_random.uniform(-15.0, 15.0)
        tx = self._rng_random.uniform(-5.0, 5.0)
        ty = self._rng_random.uniform(-5.0, 5.0)

        if cv2 is None:
            A_out, B_out, MA_out, MB_out = A2d, B2d, MA2d, MB2d
        else:
            h, w = A2d.shape
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty

            # Apply affine transform to images (linear interp) and masks (nearest)
            A_out = cv2.warpAffine(A2d, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
            B_out = cv2.warpAffine(B2d, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
            MA_out = cv2.warpAffine(MA2d, M, (w, h), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            MB_out = cv2.warpAffine(MB2d, M, (w, h), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        # Brightness jitter (±5%)
        jitter = self._rng_random.uniform(-0.05, 0.05)
        A_out = A_out * (1.0 + jitter)
        B_out = B_out * (1.0 + jitter)

        # Add Gaussian noise
        A_out = self._add_noise(A_out)
        B_out = self._add_noise(B_out)

        # Clip and binarize masks back to {0, 1}
        return (
            np.clip(A_out, -1.0, 1.0),
            np.clip(B_out, -1.0, 1.0),
            (MA_out > 0.5).astype(np.float32),
            (MB_out > 0.5).astype(np.float32)
        )

    def __getitem__(self, idx: int):
        """Retrieve a paired slice (A, B) with masks, optionally resized and augmented."""
        sid, z = self._index_map[idx]
        item = self._maybe_load_series(sid)

        A2d = item.A_norm[z].copy()
        B2d = item.B_norm[z].copy()
        MA2d = item.mask_body_A[z].copy()
        MB2d = item.mask_body_B[z].copy()

        # Resize to model input resolution if needed
        if self.upsample_to_model and (item.rows != self.model_input_resolution or item.cols != self.model_input_resolution):
            A2d = _cv2_or_ski_resize(A2d, self.model_input_resolution, self.model_input_resolution)
            B2d = _cv2_or_ski_resize(B2d, self.model_input_resolution, self.model_input_resolution)
            if cv2 is not None:
                MA2d = _cv2_or_ski_resize(MA2d, self.model_input_resolution, self.model_input_resolution,
                                         interpolation=cv2.INTER_NEAREST)
                MB2d = _cv2_or_ski_resize(MB2d, self.model_input_resolution, self.model_input_resolution,
                                         interpolation=cv2.INTER_NEAREST)
            else:
                MA2d = skimage_resize(MA2d, (self.model_input_resolution, self.model_input_resolution),
                                      order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
                MB2d = skimage_resize(MB2d, (self.model_input_resolution, self.model_input_resolution),
                                      order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
            MA2d = (MA2d > 0.5).astype(np.float32)
            MB2d = (MB2d > 0.5).astype(np.float32)

        # Apply augmentation (if enabled)
        A2d, B2d, MA2d, MB2d = self._augment_pair_mask(A2d, B2d, MA2d, MB2d)

        # Return as torch tensors with channel dimension
        return {
            "A": torch.from_numpy(A2d)[None, ...].float(),
            "B": torch.from_numpy(B2d)[None, ...].float(),
            "mask_A": torch.from_numpy(MA2d)[None, ...].float(),
            "mask_B": torch.from_numpy(MB2d)[None, ...].float(),
            "A_path": item.A_paths[z],
            "B_path": item.B_paths[z],
            "series_id": sid,
            "slice_idx": z,
        }


def worker_init_fn(worker_id):
    """
    DataLoader worker initialization: give each worker an independent random state
    to avoid repeated augmentations across workers.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if isinstance(dataset, PairedVolumeDataset):
            seed = (torch.initial_seed() + worker_id) % (2**32)
            dataset._rng_np = np.random.RandomState(seed)
            dataset._rng_random = random.Random(seed)