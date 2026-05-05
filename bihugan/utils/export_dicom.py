import os
import datetime
import logging
import numpy as np
import pydicom
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

try:
    import cv2
except Exception:
    cv2 = None  # OpenCV not available
from skimage.transform import resize as skimage_resize

logger = logging.getLogger(__name__)


def _resize_to_shape(img2d: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
    """
    Resize a 2D image to the exact target shape (out_rows, out_cols).
    Uses OpenCV if available, otherwise falls back to skimage.
    """
    if img2d.shape == (out_rows, out_cols):
        return img2d
    if cv2 is not None:
        return cv2.resize(img2d, (out_cols, out_rows), interpolation=cv2.INTER_LINEAR)
    return skimage_resize(img2d, (out_rows, out_cols), order=1,
                          preserve_range=True, anti_aliasing=True).astype(img2d.dtype)


def save_dicom_series_slice(
    image_norm: np.ndarray,
    template_dcm_path: str,
    patient_id: str,
    series_description_tag: str,
    instance_num: int,
    output_root: str,
    hu_min: float,
    hu_max: float,
    patient_uids: dict,
):
    """
    Save a single normalized 2D image as a DICOM slice, using a template DICOM for metadata.

    Args:
        image_norm: Normalized image array in [-1, 1].
        template_dcm_path: Path to the original DICOM slice used as a metadata template.
        patient_id: Patient identifier for DICOM headers.
        series_description_tag: Tag for SeriesDescription (e.g., "sRT-CT").
        instance_num: Slice instance number.
        output_root: Root output directory.
        hu_min, hu_max: HU window limits used during normalization.
        patient_uids: Dictionary caching patient-level UIDs (StudyInstanceUID, SeriesInstanceUID,
                      FrameOfReferenceUID) to keep them consistent across slices.
    Returns:
        The output DICOM file path.
    """
    # Read template DICOM for metadata
    ds = pydicom.dcmread(template_dcm_path, force=True)

    # Convert normalized image back to Hounsfield Units
    image_norm = np.clip(image_norm, -1.0, 1.0).astype(np.float32)
    hu = image_norm * ((hu_max - hu_min) / 2.0) + (hu_max + hu_min) / 2.0
    # Resize to match original slice dimensions
    hu = _resize_to_shape(hu, int(ds.Rows), int(ds.Columns))

    # Get original intercept; use a slope of 1 for consistency
    intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
    slope = 1.0
    pixel_vals = np.round((hu - intercept) / slope).astype(np.int16)

    # Handle MONOCHROME1 inversion (rare)
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_vals = np.iinfo(np.int16).max - pixel_vals
    pixel_vals = np.clip(pixel_vals, -32768, 32767).astype(np.int16)

    # Create a new DICOM dataset with explicit VR little endian transfer syntax
    new_ds = pydicom.Dataset()
    new_ds.is_little_endian = True
    new_ds.is_implicit_VR = False
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    new_ds.file_meta = file_meta

    # Copy attributes from template, skipping tags we want to override
    tags_to_skip = {
        "PixelData", "SeriesInstanceUID", "SOPInstanceUID", "SeriesDescription",
        "SeriesNumber", "InstanceNumber", "RescaleSlope", "RescaleIntercept",
        "ContentDate", "ContentTime", "SOPClassUID",
    }
    for elem in ds:
        if elem.tag == 0x7FE00010:      # Pixel Data tag – we will set it manually
            continue
        if elem.keyword in tags_to_skip or elem.keyword is None:
            continue
        setattr(new_ds, elem.keyword, elem.value)

    # Set mandatory and common DICOM attributes
    new_ds.SOPClassUID = ds.SOPClassUID
    new_ds.Rows = ds.Rows
    new_ds.Columns = ds.Columns
    new_ds.PixelSpacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    new_ds.SliceThickness = getattr(ds, "SliceThickness", 1.0)
    new_ds.ImagePositionPatient = getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])
    new_ds.ImageOrientationPatient = getattr(ds, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    new_ds.RescaleSlope = slope
    new_ds.RescaleIntercept = intercept

    # Insert pixel data
    new_ds.PixelData = pixel_vals.tobytes()
    new_ds.BitsAllocated = 16
    new_ds.BitsStored = 16
    new_ds.HighBit = 15
    new_ds.SamplesPerPixel = 1
    new_ds.PhotometricInterpretation = photometric
    new_ds.PixelRepresentation = 1
    new_ds.SeriesDescription = f"{series_description_tag} (BiHU-GAN)"

    # Manage patient-level UIDs to keep them consistent within a patient
    new_ds.PatientID = patient_id
    if patient_id not in patient_uids:
        patient_uids[patient_id] = (
            ds.StudyInstanceUID,
            generate_uid(),
            getattr(ds, "FrameOfReferenceUID", generate_uid()),
        )
    study_uid, series_uid, frame_uid = patient_uids[patient_id]
    new_ds.StudyInstanceUID = study_uid
    new_ds.SeriesInstanceUID = series_uid
    new_ds.FrameOfReferenceUID = frame_uid
    new_ds.SOPInstanceUID = generate_uid()
    new_ds.InstanceNumber = str(instance_num)

    # Derive a new SeriesNumber (original + 800) to avoid conflicts
    try:
        base_series = int(ds.SeriesNumber)
    except Exception:
        base_series = 1
    new_ds.SeriesNumber = str(base_series + 800)

    # Set current date/time
    now = datetime.datetime.now()
    new_ds.ContentDate = now.strftime("%Y%m%d")
    new_ds.ContentTime = now.strftime("%H%M%S")
    new_ds.setdefault("StudyDate", now.strftime("%Y%m%d"))
    new_ds.setdefault("StudyTime", now.strftime("%H%M%S"))

    # Write DICOM file
    out_dir = os.path.join(output_root, patient_id, series_description_tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{series_description_tag}_{instance_num:04d}.dcm")
    new_ds.save_as(out_path, write_like_original=False)
    return out_path