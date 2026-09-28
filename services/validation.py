"""Input validation for both tabs (BRD FR-3.2, FR-4.2)."""
from __future__ import annotations

import io
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from services.insurance_service import (
    AGE_RANGE, BMI_RANGE, CHILDREN_RANGE, REGION_OPTIONS, SEX_OPTIONS, SMOKER_OPTIONS,
)

ALLOWED_EXT = ("jpg", "jpeg", "png", "dcm")
MAX_UPLOAD_MB = 10
MIN_SIDE_PX = 64


def validate_insurance_inputs(age, sex, bmi, children, smoker, region) -> List[str]:
    errors = []
    if age is None or age <= 0:
        errors.append("Age must be greater than 0.")
    elif not AGE_RANGE[0] <= age <= AGE_RANGE[1]:
        errors.append(f"Age must be between {AGE_RANGE[0]} and {AGE_RANGE[1]} (the range the model was trained on).")
    if bmi is None or not BMI_RANGE[0] <= bmi <= BMI_RANGE[1]:
        errors.append(f"BMI must be between {BMI_RANGE[0]:.0f} and {BMI_RANGE[1]:.0f}.")
    if children is None or not CHILDREN_RANGE[0] <= children <= CHILDREN_RANGE[1]:
        errors.append(f"Number of children must be between {CHILDREN_RANGE[0]} and {CHILDREN_RANGE[1]}.")
    if sex not in SEX_OPTIONS or smoker not in SMOKER_OPTIONS or region not in REGION_OPTIONS:
        errors.append("Sex, smoker status and region must be selected from the given options.")
    return errors


def _read_dicom(data: bytes) -> Image.Image:
    import pydicom  # optional dependency

    ds = pydicom.dcmread(io.BytesIO(data))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):  # multi-frame -> first frame
        arr = arr[0]
    if arr.ndim == 3:  # already RGB
        arr = arr[..., :3].mean(axis=-1)
    slope, intercept = float(getattr(ds, "RescaleSlope", 1)), float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept
    lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 1.0 - arr
    return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")


def load_xray(uploaded) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Validate a Streamlit UploadedFile. Returns (RGB image, None) or (None, error message)."""
    if uploaded is None:
        return None, "No file uploaded."
    ext = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ""
    if ext not in ALLOWED_EXT:
        return None, f"Unsupported file type '.{ext}'. Allowed: JPG, PNG, DICOM."
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"File is too large ({uploaded.size / 1e6:.1f} MB). Maximum is {MAX_UPLOAD_MB} MB."
    data = uploaded.getvalue()
    try:
        if ext == "dcm":
            img = _read_dicom(data)
        else:
            probe = Image.open(io.BytesIO(data))
            probe.verify()  # detects corrupted/non-image files
            img = Image.open(io.BytesIO(data)).convert("RGB")
    except ImportError:
        return None, "DICOM support requires the 'pydicom' package."
    except Exception:
        return None, "The file could not be read as a valid chest X-ray image."
    if min(img.size) < MIN_SIDE_PX:
        return None, f"Image is too small ({img.size[0]}x{img.size[1]}). Minimum side is {MIN_SIDE_PX}px."
    return img, None
