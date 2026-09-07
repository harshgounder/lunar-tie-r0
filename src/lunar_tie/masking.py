"""UNIT-2 triple mask (valid/nodata/shadow). Gate: 'triple mask on 3 synthetic fixtures'.

Computes a boolean mask triple for a single-band image before matching:
(1) VALID   = pixels inside the actual image footprint (nonzero data area)
(2) NODATA  = pixels equal to the declared nodata constant OR saturated
              (>= declared max)
(3) SHADOW  = pixels likely in cast/permanent shadow. R0 has no SPICE, so
              shadow is approximated geometrically via an Otsu luminance
              threshold per tile: pixels at or below the Otsu low threshold
              (the darker class lower edge). Shadow is BY DEFINITION the dark
              MINORITY class: when the Otsu threshold makes (img <= thr) the
              MAJORITY of the image, the split is a saturation-edge latch, not
              a shadow verdict, and the shadow mask is left empty (a long
              bright tail makes between-class variance rise monotonically to
              the last bin, so otsu latches at the tail edge and a
              majority-dark verdict is that latch, not shadow). The interface
              is exposed so a SPICE-based shadow mask can be swapped in during
              a later rigor unit (2.x).

numpy + stdlib only. No cv2, no skimage, no pandas.
"""

import numpy as np


def otsu_threshold(img):
    """Return the Otsu threshold (float) separating dark from bright pixels.

    Implemented with numpy only. For integer dtypes a bincount over the actual
    value range is used (uint8: 256 bins; uint16: adaptive bincount over the
    observed range). For float dtypes a 256-bin histogram over the data range
    is used. Returns None when the image is single-valued (threshold undefined).
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("img must be a 2D array")
    if img.size == 0:
        return None
    flat = img.ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size < 2:
        return None
    if np.issubdtype(img.dtype, np.integer):
        lo = int(finite.min())
        hi = int(finite.max())
        if lo == hi:
            return None
        counts = np.bincount(finite - lo, minlength=hi - lo + 1).astype(np.float64)
        vals = np.arange(lo, hi + 1, dtype=np.float64)
    else:
        lo = float(finite.min())
        hi = float(finite.max())
        if lo == hi:
            return None
        counts, edges = np.histogram(finite, bins=256, range=(lo, hi))
        counts = counts.astype(np.float64)
        vals = (edges[:-1] + edges[1:]) / 2.0

    total = counts.sum()
    if total == 0:
        return None
    w = np.cumsum(counts)
    mu = np.cumsum(counts * vals)
    total_mu = mu[-1]
    denom = w * (total - w)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = total_mu * w - mu
        var = (num * num) / denom
    var[denom == 0] = 0.0
    idx = int(np.argmax(var))
    return float(vals[idx])


def _default_sat_limit(img):
    if np.issubdtype(img.dtype, np.integer):
        return int(np.iinfo(img.dtype).max)
    return float(np.finfo(img.dtype).max)


def compute_masks(img, nodata_value=0, sat_value=None):
    """Return dict of boolean masks: valid, nodata, shadow, combined.

    combined = valid AND NOT nodata AND NOT shadow.
    nodata_value=None skips nodata masking. sat_value is the declared max
    (saturation) value; default is the dtype max for img.

    Ambiguity policy (shadow): a None threshold (single-valued image) yields
    zero shadow, and so does a threshold whose dark side holds the majority
    of pixels: shadow is by definition the dark minority, so a majority-dark
    Otsu verdict is a saturation-edge latch, not shadow.
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("img must be a 2D array")
    if sat_value is None:
        sat_value = _default_sat_limit(img)

    if nodata_value is None:
        valid = img <= sat_value - 1
        nodata = np.zeros(img.shape, dtype=bool)
    else:
        valid = (img >= nodata_value + 1) & (img <= sat_value - 1)
        nodata = (img == nodata_value) | (img >= sat_value)

    thr = otsu_threshold(img)
    if thr is None:
        shadow = np.zeros(img.shape, dtype=bool)
    elif float((img <= thr).sum()) > 0.5 * img.size:
        # shadow minority rule: shadow is BY DEFINITION the dark minority
        # class. An Otsu threshold whose "dark" side is the majority is the
        # saturation-edge latch (long bright tail -> variance argmax at the
        # last bin), not a shadow verdict; fall back to no shadow mask.
        shadow = np.zeros(img.shape, dtype=bool)
    else:
        shadow = img <= thr

    combined = valid & ~nodata & ~shadow
    return {
        "valid": valid,
        "nodata": nodata,
        "shadow": shadow,
        "combined": combined,
    }


def summary_masks(img, nodata_value=0, sat_value=None):
    """Return percentage summary dict (floats 0-100) for the mask triple."""
    masks = compute_masks(img, nodata_value=nodata_value, sat_value=sat_value)
    total = float(img.size)
    if total == 0:
        return {
            "valid_pct": 0.0,
            "nodata_pct": 0.0,
            "shadow_pct": 0.0,
            "usable_pct": 0.0,
        }
    return {
        "valid_pct": 100.0 * float(masks["valid"].sum()) / total,
        "nodata_pct": 100.0 * float(masks["nodata"].sum()) / total,
        "shadow_pct": 100.0 * float(masks["shadow"].sum()) / total,
        "usable_pct": 100.0 * float(masks["combined"].sum()) / total,
    }


def save_masks(path, masks):
    """Persist a mask dict to an .npz file. Round-trip is exact."""
    np.savez(path, **masks)


def load_masks(path):
    """Load a mask dict from an .npz file written by save_masks."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
