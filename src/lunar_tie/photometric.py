"""UNIT-4 photometric ratio normalization (P3.m1/D3 / P4.m1).
Gate: 'photometric ratio on 3 synthetic fixtures'.

LUNAR-TIE must register imagery taken at different sun angles (OHRC low sun vs
TMC-2 different hour vs IIRS). R0 uses the ratio-of-convolved-pairs family:
normalize luminance and local contrast per tile so two sun-angle views of the
same terrain produce similar normalized images. This module implements the
classic locally-normalized "ratio norm": zero mean and unit variance under a
local Gaussian window.

numpy + stdlib only. No scipy, no cv2.

Memory discipline: gaussian_blur holds the factor-of-two claim exactly
(it allocates the reflect-padded array plus one work buffer, and the two
separable passes reuse them, so peak memory is O(img.size) with constant
factor two). normalize_ratio builds on it but holds MORE full-size arrays
at its peak (L, mu, L2, mu2, num, var, den, out: about eight float64
arrays before the float32 cast), so its constant is larger; it is still
O(img.size), just not the factor-of-two invariant of gaussian_blur.

Edge handling: reflect padding at image borders (np.pad mode 'symmetric').
Reflection avoids injecting artificial high-frequency energy at the border
(which zero padding would do) and keeps the local statistics well defined.
"""

import numpy as np


def gaussian_blur(img, sigma):
    """Separable Gaussian blur with reflect padding.

    Kernel radius is ceil(3*sigma), giving an odd kernel size of
    2*radius + 1. Convolution is done as two 1D passes (rows then columns)
    for speed. Border handling uses np.pad mode 'symmetric' (reflect).
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("img must be a 2D array")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = int(np.ceil(3 * sigma))
    if radius < 1:
        radius = 1
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    padded = np.pad(img, radius, mode="symmetric")
    work = np.empty_like(padded)
    for i in range(padded.shape[0]):
        work[i] = np.convolve(padded[i], kernel, mode="same")
    for j in range(padded.shape[1]):
        padded[:, j] = np.convolve(work[:, j], kernel, mode="same")
    return padded[radius:-radius, radius:-radius]


def normalize_ratio(img, sigma=8.0, eps=1e-6):
    """Locally normalize an image to zero mean and unit variance.

    L = img cast to float; mu = gaussian_blur(L, sigma);
    num = L - mu; den = sqrt(gaussian_blur(L^2, sigma) - mu^2) + eps;
    out = num / den. Output is float32, same shape as input, values roughly
    in [-1, 1]. The sqrt argument is clipped at 0 to guard against tiny
    negative values from floating point round-off.
    """
    L = np.asarray(img, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("img must be a 2D array")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    mu = gaussian_blur(L, sigma)
    L2 = L * L
    mu2 = gaussian_blur(L2, sigma)
    num = L - mu
    var = np.maximum(mu2 - mu * mu, 0.0)
    den = np.sqrt(var) + eps
    out = num / den
    return out.astype(np.float32)


def normalize_pair(imgA, imgB, sigma=8.0):
    """Normalize both images with the same sigma (shared scale for a pair)."""
    return normalize_ratio(imgA, sigma=sigma), normalize_ratio(imgB, sigma=sigma)


def quantize(img, quant_levels=6):
    """Uniform min-max quantization into integer rank bins, float64.

    Min-max scales the image to [0, quant_levels - 1] and floors each
    value to its bin index. The bins are computed after min-max scaling
    because the min-max affine map is invariant to positive gain and
    offset (a*img + b with a > 0), so the bin index is a stable
    intensity rank under multiplier gain, which is exactly what the
    RIFT-style rank transform needs. A fixed [0, 255] grid would break
    under gain. Constant images map to all zeros. Output is float64,
    same shape as the input.
    """
    L = np.asarray(img, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("img must be a 2D array")
    if quant_levels < 2:
        raise ValueError("quant_levels must be >= 2")
    lo = float(L.min())
    hi = float(L.max())
    if hi <= lo:
        return np.zeros(L.shape, dtype=np.float64)
    scaled = (L - lo) / (hi - lo) * (quant_levels - 1)
    bins = np.floor(scaled)
    np.clip(bins, 0.0, float(quant_levels) - 1.0, out=bins)
    return bins


def rank_transform(img, bins=6):
    """RIFT-style rank-transform normalizer (UNIT-4B).

    Per-pixel stable intensity rank: the image is min-max scaled and
    uniformly quantized into `bins` levels, the rank half of the
    RIFT2 rank/phase idea. Positive gain and offset map every pixel to
    the same rank, so matches survive radiometric distortion that
    breaks ratio-norm. Returns float64, same shape as the input, with
    values in [0, bins - 1].
    """
    return quantize(img, quant_levels=bins)


def apply_rank(img, quantize_bins=6):
    """Apply the rank transform with the ticket's argument spelling."""
    return rank_transform(img, bins=quantize_bins)


def correlation_preview(nA, nB):
    """Zero-mean normalized cross-correlation of two full images (float).

    A quick sanity gate, not the full matcher. Returns 0.0 if either image
    has zero variance.
    """
    a = np.asarray(nA, dtype=np.float64)
    b = np.asarray(nB, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("arrays must have the same shape")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)
