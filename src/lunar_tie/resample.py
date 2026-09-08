"""TICKET-RD05 common-GSD resample. Gate: audit L2 shape-gate fix.

crop_pair can produce different-size crops for different-resolution pair
sides (OHRC 0.25 m vs TMC-2 5 m) while run_pair requires identical shapes.
This module decimates the finer side by an integer factor so both sides
share a common GSD and common (H, W).

Memory discipline: convolutions accumulate shifted row/col slices tap by
tap in bounded chunks instead of materializing sliding-window copies, so
peak RSS stays a small multiple of the input size.

numpy + stdlib only.
"""

import numpy as np


def area_average(img, factor):
    """Block-mean decimation by an integer factor.

    The image is cropped to a multiple of the factor FIRST (the dropped
    edge pixels are the documented price of integer decimation), then
    reshaped to (-1, factor, -1, factor) and averaged over the two block
    axes. factor=1 returns img unchanged (as float32).
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor must be a positive integer")
    img = np.asarray(img, dtype=np.float32)
    if factor == 1:
        return img
    H, W = img.shape
    Hc = (H // factor) * factor
    Wc = (W // factor) * factor
    cropped = img[:Hc, :Wc]
    return cropped.reshape(
        Hc // factor, factor, Wc // factor, factor
    ).mean(axis=(1, 3))


def gauss_pyr_down(img, factor):
    """Gaussian-pyramid decimation: blur with sigma=factor*0.5, then
    area_average by factor. Energy-preserving variant (DR1 ablation)."""
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor must be a positive integer")
    img = np.asarray(img, dtype=np.float32)
    sigma = factor * 0.5
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    padded = _reflect_pad(img, radius)
    del img
    blurred = _correlate1d(padded, kernel, axis=0)
    del padded
    blurred = _correlate1d(blurred, kernel, axis=1)
    return area_average(blurred, factor)


def lanczos3_decimate(img, factor):
    """Lanczos-3 decimation by an integer factor.

    The Lanczos-3 kernel (sinc windowed by sinc(x/3)) is applied as a
    separable 1D convolution evaluated at the decimated sample positions
    (block centers). Edges are handled with reflect padding. Expensive
    variant: bake-off use only.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor must be a positive integer")
    img = np.asarray(img, dtype=np.float64)
    if factor == 1:
        return img.astype(np.float32)
    H, W = img.shape
    H_out, W_out = H // factor, W // factor
    # one output sample per block, at the block center
    pos = np.arange(H_out) * factor + (factor - 1) / 2.0
    pos_w = np.arange(W_out) * factor + (factor - 1) / 2.0
    rel = np.arange(-3 * factor, 3 * factor + 1)  # kernel taps around center
    pad = 3 * factor + 1
    padded = np.pad(img, pad, mode="reflect")
    out = _lanczos_axis(padded, pos, rel, pad, factor, axis=0)
    del padded
    out = _lanczos_axis(out, pos_w, rel, pad, factor, axis=1)
    return out.astype(np.float32)


def _lanczos_axis(padded, pos, rel, pad, factor, axis):
    """One separable Lanczos-3 pass along an axis, tap by tap.

    The subpixel phase of each tap is constant for integer factors, so a
    single weight vector derived from the effective tap offsets applies
    to every output sample.
    """
    moved = padded if axis == 0 else padded.T
    offsets = np.floor(pos[:1] + rel) - pos[:1]
    weights = _lanczos3(offsets / factor)
    weights /= weights.sum()
    out = np.zeros((pos.size, moved.shape[1]), dtype=np.float64)
    for j, off in enumerate(rel):
        rows = np.floor(pos + off).astype(np.int64) + pad
        out += weights[j] * moved[rows]
    return out if axis == 0 else out.T


def _lanczos3(x):
    """Lanczos-3 kernel: sinc(x) * sinc(x/3), zero outside |x| < 3."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    mask = np.abs(x) < 3
    xm = x[mask]
    out[mask] = np.sinc(xm) * np.sinc(xm / 3.0)
    return out


def _reflect_pad(img, pad):
    """Reflect-pad by pad pixels on all sides (np.pad 'reflect' mode)."""
    if pad <= 0:
        return img
    return np.pad(img, pad, mode="reflect")


def _correlate1d(img, kernel, axis, chunk=256):
    """Centered 1D correlation along one axis (valid region).

    Accumulates shifted slices tap by tap over bounded row chunks so no
    sliding-window copy is ever materialized.
    """
    k = kernel.size
    n = img.shape[axis]
    out_len = n - k + 1
    other = img.shape[1 - axis]
    out = np.empty((out_len, other) if axis == 0 else (other, out_len),
                   dtype=np.float32)
    for start in range(0, out_len, chunk):
        end = min(out_len, start + chunk)
        acc = np.zeros((end - start, other) if axis == 0
                       else (other, end - start), dtype=np.float32)
        for j in range(k):
            w = np.float32(kernel[j])
            if axis == 0:
                acc += w * img[start + j:end + j, :]
            else:
                acc += w * img[:, start + j:end + j]
        if axis == 0:
            out[start:end, :] = acc
        else:
            out[:, start:end] = acc
    return out


def match_shapes_to_min(a, b, gsd_ratio):
    """Bring two crops to a common GSD and common (H, W).

    gsd_ratio is the coarser-side-to-finer-side GSD ratio (e.g. 20 for
    OHRC:TMC-2). The finer side (the larger crop) is decimated by that
    integer factor, then both arrays are center-cropped to the smaller
    common shape. Returns (a_rs, b_rs, factor) where both arrays are
    float32 with identical (H, W) and factor is the decimation applied.
    """
    gsd_ratio = int(gsd_ratio)
    if gsd_ratio < 1:
        raise ValueError("gsd_ratio must be a positive integer")
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape[0] * a.shape[1] >= b.shape[0] * b.shape[1]:
        a_rs = area_average(a, gsd_ratio)
        b_rs = b
    else:
        b_rs = area_average(b, gsd_ratio)
        a_rs = a
    # center-crop both to the smaller common shape
    H = min(a_rs.shape[0], b_rs.shape[0])
    W = min(a_rs.shape[1], b_rs.shape[1])
    return _center_crop(a_rs, H, W), _center_crop(b_rs, H, W), gsd_ratio


def _center_crop(img, H, W):
    h, w = img.shape
    r0 = (h - H) // 2
    c0 = (w - W) // 2
    return img[r0:r0 + H, c0:c0 + W]