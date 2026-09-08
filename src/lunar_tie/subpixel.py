"""UNIT-7 sub-pixel refinement: parabola + phase correlation (D4 8.1-8.3 / P6.m1).
Gate: 'subpixel recovery on 3 synthetic fixtures'.

UNIT-6 consensus gives a similarity M with inliers at integer-px accuracy. The
acceptance contract needs sub-pixel. R0 provides a two-method refiner: (1)
parabola refinement of each keypoint from the DoG neighborhood scores (Lowe
style, already used inside detect.py) and (2) phase correlation on a small
patch around each match (upsampled FFT peak -> sub-pixel dx, dy), robust for
translation-like local displacement after the global similarity warp.

Navigation contract (EXACT): src_pts are the source keypoints in imgA.
dst_pts are the corresponding positions in imgB obtained by applying the
similarity warp M to src_pts; the caller applies M before calling, so
dst_pts == warped_src. Deltas are measured on imgB relative to these warped
positions: for each pair we extract a patch from imgA at src_pts and a patch
from imgB at dst_pts, and the phase shift between them is the local
displacement delta. The final refined dst position is warped_src + delta,
i.e. dst_pts + delta.

Pure numpy + stdlib. No cv2, no scipy, no skimage. Self-contained; reuses no
other module internals beyond numpy.

Ambiguity policy: a patch window that would exit the image marks that pair
valid=False without crashing (refine_matches requires the FULL half-margin
window to fit: any point within half px of the border is invalid, even
though extract_patch could reflect-pad it; the stricter rule avoids
reflection-correlated patches). The refined position is left at the input
dst and the delta is zero.
"""

import numpy as np


def bilinear_sample(img, x, y):
    """Single bilinear sample at fractional (x, y) with boundary clamp."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    x = min(max(x, 0.0), w - 1.0)
    y = min(max(y, 0.0), h - 1.0)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    top = img[y0, x0] * (1.0 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1.0 - fx) + img[y1, x1] * fx
    return float(top * (1.0 - fy) + bot * fy)


def extract_patch(img, x, y, half=16):
    """Patch of size (2*half+1)^2 centered at fractional (x, y), float32.

    The image is reflect-padded ('symmetric') by half+1 so a window that exits
    the image is filled by reflection rather than zeros. Each sample is taken
    bilinearly at the fractional coordinate.
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    pad = half + 1
    padded = np.pad(img, pad, mode="symmetric")
    size = 2 * half + 1
    patch = np.empty((size, size), dtype=np.float64)
    for j in range(size):
        py = y + (j - half) + pad
        for i in range(size):
            px = x + (i - half) + pad
            patch[j, i] = bilinear_sample(padded, px, py)
    return patch.astype(np.float32)


def _parabola(c0, c1, c2):
    """Sub-pixel offset of a parabola through (c0, c1, c2) at x=-1,0,1."""
    denom = c0 - 2.0 * c1 + c2
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (c0 - c2) / denom


def phase_shift(patchA, patchB, upsample=16, max_shift=None):
    """Phase correlation between two patches -> (dx, dy, peak_val, snr).

    Both patches are zero-meaned and Hann windowed, then the cross-power
    spectrum F_A * conj(F_B) / |.| is formed. The correlation is upsampled by
    zero-padding in the frequency domain by `upsample`, the integer peak is
    located, and a quadratic parabola fit around the peak in the upsampled
    grid gives the sub-pixel dx, dy. peak_val is the normalized peak height in
    [0, 1] and serves as a confidence proxy.

    TICKET-RD06 (audit A3+A4): max_shift restricts the argmax search to
    a [-max_shift, +max_shift] window around the surface center (the
    shift is small after consensus; lobes outside that window are
    aliases, not signal). snr = peak / median(corr_surface): a correct
    phase correlation has snr >> 5; an alias or noise floor has snr
    ~ 1-2. snr is shift-invariant (unlike peak_val which normalizes by
    the zero-shift response).
    """
    a = np.asarray(patchA, dtype=np.float64)
    b = np.asarray(patchB, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    a = a * win
    b = b * win
    FA = np.fft.fft2(a)
    FB = np.fft.fft2(b)
    cross = FA * np.conj(FB)
    denom = np.abs(cross)
    denom[denom == 0] = 1.0
    r = cross / denom

    up_h = h * upsample
    up_w = w * upsample
    r_pad = np.zeros((up_h, up_w), dtype=np.complex128)
    r_shifted = np.fft.fftshift(r)
    c0 = (up_h - h) // 2
    c1 = (up_w - w) // 2
    r_pad[c0:c0 + h, c1:c1 + w] = r_shifted
    r_pad = np.fft.ifftshift(r_pad)
    corr = np.fft.ifft2(r_pad).real

    # TICKET-RD06: the phase-corr peak for a shifted image wraps around the
    # FFT boundary: the peak is near the EDGES of the corr surface, not the
    # center. the max_shift check must be applied AFTER the unwrapping (the
    # dx/dy computation), not as a window on the argmax.
    py, px = np.unravel_index(np.argmax(corr), corr.shape)

    y0 = max(py - 1, 0)
    y1 = min(py + 1, up_h - 1)
    x0 = max(px - 1, 0)
    x1 = min(px + 1, up_w - 1)
    dx_up = _parabola(corr[py, x0], corr[py, px], corr[py, x1])
    dy_up = _parabola(corr[y0, px], corr[py, px], corr[y1, px])

    dx = (up_w - (px + dx_up)) / upsample
    dy = (up_h - (py + dy_up)) / upsample
    if dx > w / 2.0:
        dx -= w
    if dy > h / 2.0:
        dy -= h
    peak_val = float(min(max(corr[py, px] * upsample * upsample, 0.0), 1.0))
    # TICKET-RD06: surface SNR (shift-invariant confidence)
    med = float(np.median(corr))
    snr = float(corr[py, px] / med) if abs(med) > 1e-12 else 0.0
    return float(dx), float(dy), peak_val, snr


def refine_matches(imgA, imgB, src_pts, dst_pts, half=16, upsample=16,
                   max_phase=8.0):
    """Refine matched pairs to sub-pixel accuracy via phase correlation.

    See the module docstring for the navigation contract: dst_pts are the
    similarity-warped positions of src_pts (caller applies M), deltas are
    measured on imgB relative to those warped positions, and the refined dst
    is dst_pts + delta.

    Returns a dict with 'src' (fractional refined source, unchanged), 'dst'
    (refined positions = dst_pts + delta), 'deltas' (N,2) applied shifts,
    'peak_vals' (N,) confidences, and 'valid' (N,) bool mask where
    |delta| <= max_phase and peak_val >= 0.2. Edge rule (stricter than
    extract_patch's reflection capability): a point whose half-margin
    window does not fully fit inside the image (within half px of any
    border) is marked invalid with delta zero, without crashing.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    n = src.shape[0]
    h, w = imgA.shape
    refined_src = src.copy()
    refined_dst = dst.copy()
    deltas = np.zeros((n, 2), dtype=np.float64)
    peak_vals = np.zeros(n, dtype=np.float64)
    snrs = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        x, y = dst[i]
        if x - half < 0 or y - half < 0 or x + half > w - 1 or y + half > h - 1:
            continue
        patchA = extract_patch(imgA, src[i, 0], src[i, 1], half)
        patchB = extract_patch(imgB, x, y, half)
        dx, dy, pv, snr = phase_shift(patchA, patchB, upsample,
                                      max_shift=max_phase)
        deltas[i] = (dx, dy)
        peak_vals[i] = pv
        snrs[i] = snr
        refined_dst[i] = (x + dx, y + dy)
        # TICKET-RD06: SNR gate replaces the peak_val >= 0.2 gate
        # (peak_val collapses with shift: pv(4,4)=0.087 for a CORRECT match)
        if abs(dx) <= max_phase and abs(dy) <= max_phase and snr >= 5.0:
            valid[i] = True
    return {
        "src": refined_src,
        "dst": refined_dst,
        "deltas": deltas,
        "peak_vals": peak_vals,
        "snrs": snrs,
        "valid": valid,
    }


def rms_subpx(errors):
    """RMS of an error array over the valid mask: sqrt(mean(e^2))."""
    err = np.asarray(errors, dtype=np.float64)
    if err.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(err ** 2)))
