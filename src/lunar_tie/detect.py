"""UNIT-5 keypoint detect + describe: SIFT and RD-SIFT (D4 5.1-5.2 / P4.m2).
Gate: 'SIFT blob detection + polarity flip on 3 synthetic fixtures'.

LUNAR-TIE must register imagery taken at different sun angles, where contrast
polarity can flip (I1). SIFT keypoints and descriptors are gradient based and
gain invariant, but a full polarity flip rotates every gradient direction by
180 degrees, which defeats signed SIFT histograms. RD-SIFT keeps the SAME
detector but folds a polarity bit into the descriptor (see below) so it
survives contrast reversal.

Pure numpy + stdlib. No cv2, no skimage, no scipy. The Gaussian scale space
reuses gaussian_blur from photometric.py (own separable blur), never
reimplemented.

Descriptor layouts:
  SIFT:    4x4 cells x 8 signed orientation bins over [0, 2pi), Gaussian
    weighted (sigma = 1.5 x cell size), L2 normalized -> 128 dims.
  RD-SIFT: the SAME 4x4x8 signed histogram, but every cell is rolled by PI
    (4 bins) when the keypoint DoG response is negative. The polarity bit is
    the sign of the interpolated DoG value at the keypoint. Folding the bit
    into the bins keeps the descriptor 128 dims and makes two views of the
    same terrain that differ only by contrast sign (255-x) produce near
    identical descriptors.

Ambiguity: an image smaller than 32 px on either side returns an empty
keypoint list and a (0, 128) descriptor array without crashing.
"""

import numpy as np

from lunar_tie.photometric import gaussian_blur

_PI2 = 2.0 * np.pi


def _is_extremum_mask(dog_low, dog_mid, dog_high):
    """Boolean mask (interior) where dog_mid is a 3x3x3 local extremum."""
    shape = dog_mid.shape
    interior = dog_mid[1:-1, 1:-1]
    maxm = interior.copy()
    minm = interior.copy()
    for s in (dog_low, dog_mid, dog_high):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if s is dog_mid and dy == 0 and dx == 0:
                    continue
                n = s[1 + dy:1 + dy + shape[0] - 2,
                      1 + dx:1 + dx + shape[1] - 2]
                np.maximum(maxm, n, out=maxm)
                np.minimum(minm, n, out=minm)
    return (interior == maxm) | (interior == minm)


def _refine(dog_low, dog_mid, dog_high, x, y):
    """Sub-pixel offset via quadratic fit (Lowe). Returns tuple or None."""
    val = dog_mid[y, x]
    dx0 = 0.5 * (dog_mid[y, x + 1] - dog_mid[y, x - 1])
    dy0 = 0.5 * (dog_mid[y + 1, x] - dog_mid[y - 1, x])
    ds0 = 0.5 * (dog_high[y, x] - dog_low[y, x])
    dxx = dog_mid[y, x + 1] - 2.0 * val + dog_mid[y, x - 1]
    dyy = dog_mid[y + 1, x] - 2.0 * val + dog_mid[y - 1, x]
    dss = dog_high[y, x] - 2.0 * val + dog_low[y, x]
    dxy = 0.25 * (dog_mid[y + 1, x + 1] - dog_mid[y + 1, x - 1] -
                  dog_mid[y - 1, x + 1] + dog_mid[y - 1, x - 1])
    dxs = 0.25 * (dog_high[y, x + 1] - dog_high[y, x - 1] -
                  dog_low[y, x + 1] + dog_low[y, x - 1])
    dys = 0.25 * (dog_high[y + 1, x] - dog_high[y - 1, x] -
                  dog_low[y + 1, x] + dog_low[y - 1, x])

    hess = np.array([[dxx, dxy, dxs],
                     [dxy, dyy, dys],
                     [dxs, dys, dss]], dtype=np.float64)
    grad = np.array([dx0, dy0, ds0], dtype=np.float64)
    try:
        off = -np.linalg.solve(hess, grad)
    except np.linalg.LinAlgError:
        return None
    if np.any(np.abs(off) > 0.5):
        return None
    dog_val = val + 0.5 * grad @ off
    return float(off[0]), float(off[1]), float(off[2]), float(dog_val), hess


def _edge_ok(hess, edge_threshold):
    """Lowe edge criterion from the 2x2 Hessian in (x, y); True keeps point."""
    a, d = hess[0, 0], hess[1, 1]
    b = hess[0, 1]
    tr = a + d
    det = a * d - b * b
    if det <= 0:
        return False
    r = edge_threshold
    return (tr * tr) / det < ((r + 1.0) ** 2) / r


def _assign_angle(oct_img, x, y, sigma_oct):
    """Dominant orientation from a 36-bin signed gradient histogram."""
    gy, gx = np.gradient(oct_img)
    mag = np.sqrt(gx * gx + gy * gy)
    dirr = np.arctan2(gy, gx)
    radius = int(round(3.0 * sigma_oct))
    h, w = oct_img.shape
    bin_w = _PI2 / 36.0
    sigma = 1.5 * sigma_oct
    hist = np.zeros(36)
    xx = np.arange(max(0, x - radius), min(w, x + radius + 1))
    yy = np.arange(max(0, y - radius), min(h, y + radius + 1))
    for jj in range(len(yy)):
        py = yy[jj]
        for ii in range(len(xx)):
            px = xx[ii]
            ox = px - x
            oy = py - y
            d2 = ox * ox + oy * oy
            wgt = mag[py, px] * np.exp(-d2 / (2.0 * sigma * sigma))
            ob = ((dirr[py, px] / _PI2) % 1.0) * 36.0
            b0 = int(np.floor(ob)) % 36
            frac = ob - np.floor(ob)
            hist[b0] += wgt * (1.0 - frac)
            hist[(b0 + 1) % 36] += wgt * frac
    best = int(np.argmax(hist))
    return best * bin_w + bin_w * 0.5


def _signed_descriptor(oct_img, x, y, sigma_oct, angle, n_cells=4, n_bins=8):
    """4x4x8 signed SIFT descriptor, L2 normalized -> 128 dims."""
    gy, gx = np.gradient(oct_img)
    mag = np.sqrt(gx * gx + gy * gy)
    dirr = np.arctan2(gy, gx)
    cosp, sinp = np.cos(angle), np.sin(angle)
    hist = np.zeros(n_cells * n_cells * n_bins, dtype=np.float64)
    bin_w = _PI2 / n_bins
    sigma = 1.5 * sigma_oct
    half = n_cells * sigma_oct / 2.0
    h, w = oct_img.shape

    xmin = int(np.floor(x - half))
    xmax = int(np.ceil(x + half))
    ymin = int(np.floor(y - half))
    ymax = int(np.ceil(y + half))

    for yy in range(ymin, ymax + 1):
        if not (0 <= yy < h):
            continue
        for xx in range(xmin, xmax + 1):
            if not (0 <= xx < w):
                continue
            ox = xx - x
            oy = yy - y
            rx = cosp * ox + sinp * oy
            ry = -sinp * ox + cosp * oy
            cbin = rx / sigma_oct + n_cells / 2.0 - 0.5
            rbin = ry / sigma_oct + n_cells / 2.0 - 0.5
            if not (0.0 <= cbin < n_cells and 0.0 <= rbin < n_cells):
                continue
            da = (dirr[yy, xx] - angle) % _PI2
            obin = da / bin_w
            wgt = mag[yy, xx] * np.exp(-(ox * ox + oy * oy) / (2.0 * sigma * sigma))
            c0 = int(np.floor(cbin))
            r0 = int(np.floor(rbin))
            o0 = int(np.floor(obin)) % n_bins
            dc = cbin - c0
            dr = rbin - r0
            dof = obin - np.floor(obin)
            for r_off in (0, 1):
                ri = r0 + r_off
                if not (0 <= ri < n_cells):
                    continue
                wr = dr if r_off == 1 else 1.0 - dr
                for c_off in (0, 1):
                    ci = c0 + c_off
                    if not (0 <= ci < n_cells):
                        continue
                    wc = dc if c_off == 1 else 1.0 - dc
                    for o_off in (0, 1):
                        oi = (o0 + o_off) % n_bins
                        wo = dof if o_off == 1 else 1.0 - dof
                        idx = ri * n_cells * n_bins + ci * n_bins + oi
                        hist[idx] += wgt * wr * wc * wo

    return _normalize_descriptor(hist)


def _normalize_descriptor(hist):
    """L2 normalize, clip to 0.2, renormalize (SIFT style)."""
    vec = hist.astype(np.float64)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros(128, dtype=np.float32)
    vec /= norm
    np.clip(vec, 0.0, 0.2, out=vec)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.astype(np.float32)


_RD_ROLL = 4


def _rd_descriptor(oct_img, x, y, sigma_oct, angle, dog_val):
    """Signed descriptor with polarity bit folded as a per-cell PI roll."""
    desc = _signed_descriptor(oct_img, x, y, sigma_oct, angle)
    if dog_val < 0:
        view = desc.reshape(4, 4, 8)
        desc = np.roll(view, _RD_ROLL, axis=2).reshape(-1)
    return desc


def _detect(img, n_octaves=4, n_scales=3, sigma0=1.6,
            contrast_threshold=0.03, edge_threshold=10.0, signed=True):
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("img must be a 2D array")
    arr = img.astype(np.float64)
    h, w = arr.shape
    if min(h, w) < 32:
        return [], np.empty((0, 128), dtype=np.float32)

    thresh = contrast_threshold * n_scales
    n_levels = n_scales + 3
    k = 2.0 ** (1.0 / n_scales)

    keypoints = []
    descriptors = []

    cur = gaussian_blur(arr, sigma0)
    for o in range(n_octaves):
        if min(cur.shape) < 16:
            break
        abs_base = sigma0 * (2.0 ** o)
        oct_blurred = [gaussian_blur(cur, abs_base * (k ** b))
                       for b in range(n_levels)]
        dogs = [oct_blurred[i + 1] - oct_blurred[i]
                for i in range(n_levels - 1)]

        for b in range(1, len(dogs) - 1):
            dmid = dogs[b]
            idx = np.argwhere(_is_extremum_mask(dogs[b - 1], dmid, dogs[b + 1]))
            for (iy, ix) in idx:
                y = iy + 1
                x = ix + 1
                ref = _refine(dogs[b - 1], dmid, dogs[b + 1], x, y)
                if ref is None:
                    continue
                dx, dy, ds, dog_val, hess = ref
                if abs(dog_val) <= thresh:
                    continue
                if not _edge_ok(hess, edge_threshold):
                    continue
                fx = x + dx
                fy = y + dy
                sigma_abs = abs_base * (k ** (b + 1))
                sigma_oct = sigma_abs / (2.0 ** o)
                gauss_img = oct_blurred[b + 1]
                ang = _assign_angle(gauss_img, int(round(fx)), int(round(fy)),
                                    sigma_oct)
                if signed:
                    desc = _signed_descriptor(gauss_img, fx, fy, sigma_oct, ang)
                else:
                    desc = _rd_descriptor(gauss_img, fx, fy, sigma_oct, ang,
                                          dog_val)
                keypoints.append({
                    "x": float((x + dx) * (2.0 ** o)),
                    "y": float((y + dy) * (2.0 ** o)),
                    "octave": o,
                    "scale": float(sigma_abs),
                    "response": float(dog_val),
                    "angle": float(ang),
                })
                descriptors.append(desc)

        if o < n_octaves - 1:
            cur = oct_blurred[n_scales][::2, ::2]

    if not descriptors:
        return [], np.empty((0, 128), dtype=np.float32)
    return keypoints, np.array(descriptors, dtype=np.float32)


def detect_keypoints_sift(img, n_octaves=4, n_scales=3, sigma0=1.6,
                          contrast_threshold=0.03, edge_threshold=10.0):
    """SIFT keypoints + 128-d descriptors. Pure numpy, no opencv."""
    return _detect(img, n_octaves=n_octaves, n_scales=n_scales, sigma0=sigma0,
                   contrast_threshold=contrast_threshold,
                   edge_threshold=edge_threshold, signed=True)


def detect_keypoints_rdsift(img, n_octaves=4, n_scales=3, sigma0=1.6,
                            contrast_threshold=0.03, edge_threshold=10.0):
    """RD-SIFT: same detector, polarity-robust 128-d descriptors."""
    return _detect(img, n_octaves=n_octaves, n_scales=n_scales, sigma0=sigma0,
                   contrast_threshold=contrast_threshold,
                   edge_threshold=edge_threshold, signed=False)


def match_descriptors(descA, descB, ratio=0.8):
    """Lowe ratio test on L2 distance. Returns [(i, j), ...] matches."""
    descA = np.asarray(descA, dtype=np.float64)
    descB = np.asarray(descB, dtype=np.float64)
    if descA.shape[0] == 0 or descB.shape[0] == 0:
        return []
    if descA.shape[1] != descB.shape[1]:
        raise ValueError("descriptor matrices must have the same width")
    a_n2 = (descA ** 2).sum(1)[:, None]
    b_n2 = (descB ** 2).sum(1)[None, :]
    cross = descA @ descB.T
    dist = np.sqrt(np.maximum(a_n2 + b_n2 - 2.0 * cross, 0.0))
    k = 2
    if dist.shape[1] < k:
        k = 1
    best_idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
    d0 = dist[np.arange(dist.shape[0]), best_idx[:, 0]]
    d1 = dist[np.arange(dist.shape[0]), best_idx[:, 1]]
    keep = np.where((best_idx[:, 0] != best_idx[:, 1]) & (d0 < ratio * d1))[0]
    return [(int(i), int(best_idx[i, 0])) for i in keep]
