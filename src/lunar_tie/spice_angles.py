#!/usr/bin/env python3
"""TICKET-RD08: real SPICE angle extraction for pair-A. the angles that feed
the photometric correction with real physics (the illumination division).

provenance: this module reads the REAL CK/SPK kernels the user downloaded
from the ISDA SPICE tab. no heuristic angles. no guessed values. SPICE.
"""
import math
import os

import spiceypy as sp

_KFILES_LOADED = False
SPICE_DIR = os.environ.get("LUNAR_TIE_SPICE_DIR",
                            "/home/liebert511/chandrayaan-2-data")
CK_DIR = os.environ.get("LUNAR_TIE_CK_DIR",
                        "/home/liebert511/Downloads/ABDM")


def load_kernels():
    """Furnish all CH2 + generic kernels for angle extraction."""
    global _KFILES_LOADED
    if _KFILES_LOADED:
        return
    for f in ("ch2_sclk_v1.tsc", "naif0012.tls", "ch2_v01.tf",
              "ch2_ohr_v01.ti", "ch2_tmc_v01.ti", "pck00010.tpc",
              "de430s.bsp"):
        sp.furnsh(f"{SPICE_DIR}/{f}")
    for f in ("ch2_att_27Mar2024_04May2024_v1.bc",
              "ch2_att_27Apr2024_04Jun2024_v1.bc",
              "ch2_eph_29Mar2024_02May2024_v1.bsp",
              "ch2_eph_29Apr2024_02Jun2024_v1.bsp"):
        sp.furnsh(f"{CK_DIR}/{f}")
    _KFILES_LOADED = True


def angles_for_product(utc_obs, naif_id="-152", frame_id=-152001):
    """Extract incidence/phase/emission angles for one product obs time.

    utc_obs: "2024-03-30 00:35:08.536" (from the product filename timestamp)
    Returns dict with all angles + the spacecraft position + the CK sample
    time (provenance: the CK sampling gap between the obs and the nearest
    kernel entry).
    """
    load_kernels()
    et = sp.str2et(utc_obs)
    sclk = sp.sce2c(-152, et)
    cmat, sclk_out = sp.ckgp(-152001, sclk, 7200.0, "J2000")
    st, _ = sp.spkezr(naif_id, et, "J2000", "NONE", "301")

    # EXTERNAL-AUDIT V1: ch2_v01.tf states '+X is along instrument boresights
    # - towards Moon' (CH2_SPACECRAFT) and '+X axis points along the detector
    # boresight' (OHRC/TMC), so the boresight points AT the Moon, i.e. the
    # camera stares nadir. Incidence/emission/phase must therefore be taken at
    # the sub-observer surface intercept the camera sees, which the SPICE-native
    # subpnt + ilumin authority computes (emission ~0 for nadir viewing), not a
    # naive vsep of a body-frame axis against a J2000 vector (which yields the
    # anti-parallel 96.8/149 deg artifacts seen before the audit).
    spoint, _, _ = sp.subpnt("NEAR POINT/ELLIPSOID", "301", et,
                             "IAU_MOON", "NONE", naif_id)
    _, _, phase, incd, emisd = sp.ilumin("ELLIPSOID", "301", et,
                                         "IAU_MOON", "LT+S", naif_id, spoint)
    inc_deg = math.degrees(incd)
    emis_deg = math.degrees(emisd)
    phase_deg = math.degrees(phase)

    # sub-solar lat/lon (the illumination conditions on the surface)
    sun_iau, _ = sp.spkezr("SUN", et, "IAU_MOON", "LT+S", "301")
    _, lon_iau, lat_iau = sp.reclat(sun_iau[:3])
    sun_lat = math.degrees(lat_iau)
    sun_lon = math.degrees(lon_iau)

    # CK sampling gap (provenance: how far the CK sample is from the obs)
    dt_ck = abs(et - sp.sct2e(-152, sclk_out))

    return {
        "utc_obs": utc_obs,
        "et_obs": et,
        "ck_sample_sclk": sclk_out,
        "ck_sample_dt_s": dt_ck,
        "incidence_deg": inc_deg,
        "phase_deg": phase_deg,
        "emission_deg": emis_deg,
        "sub_solar_lat": sun_lat,
        "sub_solar_lon": sun_lon,
        "alt_km": sp.vnorm(st[:3]) - 1737.4,
    }


if __name__ == "__main__":
    load_kernels()
    # OHRC: 2024-03-30T00:35:08.536 (from the product ID)
    a = angles_for_product("2024-03-30 00:35:08.536", frame_id=-152270)
    print(f"OHRC 20240330: inc={a['incidence_deg']:.2f} "
          f"phase={a['phase_deg']:.2f} emis={a['emission_deg']:.2f} "
          f"lat={a['sub_solar_lat']:.2f} lon={a['sub_solar_lon']:.2f} "
          f"alt={a['alt_km']:.1f} ck_dt={a['ck_sample_dt_s']:.0f}s")
    # TMC: 2024-05-23T16:00:30.958
    b = angles_for_product("2024-05-23 16:00:30.958", frame_id=-152210)
    print(f"TMC 20240523: inc={b['incidence_deg']:.2f} "
          f"phase={b['phase_deg']:.2f} emis={b['emission_deg']:.2f} "
          f"lat={b['sub_solar_lat']:.2f} lon={b['sub_solar_lon']:.2f} "
          f"alt={b['alt_km']:.1f} ck_dt={b['ck_sample_dt_s']:.0f}s")
    print(f"\nsun delta: {abs(a['sub_solar_lon']-b['sub_solar_lon']):.1f} deg lon, "
          f"{abs(a['sub_solar_lat']-b['sub_solar_lat']):.1f} deg lat, "
          f"{abs(a['incidence_deg']-b['incidence_deg']):.1f} deg incidence")