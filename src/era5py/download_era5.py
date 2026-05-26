"""
download_era5.py — ERA 5 Downloader · Post-processor 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER CONFIGURATION  ← Edit this section before running
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── Time ─────────────────────────────────────────────────────────────────────
YEAR_i  = 2020       # int  – year  to download  (2003 – 2024)
YEAR_f  = 2021
NAME = 'bolivia'
# ── Variable ──────────────────────────────────────────────────────────────────
# Choose ONE key from the catalogue below:
VARIABLE = "2m_temperature"
        # "2m_temperature",
        # "2m_dewpoint_temperature",
        #  "total_precipitation",
        # "10m_u_component_of_wind",
        # "10m_v_component_of_wind",
        # "100m_u_component_of_wind",
        # "100m_v_component_of_wind",
# ── Domain mode ──────────────────────────────────────────────────────────────
# "point"  → download data for a single lat/lon (nearest grid cell)
# "area"   → download data for a bounding box
MODE = "area"
# Point settings (used when MODE = "point")
LAT =  -16.5    # Latitude  of point
LON =  -68.15   # Longitude of point

# Area settings (used when MODE = "area")  [N, W, S, E]
AREA_N =  -8.0
AREA_W =  -70.0
AREA_S =  -24.0
AREA_E =  -57.0

# ── Output ───────────────────────────────────────────────────────────────────
OUTDIR = "era5_output_bolivia"   # all files land here
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROUGHNESS_LENGTH = 0.03   # m  (open farmland – adjust for your site)
REF_HEIGHT_LOW   = 10.0   # m  (ERA-5 lowest wind level)
REF_HEIGHT_HIGH  = 100.0  # m  (ERA-5 highest single-level wind level)
TARGET_HEIGHTS   = [10, 30, 50]  # m

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import argparse
import os
import sys
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

client = cdsapi.Client()
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log_wind_speed(u_ref, v_ref, z_ref: float, z_target: float,
                   z0: float = ROUGHNESS_LENGTH) -> np.ndarray:
    """
    Estimate wind speed at z_target using the logarithmic wind profile.

        U(z) = U_ref * ln(z / z0) / ln(z_ref / z0)

    Parameters
    ----------
    u_ref, v_ref : array-like   U and V components at reference height
    z_ref        : float        Reference height [m]
    z_target     : float        Target height [m]
    z0           : float        Aerodynamic roughness length [m]
    """
    u_ref  = np.asarray(u_ref)
    v_ref  = np.asarray(v_ref)
    scale  = np.log(z_target / z0) / np.log(z_ref / z0)
    u_tgt  = u_ref * scale
    v_tgt  = v_ref * scale
    return np.sqrt(u_tgt**2 + v_tgt**2)


def build_year_chunks(start: int, end: int, chunk: int = 5):
    """Split [start, end] into chunks of `chunk` years for smaller CDS requests."""
    years = list(range(start, end + 1))
    return [years[i:i + chunk] for i in range(0, len(years), chunk)]

def _area_from_config(mode: str) -> list[float]:
    """Return [N, W, S, E] depending on mode."""
    if mode == "point":
        pad = 1.0
        return [LAT + pad, LON - pad, LAT - pad, LON + pad]
    else:
        return [AREA_N, AREA_W, AREA_S, AREA_E]

# ---------------------------------------------------------------------------
# CDS download
# ---------------------------------------------------------------------------
def download_era5(client: cdsapi.Client, year: int,
                        LAT: float, LON: float, OUTDIR: Path, VARIABLE: str, MODE: str):
    """Download one chunk of years from ERA-5 single-levels."""
    str_years  = [str(year)]
    str_months = [f"{m:02d}" for m in range(1, 13)]
    str_days   = [f"{d:02d}" for d in range(1, 32)]
    str_times  = [f"{h:02d}:00" for h in range(0, 24)]

    # Build a bounding box tight around the requested point
    # (ERA-5 ~0.25° grid; ±0.5° guarantees we get the nearest cell)
    # area = [lat + 0.5, lon - 0.5, lat - 0.5, lon + 0.5]   # N W S E // delete -> not used anymore
    area   = _area_from_config(MODE)

    request = {
        "product_type": "reanalysis",
        "variable": [ VARIABLE
            #"2m_temperature",
            # "2m_dewpoint_temperature",
            #  "total_precipitation",
            # "10m_u_component_of_wind",
            # "10m_v_component_of_wind",
            # "100m_u_component_of_wind",
            # "100m_v_component_of_wind",
        ],
        "year":  str_years,
        "month": str_months,
        "day":   str_days,
        "time":  str_times,
        "area":  area,
        "format": "netcdf",
    }

    print(f"  Requesting year {year} …")
    client.retrieve("reanalysis-era5-single-levels", request, str(OUTDIR))
    print(f"  Saved → {OUTDIR}")

def download():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nERA-5 downloader")
    print(f"  Period : {YEAR_i} – {YEAR_f} ")
    print(f"  Output : {outdir.resolve()}\n")

    for yy in range(YEAR_i, YEAR_f+1):
        tag  = f"_{yy}_{VARIABLE}_{NAME}"
        nc_p = outdir / f"era5_raw_{tag}.nc"
        if nc_p.exists():
            print(f"  {nc_p.name} already exists – skipping download.")
        else:
            download_era5(client, yy, LAT, LON, nc_p, VARIABLE, MODE)

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def process_dataset(nc_path: Path, lat: float, lon: float) -> pd.DataFrame:
    """
    Open a downloaded NetCDF, select the nearest grid point, compute derived
    variables and return a tidy hourly DataFrame.
    """
    ds = xr.open_dataset(nc_path)

    # Select nearest grid point
    ds = ds.sel(latitude=lat, longitude=lon, method="nearest")

    df = pd.DataFrame(index=pd.to_datetime(ds["valid_time"].values))
    df.index.name = "time"

    # ── Single-level variables ───────────────────────────────────────────────
    df["t2m_K"]      = ds["t2m"].values          # 2-m temperature [K]
    df["t2m_C"]      = df["t2m_K"] - 273.15      # … [°C]
    df["d2m_K"]      = ds["d2m"].values           # 2-m dewpoint [K]
    df["precip_m"]   = ds["tp"].values            # total precipitation [m]
    df["precip_mm"]  = df["precip_m"] * 1000      # … [mm]

    # Specific humidity from dewpoint (Bolton 1980 approximation)
    e_s = 6.112 * np.exp(17.67 * (df["d2m_K"] - 273.15) /
                          ((df["d2m_K"] - 273.15) + 243.5))   # hPa
    df["specific_humidity_g_kg"] = (0.622 * e_s) / (1013.25 - 0.378 * e_s) * 1000

    # ── Wind at native ERA-5 heights ─────────────────────────────────────────
    u10  = ds["u10"].values
    v10  = ds["v10"].values
    u100 = ds["u100"].values
    v100 = ds["v100"].values

    df["wind_speed_10m_ms"]  = np.sqrt(u10**2  + v10**2)
    df["wind_speed_100m_ms"] = np.sqrt(u100**2 + v100**2)

    # ── Wind at 10 / 30 / 50 m via log-wind profile ─────────────────────────
    # Use 10 m as reference for 30 m & 50 m (same boundary-layer regime)
    for z in TARGET_HEIGHTS:
        col = f"wind_speed_{z}m_ms"
        if z == 10:
            df[col] = df["wind_speed_10m_ms"]
        else:
            df[col] = log_wind_speed(u10, v10, REF_HEIGHT_LOW, z)

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPERATIONS = {
    "download":      download,
}
def main():
    parser = argparse.ArgumentParser(
        description="Download ERA-5 hourly data for a lat/lon point or area.")

    parser.add_argument(
        "-o", "--operation",
        required=True,
        choices=["download", "postprocess", "visualize", "all"],
        help="Operation to run")

    args = parser.parse_args()

    if args.operation == "all":
        for op_name, op_fn in OPERATIONS.items():
            op_fn()
    else:
        OPERATIONS[args.operation]()


if __name__ == "__main__":
    main()