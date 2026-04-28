

import argparse
import os
import sys
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROUGHNESS_LENGTH = 0.03   # m  (open farmland – adjust for your site)
REF_HEIGHT_LOW   = 10.0   # m  (ERA-5 lowest wind level)
REF_HEIGHT_HIGH  = 100.0  # m  (ERA-5 highest single-level wind level)
TARGET_HEIGHTS   = [10, 30, 50]  # m


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


# ---------------------------------------------------------------------------
# CDS download
# ---------------------------------------------------------------------------
def download_era5_chunk(client: cdsapi.Client, years: list[int],
                        lat: float, lon: float, out_path: Path):
    """Download one chunk of years from ERA-5 single-levels."""
    str_years  = [str(y) for y in years]
    str_months = [f"{m:02d}" for m in range(1, 13)]
    str_days   = [f"{d:02d}" for d in range(1, 32)]
    str_times  = [f"{h:02d}:00" for h in range(0, 24)]

    # Build a bounding box tight around the requested point
    # (ERA-5 ~0.25° grid; ±0.5° guarantees we get the nearest cell)
    area = [lat + 0.5, lon - 0.5, lat - 0.5, lon + 0.5]   # N W S E

    request = {
        "product_type": "reanalysis",
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "total_precipitation",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
        ],
        "year":  str_years,
        "month": str_months,
        "day":   str_days,
        "time":  str_times,
        "area":  area,
        "format": "netcdf",
    }

    print(f"  Requesting years {years[0]}–{years[-1]} …")
    client.retrieve("reanalysis-era5-single-levels", request, str(out_path))
    print(f"  Saved → {out_path}")


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download ERA-5 hourly data for a lat/lon point.")
    parser.add_argument("--lat",   type=float, default=51.5,
                        help="Latitude  (default: 51.5 – London)")
    parser.add_argument("--lon",   type=float, default=-0.12,
                        help="Longitude (default: -0.12 – London)")
    parser.add_argument("--start", type=int,   default=2010,
                        help="First year (default: 2010)")
    parser.add_argument("--end",   type=int,   default=2023,
                        help="Last year  (default: 2023)")
    parser.add_argument("--outdir", type=str,  default="era5_output",
                        help="Output directory (default: era5_output)")
    parser.add_argument("--chunk", type=int,   default=5,
                        help="Years per CDS request chunk (default: 5)")
    args = parser.parse_args()

    if args.end - args.start < 9:
        print("WARNING: period is shorter than 10 years. "
              "Use --start / --end to specify a longer range.")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nERA-5 downloader")
    print(f"  Point  : lat={args.lat}, lon={args.lon}")
    print(f"  Period : {args.start} – {args.end}  "
          f"({args.end - args.start + 1} years)")
    print(f"  Output : {out_dir.resolve()}\n")

    # ── Download in year chunks ──────────────────────────────────────────────
    client = cdsapi.Client()
    chunks = build_year_chunks(args.start, args.end, chunk=args.chunk)
    nc_files: list[Path] = []

    for chunk_years in chunks:
        tag  = f"{chunk_years[0]}-{chunk_years[-1]}"
        nc_p = out_dir / f"era5_raw_{tag}.nc"
        if nc_p.exists():
            print(f"  {nc_p.name} already exists – skipping download.")
        else:
            download_era5_chunk(client, chunk_years, args.lat, args.lon, nc_p)
        nc_files.append(nc_p)

    # ── Post-process & merge ─────────────────────────────────────────────────
    print("\nPost-processing downloaded files …")
    frames: list[pd.DataFrame] = []
    for nc_p in nc_files:
        print(f"  Processing {nc_p.name} …")
        frames.append(process_dataset(nc_p, args.lat, args.lon))

    df_all = pd.concat(frames).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="first")]

    # ── Save outputs ─────────────────────────────────────────────────────────
    csv_path = out_dir / f"era5_point_lat{args.lat}_lon{args.lon}.csv"
    df_all.to_csv(csv_path)
    print(f"\nMerged CSV saved → {csv_path}")

    # Quick summary
    print("\n── Summary statistics ──────────────────────────────────────────")
    cols_summary = [
        "t2m_C", "specific_humidity_g_kg", "precip_mm",
        "wind_speed_10m_ms", "wind_speed_30m_ms", "wind_speed_50m_ms",
    ]
    print(df_all[cols_summary].describe().round(3).to_string())
    print("\nDone ✓")


if __name__ == "__main__":
    main()
