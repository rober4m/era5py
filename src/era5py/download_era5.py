# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import logging
import time
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log_wind_speed(u_ref, v_ref, z_ref: float, z_target: float,
                   z0: float = 0.03) -> np.ndarray:
    """
    Estimate wind speed at z_target using the logarithmic wind profile.

        U(z) = U_ref * ln(z / z0) / ln(z_ref / z0)
    """
    u_ref = np.asarray(u_ref)
    v_ref = np.asarray(v_ref)
    scale = np.log(z_target / z0) / np.log(z_ref / z0)
    return np.sqrt((u_ref * scale) ** 2 + (v_ref * scale) ** 2)


def build_year_chunks(start: int, end: int, chunk: int = 5) -> list[list[int]]:
    """Split [start, end] into chunks of `chunk` years for smaller CDS requests."""
    years = list(range(start, end + 1))
    return [years[i:i + chunk] for i in range(0, len(years), chunk)]


def area_from_config(cfg: dict) -> list[float]:
    """Return [N, W, S, E] depending on mode."""
    if cfg["mode"] == "point":
        lat, lon, pad = cfg["lat"], cfg["lon"], 1.0
        return [lat + pad, lon - pad, lat - pad, lon + pad]
    a = cfg["area"]
    return [a["north"], a["west"], a["south"], a["east"]]


# ---------------------------------------------------------------------------
# CDS download
# ---------------------------------------------------------------------------
def _build_request(year: int, variable: str, area: list[float]) -> dict:
    return {
        "product_type": "reanalysis",
        "variable":     [variable],
        "year":         [str(year)],
        "month":        [f"{m:02d}" for m in range(1, 13)],
        "day":          [f"{d:02d}" for d in range(1, 32)],
        "time":         [f"{h:02d}:00" for h in range(0, 24)],
        "area":         area,
        "format":       "netcdf",
    }


def download_year(client: cdsapi.Client, year: int, out_path: Path,
                  cfg: dict, max_retries: int = 3) -> None:
    """Download one year from ERA-5 single-levels with retry logic."""
    area     = area_from_config(cfg)
    variable = cfg["variable"]
    request  = _build_request(year, variable, area)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Requesting year %d (attempt %d/%d) …", year, attempt, max_retries)
            client.retrieve("reanalysis-era5-single-levels", request, str(out_path))
            logger.info("Saved → %s", out_path)
            return
        except Exception as exc:
            logger.warning("Download failed for year %d: %s", year, exc)
            if attempt == max_retries:
                logger.error("Giving up on year %d after %d attempts.", year, max_retries)
                raise
            time.sleep(30 * attempt)


def download_all(client: cdsapi.Client, cfg: dict) -> None:
    """Download all years defined in cfg, skipping files that already exist."""
    outdir   = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    name     = cfg.get("name", "")
    variable = cfg["variable"]

    logger.info("ERA-5 downloader")
    logger.info("  Period   : %d – %d", cfg["year_start"], cfg["year_end"])
    logger.info("  Variable : %s", variable)
    logger.info("  Mode     : %s", cfg["mode"])
    logger.info("  Output   : %s", outdir.resolve())

    for year in range(cfg["year_start"], cfg["year_end"] + 1):
        tag     = f"_{year}_{variable}_{name}"
        nc_path = outdir / f"era5_raw_{tag}.nc"
        if nc_path.exists():
            logger.info("%s already exists – skipping.", nc_path.name)
        else:
            download_year(client, year, nc_path, cfg)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def process_dataset(nc_path: Path, lat: float, lon: float, cfg: dict) -> pd.DataFrame:
    """
    Open a downloaded NetCDF, select the nearest grid point, compute derived
    variables and return a tidy hourly DataFrame.
    """
    roughness      = cfg.get("roughness_length", 0.03)
    z_low          = cfg.get("ref_height_low",   10.0)
    target_heights = cfg.get("target_heights",   [10, 30, 50])

    ds = xr.open_dataset(nc_path)
    ds = ds.sel(latitude=lat, longitude=lon, method="nearest")

    df            = pd.DataFrame(index=pd.to_datetime(ds["valid_time"].values))
    df.index.name = "time"

    # ── Single-level variables ────────────────────────────────────────────────
    df["t2m_K"]     = ds["t2m"].values
    df["t2m_C"]     = df["t2m_K"] - 273.15
    df["d2m_K"]     = ds["d2m"].values
    df["precip_m"]  = ds["tp"].values
    df["precip_mm"] = df["precip_m"] * 1000

    # Specific humidity from dewpoint (Bolton 1980 approximation)
    e_s = 6.112 * np.exp(17.67 * (df["d2m_K"] - 273.15) /
                         ((df["d2m_K"] - 273.15) + 243.5))   # hPa
    df["specific_humidity_g_kg"] = (0.622 * e_s) / (1013.25 - 0.378 * e_s) * 1000

    # ── Wind at native ERA-5 heights ──────────────────────────────────────────
    u10  = ds["u10"].values
    v10  = ds["v10"].values
    u100 = ds["u100"].values
    v100 = ds["v100"].values

    df["wind_speed_10m_ms"]  = np.sqrt(u10**2  + v10**2)
    df["wind_speed_100m_ms"] = np.sqrt(u100**2 + v100**2)

    # ── Wind at target heights via log-wind profile ───────────────────────────
    for z in target_heights:
        col = f"wind_speed_{z}m_ms"
        if z == 10:
            df[col] = df["wind_speed_10m_ms"]
        else:
            df[col] = log_wind_speed(u10, v10, z_low, z, roughness)

    return df
