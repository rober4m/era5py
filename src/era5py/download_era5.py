# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import logging
import time
import zipfile
from datetime import datetime, date as Date, timedelta
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def _unzip_if_needed(path: Path) -> Path:
    """Extract a .nc file from a ZIP archive delivered by the CDS API, replacing the zip in place."""
    if not zipfile.is_zipfile(path):
        return path
    with zipfile.ZipFile(path) as zf:
        nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
        if not nc_names:
            logger.warning("ZIP at %s contains no .nc files; leaving as-is.", path.name)
            return path
        extracted = Path(zf.extract(nc_names[0], path.parent))
    extracted.replace(path)
    logger.info("Extracted NetCDF from ZIP → %s", path.name)
    return path


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


def parse_date(s: str) -> Date:
    """Parse DD-MM-YYYY string to a date object."""
    return datetime.strptime(s, "%d-%m-%Y").date()


def build_year_chunks(start: int, end: int, chunk: int = 5) -> list[list[int]]:
    """Split [start, end] into chunks of `chunk` years for smaller CDS requests."""
    years = list(range(start, end + 1))
    return [years[i:i + chunk] for i in range(0, len(years), chunk)]


def _months_and_days_for_year(year: int, date_start: Date, date_end: Date):
    """Return (months, days) covering the slice of `year` within [date_start, date_end]."""
    slice_start = max(date_start, Date(year, 1, 1))
    slice_end   = min(date_end,   Date(year, 12, 31))
    months = list(range(slice_start.month, slice_end.month + 1))
    days: set[int] = set()
    cur = slice_start
    while cur <= slice_end:
        days.add(cur.day)
        cur += timedelta(days=1)
    return months, sorted(days)


def area_from_config(cfg: dict) -> list[float]:
    """Return [N, W, S, E] depending on mode."""
    if cfg["mode"] == "point":
        lat, lon, pad = cfg["lat"], cfg["lon"], 1.0
        return [lat + pad, lon - pad, lat - pad, lon + pad]
    a = cfg["area"]
    return [a["north"], a["west"], a["south"], a["east"]]


# ---------------------------------------------------------------------------
# CDS download — standard backend (reanalysis-era5-single-levels)
# ---------------------------------------------------------------------------
def _build_request(year: int, variable: str, area: list[float],
                   date_start: Date, date_end: Date) -> dict:
    months, days = _months_and_days_for_year(year, date_start, date_end)
    return {
        "product_type": "reanalysis",
        "variable":     [variable],
        "year":         [str(year)],
        "month":        [f"{m:02d}" for m in months],
        "day":          [f"{d:02d}" for d in days],
        "time":         [f"{h:02d}:00" for h in range(0, 24)],
        "area":         area,
        "format":       "netcdf",
    }


def download_year(client: cdsapi.Client, year: int, out_path: Path,
                  cfg: dict, variable: str,
                  date_start: Date, date_end: Date,
                  max_retries: int = 3) -> None:
    """Download one year (or partial year) from ERA-5 single-levels with retry logic."""
    area    = area_from_config(cfg)
    request = _build_request(year, variable, area, date_start, date_end)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Requesting year %d (attempt %d/%d) …", year, attempt, max_retries)
            client.retrieve("reanalysis-era5-single-levels", request, str(out_path))
            _unzip_if_needed(out_path)
            logger.info("Saved → %s", out_path)
            return
        except Exception as exc:
            logger.warning("Download failed for year %d: %s", year, exc)
            if attempt == max_retries:
                logger.error("Giving up on year %d after %d attempts.", year, max_retries)
                raise
            time.sleep(30 * attempt)


def _download_all_cds(client: cdsapi.Client, cfg: dict,
                      outdir: Path, name: str, variables: list) -> None:
    """Year-by-year downloads via reanalysis-era5-single-levels (default)."""
    d_start = parse_date(cfg["date_start"])
    d_end   = parse_date(cfg["date_end"])
    for variable in variables:
        for year in range(d_start.year, d_end.year + 1):
            tag     = f"_{year}_{variable}_{name}"
            nc_path = outdir / f"era5_raw_{tag}.nc"
            if nc_path.exists():
                logger.info("%s already exists – skipping.", nc_path.name)
            else:
                download_year(client, year, nc_path, cfg, variable, d_start, d_end)


# ---------------------------------------------------------------------------
# Timeseries backend (reanalysis-era5-single-levels-timeseries)
# Optimised for single-point retrievals over long periods (ARCO/Zarr backed).
# https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries
# ---------------------------------------------------------------------------
def _build_timeseries_request(cfg: dict, variable: str) -> dict:
    """Single request covering the full period for a point location."""
    d_start    = parse_date(cfg["date_start"])
    d_end      = parse_date(cfg["date_end"])
    date_range = f"{d_start.strftime('%Y-%m-%d')}/{d_end.strftime('%Y-%m-%d')}"
    return {
        "product_type": "reanalysis",
        "variable":     [variable],
        "date":         date_range,
        "location":     {"latitude": cfg["lat"], "longitude": cfg["lon"]},
        "format":       "netcdf",
    }


def download_timeseries(client: cdsapi.Client, out_path: Path,
                        cfg: dict, variable: str,
                        max_retries: int = 3) -> None:
    """One CDS request for the full period using the timeseries-optimised dataset."""
    request = _build_timeseries_request(cfg, variable)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Timeseries request for '%s' %s–%s (attempt %d/%d) …",
                variable, cfg["date_start"], cfg["date_end"], attempt, max_retries,
            )
            client.retrieve(
                "reanalysis-era5-single-levels-timeseries", request, str(out_path)
            )
            _unzip_if_needed(out_path)
            logger.info("Saved → %s", out_path)
            return
        except Exception as exc:
            logger.warning("Timeseries download failed for '%s': %s", variable, exc)
            if attempt == max_retries:
                logger.error(
                    "Giving up on '%s' after %d attempts.", variable, max_retries
                )
                raise
            time.sleep(30 * attempt)


def _download_all_timeseries(client: cdsapi.Client, cfg: dict,
                              outdir: Path, name: str, variables: list) -> None:
    """One request per variable covering all years (timeseries backend)."""
    for variable in variables:
        tag     = f"_{cfg['date_start']}_{cfg['date_end']}_{variable}_{name}"
        nc_path = outdir / f"era5_ts_{tag}.nc"
        if nc_path.exists():
            logger.info("%s already exists – skipping.", nc_path.name)
        else:
            download_timeseries(client, nc_path, cfg, variable)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def download_all(client: cdsapi.Client, cfg: dict) -> None:
    """Download all years/variables defined in cfg."""
    outdir    = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    name      = cfg.get("name", "")
    variables = list(cfg["variables"])
    backend   = cfg.get("download_backend", "cds")

    logger.info("ERA-5 downloader  [backend: %s]", backend)
    logger.info("  Period    : %s – %s", cfg["date_start"], cfg["date_end"])
    logger.info("  Variables : %s", variables)
    logger.info("  Mode      : %s", cfg["mode"])
    logger.info("  Output    : %s", outdir.resolve())

    if backend == "timeseries":
        if cfg.get("mode") != "point":
            logger.warning(
                "timeseries backend is optimised for mode=point; "
                "current mode is '%s'. Falling back to cds backend.",
                cfg.get("mode"),
            )
            _download_all_cds(client, cfg, outdir, name, variables)
        else:
            _download_all_timeseries(client, cfg, outdir, name, variables)
    else:
        if backend != "cds":
            logger.warning("Unknown download_backend '%s'; using 'cds'.", backend)
        _download_all_cds(client, cfg, outdir, name, variables)


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

    nc_path = _unzip_if_needed(nc_path)
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    if ds["latitude"].dims:
        ds = ds.sel(latitude=lat, longitude=lon, method="nearest")

    df            = pd.DataFrame(index=pd.to_datetime(ds["valid_time"].values))
    df.index.name = "time"

    # ── Single-level variables ────────────────────────────────────────────────
    if "t2m" in ds:
        df["t2m_K"]     = ds["t2m"].values
        df["t2m_C"]     = df["t2m_K"] - 273.15
    if "d2m" in ds:
        df["d2m_K"]     = ds["d2m"].values
        # Specific humidity from dewpoint (Bolton 1980 approximation)
        e_s = 6.112 * np.exp(17.67 * (df["d2m_K"] - 273.15) /
                             ((df["d2m_K"] - 273.15) + 243.5))   # hPa
        df["specific_humidity_g_kg"] = (0.622 * e_s) / (1013.25 - 0.378 * e_s) * 1000
    if "tp" in ds:
        df["precip_m"]  = ds["tp"].values
        df["precip_mm"] = df["precip_m"] * 1000

    # ── Surface temperature ───────────────────────────────────────────────────
    if "skt" in ds:
        df["skt_K"] = ds["skt"].values
        df["skt_C"] = df["skt_K"] - 273.15

    # ── Solar radiation ───────────────────────────────────────────────────────
    # ssrd / ssr are hourly accumulated J/m²; divide by 3600 → mean W/m²
    if "ssrd" in ds:
        df["ssrd_J_m2"] = ds["ssrd"].values
        df["ssrd_W_m2"] = df["ssrd_J_m2"] / 3600.0
    if "ssr" in ds:
        df["ssr_J_m2"] = ds["ssr"].values
        df["ssr_W_m2"] = df["ssr_J_m2"] / 3600.0

    # ── Wind at native ERA-5 heights ──────────────────────────────────────────
    if "u10" in ds:
        df["u10_ms"] = ds["u10"].values
    if "v10" in ds:
        df["v10_ms"] = ds["v10"].values
    if "u10" in ds and "v10" in ds:
        u10 = df["u10_ms"].values
        v10 = df["v10_ms"].values
        df["wind_speed_10m_ms"] = np.sqrt(u10**2 + v10**2)
        df["wind_dir_10m_deg"]  = (np.degrees(np.arctan2(-u10, -v10)) + 360) % 360

    if "u100" in ds:
        df["u100_ms"] = ds["u100"].values
    if "v100" in ds:
        df["v100_ms"] = ds["v100"].values
    if "u100" in ds and "v100" in ds:
        u100 = df["u100_ms"].values
        v100 = df["v100_ms"].values
        df["wind_speed_100m_ms"] = np.sqrt(u100**2 + v100**2)
        df["wind_dir_100m_deg"]  = (np.degrees(np.arctan2(-u100, -v100)) + 360) % 360

    # ── Wind at target heights via log-wind profile ───────────────────────────
    if "u10" in ds and "v10" in ds:
        for z in target_heights:
            col = f"wind_speed_{z}m_ms"
            if z == 10:
                df[col] = df["wind_speed_10m_ms"]
            else:
                df[col] = log_wind_speed(u10, v10, z_low, z, roughness)

    # ── Date range filter from config ─────────────────────────────────────────
    if "date_start" in cfg and "date_end" in cfg:
        d_start = pd.Timestamp(parse_date(cfg["date_start"]))
        d_end   = pd.Timestamp(parse_date(cfg["date_end"]))
        df      = df.loc[d_start:d_end]

    return df
