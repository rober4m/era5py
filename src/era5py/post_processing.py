# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wind direction
# ---------------------------------------------------------------------------
def compute_wind_direction(df: pd.DataFrame, u_col: str, v_col: str,
                           out_col: str = "wind_direction_deg") -> pd.DataFrame:
    """Add meteorological wind direction (degrees from N, clockwise) from u/v."""
    u  = df[u_col].values
    v  = df[v_col].values
    wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    df = df.copy()
    df[out_col] = wd
    return df


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Count, mean, std, percentiles (5/25/50/75/95), max and CV per column."""
    numeric = df.select_dtypes(include="number")
    summary = numeric.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    summary["cv"] = summary["std"] / summary["mean"].abs()
    return summary


def monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly mean and std for all numeric columns."""
    return df.select_dtypes(include="number").resample("ME").agg(["mean", "std"])


def seasonal_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Quarterly mean, std, min, max for all numeric columns."""
    return df.select_dtypes(include="number").resample("QE").agg(["mean", "std", "min", "max"])


def rolling_stats(df: pd.DataFrame, window: int = 720) -> pd.DataFrame:
    """Rolling mean and std."""
    numeric = df.select_dtypes(include="number")
    roll    = numeric.rolling(window, min_periods=1)
    return pd.concat(
        [roll.mean().add_suffix("_roll_mean"),
         roll.std().add_suffix("_roll_std")],
        axis=1,
    )


def linear_trend(series: pd.Series) -> dict:
    """Slope, intercept, r², p-value from a least-squares fit on the series."""
    y   = series.dropna().values
    x   = np.arange(len(y), dtype=float)
    res = stats.linregress(x, y)
    return {
        "slope":     res.slope,
        "intercept": res.intercept,
        "r2":        res.rvalue ** 2,
        "p_value":   res.pvalue,
    }


def trend_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Linear trend statistics for each numeric column."""
    rows = {col: linear_trend(df[col]) for col in df.select_dtypes(include="number").columns}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Uncertainty analysis
# ---------------------------------------------------------------------------
def bootstrap_ci(series: pd.Series, stat_func=np.mean,
                 n_boot: int = 500, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval for a statistic."""
    rng   = np.random.default_rng(42)
    data  = series.dropna().values
    boot  = [stat_func(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1.0 - alpha))


def uncertainty_report(df: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    """Bootstrap 95 % CI for mean, p5, and p95 for each numeric column."""
    rows = {}
    for col in df.select_dtypes(include="number").columns:
        s             = df[col].dropna()
        lo_m,  hi_m  = bootstrap_ci(s, np.mean,                            n_boot)
        lo_p5, hi_p5 = bootstrap_ci(s, lambda x: np.percentile(x,  5),    n_boot)
        lo_p95,hi_p95 = bootstrap_ci(s, lambda x: np.percentile(x, 95),   n_boot)
        rows[col] = {
            "mean_ci_lo": lo_m,   "mean_ci_hi": hi_m,
            "p5_ci_lo":   lo_p5,  "p5_ci_hi":   hi_p5,
            "p95_ci_lo":  lo_p95, "p95_ci_hi":  hi_p95,
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _detect_uv_pairs(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return (u_col, v_col, out_col) pairs found in df."""
    candidates = [
        ("u10_ms",  "v10_ms",  "wind_dir_10m_deg"),
        ("u100_ms", "v100_ms", "wind_dir_100m_deg"),
    ]
    return [(u, v, o) for u, v, o in candidates
            if u in df.columns and v in df.columns]


def run_stats(cfg: dict) -> None:
    """Compute and save statistics + uncertainty reports for all processed CSVs."""
    outdir         = Path(cfg["output_dir"])
    name           = cfg.get("name", "")
    variables      = list(cfg["variables"])
    rolling_window = int(cfg.get("rolling_window", 720))
    n_boot         = int(cfg.get("n_boot", 500))

    for variable in variables:
        csv_files = sorted(outdir.glob(f"era5_raw_*{variable}*{name}*.csv"))
        if not csv_files:
            logger.warning(
                "No CSV files for '%s' in %s. Run 'process' first.", variable, outdir)
            continue

        for csv_path in csv_files:
            logger.info("Running statistics on %s …", csv_path.name)
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

            # ── Wind direction ────────────────────────────────────────────────
            for u_col, v_col, out_col in _detect_uv_pairs(df):
                df = compute_wind_direction(df, u_col, v_col, out_col)
                logger.info("Added %s from %s / %s.", out_col, u_col, v_col)

            # ── Save enhanced CSV with wind direction ─────────────────────────
            df.to_csv(csv_path)

            # ── Statistical reports ───────────────────────────────────────────
            stem = csv_path.stem
            _write(descriptive_stats(df),              outdir / f"{stem}_stats_descriptive.csv")
            _write(monthly_stats(df),                  outdir / f"{stem}_stats_monthly.csv")
            _write(seasonal_stats(df),                 outdir / f"{stem}_stats_seasonal.csv")
            _write(rolling_stats(df, rolling_window),  outdir / f"{stem}_stats_rolling.csv")
            _write(trend_summary(df),                  outdir / f"{stem}_stats_trends.csv")
            _write(uncertainty_report(df, n_boot),     outdir / f"{stem}_uncertainty.csv")

    logger.info("Statistics complete → %s", outdir.resolve())


def _write(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)
    logger.info("Saved → %s", path.name)
