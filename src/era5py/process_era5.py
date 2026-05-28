# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

_COLORSCALES: dict[str, str] = {
    "t2m":  "RdBu_r", "d2m":  "RdBu_r", "skt": "RdBu_r",
    "tp":   "Blues",
    "ssrd": "YlOrRd", "ssr":  "YlOrRd",
    "u10":  "RdBu",   "v10":  "RdBu",
    "u100": "RdBu",   "v100": "RdBu",
}

# ---------------------------------------------------------------------------
# Point: time-series
# ---------------------------------------------------------------------------
def plot_timeseries(csv_path: Path, fig_dir: Path) -> None:
    """One interactive HTML line chart per numeric column in the processed CSV."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if df.empty:
        logger.warning("Empty DataFrame from %s – skipping.", csv_path.name)
        return

    for col in df.select_dtypes(include="number").columns:
        fig = px.line(
            df, y=col,
            title=f"{col}  |  {csv_path.stem}",
            labels={"index": "Time", col: col},
        )
        fig.update_layout(xaxis_title="Time", yaxis_title=col)
        out = fig_dir / f"{csv_path.stem}_{col}.html"
        fig.write_html(str(out))
        logger.info("Saved → %s", out.name)


def _plot_all_point(csv_files: list[Path], fig_dir: Path) -> None:
    for csv_path in csv_files:
        logger.info("Plotting time-series for %s …", csv_path.name)
        plot_timeseries(csv_path, fig_dir)


# ---------------------------------------------------------------------------
# Area: animated map
# ---------------------------------------------------------------------------
def _pick_colorscale(var_name: str) -> str:
    for key, cs in _COLORSCALES.items():
        if key in var_name.lower():
            return cs
    return "Viridis"


def plot_area_map(nc_path: Path, variable: str, fig_dir: Path,
                  n_frames: int = 48) -> None:
    """Animated Heatmap with a time slider for one gridded ERA-5 variable."""
    ds = xr.open_dataset(nc_path)
    if variable not in ds:
        avail = list(ds.data_vars)
        logger.warning("'%s' not in %s. Available: %s", variable, nc_path.name, avail)
        return

    da    = ds[variable]
    times = da["valid_time"].values
    step  = max(1, len(times) // n_frames)
    sel   = times[::step]
    cs    = _pick_colorscale(variable)
    lons  = da["longitude"].values
    lats  = da["latitude"].values

    frames        = []
    slider_steps  = []
    for t in sel:
        snap  = da.sel(valid_time=t, method="nearest")
        label = str(pd.Timestamp(t))[:10]
        frames.append(go.Frame(
            data=[go.Heatmap(z=snap.values, x=lons, y=lats,
                             colorscale=cs, colorbar=dict(title=variable),
                             zsmooth="best")],
            name=label,
        ))
        slider_steps.append(dict(
            args=[[label], {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}],
            label=label, method="animate",
        ))

    snap0 = da.sel(valid_time=sel[0], method="nearest")
    fig   = go.Figure(
        data=[go.Heatmap(z=snap0.values, x=lons, y=lats,
                         colorscale=cs, colorbar=dict(title=variable),
                         zsmooth="best")],
        frames=frames,
    )
    fig.update_layout(
        title=f"{variable}  |  {nc_path.stem}",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.15, x=0.5, xanchor="center",
            buttons=[
                dict(label="Play",  method="animate",
                     args=[None, {"frame": {"duration": 100, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(steps=slider_steps, x=0, len=1.0,
                      currentvalue=dict(prefix="Date: ", visible=True))],
    )
    out = fig_dir / f"{nc_path.stem}_{variable}_map.html"
    fig.write_html(str(out))
    logger.info("Saved → %s", out.name)


def _plot_all_area(nc_files: list[Path], fig_dir: Path) -> None:
    for nc_path in nc_files:
        ds = xr.open_dataset(nc_path)
        for var in ds.data_vars:
            logger.info("Mapping %s / %s …", nc_path.name, var)
            plot_area_map(nc_path, var, fig_dir)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def visualize(cfg: dict) -> None:
    """Route to point time-series or area animated maps based on cfg.mode."""
    outdir    = Path(cfg["output_dir"])
    name      = cfg.get("name", "")
    variables = list(cfg["variables"])
    mode      = cfg.get("mode", "point")
    fig_dir   = Path(cfg.get("results_dir", outdir / "figures"))
    fig_dir.mkdir(parents=True, exist_ok=True)

    if mode == "point":
        files = []
        for variable in variables:
            files.extend(sorted(outdir.glob(f"era5_raw_*{variable}*{name}.csv")))
        if not files:
            logger.error("No CSV files in %s. Run 'process' first.", outdir)
            return
        _plot_all_point(files, fig_dir)
    else:
        files = []
        for variable in variables:
            files.extend(sorted(outdir.glob(f"era5_raw_*{variable}*{name}.nc")))
        if not files:
            logger.error("No NetCDF files in %s.", outdir)
            return
        _plot_all_area(files, fig_dir)

    logger.info("All figures saved in %s", fig_dir.resolve())


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def _load_all_years(csv_files: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(f, index_col="time", parse_dates=True) for f in csv_files]
    return pd.concat(frames).sort_index()


def _primary_col(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include="number").columns.tolist()
    celsius = [c for c in numeric if c.endswith("_C")]
    return celsius[0] if celsius else numeric[0]


def _build_variable_dashboard(
    csv_files: list[Path], variable: str, name: str, stem: str, fig_dir: Path
) -> None:
    all_data = _load_all_years(csv_files)
    col      = _primary_col(all_data)
    series   = all_data[col]

    # Panel 1: monthly mean + rolling ±1σ band across the full record
    monthly   = series.resample("ME").mean()
    roll_mean = monthly.rolling(12, center=True, min_periods=6).mean()
    roll_std  = monthly.rolling(12, center=True, min_periods=6).std()
    band_x = list(monthly.index) + list(monthly.index[::-1])
    band_y = list(roll_mean + roll_std) + list((roll_mean - roll_std)[::-1])

    # Panel 2: single-year 2018 hourly series
    years_available = series.index.year.unique()
    series_2018 = series[series.index.year == 2018] if 2018 in years_available else None
    if series_2018 is None:
        logger.warning("No 2018 data for '%s'; year panel will be empty.", variable)

    # Panel 3: daily mean profile (hour 0–23) ± 1σ
    hourly_mean = series.groupby(series.index.hour).mean()
    hourly_std  = series.groupby(series.index.hour).std()
    hours = list(hourly_mean.index)
    prof_band_x = hours + hours[::-1]
    prof_band_y = (
        list(hourly_mean + hourly_std) + list((hourly_mean - hourly_std)[::-1])
    )

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "All years – monthly mean ± 1σ",
            "Year 2018",
            "Daily average profile (hour 0–23)",
        ],
        vertical_spacing=0.1,
    )

    # Row 1 – all-years variability
    fig.add_trace(go.Scatter(
        x=monthly.index, y=monthly.values,
        mode="lines", name="Monthly mean",
        line=dict(color="royalblue", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=band_x, y=band_y,
        fill="toself", fillcolor="rgba(65,105,225,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ", showlegend=True,
    ), row=1, col=1)

    # Row 2 – 2018
    if series_2018 is not None:
        fig.add_trace(go.Scatter(
            x=series_2018.index, y=series_2018.values,
            mode="lines", name="2018",
            line=dict(color="darkorange", width=1),
        ), row=2, col=1)

    # Row 3 – daily profile
    fig.add_trace(go.Scatter(
        x=hours, y=hourly_mean.values,
        mode="lines+markers", name="Mean profile",
        line=dict(color="seagreen", width=2),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=prof_band_x, y=prof_band_y,
        fill="toself", fillcolor="rgba(46,139,87,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ daily", showlegend=True,
    ), row=3, col=1)

    fig.update_layout(
        height=900,
        title_text=f"ERA-5 Summary – {name.upper()}  |  {variable}",
        title_font_size=16,
    )
    fig.update_xaxes(title_text="Date",        row=1, col=1)
    fig.update_xaxes(title_text="Date",        row=2, col=1)
    fig.update_xaxes(title_text="Hour of day", row=3, col=1)
    fig.update_yaxes(title_text=col, row=1, col=1)
    fig.update_yaxes(title_text=col, row=2, col=1)
    fig.update_yaxes(title_text=col, row=3, col=1)

    out = fig_dir / f"{stem}.html"
    fig.write_html(str(out))
    logger.info("Dashboard saved → %s", out)


def build_dashboard(cfg: dict) -> None:
    """Build a 3-panel summary dashboard per variable: all-years, 2018, daily profile."""
    outdir    = Path(cfg["output_dir"])
    name      = cfg.get("name", "")
    variables = list(cfg["variables"])
    fig_dir   = Path(cfg.get("results_dir", outdir / "figures"))
    fig_dir.mkdir(parents=True, exist_ok=True)

    for variable in variables:
        csv_files = sorted(
            f for f in outdir.glob(f"era5_raw_*{variable}*{name}.csv")
            if "_stats_" not in f.name and "_uncertainty" not in f.name
        )
        if not csv_files:
            logger.warning("No CSV files for '%s' – skipping dashboard.", variable)
            continue
        stem = (
            f"summary_{name}" if len(variables) == 1
            else f"summary_{name}_{variable.replace(' ', '_')}"
        )
        _build_variable_dashboard(csv_files, variable, name, stem, fig_dir)
        logger.info("Dashboard complete for '%s'.", variable)
