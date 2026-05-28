# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import argparse
import logging
import sys
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import cdsapi
import yaml

from src.era5py.download_era5 import download_all, process_dataset
from src.era5py.process_era5 import visualize as _visualize, build_dashboard
from src.era5py.post_processing import run_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(
        description="Download and process ERA-5 hourly data for a lat/lon point or area.")

    parser.add_argument(
        "-o", "--operation",
        required=True,
        choices=["download", "process", "visualize", "stats"],
        help="Operation to run")

    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML config file in etc/ ")

    args = parser.parse_args()

    cfg = load_config(args.config)
    OPERATIONS[args.operation](cfg)

# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------
def download(cfg: dict) -> None:
    client = cdsapi.Client()
    download_all(client, cfg)


def visualize(cfg: dict) -> None:
    _visualize(cfg)


def stats(cfg: dict) -> None:
    run_stats(cfg)


def process(cfg: dict) -> None:
    outdir    = Path(cfg["output_dir"])
    name      = cfg.get("name", "")
    variables = list(cfg["variables"])
    lat       = cfg["lat"]
    lon       = cfg["lon"]

    for variable in variables:
        nc_files = sorted(outdir.glob(f"era5_raw_*{variable}*{name}*.nc"))
        if not nc_files:
            logger.error("No NetCDF files found in %s for variable '%s'.", outdir, variable)
            sys.exit(1)

        for nc_path in nc_files:
            logger.info("Processing %s …", nc_path.name)
            df       = process_dataset(nc_path, lat, lon, cfg)
            csv_path = nc_path.with_suffix(".csv")
            df.to_csv(csv_path)
            logger.info("Saved → %s", csv_path)

    build_dashboard(cfg)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    config_path = Path('etc/') / path
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    return OmegaConf.load(config_path)


OPERATIONS = {
    "download":  download,
    "process":   process,
    "visualize": visualize,
    "stats":     stats,
}

if __name__ == "__main__":
    main()
