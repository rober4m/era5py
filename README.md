# ERA-5 Data Downloader
=====================
Downloads ERA-5 reanalysis data for a single point (lat/lon) over a multi-year period.

Variables downloaded:
  - 2m temperature           (t2m)
  - 2m dewpoint temperature  (d2m)  → proxy for moisture / specific humidity
  - Total precipitation      (tp)
  - 10m U & V wind           (u10, v10)
  - 100m U & V wind          (u100, v100)  ← closest ERA-5 level to 30 m & 50 m
  - Wind speed reconstructed at 10 / 30 / 50 m using the log-wind profile

## Requirements
------------
    pip install cdsapi xarray netCDF4 numpy pandas

You also need a CDS API key:
  1. Register at https://cds.climate.copernicus.eu
  2. Follow https://cds.climate.copernicus.eu/how-to-api
     to create ~/.cdsapirc (or set env vars CDS_URL + CDS_API_KEY)

## Usage
-----
    python download_era5.py                           # default point & period
    python download_era5.py --lat 40.42 --lon -3.70  # Madrid
    python download_era5.py --lat 48.85 --lon 2.35 --start 2000 --end 2023