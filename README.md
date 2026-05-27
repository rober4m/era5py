# era5py

Download, post-process and analyse ERA-5 reanalysis data via the CDS API.

## Variables supported

| Short name | ERA-5 variable |
|---|---|
| `t2m` | 2 m temperature |
| `d2m` | 2 m dewpoint temperature |
| `tp` | Total precipitation |
| `u10` / `v10` | 10 m U/V wind components |
| `u100` / `v100` | 100 m U/V wind components |
| wind at target heights | Reconstructed via log-wind profile |

Derived outputs: temperature in °C, precipitation in mm, specific humidity (g kg⁻¹), wind speed at 10 / 30 / 50 m.

## Project layout

```
era5py/
├── run_era5.py              # Entry point
├── settings_era5.yml        # User configuration
├── requirements.txt
└── src/era5py/
    ├── __init__.py
    └── download_era5.py     # Download + post-processing functions
```

## Requirements

Python 3.12 and:

```bash
pip install -r requirements.txt
```

### CDS API key

1. Register at <https://cds.climate.copernicus.eu>
2. Follow the [how-to-api guide](https://cds.climate.copernicus.eu/how-to-api) to create `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <YOUR_API_KEY>
```

## Configuration

Edit `settings_era5.yml` before running:

```yaml
year_start: 2020
year_end:   2021
name:       bolivia

variable: "2m_temperature"
mode:     area            # "point" or "area"

lat:  -16.5               # used when mode = point
lon:  -68.15

area:                     # used when mode = area
  north:  -8.0
  west:  -70.0
  south: -24.0
  east:  -57.0

output_dir: era5_output_bolivia
```

## Usage

```bash
# Download data defined in settings_era5.yml
python run_era5.py -o download

# Post-process downloaded NetCDF files → CSV
python run_era5.py -o process

# Use a custom config file
python run_era5.py -o download -c my_settings.yml
```

Files are saved as `era5_raw_<year>_<variable>_<name>.nc` inside `output_dir`. Processed CSVs are written alongside the NetCDF files.
