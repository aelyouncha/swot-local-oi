# FAROX25 SWOT local OI

Minimal Python implementation of the local space-time optimal interpolation (OI)
used to map SWOT Level-3 sea-level anomaly (SLA) observations for the FAROX25
SWOT validation study.

This repository is intended to document the local OI method used in the
manuscript:

> Elyouncha et al., *Shipboard Validation of SWOT-Resolved Mesoscale Sea-Level
> Variability at High Latitudes*, submitted to *Geophysical Research Letters*.

## What the code does

The script `swot_local_oi.py`:

1. reads SWOT Level-3 NetCDF files,
2. extracts `longitude`, `latitude`, `ssha_filtered`, `time`, and
   `cross_track_distance`,
3. applies a simple cross-track observation-error model,
4. subsets the observations to either the Iceland--Faroe Ridge (IFR) or
   Faroe--Shetland Channel (FSC) region,
5. performs a localized space-time OI using a Gaussian covariance model,
6. writes gridded `sla_oi`, `sla_oi_var`, and `mdt_oi` fields to NetCDF.

The code is intentionally simple and close to the analysis version used for the
paper. It is not yet a general software package.

## Example usage

For the Iceland--Faroe Ridge:

```bash
python swot_local_oi.py \
    --region IFR \
    --swot-dir /path/to/SWOT_L3_files \
    --background-file /path/to/swot_farox_uvh_binned_timeseries.nc \
    --output swot_oi_IFR.nc
```

For the Faroe--Shetland Channel:

```bash
python swot_local_oi.py \
    --region FSC \
    --swot-dir /path/to/SWOT_L3_files \
    --background-file /path/to/swot_farox_uvh_binned_timeseries_FSC.nc \
    --output swot_oi_FSC.nc
```

The default OI parameters are:

```text
Lx = 30 km
Ly = 30 km
Lt = 72 h
sigma_R = 0.07 m
sigma_B = 0.18 m
max_obs = 1000
```

## Input data

The SWOT Level-3 files are expected to contain at least:

- `longitude`
- `latitude`
- `ssha_filtered`
- `cross_track_distance`
- `time`, or the global attribute `firstMeasurementTime`

The background file is expected to contain `mdt`. If `mdt` is absent, the script
will still run but will write `mdt_oi` as NaN.

## Output

The output NetCDF file contains:

- `sla_oi(time, lat, lon)`: local OI SLA field
- `sla_oi_var(time, lat, lon)`: posterior variance
- `mdt_oi(lat, lon)`: MDT from the background file

## Data availability

The SWOT-OI fields used in the manuscript are archived on Zenodo:

https://doi.org/10.5281/zenodo.23283920

Raw and processed FAROX25 observational data are described in the manuscript
Open Research statement.

## License

This code is released under the MIT License.
