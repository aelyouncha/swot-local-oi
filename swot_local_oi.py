#!/usr/bin/env python3
"""
Local SWOT optimal interpolation for FAROX25.

Cleaned minimal version of the analysis script used for the FAROX25 SWOT
validation study. It performs localized space-time optimal interpolation (OI)
of SWOT Level-3 sea-level anomaly (SLA) observations onto a regular grid.
"""

from __future__ import annotations

import argparse
import glob
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from global_land_mask import globe


@dataclass(frozen=True)
class RegionConfig:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    start: str
    stop: str
    dlon: float = 0.07
    dlat: float = 0.05


REGIONS = {
    "IFR": RegionConfig(
        lon_min=-13.0,
        lon_max=-8.0,
        lat_min=62.25,
        lat_max=64.75,
        start="2025-07-11T00:00:00",
        stop="2025-07-13T00:00:00",
    ),
    "FSC": RegionConfig(
        lon_min=-5.8,
        lon_max=-0.6,
        lat_min=60.0,
        lat_max=62.4,
        start="2025-07-04T00:00:00",
        stop="2025-07-10T00:00:00",
    ),
}


def _km_per_deg_lon(lat_deg):
    return 111.32 * np.cos(np.deg2rad(lat_deg))


def _dxdy_km(lon_obs, lat_obs, lon0, lat0):
    """Approximate km offsets from (lon0, lat0) using a local tangent plane."""
    dy = (lat_obs - lat0) * 111.32
    dx = (lon_obs - lon0) * _km_per_deg_lon(lat0)
    return dx, dy


def wrap180(lon):
    """Wrap longitudes to [-180, 180]."""
    lon = np.asarray(lon)
    return np.where(lon > 180, lon - 360, lon)


def to_hours(t, t0):
    """Convert datetime64 values to hours relative to t0."""
    t = np.asarray(t, dtype="datetime64[ns]")
    t0 = np.datetime64(t0, "ns")
    return (t - t0).astype("timedelta64[s]").astype(float) / 3600.0


def make_time_grid(start, stop, delta_hours=24):
    """Create a regular datetime64 time grid."""
    start = np.datetime64(start, "ns")
    stop = np.datetime64(stop, "ns")
    return np.arange(
        start,
        stop + np.timedelta64(delta_hours, "h"),
        np.timedelta64(delta_hours, "h"),
    ).astype("datetime64[ns]")


def oi_space_time(
    lon_obs,
    lat_obs,
    t_obs_hours,
    y_obs,
    lon_grid,
    lat_grid,
    t_grid_hours,
    Lx_km=30.0,
    Ly_km=30.0,
    Lt_h=72.0,
    sigma_R=0.07,
    sigma_B=0.18,
    sigma_R_obs=None,
    radius_space_km=None,
    radius_time_h=None,
    max_obs=1000,
    compute_mask=None,
):
    """
    Localized space-time optimal interpolation onto a regular grid.

    The covariance model is a separable Gaussian in zonal distance,
    meridional distance, and time. The background mean is assumed to be zero,
    so the input observations should be anomalies.
    """
    lon_obs = np.asarray(lon_obs).ravel()
    lat_obs = np.asarray(lat_obs).ravel()
    t_obs_hours = np.asarray(t_obs_hours).ravel()
    y_obs = np.asarray(y_obs).ravel()
    sigma_R_obs = None if sigma_R_obs is None else np.asarray(sigma_R_obs).ravel()

    good = (
        np.isfinite(lon_obs)
        & np.isfinite(lat_obs)
        & np.isfinite(t_obs_hours)
        & np.isfinite(y_obs)
    )
    lon_obs = lon_obs[good]
    lat_obs = lat_obs[good]
    t_obs_hours = t_obs_hours[good]
    y_obs = y_obs[good]
    if sigma_R_obs is not None:
        sigma_R_obs = sigma_R_obs[good]

    ny, nx = lon_grid.shape
    nt = len(t_grid_hours)

    if lon_obs.size == 0:
        return np.full((nt, ny, nx), np.nan), np.full((nt, ny, nx), np.nan)

    if radius_space_km is None:
        radius_space_km = 2.5 * max(Lx_km, Ly_km)
    if radius_time_h is None:
        radius_time_h = 2.0 * Lt_h

    x_hat = np.full((nt, ny, nx), np.nan, dtype=float)
    p_hat = np.full((nt, ny, nx), np.nan, dtype=float)

    sigB2 = float(sigma_B) ** 2
    sigR2 = float(sigma_R) ** 2

    if compute_mask is None:
        jj, ii = np.indices(lon_grid.shape)
        jj, ii = jj.ravel(), ii.ravel()
    else:
        jj, ii = np.where(compute_mask)

    for it, t0 in enumerate(t_grid_hours):
        mt = np.abs(t_obs_hours - t0) <= radius_time_h
        if not mt.any():
            continue

        lon_t = lon_obs[mt]
        lat_t = lat_obs[mt]
        t_t = t_obs_hours[mt]
        y_t = y_obs[mt]
        sig_t = sigma_R_obs[mt] if sigma_R_obs is not None else None

        for j, i in zip(jj, ii):
            lon0 = float(lon_grid[j, i])
            lat0 = float(lat_grid[j, i])

            dx, dy = _dxdy_km(lon_t, lat_t, lon0, lat0)
            dt = t_t - t0

            ms = (dx * dx + dy * dy) <= (radius_space_km**2)
            if ms.sum() < 3:
                continue

            dxl, dyl, dtl = dx[ms], dy[ms], dt[ms]
            yl = y_t[ms]
            sigl = sig_t[ms] if sig_t is not None else None

            if yl.size > max_obs:
                dnorm = (dxl / Lx_km) ** 2 + (dyl / Ly_km) ** 2 + (dtl / Lt_h) ** 2
                idx = np.argpartition(dnorm, max_obs)[:max_obs]
                dxl, dyl, dtl, yl = dxl[idx], dyl[idx], dtl[idx], yl[idx]
                if sigl is not None:
                    sigl = sigl[idx]

            m = yl.size

            c = sigB2 * np.exp(
                -((dxl / Lx_km) ** 2 + (dyl / Ly_km) ** 2 + (dtl / Lt_h) ** 2)
            )

            Dx = (dxl[:, None] - dxl[None, :]) / Lx_km
            Dy = (dyl[:, None] - dyl[None, :]) / Ly_km
            Dt = (dtl[:, None] - dtl[None, :]) / Lt_h
            Coo = sigB2 * np.exp(-(Dx * Dx + Dy * Dy + Dt * Dt))

            if sigl is None:
                Coo.flat[:: m + 1] += sigR2
            else:
                Coo.flat[:: m + 1] += sigl**2

            try:
                w = np.linalg.solve(Coo, yl)
            except np.linalg.LinAlgError:
                Coo.flat[:: m + 1] += 0.05 * sigR2 + 1e-12
                w = np.linalg.solve(Coo, yl)

            x_hat[it, j, i] = c @ w

            try:
                v = np.linalg.solve(Coo, c)
            except np.linalg.LinAlgError:
                Coo.flat[:: m + 1] += 0.05 * sigR2 + 1e-12
                v = np.linalg.solve(Coo, c)

            p_hat[it, j, i] = max(sigB2 - (c @ v), 0.0)

    return x_hat, p_hat


def read_swot_l3(swot_dir, pattern="SWOT*.nc"):
    """Read SWOT Level-3 files and return flattened SLA observations."""
    lon_all, lat_all, t_all, y_all, sig_all = [], [], [], [], []

    file_list = np.sort(glob.glob(str(Path(swot_dir) / pattern)))
    if len(file_list) == 0:
        raise FileNotFoundError(f"No SWOT files found in {swot_dir!s} with pattern {pattern!r}")

    for f in file_list:
        with xr.open_dataset(f) as ds:
            if "time" in ds.coords or "time" in ds.variables:
                t = ds["time"].values
                if isinstance(t, np.ndarray):
                    t = t.ravel()[0]
                t = np.datetime64(t, "ns")
            else:
                tstr = ds.attrs["firstMeasurementTime"].rstrip("Z")
                t = None
                for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        t = np.datetime64(datetime.strptime(tstr, fmt), "ns")
                        break
                    except ValueError:
                        pass
                if t is None:
                    raise ValueError(f"Unrecognized time in {f}")

            lon = ds["longitude"].values
            lat = ds["latitude"].values
            y = ds["ssha_filtered"].values

            xtrk = ds["cross_track_distance"].values
            xtrk2 = np.broadcast_to(xtrk[None, :], y.shape)

            # Simple cross-track observation-error model, in meters.
            x1 = np.nanmax(np.abs(xtrk2))
            sigma0, sigma1 = 0.03, 0.15
            r = np.abs(xtrk2) / x1
            sig = sigma0 + (sigma1 - sigma0) * (r**2)

            m = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(y)

            lon_all.append(lon[m].ravel())
            lat_all.append(lat[m].ravel())
            y_all.append(y[m].ravel())
            sig_all.append(sig[m].ravel())
            t_all.append(np.full(m.sum(), t, dtype="datetime64[ns]"))

    return (
        np.concatenate(lon_all),
        np.concatenate(lat_all),
        np.concatenate(t_all),
        np.concatenate(y_all),
        np.concatenate(sig_all),
    )


def build_grid(region):
    """Build regular grid and ocean mask for a FAROX25 region."""
    lon_edges = np.arange(region.lon_min, region.lon_max + region.dlon, region.dlon)
    lat_edges = np.arange(region.lat_min, region.lat_max + region.dlat, region.dlat)

    lonc = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    latc = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    lon2, lat2 = np.meshgrid(lonc, latc)
    ocean = ~globe.is_land(lat2, lon2)

    return lonc, latc, lon2, lat2, ocean


def subset_observations(lon_obs, lat_obs, t_obs, y_obs, sig_obs, lon2, lat2, pad=0.5):
    """Subset observations to the analysis domain and remove land points."""
    lon_obs = wrap180(lon_obs)

    dom = (
        (lon_obs >= lon2.min() - pad)
        & (lon_obs <= lon2.max() + pad)
        & (lat_obs >= lat2.min() - pad)
        & (lat_obs <= lat2.max() + pad)
    )

    lon_obs = lon_obs[dom]
    lat_obs = lat_obs[dom]
    t_obs = t_obs[dom]
    y_obs = y_obs[dom]
    sig_obs = sig_obs[dom]

    ocean_obs = ~globe.is_land(lat_obs, lon_obs)

    lon_obs = lon_obs[ocean_obs]
    lat_obs = lat_obs[ocean_obs]
    t_obs = t_obs[ocean_obs]
    y_obs = y_obs[ocean_obs]
    sig_obs = sig_obs[ocean_obs]

    good = (
        np.isfinite(lon_obs)
        & np.isfinite(lat_obs)
        & np.isfinite(y_obs)
        & np.isfinite(sig_obs)
    )

    return lon_obs[good], lat_obs[good], t_obs[good], y_obs[good], sig_obs[good]


def run(args):
    """Run local SWOT OI."""
    region = REGIONS[args.region]

    tic = time.perf_counter()
    lon_obs, lat_obs, t_obs_dt, y_obs, sig_obs = read_swot_l3(args.swot_dir, args.pattern)
    print(f"Read SWOT observations: n={len(y_obs):,}")

    lonc, latc, lon2, lat2, ocean = build_grid(region)

    lon_obs, lat_obs, t_obs_dt, y_obs, sig_obs = subset_observations(
        lon_obs, lat_obs, t_obs_dt, y_obs, sig_obs, lon2, lat2, pad=args.pad
    )
    print(f"Observations after domain/ocean filtering: n={len(y_obs):,}")

    t_grid_dt = make_time_grid(region.start, region.stop, delta_hours=args.delta_hours)
    t_ref = np.datetime64(region.start, "ns")

    t_grid_hours = to_hours(t_grid_dt, t_ref)
    t_obs_hours = to_hours(t_obs_dt, t_ref)

    xhat, phat = oi_space_time(
        lon_obs,
        lat_obs,
        t_obs_hours,
        y_obs,
        lon2,
        lat2,
        t_grid_hours,
        Lx_km=args.Lx_km,
        Ly_km=args.Ly_km,
        Lt_h=args.Lt_h,
        sigma_R=args.sigma_R,
        sigma_B=args.sigma_B,
        sigma_R_obs=sig_obs,
        max_obs=args.max_obs,
        compute_mask=ocean,
    )

    with xr.open_dataset(args.background_file) as ds_bg:
        if "mdt" in ds_bg:
            mdt = ds_bg["mdt"].mean("time").values.astype(np.float32)
        else:
            mdt = np.full(lon2.shape, np.nan, dtype=np.float32)

    ds_oi = xr.Dataset(
        data_vars={
            "sla_oi": (("time", "lat", "lon"), xhat.astype(np.float32)),
            "sla_oi_var": (("time", "lat", "lon"), phat.astype(np.float32)),
            "mdt_oi": (("lat", "lon"), mdt),
        },
        coords={
            "time": ("time", t_grid_dt),
            "lon": ("lon", lonc),
            "lat": ("lat", latc),
        },
        attrs={
            "title": "Local SWOT optimal interpolation for FAROX25",
            "region": args.region,
            "Lx_km": args.Lx_km,
            "Ly_km": args.Ly_km,
            "Lt_h": args.Lt_h,
            "sigma_R": args.sigma_R,
            "sigma_B": args.sigma_B,
            "max_obs": args.max_obs,
        },
    )

    ds_oi["sla_oi"].attrs.update({"long_name": "SLA from local SWOT optimal interpolation", "units": "m"})
    ds_oi["sla_oi_var"].attrs.update({"long_name": "Local OI posterior variance", "units": "m2"})
    ds_oi["mdt_oi"].attrs.update({"long_name": "Mean dynamic topography from background file", "units": "m"})

    ds_oi.to_netcdf(args.output)
    toc = time.perf_counter()

    print(f"Wrote {args.output}")
    print(f"Total runtime: {(toc - tic) / 60:.2f} min")


def parse_args():
    parser = argparse.ArgumentParser(description="Local space-time OI of SWOT Level-3 SLA for FAROX25.")

    parser.add_argument("--region", choices=sorted(REGIONS), required=True)
    parser.add_argument("--swot-dir", required=True, help="Directory containing SWOT Level-3 NetCDF files.")
    parser.add_argument("--pattern", default="SWOT*.nc", help="SWOT filename pattern.")
    parser.add_argument("--background-file", required=True, help="NetCDF file containing MDT variable.")
    parser.add_argument("--output", required=True, help="Output NetCDF file.")

    parser.add_argument("--Lx-km", type=float, default=30.0)
    parser.add_argument("--Ly-km", type=float, default=30.0)
    parser.add_argument("--Lt-h", type=float, default=72.0)
    parser.add_argument("--sigma-R", type=float, default=0.07)
    parser.add_argument("--sigma-B", type=float, default=0.18)
    parser.add_argument("--max-obs", type=int, default=1000)
    parser.add_argument("--delta-hours", type=int, default=24)
    parser.add_argument("--pad", type=float, default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
