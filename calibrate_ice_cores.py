"""
calibrate_ice_cores.py -- Per-Site Linear Regression Calibration
Antarctic Climate Reconstruction from Ice Core Data

Calibrates raw ice core measurements to ERA5 units via per-site linear regression
over the overlap period (1979-2000):
  - dD (permil) -> 2m temperature (K)   [direct regression, no d18O intermediate]
  - accumulation (m ice eq/yr) -> total precipitation (m/yr)

CSV format (both Water and Accum files):
  Coords: no header, columns = short_id, long_name, lat, lon
  Data:   row 1 = numeric site IDs (ignored)
          row 2 = short site names (match coords short_id)
          row 3+ = year, val1, val2, ... (years descending from 2020)

Outputs saved to cache/calibration/:
  coefficients.npz       -- per-site regression coefficients (slope, intercept, r2, n)
  calibrated_iso.npy     -- temperature time series (n_iso_sites, n_years) in K
  calibrated_accum.npy   -- precip time series (n_accum_sites, n_years) in m/yr
  calibration_meta.npz   -- site coords, IDs, year index for both arrays

Usage:
  python calibrate_ice_cores.py [--dry-run]
"""

import os
import sys
import csv
import time
import argparse
import logging
import resource
from pathlib import Path

import numpy as np

# ------------------------------------
# Paths
# ------------------------------------
RECON_DIR  = Path("/glade/derecho/scratch/advike/graphcast_recon")
DATA_DIR   = RECON_DIR / "data"
CACHE_DIR  = RECON_DIR / "cache"
CALIB_DIR  = CACHE_DIR / "calibration"
TGT_DIR    = CACHE_DIR / "era5_targets"
LOG_DIR    = RECON_DIR / "logs"

CALIB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ACCUM_COORDS = DATA_DIR / "AccumCoresCoords.csv"
ACCUM_DATA   = DATA_DIR / "AccumCoresData.csv"
WATER_COORDS = DATA_DIR / "WaterCoresCoords.csv"
WATER_DATA   = DATA_DIR / "WaterCoresData.csv"

OVERLAP_START = 1979
OVERLAP_END   = 2000
RECON_START   = 1801
RECON_END     = 2000

LAT_VALS = np.arange(-60, -91, -1, dtype=np.float32)
LON_VALS = np.arange(0,   360,  1, dtype=np.float32)


# ------------------------------------
# Logging
# ------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("calibrate")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def log_memory(logger: logging.Logger, tag: str = ""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    mb = usage.ru_maxrss / 1024
    logger.info(f"[MEM{' '+tag if tag else ''}] peak RSS = {mb:.0f} MB")


# ------------------------------------
# Grid helpers
# ------------------------------------
def snap_to_grid(lat: float, lon: float) -> tuple[float, float]:
    slat = float(LAT_VALS[np.argmin(np.abs(LAT_VALS - lat))])
    slon = float(LON_VALS[np.argmin(np.abs(LON_VALS - lon))])
    return slat, slon


def grid_node_index(lat: float, lon: float) -> int:
    lat_idx = int(np.argmin(np.abs(LAT_VALS - lat)))
    lon_idx = int(np.argmin(np.abs(LON_VALS - lon)))
    return lat_idx * 360 + lon_idx


# ------------------------------------
# CSV loaders
# ------------------------------------
def load_csv_coords(path: Path) -> dict:
    """
    No header row. Columns: short_id, long_name, lat, lon
    Returns dict: short_id -> {'lat': float, 'lon': float}
    """
    sites = {}
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            sid = row[0].strip()
            lat = float(row[2])
            lon = float(row[3]) % 360
            sites[sid] = {'lat': lat, 'lon': lon}
    return sites


def load_csv_data(path: Path, name_map: dict = None) -> tuple[np.ndarray, dict]:
    """
    Row 1: numeric site IDs (ignored)
    Row 2: short site names (keys matching coords short_id)
    Row 3+: year col 0, values col 1+ (empty string = NaN, years descending)

    name_map: optional dict to rename specific site IDs in row 2 (e.g. long->short)

    Returns:
      years  np.ndarray (N_years,) int32   sorted ascending
      data   dict: short_id -> np.ndarray (N_years,) float32
    """
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)                       # row 1: numeric IDs, skip
        row2     = next(reader)            # row 2: short site names
        site_ids = [s.strip() for s in row2[1:]]
        if name_map:
            site_ids = [name_map.get(s, s) for s in site_ids]

        years_list, rows = [], []
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                yr = int(row[0].strip())
            except ValueError:
                continue
            years_list.append(yr)
            vals = []
            for v in row[1:]:
                v = v.strip()
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(np.nan)
            while len(vals) < len(site_ids):
                vals.append(np.nan)
            rows.append(vals[:len(site_ids)])

    years = np.array(years_list, dtype=np.int32)
    arr   = np.array(rows, dtype=np.float32)

    order = np.argsort(years)
    years = years[order]
    arr   = arr[order]

    data = {sid: arr[:, i] for i, sid in enumerate(site_ids)}
    return years, data


# ------------------------------------
# ERA5 loader
# ------------------------------------
def load_era5_at_node(node_idx: int, years: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load ERA5 T and precip at a single grid node for given years.
    targets_YYYY.npy shape: (11160, 2) -- col 0=temp(K), col 1=precip(m/yr)
    Returns NaN for missing years.
    """
    temps, precips = [], []
    for yr in years:
        path = TGT_DIR / f"targets_{yr}.npy"
        if path.exists():
            tgt = np.load(path)
            temps.append(float(tgt[node_idx, 0]))
            precips.append(float(tgt[node_idx, 1]))
        else:
            temps.append(np.nan)
            precips.append(np.nan)
    return np.array(temps, dtype=np.float32), np.array(precips, dtype=np.float32)


# ------------------------------------
# Linear regression (no scipy)
# ------------------------------------
def linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    """
    OLS: y = slope * x + intercept. Ignores NaN pairs.
    Returns: slope, intercept, r2, n_valid
    """
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 2:
        return np.nan, np.nan, np.nan, n

    xv = x[mask].astype(np.float64)
    yv = y[mask].astype(np.float64)
    xm, ym = xv.mean(), yv.mean()
    ss_xx = ((xv - xm) ** 2).sum()
    ss_xy = ((xv - xm) * (yv - ym)).sum()

    if ss_xx == 0:
        return 0.0, float(ym), np.nan, n

    slope     = ss_xy / ss_xx
    intercept = ym - slope * xm
    y_pred    = slope * xv + intercept
    ss_res    = ((yv - y_pred) ** 2).sum()
    ss_tot    = ((yv - ym) ** 2).sum()
    r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(slope), float(intercept), float(r2), n


def apply_regression(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float32)
    valid = np.isfinite(x)
    out[valid] = slope * x[valid] + intercept
    return out


# ------------------------------------
# Main calibration
# ------------------------------------
def calibrate(args, logger: logging.Logger):

    logger.info("Loading ice core CSVs...")
    iso_coords   = load_csv_coords(WATER_COORDS)
    accum_coords = load_csv_coords(ACCUM_COORDS)
    iso_years,   iso_data   = load_csv_data(WATER_DATA, name_map={'Mount Brown South': 'MBS', 'Mount Brown South dD': 'MBS dD'})
    accum_years, accum_data = load_csv_data(ACCUM_DATA)

    logger.info(f"  Isotope sites: {len(iso_coords)}, "
                f"years: {iso_years[0]}-{iso_years[-1]}, "
                f"data columns: {len(iso_data)}")
    logger.info(f"  Accum sites:   {len(accum_coords)}, "
                f"years: {accum_years[0]}-{accum_years[-1]}, "
                f"data columns: {len(accum_data)}")

    iso_matched   = set(iso_coords.keys()) & set(iso_data.keys())
    accum_matched = set(accum_coords.keys()) & set(accum_data.keys())
    logger.info(f"  Isotope coords/data matches: {len(iso_matched)}/{len(iso_coords)}")
    logger.info(f"  Accum coords/data matches:   {len(accum_matched)}/{len(accum_coords)}")

    overlap_years = list(range(OVERLAP_START, OVERLAP_END + 1))
    recon_years   = list(range(RECON_START, RECON_END + 1))
    n_recon       = len(recon_years)
    logger.info(f"  Overlap period: {OVERLAP_START}-{OVERLAP_END} ({len(overlap_years)} yrs)")
    logger.info(f"  Reconstruction period: {RECON_START}-{RECON_END} ({n_recon} yrs)")

    # --------------------------------------------------------
    # Isotope sites: dD -> temperature (K)
    # --------------------------------------------------------
    logger.info("Calibrating isotope sites (dD -> temperature K)...")

    iso_site_ids   = sorted(iso_coords.keys())
    n_iso          = len(iso_site_ids)
    iso_calib_ts   = np.full((n_iso, n_recon), np.nan, dtype=np.float32)
    iso_slopes     = np.full(n_iso, np.nan, dtype=np.float64)
    iso_intercepts = np.full(n_iso, np.nan, dtype=np.float64)
    iso_r2         = np.full(n_iso, np.nan, dtype=np.float64)
    iso_n          = np.zeros(n_iso, dtype=np.int32)
    iso_lats       = np.zeros(n_iso, dtype=np.float32)
    iso_lons       = np.zeros(n_iso, dtype=np.float32)

    for i, sid in enumerate(iso_site_ids):
        lat, lon   = iso_coords[sid]['lat'], iso_coords[sid]['lon']
        slat, slon = snap_to_grid(lat, lon)
        node_idx   = grid_node_index(slat, slon)
        iso_lats[i] = slat
        iso_lons[i] = slon

        raw_series = iso_data.get(sid, None)
        if raw_series is None:
            logger.warning(f"  ISO {sid}: no data column, skipping")
            continue

        iso_overlap = np.full(len(overlap_years), np.nan, dtype=np.float32)
        for j, yr in enumerate(overlap_years):
            idx = np.where(iso_years == yr)[0]
            if len(idx) > 0:
                iso_overlap[j] = raw_series[idx[0]]

        n_obs = int(np.sum(np.isfinite(iso_overlap)))
        era5_temp, _ = load_era5_at_node(node_idx, overlap_years)

        slope, intercept, r2, n_valid = linear_regression(iso_overlap, era5_temp)
        iso_slopes[i]     = slope
        iso_intercepts[i] = intercept
        iso_r2[i]         = r2
        iso_n[i]          = n_valid

        if np.isnan(slope):
            logger.warning(f"  ISO {sid}: regression failed (n_ice={n_obs}, n_valid={n_valid})")
            continue

        logger.info(f"  ISO {sid}: slope={slope:.4f}, intercept={intercept:.2f}, "
                    f"r2={r2:.3f}, n={n_valid}")

        full_series = np.full(n_recon, np.nan, dtype=np.float32)
        for j, yr in enumerate(recon_years):
            idx = np.where(iso_years == yr)[0]
            if len(idx) > 0:
                full_series[j] = raw_series[idx[0]]
        iso_calib_ts[i] = apply_regression(full_series, slope, intercept)

        if args.dry_run and i >= 4:
            logger.info("  [DRY RUN] stopping after 5 isotope sites")
            break

    # --------------------------------------------------------
    # Accumulation sites: accum -> precipitation (m/yr)
    # --------------------------------------------------------
    logger.info("Calibrating accumulation sites (accum -> precip m/yr)...")

    accum_site_ids   = sorted(accum_coords.keys())
    n_accum          = len(accum_site_ids)
    accum_calib_ts   = np.full((n_accum, n_recon), np.nan, dtype=np.float32)
    accum_slopes     = np.full(n_accum, np.nan, dtype=np.float64)
    accum_intercepts = np.full(n_accum, np.nan, dtype=np.float64)
    accum_r2         = np.full(n_accum, np.nan, dtype=np.float64)
    accum_n          = np.zeros(n_accum, dtype=np.int32)
    accum_lats       = np.zeros(n_accum, dtype=np.float32)
    accum_lons       = np.zeros(n_accum, dtype=np.float32)

    for i, sid in enumerate(accum_site_ids):
        lat, lon   = accum_coords[sid]['lat'], accum_coords[sid]['lon']
        slat, slon = snap_to_grid(lat, lon)
        node_idx   = grid_node_index(slat, slon)
        accum_lats[i] = slat
        accum_lons[i] = slon

        raw_series = accum_data.get(sid, None)
        if raw_series is None:
            logger.warning(f"  ACCUM {sid}: no data column, skipping")
            continue

        accum_overlap = np.full(len(overlap_years), np.nan, dtype=np.float32)
        for j, yr in enumerate(overlap_years):
            idx = np.where(accum_years == yr)[0]
            if len(idx) > 0:
                accum_overlap[j] = raw_series[idx[0]]

        n_obs = int(np.sum(np.isfinite(accum_overlap)))
        _, era5_precip = load_era5_at_node(node_idx, overlap_years)

        slope, intercept, r2, n_valid = linear_regression(accum_overlap, era5_precip)
        accum_slopes[i]     = slope
        accum_intercepts[i] = intercept
        accum_r2[i]         = r2
        accum_n[i]          = n_valid

        if np.isnan(slope):
            logger.warning(f"  ACCUM {sid}: regression failed (n_ice={n_obs}, n_valid={n_valid})")
            continue

        logger.info(f"  ACCUM {sid}: slope={slope:.4f}, intercept={intercept:.4f}, "
                    f"r2={r2:.3f}, n={n_valid}")

        full_series = np.full(n_recon, np.nan, dtype=np.float32)
        for j, yr in enumerate(recon_years):
            idx = np.where(accum_years == yr)[0]
            if len(idx) > 0:
                full_series[j] = raw_series[idx[0]]
        accum_calib_ts[i] = apply_regression(full_series, slope, intercept)

        if args.dry_run and i >= 4:
            logger.info("  [DRY RUN] stopping after 5 accum sites")
            break

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    logger.info("=" * 50)
    logger.info("Calibration summary:")
    iso_valid   = int(np.sum(np.isfinite(iso_slopes)))
    accum_valid = int(np.sum(np.isfinite(accum_slopes)))
    logger.info(f"  Isotope sites with valid regression: {iso_valid}/{n_iso}")
    logger.info(f"  Accum sites with valid regression:   {accum_valid}/{n_accum}")
    if iso_valid > 0:
        r2v = iso_r2[np.isfinite(iso_r2)]
        logger.info(f"  Isotope R2: mean={r2v.mean():.3f}, min={r2v.min():.3f}, max={r2v.max():.3f}")
    if accum_valid > 0:
        r2v = accum_r2[np.isfinite(accum_r2)]
        logger.info(f"  Accum R2:   mean={r2v.mean():.3f}, min={r2v.min():.3f}, max={r2v.max():.3f}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    if args.dry_run:
        logger.info("[DRY RUN] skipping file saves")
    else:
        logger.info("Saving outputs...")

        coeff_path = CALIB_DIR / "coefficients.npz"
        tmp = str(coeff_path).replace('.npz', '.tmp.npz')
        np.savez_compressed(tmp,
            iso_slopes=iso_slopes, iso_intercepts=iso_intercepts,
            iso_r2=iso_r2, iso_n=iso_n,
            accum_slopes=accum_slopes, accum_intercepts=accum_intercepts,
            accum_r2=accum_r2, accum_n=accum_n,
        )
        os.rename(tmp, coeff_path)
        logger.info(f"  Saved: {coeff_path}")

        iso_ts_path = CALIB_DIR / "calibrated_iso.npy"
        tmp = str(iso_ts_path).replace('.npy', '.tmp.npy')
        np.save(tmp, iso_calib_ts)
        os.rename(tmp, iso_ts_path)
        logger.info(f"  Saved: {iso_ts_path}  shape={iso_calib_ts.shape}")

        accum_ts_path = CALIB_DIR / "calibrated_accum.npy"
        tmp = str(accum_ts_path).replace('.npy', '.tmp.npy')
        np.save(tmp, accum_calib_ts)
        os.rename(tmp, accum_ts_path)
        logger.info(f"  Saved: {accum_ts_path}  shape={accum_calib_ts.shape}")

        meta_path = CALIB_DIR / "calibration_meta.npz"
        tmp = str(meta_path).replace('.npz', '.tmp.npz')
        np.savez_compressed(tmp,
            iso_site_ids=np.array(iso_site_ids),
            iso_lats=iso_lats, iso_lons=iso_lons,
            accum_site_ids=np.array(accum_site_ids),
            accum_lats=accum_lats, accum_lons=accum_lons,
            recon_years=np.array(recon_years, dtype=np.int32),
        )
        os.rename(tmp, meta_path)
        logger.info(f"  Saved: {meta_path}")

    log_memory(logger, "post-calibration")
    logger.info("Calibration complete.")


# ------------------------------------
# Entry point
# ------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Calibrate ice core data to ERA5 units")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process first 5 sites only, skip saving")
    args = parser.parse_args()

    log_path = LOG_DIR / f"calibrate_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("Ice Core Calibration -- dD/accum -> T/precip (ERA5 units)")
    logger.info(f"  Args: {vars(args)}")
    logger.info("=" * 60)

    for p, desc in [
        (ACCUM_COORDS, "accumulation coords CSV"),
        (ACCUM_DATA,   "accumulation data CSV"),
        (WATER_COORDS, "isotope coords CSV"),
        (WATER_DATA,   "isotope data CSV"),
    ]:
        if not p.exists():
            logger.error(f"Required file missing: {p} ({desc})")
            sys.exit(1)

    if not (TGT_DIR / "targets_1979.npy").exists():
        logger.error(f"ERA5 targets not found at {TGT_DIR} -- run era5_targets.py first")
        sys.exit(1)

    calibrate(args, logger)


if __name__ == "__main__":
    main()
