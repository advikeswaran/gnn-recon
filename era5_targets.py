"""
era5_targets.py
---------------
Extracts annual-mean ERA5 2m temperature and total precipitation at the
11,160 Antarctic grid nodes (south of 60°S) for each training year 1979–2017.

Strategy:
  - Weekly subsampling (~52 snapshots/year), same dates as extract_embeddings.py
  - Average 2T and TP across all valid snapshots -> annual mean
  - Subsample to Antarctic 1° grid: lat rows 150:181 of (181,360) ERA5 grid
    (lat = -60 to -90 inclusive = 31 rows x 360 cols = 11,160 nodes)
  - Save as (11160, 2) float32: col 0 = temperature (K), col 1 = precip (m/year)
  - After all years done, compute and save normalization stats

Output files:
  cache/era5_targets/targets_YYYY.npy   shape (11160, 2), float32
  cache/era5_targets/norm_stats.npz     mean/std for temp and precip

Node ordering matches extract_embeddings.py:
  node i = lat_idx * 360 + lon_idx
  lat_idx 0 = -60 deg, lat_idx 30 = -90 deg
  lon_idx 0 = 0 deg,   lon_idx 359 = 359 deg

Usage:
  python3 era5_targets.py --year 1979    # single year
  python3 era5_targets.py --all          # all 38 years
  python3 era5_targets.py --norm         # compute norm stats from saved targets
  python3 era5_targets.py --test 1979    # verify one year output
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, '/glade/derecho/scratch/advike/graphcast_recon')
from era5_loader import ERA5Loader

# -- Paths ---------------------------------------------------------------------
RECON_DIR   = '/glade/derecho/scratch/advike/graphcast_recon'
TARGET_DIR  = f'{RECON_DIR}/cache/era5_targets'

# -- Constants -----------------------------------------------------------------
TRAIN_YEARS  = list(range(1979, 2018))   # 38 years
N_NODES      = 11160                     # 31 lat x 360 lon
WEEK_STRIDE  = 7                         # days between snapshots

# Antarctic slice of the (181, 360) ERA5/GraphCast 1 deg grid
# GC_LAT = linspace(90, -90, 181): GC_LAT[i] = 90 - i
# GC_LAT[150] = -60 deg, GC_LAT[180] = -90 deg
ANTARCTIC_LAT_START = 150
ANTARCTIC_LAT_END   = 181
N_LAT = ANTARCTIC_LAT_END - ANTARCTIC_LAT_START   # 31
N_LON = 360


def get_snapshots(year):
    start     = datetime(year, 1, 1, 0)
    end       = datetime(year, 12, 31, 0)
    snapshots = []
    dt = start
    while dt <= end:
        snapshots.append(dt)
        dt += timedelta(days=WEEK_STRIDE)
    return snapshots


def extract_year(year, verbose=True):
    out_path = f'{TARGET_DIR}/targets_{year:04d}.npy'
    if os.path.exists(out_path):
        if verbose:
            print(f"Year {year}: already cached, skipping.")
        return np.load(out_path)

    snapshots = get_snapshots(year)
    if verbose:
        print(f"Year {year}: {len(snapshots)} snapshots")

    era5 = ERA5Loader(verbose=False)

    temp_sum = np.zeros((N_LAT, N_LON), dtype=np.float64)
    prec_sum = np.zeros((N_LAT, N_LON), dtype=np.float64)
    n_valid  = 0

    for i, dt in enumerate(snapshots):
        try:
            if verbose:
                print(f"  [{i+1}/{len(snapshots)}] {dt.strftime('%Y-%m-%d')}", flush=True)

            temp = era5.load_2t(dt)
            prec = era5.load_tp(dt)

            temp_ant = temp[ANTARCTIC_LAT_START:ANTARCTIC_LAT_END, :]
            prec_ant = prec[ANTARCTIC_LAT_START:ANTARCTIC_LAT_END, :]

            if np.isnan(temp_ant).any() or np.isnan(prec_ant).any():
                if verbose:
                    print(f"  WARNING: NaNs in snapshot {dt} -- skipping")
                continue

            temp_sum += temp_ant.astype(np.float64)
            prec_sum += prec_ant.astype(np.float64)
            n_valid  += 1

        except Exception as e:
            if verbose:
                print(f"  WARNING: snapshot {dt} failed: {e} -- skipping")
            continue

    if n_valid == 0:
        raise RuntimeError(f"All snapshots failed for year {year}")

    temp_mean   = (temp_sum / n_valid).astype(np.float32)
    prec_mean   = (prec_sum / n_valid).astype(np.float32)
    prec_annual = prec_mean * 365.0

    temp_flat = temp_mean.reshape(-1)
    prec_flat = prec_annual.reshape(-1)

    targets = np.stack([temp_flat, prec_flat], axis=1).astype(np.float32)

    if verbose:
        print(f"Year {year}: averaged {n_valid} snapshots")
        print(f"  Temp range: [{targets[:,0].min():.2f}, {targets[:,0].max():.2f}] K")
        print(f"  Prec range: [{targets[:,1].min():.4f}, {targets[:,1].max():.4f}] m/yr")

    os.makedirs(TARGET_DIR, exist_ok=True)
    tmp_path = out_path.replace('.npy', '.tmp')
    np.save(tmp_path, targets)
    os.rename(tmp_path, out_path)

    if verbose:
        print(f"Year {year}: saved to {out_path}")

    return targets


def compute_norm_stats(verbose=True):
    paths = sorted([
        f for f in os.listdir(TARGET_DIR)
        if f.startswith('targets_') and f.endswith('.npy')
    ])
    if not paths:
        raise RuntimeError(f"No target files found in {TARGET_DIR}")

    if verbose:
        print(f"Computing norm stats from {len(paths)} years...")

    all_temp = []
    all_prec = []

    for fname in paths:
        targets = np.load(f'{TARGET_DIR}/{fname}')
        all_temp.append(targets[:, 0])
        all_prec.append(targets[:, 1])

    all_temp = np.concatenate(all_temp)
    all_prec = np.concatenate(all_prec)

    stats = {
        'temp_mean': float(all_temp.mean()),
        'temp_std':  float(all_temp.std()),
        'prec_mean': float(all_prec.mean()),
        'prec_std':  float(all_prec.std()),
    }

    np.savez(f'{TARGET_DIR}/norm_stats.npz', **stats)

    if verbose:
        print(f"  Temp: mean={stats['temp_mean']:.2f} K, std={stats['temp_std']:.2f} K")
        print(f"  Prec: mean={stats['prec_mean']:.4f} m/yr, std={stats['prec_std']:.4f} m/yr")
        print(f"  Saved to {TARGET_DIR}/norm_stats.npz")

    return stats


def verify_year(year):
    path = f'{TARGET_DIR}/targets_{year:04d}.npy'
    if not os.path.exists(path):
        print(f"FAIL: {path} does not exist")
        return False

    targets = np.load(path)

    checks = [
        (targets.shape == (N_NODES, 2),
            f"Shape {targets.shape} != ({N_NODES}, 2)"),
        (targets.dtype == np.float32,
            f"dtype {targets.dtype} != float32"),
        (not np.isnan(targets).any(),
            f"{np.isnan(targets).sum()} NaNs found"),
        (targets[:, 0].min() > 180.0,
            f"Temp min {targets[:,0].min():.2f} K suspiciously low"),
        (targets[:, 0].max() < 320.0,
            f"Temp max {targets[:,0].max():.2f} K suspiciously high"),
        (targets[:, 1].min() >= 0.0,
            f"Precip min {targets[:,1].min():.6f} m/yr is negative"),
        (targets[:, 1].max() < 20.0,
            f"Precip max {targets[:,1].max():.4f} m/yr suspiciously high"),
    ]

    all_pass = True
    for passed, msg in checks:
        if not passed:
            print(f"  FAIL: {msg}")
            all_pass = False

    if all_pass:
        print(f"  OK   targets_{year}.npy -- "
              f"temp=[{targets[:,0].min():.1f}, {targets[:,0].max():.1f}] K  "
              f"prec=[{targets[:,1].min():.4f}, {targets[:,1].max():.4f}] m/yr")
    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--test', type=int, default=None, metavar='YEAR')
    args = parser.parse_args()

    if args.year is not None:
        extract_year(args.year)
    elif args.all:
        print(f"Extracting ERA5 targets for {len(TRAIN_YEARS)} years...")
        failed = []
        for year in TRAIN_YEARS:
            try:
                extract_year(year)
            except Exception as e:
                print(f"ERROR year {year}: {e}")
                failed.append(year)
        if failed:
            print(f"\nFailed years: {failed}")
        else:
            print("\nAll years complete.")
    elif args.norm:
        compute_norm_stats()
    elif args.test is not None:
        print(f"Verifying targets_{args.test}.npy...")
        verify_year(args.test)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
