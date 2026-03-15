"""
extract_embeddings.py

Runs frozen GraphCast on ERA5 snapshots and extracts Antarctic node embeddings.

Strategy:
  - 38 ERA5 years (1979-2017), rolling 12-month windows -> 456 training samples
  - Weekly subsampling (~52 snapshots/window)
  - Each snapshot: run GraphCast with normalization -> extract latent_grid_nodes
  - Average embeddings within window -> one (11160, 512) array per year-window
  - Save to cache/embeddings/window_YYYY.npy

Antarctic nodes: first 11160 of 65160 grid nodes (lats -90 to -60, row-major)

Usage:
    python extract_embeddings.py --test        # single snapshot test
    python extract_embeddings.py --year 1979   # one year
    python extract_embeddings.py               # all 38 years
"""

import os
import sys
import glob
import argparse
import dataclasses
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import haiku as hk
import jax
import jax.numpy as jnp

sys.path.insert(0, '/glade/derecho/scratch/advike/graphcast_test/graphcast')

from graphcast import graphcast, checkpoint, data_utils, normalization, casting

# ── Paths ──────────────────────────────────────────────────────────────────────
WEIGHTS_DIR  = '/glade/derecho/scratch/advike/graphcast_weights'
RECON_DIR    = '/glade/derecho/scratch/advike/graphcast_recon'
EMBED_DIR    = f'{RECON_DIR}/cache/embeddings'
ERA5_TARGETS = f'{RECON_DIR}/cache/era5_targets'

sys.path.insert(0, RECON_DIR)
from era5_loader import ERA5Loader

# ── Constants ──────────────────────────────────────────────────────────────────
N_ANTARCTIC  = 11160   # grid nodes south of -60 (31 lat rows x 360 lon)
LATENT_SIZE  = 512
TRAIN_YEARS  = list(range(1979, 2018))   # 38 years
VAL_YEARS    = list(range(2011, 2018))   # held-out
GC_LEVELS    = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


# ══════════════════════════════════════════════════════════════════════════════
class EmbeddingExtractor:
    """Runs frozen GraphCast and extracts Antarctic latent_grid_nodes."""

    def __init__(self, verbose=True):
        self.verbose  = verbose
        self._loaded  = False

    def _load_model(self):
        """Load weights, normalization stats, and build the haiku transform."""
        if self._loaded:
            return
        self._log("Loading GraphCast weights...")
        with open(f"{WEIGHTS_DIR}/GraphCast_small.npz", "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
        self.params       = ckpt.params
        self.model_config = ckpt.model_config
        self.task_config  = ckpt.task_config

        self._log("Loading normalization stats...")
        with open(f"{WEIGHTS_DIR}/mean_by_level.nc", "rb") as f:
            self.mean_by_level = xr.load_dataset(f).compute()
        with open(f"{WEIGHTS_DIR}/stddev_by_level.nc", "rb") as f:
            self.stddev_by_level = xr.load_dataset(f).compute()
        with open(f"{WEIGHTS_DIR}/diffs_stddev_by_level.nc", "rb") as f:
            self.diffs_stddev_by_level = xr.load_dataset(f).compute()

        self._log("Loading sample data (for static fields)...")
        with open(f"{WEIGHTS_DIR}/sample_data.nc", "rb") as f:
            self._sample_batch = xr.load_dataset(f).compute()

        self._log("Building haiku transform...")
        self._run_extract = hk.transform_with_state(self._extract_fn)
        self._state       = {}
        self._loaded      = True
        self._log("Model ready.")

    def _extract_fn(self, model_config, task_config, inputs, targets_template, forcings):
        """
        Haiku-transformed function: normalize inputs then extract latent_grid_nodes.
        """
        normed_inputs, normed_forcings = self._normalize_inputs(inputs, forcings)

        predictor = graphcast.GraphCast(model_config, task_config)
        predictor._maybe_init(normed_inputs)
        grid_node_features = predictor._inputs_to_grid_node_features(
            normed_inputs, normed_forcings)
        latent_mesh_nodes, latent_grid_nodes = predictor._run_grid2mesh_gnn(
            grid_node_features)

        # Antarctic nodes: lats -90..-60 = first 31 rows = indices 0:11160
        return latent_grid_nodes[:N_ANTARCTIC, :, :]  # (11160, batch, 512)

    def _normalize_inputs(self, inputs, forcings):
        """Apply GraphCast's standard input normalization."""
        normed_inputs = inputs.copy()
        for var in inputs.data_vars:
            if var in self.mean_by_level:
                normed_inputs[var] = (
                    (inputs[var] - self.mean_by_level[var])
                    / self.stddev_by_level[var]
                )
        return normed_inputs, forcings

    def extract_snapshot(self, inputs_xr, targets_xr, forcings_xr) -> np.ndarray:
        """
        Extract Antarctic embeddings for one snapshot.
        Returns (11160, 512) float32.
        """
        self._load_model()
        embeddings, self._state = self._run_extract.apply(
            self.params, self._state,
            jax.random.PRNGKey(0),
            self.model_config, self.task_config,
            inputs_xr, targets_xr, forcings_xr
        )
        return np.array(embeddings[:, 0, :])  # (11160, 512)

    def get_sample_inputs_targets_forcings(self):
        """Return inputs/targets/forcings from sample data."""
        self._load_model()
        return data_utils.extract_inputs_targets_forcings(
            self._sample_batch,
            target_lead_times=slice("6h", "6h"),
            **dataclasses.asdict(self.task_config)
        )

    # ── ERA5 → GraphCast xarray conversion ────────────────────────────────────

    def _era5_to_graphcast_xr(self, era5_data, era5_loader, sample_inputs, dt):
        """
        Convert ERA5 numpy arrays into xarray Datasets matching GraphCast's
        expected input format.

        Uses 4 time slots matching the fake-batch diagnostic layout:
            slot 0: 0h   -> t-12h dummy (duplicate, not used as input)
            slot 1: 6h   -> t-12h (first real input)
            slot 2: 12h  -> t0    (second real input)
            slot 3: 18h  -> t+6h  (target, filled with t0 as dummy)

        After extract_inputs_targets_forcings reindexes relative to slot 2,
        the input slice (-12h, 0] grabs slots 1 and 2 — exactly 2 timesteps.
        Slot 3 becomes the +6h target.
        """
        dt_m12 = dt - timedelta(hours=12)
        dt_0   = dt
        dt_p6  = dt + timedelta(hours=6)

        # 4-slot time axis: timedelta64 offsets (ns)
        times = np.array(
            [0, 6*3600, 12*3600, 18*3600], dtype='timedelta64[s]'
        ).astype('timedelta64[ns]')

        # Wall-clock datetimes — slot 0 duplicates slot 1 (both t-12h)
        datetimes = pd.to_datetime([dt_m12, dt_m12, dt_0, dt_p6])

        lats   = era5_data['lat'][::-1]   # flip 90→-90 to -90→+90
        lons   = era5_data['lon']
        levels = np.array(GC_LEVELS, dtype=np.int32)

        # ── Load t-12h ERA5 fields ─────────────────────────────────────────────
        self._log(f"    Loading t-12h snapshot: {dt_m12}")
        era5_m12 = era5_loader.load_snapshot(dt_m12)

        # ── Helpers ────────────────────────────────────────────────────────────

        def flip(arr):
            """Flip lat axis from 90→-90 to -90→+90."""
            if arr.ndim == 2:
                return arr[::-1, :]        # (181, 360)
            elif arr.ndim == 3:
                return arr[:, ::-1, :]     # (13, 181, 360)
            return arr

        def stack4(key):
            """Stack [dummy_t-12h, t-12h, t0, dummy_t+6h] into (4, ...) array."""
            m12 = flip(era5_m12[key])
            t0  = flip(era5_data[key])
            return np.stack([m12, m12, t0, t0], axis=0)

        # ── Surface variables (batch=1, time=4, lat=181, lon=360) ─────────────
        surf_vars = {
            '2m_temperature':          stack4('2t'),
            'mean_sea_level_pressure': stack4('msl'),
            '10m_u_component_of_wind': stack4('u10'),
            '10m_v_component_of_wind': stack4('v10'),
            'total_precipitation_6hr': stack4('tp'),
        }

        # ── Pressure-level variables (batch=1, time=4, level=13, lat=181, lon=360)
        pl_vars = {
            'temperature':         stack4('t'),
            'geopotential':        stack4('z'),
            'u_component_of_wind': stack4('u'),
            'v_component_of_wind': stack4('v'),
            'specific_humidity':   stack4('q'),
            'vertical_velocity':   stack4('w'),
        }

        # ── Assemble data_vars ─────────────────────────────────────────────────
        data_vars = {}

        for name, arr in surf_vars.items():
            data_vars[name] = xr.Variable(
                dims=('batch', 'time', 'lat', 'lon'),
                data=arr[np.newaxis]        # (1, 4, 181, 360)
            )

        for name, arr in pl_vars.items():
            data_vars[name] = xr.Variable(
                dims=('batch', 'time', 'level', 'lat', 'lon'),
                data=arr[np.newaxis]        # (1, 4, 13, 181, 360)
            )

        # Static fields — borrow from sample_inputs (no time/batch dims)
        data_vars['geopotential_at_surface'] = sample_inputs['geopotential_at_surface']
        data_vars['land_sea_mask']           = sample_inputs['land_sea_mask']

        # datetime coord: (batch=1, time=4)
        datetime_vals = np.array(datetimes, dtype='datetime64[ns]')[np.newaxis, :]

        # ── Build Dataset ──────────────────────────────────────────────────────
        batch = xr.Dataset(
            data_vars=data_vars,
            coords={
                'lat':      lats,
                'lon':      lons,
                'level':    levels,
                'time':     times,           # timedelta64 offsets
                'batch':    np.array([0]),
                'datetime': xr.Variable(
                    dims=('batch', 'time'), data=datetime_vals),
            }
        )

        # ── Add temporal forcings in place ─────────────────────────────────────
        data_utils.add_derived_vars(batch)   # year/day progress sin/cos
        data_utils.add_tisr_var(batch)       # toa_incident_solar_radiation

        # ── Split into inputs / targets / forcings ─────────────────────────────
        inputs_xr, targets_xr, forcings_xr = data_utils.extract_inputs_targets_forcings(
            batch,
            target_lead_times=slice("6h", "6h"),
            **dataclasses.asdict(self.task_config)
        )

        return inputs_xr, targets_xr, forcings_xr

    # ── Window-level extraction ────────────────────────────────────────────────

    def extract_window(self, year: int, week_stride: int = 7) -> np.ndarray:
        """
        Extract mean embedding for one 12-month window starting Jan 1 of year.
        Subsamples weekly (~52 snapshots). Returns (11160, 512) float32.
        """
        out_path = f"{EMBED_DIR}/window_{year:04d}.npy"
        if os.path.exists(out_path):
            self._log(f"Year {year}: already cached at {out_path}, skipping.")
            return np.load(out_path)

        self._load_model()

        # Weekly snapshots for this year
        start     = datetime(year, 1, 1, 0)
        end       = datetime(year, 12, 31, 0)
        snapshots = []
        dt = start
        while dt <= end:
            snapshots.append(dt)
            dt += timedelta(days=week_stride)
        self._log(f"Year {year}: {len(snapshots)} snapshots")

        # Static fields from sample data
        sample_inputs, _, _ = self.get_sample_inputs_targets_forcings()

        running_sum = np.zeros((N_ANTARCTIC, LATENT_SIZE), dtype=np.float64)
        n_valid     = 0

        era5 = ERA5Loader(verbose=False)

        for i, dt in enumerate(snapshots):
            try:
                self._log(f"  [{i+1}/{len(snapshots)}] {dt.strftime('%Y-%m-%d')}")

                era5_data = era5.load_snapshot(dt)

                inputs_xr, targets_xr, forcings_xr = self._era5_to_graphcast_xr(
                    era5_data, era5, sample_inputs, dt)

                emb = self.extract_snapshot(inputs_xr, targets_xr, forcings_xr)
                running_sum += emb.astype(np.float64)
                n_valid += 1

            except Exception as e:
                self._log(f"  WARNING: snapshot {dt} failed: {e} — skipping")
                continue

        if n_valid == 0:
            raise RuntimeError(f"All snapshots failed for year {year}")

        mean_emb = (running_sum / n_valid).astype(np.float32)
        self._log(f"Year {year}: averaged {n_valid} snapshots -> shape {mean_emb.shape}")

        os.makedirs(EMBED_DIR, exist_ok=True)
        np.save(out_path, mean_emb)
        self._log(f"Year {year}: saved to {out_path}")
        return mean_emb

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                        help='Run single snapshot test with sample data')
    parser.add_argument('--year', type=int, default=None,
                        help='Extract embeddings for one year')
    parser.add_argument('--all', action='store_true',
                        help='Extract all 38 training years')
    args = parser.parse_args()

    extractor = EmbeddingExtractor(verbose=True)

    if args.test:
        print("=" * 60)
        print("Single snapshot test using sample data")
        print("=" * 60)
        extractor._load_model()
        inputs, targets, forcings = extractor.get_sample_inputs_targets_forcings()
        emb = extractor.extract_snapshot(inputs, targets, forcings)
        print(f"\nEmbedding shape : {emb.shape}")
        print(f"dtype           : {emb.dtype}")
        print(f"range           : [{emb.min():.4f}, {emb.max():.4f}]")
        print(f"mean            : {emb.mean():.4f}")
        print(f"any NaN         : {np.isnan(emb).any()}")
        print("\nPASSED" if not np.isnan(emb).any() else "\nFAILED — NaNs detected")

    elif args.year is not None:
        extractor.extract_window(args.year)

    elif args.all:
        print(f"Extracting embeddings for {len(TRAIN_YEARS)} years...")
        for year in TRAIN_YEARS:
            try:
                extractor.extract_window(year)
            except Exception as e:
                print(f"ERROR year {year}: {e}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
