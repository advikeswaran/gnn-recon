"""
ice_core_loader.py
------------------
Loads and preprocesses Antarctic ice core data (accumulation + water isotope)
for use as observation nodes in the reconstruction GNN.

Values served are calibrated to ERA5 units via per-site linear regression
(see calibrate_ice_cores.py), then expressed as ANOMALIES relative to each
site's 1979-2000 mean. This ensures the GNN is trained on and applied to
anomaly fields, avoiding artifacts from changing obs network composition.

Anomaly definition:
  iso_anom(yr)   = calib_iso(yr)   - mean(calib_iso[1979-2000])   [K]
  accum_anom(yr) = calib_accum(yr) - mean(calib_accum[1979-2000]) [m/yr]

These anomalies are normalised by ERA5 std only (no mean shift, since
anomalies are already centered near zero).

Node feature layout (518-dim per observation node):
  [0]     iso_anom_norm    (normalised temperature anomaly, or 0.0 if unavailable)
  [1]     iso_avail        (1.0 if available, 0.0 otherwise)
  [2]     accum_anom_norm  (normalised precip anomaly, or 0.0 if unavailable)
  [3]     accum_avail      (1.0 if available, 0.0 otherwise)
  [4]     lat              (grid-snapped, degrees)
  [5]     lon              (grid-snapped, degrees)
  [6:518] embedding        (512-dim climatological mean GraphCast embedding)

Usage:
  loader = IceCoreLoader(data_dir, embeddings_dir, calibration_dir,
                         temp_mean, temp_std, prec_mean, prec_std)
  obs = loader.get_year(1990)
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_LATS = np.arange(-60, -91, -1, dtype=np.float32)
TARGET_LONS = np.arange(0, 360, 1, dtype=np.float32)

N_LAT   = len(TARGET_LATS)
N_LON   = len(TARGET_LONS)
N_NODES = N_LAT * N_LON

# 1979-2000 overlap indices within the 1801-2000 calibration array
OVERLAP_START_IDX = 178   # index of 1979 in recon_years
OVERLAP_END_IDX   = 200   # exclusive, slice [178:200] = 22 years

WATER_COORD_FIXES = {
    'Mount Brown South': 'MBS',
}
WATER_DATA_FIXES = {
    'Mount Brown South':    'MBS',
    'Mount Brown South dD': 'MBS dD',
}


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def snap_to_grid(lat: float, lon: float):
    lon = lon % 360.0
    if lat > -60.0 or lat < -91.0:
        return None
    lat_idx = int(np.argmin(np.abs(TARGET_LATS - lat)))
    lon_idx = int(np.argmin(np.abs(TARGET_LONS - lon)))
    return (float(TARGET_LATS[lat_idx]),
            float(TARGET_LONS[lon_idx]),
            lat_idx * N_LON + lon_idx)


def grid_index_to_latlon(grid_idx: int):
    lat_idx = grid_idx // N_LON
    lon_idx = grid_idx % N_LON
    return float(TARGET_LATS[lat_idx]), float(TARGET_LONS[lon_idx])


# ---------------------------------------------------------------------------
# IceCoreLoader
# ---------------------------------------------------------------------------

class IceCoreLoader:

    def __init__(self, data_dir: str, embeddings_dir: str, calibration_dir: str,
                 temp_mean: float, temp_std: float,
                 prec_mean: float, prec_std: float):
        """
        temp_mean, temp_std, prec_mean, prec_std must be passed explicitly
        from norm_stats.npz — no defaults to avoid silent errors.
        Note: temp_mean and prec_mean are not used for normalisation in
        anomaly mode, but are retained for reference/logging.
        """
        self.data_dir        = Path(data_dir)
        self.embeddings_dir  = Path(embeddings_dir)
        self.calibration_dir = Path(calibration_dir)
        self.temp_mean = temp_mean
        self.temp_std  = temp_std
        self.prec_mean = prec_mean
        self.prec_std  = prec_std

        print("Loading ice core data...")
        self._load_coords()
        self._load_calibration()
        self._compute_overlap_means()
        self._build_site_registry()
        self._load_clim_embedding()
        print(f"Ready: {len(self.site_registry)} unique grid nodes, "
              f"{len(self.iso_grid_map)} iso nodes, "
              f"{len(self.accum_grid_map)} accum nodes")

    def _load_coords(self):
        def read_coords(path):
            df = pd.read_csv(path, header=None,
                             names=['site_id', 'site_name', 'lat', 'lon'],
                             engine='python')
            df['site_id'] = df['site_id'].str.strip()
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            return df

        self.accum_coords = read_coords(self.data_dir / 'AccumCoresCoords.csv')
        self.water_coords = read_coords(self.data_dir / 'WaterCoresCoords.csv')
        self.water_coords['site_id'] = (
            self.water_coords['site_id'].replace(WATER_COORD_FIXES)
        )
        print(f"  Coords: {len(self.accum_coords)} accum, "
              f"{len(self.water_coords)} water sites")

    def _load_calibration(self):
        iso_path   = self.calibration_dir / 'calibrated_iso.npy'
        accum_path = self.calibration_dir / 'calibrated_accum.npy'
        meta_path  = self.calibration_dir / 'calibration_meta.npz'

        for p in [iso_path, accum_path, meta_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Calibration file missing: {p}\n"
                    f"Run calibrate_ice_cores.py first."
                )

        self.calib_iso   = np.load(iso_path)    # (80, 200)
        self.calib_accum = np.load(accum_path)  # (84, 200)

        meta = np.load(meta_path, allow_pickle=True)
        self.calib_iso_site_ids   = [s for s in meta['iso_site_ids']]
        self.calib_iso_lats       = meta['iso_lats']
        self.calib_iso_lons       = meta['iso_lons']
        self.calib_accum_site_ids = [s for s in meta['accum_site_ids']]
        self.calib_accum_lats     = meta['accum_lats']
        self.calib_accum_lons     = meta['accum_lons']
        self.calib_years          = meta['recon_years']

        self.calib_year_to_idx = {int(yr): i
                                   for i, yr in enumerate(self.calib_years)}
        self.iso_site_to_row   = {sid: i
                                   for i, sid in enumerate(self.calib_iso_site_ids)}
        self.accum_site_to_row = {sid: i
                                   for i, sid in enumerate(self.calib_accum_site_ids)}

        print(f"  Calibration: {len(self.calib_iso_site_ids)} iso sites, "
              f"{len(self.calib_accum_site_ids)} accum sites, "
              f"years {self.calib_years[0]}-{self.calib_years[-1]}")

    def _compute_overlap_means(self):
        """
        Per-site mean over 1979-2000 (indices 178:200).
        Sites with all-NaN overlap get NaN mean and are excluded from
        the site registry entirely.
        """
        overlap = slice(OVERLAP_START_IDX, OVERLAP_END_IDX)
        self.iso_site_means   = np.nanmean(
            self.calib_iso[:, overlap], axis=1).astype(np.float32)   # (80,)
        self.accum_site_means = np.nanmean(
            self.calib_accum[:, overlap], axis=1).astype(np.float32) # (84,)

        # compute anomaly std across all sites and years for normalisation
        iso_anom   = self.calib_iso   - self.iso_site_means[:, None]
        accum_anom = self.calib_accum - self.accum_site_means[:, None]
        self.iso_anom_std   = float(np.nanstd(iso_anom))
        self.accum_anom_std = float(np.nanstd(accum_anom))

        n_iso_valid   = int(np.sum(np.isfinite(self.iso_site_means)))
        n_accum_valid = int(np.sum(np.isfinite(self.accum_site_means)))
        print(f"  Overlap means (1979-2000): "
              f"{n_iso_valid}/{len(self.iso_site_means)} iso valid, "
              f"{n_accum_valid}/{len(self.accum_site_means)} accum valid")
        print(f"  Anomaly stds: iso={self.iso_anom_std:.4f}K  "
              f"accum={self.accum_anom_std:.4f}m/yr")

        # save anomaly stds so validate.py and apply.py can load them
        # without reinstantiating the full loader
        import os
        anom_std_path = self.calibration_dir / 'anom_stds.npz'
        tmp = str(anom_std_path).replace('.npz', '.tmp.npz')
        np.savez(tmp,
                 iso_anom_std=np.float32(self.iso_anom_std),
                 accum_anom_std=np.float32(self.accum_anom_std))
        os.rename(tmp, str(anom_std_path))

    def _build_site_registry(self):
        self.iso_grid_map = {}
        for sid in self.calib_iso_site_ids:
            i = self.iso_site_to_row[sid]
            # exclude sites with no valid 1979-2000 overlap mean
            if not np.isfinite(self.iso_site_means[i]):
                continue
            lat = float(self.calib_iso_lats[i])
            lon = float(self.calib_iso_lons[i])
            result = snap_to_grid(lat, lon)
            if result is None:
                continue
            _, _, gidx = result
            self.iso_grid_map.setdefault(gidx, []).append(sid)

        self.accum_grid_map = {}
        for sid in self.calib_accum_site_ids:
            i = self.accum_site_to_row[sid]
            if not np.isfinite(self.accum_site_means[i]):
                continue
            lat = float(self.calib_accum_lats[i])
            lon = float(self.calib_accum_lons[i])
            result = snap_to_grid(lat, lon)
            if result is None:
                continue
            _, _, gidx = result
            self.accum_grid_map.setdefault(gidx, []).append(sid)

        self.site_registry = sorted(
            set(self.iso_grid_map.keys()) | set(self.accum_grid_map.keys())
        )

    def _load_clim_embedding(self):
        path = self.embeddings_dir / 'embedding_clim.npy'
        if not path.exists():
            raise FileNotFoundError(
                f"Climatological embedding not found: {path}\n"
                f"Run compute_clim_embedding.py first."
            )
        emb = np.load(path)
        assert emb.shape == (N_NODES, 512), \
            f"Unexpected embedding shape {emb.shape}, expected ({N_NODES}, 512)"
        self.clim_embedding = emb.astype(np.float32)
        print(f"  Embedding loaded: shape {self.clim_embedding.shape}")

    def _get_iso_value(self, grid_idx: int, year: int):
        site_ids = self.iso_grid_map.get(grid_idx, [])
        if not site_ids:
            return 0.0, False
        yr_idx = self.calib_year_to_idx.get(year, None)
        if yr_idx is None:
            return 0.0, False
        anoms = []
        for sid in site_ids:
            row  = self.iso_site_to_row[sid]
            v    = self.calib_iso[row, yr_idx]
            mean = self.iso_site_means[row]
            if np.isfinite(v) and np.isfinite(mean):
                anoms.append(float(v) - float(mean))
        if not anoms:
            return 0.0, False
        # divide by std only — anomalies are already centered near zero
        norm_val = float(np.mean(anoms)) / self.iso_anom_std
        return norm_val, True

    def _get_accum_value(self, grid_idx: int, year: int):
        site_ids = self.accum_grid_map.get(grid_idx, [])
        if not site_ids:
            return 0.0, False
        yr_idx = self.calib_year_to_idx.get(year, None)
        if yr_idx is None:
            return 0.0, False
        anoms = []
        for sid in site_ids:
            row  = self.accum_site_to_row[sid]
            v    = self.calib_accum[row, yr_idx]
            mean = self.accum_site_means[row]
            if np.isfinite(v) and np.isfinite(mean):
                anoms.append(float(v) - float(mean))
        if not anoms:
            return 0.0, False
        norm_val = float(np.mean(anoms)) / self.accum_anom_std
        return norm_val, True

    def get_year(self, year: int) -> dict:
        features_list     = []
        grid_indices_list = []
        lats_list         = []
        lons_list         = []

        for grid_idx in self.site_registry:
            iso_val,   iso_avail   = self._get_iso_value(grid_idx, year)
            accum_val, accum_avail = self._get_accum_value(grid_idx, year)

            if not iso_avail and not accum_avail:
                continue

            slat, slon = grid_index_to_latlon(grid_idx)
            emb = self.clim_embedding[grid_idx]

            feat = np.concatenate([
                np.array([iso_val, float(iso_avail),
                          accum_val, float(accum_avail),
                          slat, slon], dtype=np.float32),
                emb,
            ])

            features_list.append(feat)
            grid_indices_list.append(grid_idx)
            lats_list.append(slat)
            lons_list.append(slon)

        if not features_list:
            raise ValueError(f"No observation nodes for year {year}.")

        return {
            'features':     np.stack(features_list).astype(np.float32),
            'grid_indices': np.array(grid_indices_list, dtype=np.int32),
            'lats':         np.array(lats_list, dtype=np.float32),
            'lons':         np.array(lons_list, dtype=np.float32),
            'n_obs':        len(features_list),
            'year':         year,
        }

    def get_available_years(self) -> list:
        return [int(y) for y in self.calib_years]

    def summary(self, year: int = None):
        print(f"\n{'='*60}")
        print(f"IceCoreLoader summary (anomaly mode, ref: 1979-2000)")
        print(f"{'='*60}")
        print(f"Iso grid nodes:    {len(self.iso_grid_map)}")
        print(f"Accum grid nodes:  {len(self.accum_grid_map)}")
        print(f"Total grid nodes:  {len(self.site_registry)}")
        print(f"Calib year range:  {self.calib_years[0]}-{self.calib_years[-1]}")
        if year is not None:
            obs = self.get_year(year)
            iso_avail   = obs['features'][:, 1]
            accum_avail = obs['features'][:, 3]
            print(f"\nCoverage for year {year}:")
            print(f"  Total obs nodes:         {obs['n_obs']}")
            print(f"  Nodes with iso:          {int(iso_avail.sum())}")
            print(f"  Nodes with accum:        {int(accum_avail.sum())}")
            print(f"  iso_anom  range (norm):  [{obs['features'][:,0].min():.3f}, "
                  f"{obs['features'][:,0].max():.3f}]")
            print(f"  accum_anom range (norm): [{obs['features'][:,2].min():.4f}, "
                  f"{obs['features'][:,2].max():.4f}]")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    RECON_DIR       = '/glade/derecho/scratch/advike/graphcast_recon'
    DATA_DIR        = f'{RECON_DIR}/data'
    EMBEDDINGS_DIR  = f'{RECON_DIR}/cache/embeddings'
    CALIBRATION_DIR = f'{RECON_DIR}/cache/calibration'
    NORM_STATS_PATH = f'{RECON_DIR}/cache/era5_targets/norm_stats.npz'

    ns = np.load(NORM_STATS_PATH)

    print("=== IceCoreLoader test (anomaly mode) ===\n")
    loader = IceCoreLoader(
        data_dir=DATA_DIR, embeddings_dir=EMBEDDINGS_DIR,
        calibration_dir=CALIBRATION_DIR,
        temp_mean=float(ns['temp_mean']), temp_std=float(ns['temp_std']),
        prec_mean=float(ns['prec_mean']), prec_std=float(ns['prec_std']),
    )
    loader.summary(year=1990)

    print("Testing get_year(1990)...")
    obs = loader.get_year(1990)

    print(f"\nResult for year 1990:")
    print(f"  n_obs:                  {obs['n_obs']}")
    print(f"  features shape:         {obs['features'].shape}")
    print(f"  iso_anom  (norm): min={obs['features'][:,0].min():.3f}, "
          f"max={obs['features'][:,0].max():.3f}")
    print(f"  accum_anom (norm): min={obs['features'][:,2].min():.4f}, "
          f"max={obs['features'][:,2].max():.4f}")

    # check mean anomaly over overlap period is near zero
    print("\nChecking 1979-2000 mean anomaly (should be ~0)...")
    iso_anoms, accum_anoms = [], []
    for yr in range(1979, 2001):
        o = loader.get_year(yr)
        iso_mask   = o['features'][:, 1] == 1.0
        accum_mask = o['features'][:, 3] == 1.0
        if iso_mask.any():
            iso_anoms.append(o['features'][:, 0][iso_mask].mean())
        if accum_mask.any():
            accum_anoms.append(o['features'][:, 2][accum_mask].mean())
    print(f"  Mean iso anom 1979-2000:   {float(np.mean(iso_anoms)):.4f} (norm)")
    print(f"  Mean accum anom 1979-2000: {float(np.mean(accum_anoms)):.4f} (norm)")

    assert obs['features'].shape[1] == 518,       "Feature dim should be 518"
    assert obs['features'].dtype == np.float32,   "Wrong dtype"
    assert not np.any(np.isnan(obs['features'])), "NaNs in features!"
    assert np.all(obs['lats'] <= -60.0),           "Node outside Antarctic domain!"
    assert np.all(obs['grid_indices'] < N_NODES),  "Grid index out of range!"

    iso_avail   = obs['features'][:, 1]
    accum_avail = obs['features'][:, 3]
    assert np.all(np.isin(iso_avail,   [0.0, 1.0])), "iso_avail not binary!"
    assert np.all(np.isin(accum_avail, [0.0, 1.0])), "accum_avail not binary!"

    print("\n=== All checks passed ===")
