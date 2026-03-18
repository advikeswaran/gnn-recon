"""
ice_core_loader.py
------------------
Loads and preprocesses Antarctic ice core data (accumulation + water isotope)
for use as observation nodes in the reconstruction GNN.

Values served are calibrated to ERA5 units via per-site linear regression
(see calibrate_ice_cores.py):
  - iso_value   : 2m temperature (K),   calibrated from dD
  - accum_value : precipitation (m/yr), calibrated from accumulation

Raw CSV files are still used for site coordinates and grid snapping.
Calibrated time series are loaded from cache/calibration/.

Design decisions:
  - Sites are grouped by nearest 1deg grid node; values averaged within each group
  - Node position: snapped to nearest 1deg grid node center
  - Missing data: site excluded from graph for that year
  - Embeddings: climatological mean (embedding_clim.npy) used for ALL years

Node feature layout (518-dim per observation node):
  [0]     iso_value    (calibrated temperature in K, or 0.0 if unavailable)
  [1]     iso_avail    (1.0 if available, 0.0 otherwise)
  [2]     accum_value  (calibrated precip in m/yr, or 0.0 if unavailable)
  [3]     accum_avail  (1.0 if available, 0.0 otherwise)
  [4]     lat          (grid-snapped, degrees)
  [5]     lon          (grid-snapped, degrees)
  [6:518] embedding    (512-dim climatological mean GraphCast embedding)

Usage:
  loader = IceCoreLoader(data_dir, embeddings_dir, calibration_dir)
  obs = loader.get_year(1990)
  # obs is a dict:
  #   'features'     : np.ndarray (N_obs, 518)
  #   'grid_indices' : np.ndarray (N_obs,) int
  #   'lats'         : np.ndarray (N_obs,)
  #   'lons'         : np.ndarray (N_obs,)
  #   'n_obs'        : int
  #   'year'         : int
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
                 temp_mean: float = 248.26, temp_std: float = 17.29,
                 prec_mean: float = 0.3608, prec_std: float = 0.3656):
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

        self.calib_iso   = np.load(iso_path)
        self.calib_accum = np.load(accum_path)

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

    def _build_site_registry(self):
        self.iso_grid_map = {}
        for sid in self.calib_iso_site_ids:
            i   = self.iso_site_to_row[sid]
            lat = float(self.calib_iso_lats[i])
            lon = float(self.calib_iso_lons[i])
            result = snap_to_grid(lat, lon)
            if result is None:
                continue
            _, _, gidx = result
            self.iso_grid_map.setdefault(gidx, []).append(sid)

        self.accum_grid_map = {}
        for sid in self.calib_accum_site_ids:
            i   = self.accum_site_to_row[sid]
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
        vals = []
        for sid in site_ids:
            row = self.iso_site_to_row[sid]
            v   = self.calib_iso[row, yr_idx]
            if np.isfinite(v):
                vals.append(float(v))
        if not vals:
            return 0.0, False
        norm_val = (float(np.mean(vals)) - self.temp_mean) / self.temp_std
        return norm_val, True

    def _get_accum_value(self, grid_idx: int, year: int):
        site_ids = self.accum_grid_map.get(grid_idx, [])
        if not site_ids:
            return 0.0, False
        yr_idx = self.calib_year_to_idx.get(year, None)
        if yr_idx is None:
            return 0.0, False
        vals = []
        for sid in site_ids:
            row = self.accum_site_to_row[sid]
            v   = self.calib_accum[row, yr_idx]
            if np.isfinite(v):
                vals.append(float(v))
        if not vals:
            return 0.0, False
        norm_val = (float(np.mean(vals)) - self.prec_mean) / self.prec_std
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
        print(f"IceCoreLoader summary (calibrated values)")
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
            print(f"  Total obs nodes:       {obs['n_obs']}")
            print(f"  Nodes with iso:        {int(iso_avail.sum())}")
            print(f"  Nodes with accum:      {int(accum_avail.sum())}")
            print(f"  iso_value  range (K):  [{obs['features'][:,0].min():.1f}, "
                  f"{obs['features'][:,0].max():.1f}]")
            print(f"  accum_value range (m): [{obs['features'][:,2].min():.4f}, "
                  f"{obs['features'][:,2].max():.4f}]")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    DATA_DIR        = '/glade/derecho/scratch/advike/graphcast_recon/data'
    EMBEDDINGS_DIR  = '/glade/derecho/scratch/advike/graphcast_recon/cache/embeddings'
    CALIBRATION_DIR = '/glade/derecho/scratch/advike/graphcast_recon/cache/calibration'

    print("=== IceCoreLoader test (calibrated) ===\n")
    loader = IceCoreLoader(DATA_DIR, EMBEDDINGS_DIR, CALIBRATION_DIR)
    loader.summary(year=1990)

    print("Testing get_year(1990)...")
    obs = loader.get_year(1990)

    print(f"\nResult for year 1990:")
    print(f"  n_obs:           {obs['n_obs']}")
    print(f"  features shape:  {obs['features'].shape}")
    print(f"  iso_value  (K):  min={obs['features'][:,0].min():.1f}, "
          f"max={obs['features'][:,0].max():.1f}")
    print(f"  accum_value(m):  min={obs['features'][:,2].min():.4f}, "
          f"max={obs['features'][:,2].max():.4f}")

    assert obs['features'].shape[1] == 518,      "Feature dim should be 518"
    assert obs['features'].dtype == np.float32,  "Wrong dtype"
    assert not np.any(np.isnan(obs['features'])), "NaNs in features!"
    assert np.all(obs['lats'] <= -60.0),          "Node outside Antarctic domain!"
    assert np.all(obs['grid_indices'] < N_NODES), "Grid index out of range!"

    iso_avail   = obs['features'][:, 1]
    accum_avail = obs['features'][:, 3]
    assert np.all(np.isin(iso_avail,   [0.0, 1.0])), "iso_avail not binary!"
    assert np.all(np.isin(accum_avail, [0.0, 1.0])), "accum_avail not binary!"

    iso_vals = obs['features'][:, 0][iso_avail == 1.0]
    if len(iso_vals) > 0:
        assert iso_vals.min() > 180.0, f"Suspiciously cold T: {iso_vals.min():.1f} K"
        assert iso_vals.max() < 310.0, f"Suspiciously warm T: {iso_vals.max():.1f} K"

    print("\n=== All checks passed ===")
