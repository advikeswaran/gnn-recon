"""
ice_core_loader.py
------------------
Loads and preprocesses Antarctic ice core data (accumulation + water isotope)
for use as observation nodes in the reconstruction GNN.

Design decisions:
  - Sites are grouped by nearest 1° grid node; values averaged within each group
  - Isotope priority: δ18O preferred; dD-only converted via δ18O ≈ δD/8
  - If both δ18O and converted-dD available for a group, average them
  - Treated identically in node features regardless of conversion
  - Node position: snapped to nearest 1° grid node center
  - Missing data: site excluded from graph for that year
  - Water coord fix: 'Mount Brown South' hardcoded → 'MBS'
  - Embeddings: climatological mean (embedding_clim.npy) used for ALL years,
    both during ERA5 training (1979–2017) and reconstruction (1801–2000).
    Generate embedding_clim.npy first with compute_clim_embedding.py.

Node feature layout (516-dim per observation node):
  [0]     iso_value    (δ18O in ‰, or 0.0 if unavailable)
  [1]     iso_avail    (1.0 if available, 0.0 otherwise)
  [2]     accum_value  (mm/year, or 0.0 if unavailable)
  [3]     accum_avail  (1.0 if available, 0.0 otherwise)
  [4]     lat          (grid-snapped, degrees)
  [5]     lon          (grid-snapped, degrees)
  [6:518] embedding    (512-dim climatological mean GraphCast embedding)  # indices 6..517

Usage:
  loader = IceCoreLoader(data_dir, embeddings_dir)
  obs = loader.get_year(1990)
  # obs is a dict:
  #   'features'     : np.ndarray (N_obs, 518)
  #   'grid_indices' : np.ndarray (N_obs,) int — index into 11160-node target grid
  #   'lats'         : np.ndarray (N_obs,)
  #   'lons'         : np.ndarray (N_obs,)
  #   'n_obs'        : int
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Antarctic target grid: lat from -60 to -90 (step -1), lon from 0 to 359 (step 1)
# Matches extract_embeddings.py convention
TARGET_LATS = np.arange(-60, -91, -1, dtype=np.float32)   # shape (31,)
TARGET_LONS = np.arange(0, 360, 1, dtype=np.float32)       # shape (360,)

N_LAT = len(TARGET_LATS)   # 31
N_LON = len(TARGET_LONS)   # 360
N_NODES = N_LAT * N_LON    # 11160

# δD → δ18O conversion (Global Meteoric Water Line slope)
DD_TO_D18O = 1.0 / 8.0

# Hardcoded water coord name fixes (coord name → data column name)
WATER_COORD_FIXES = {
    'Mount Brown South': 'MBS',
    # 'MBS dD' already matches 'MBS dD' in data — no fix needed
}

# Renames applied to water DATA columns (not coords)
WATER_DATA_FIXES = {
    'Mount Brown South': 'MBS',
    'Mount Brown South dD': 'MBS dD',
}


# ---------------------------------------------------------------------------
# Helper: grid utilities
# ---------------------------------------------------------------------------

def snap_to_grid(lat: float, lon: float):
    """
    Snap a lat/lon to the nearest 1° Antarctic target grid node.
    Returns (snapped_lat, snapped_lon, grid_index) or None if outside domain.
    lon is wrapped to [0, 360) before snapping.
    """
    # Wrap lon to [0, 360)
    lon = lon % 360.0

    # Find nearest lat in TARGET_LATS (all <= -60)
    if lat > -60.0 or lat < -91.0:
        return None  # outside Antarctic domain

    lat_idx = int(np.argmin(np.abs(TARGET_LATS - lat)))
    lon_idx = int(np.argmin(np.abs(TARGET_LONS - lon)))

    snapped_lat = float(TARGET_LATS[lat_idx])
    snapped_lon = float(TARGET_LONS[lon_idx])
    grid_idx = lat_idx * N_LON + lon_idx

    return snapped_lat, snapped_lon, grid_idx


def grid_index_to_latlon(grid_idx: int):
    lat_idx = grid_idx // N_LON
    lon_idx = grid_idx % N_LON
    return float(TARGET_LATS[lat_idx]), float(TARGET_LONS[lon_idx])


# ---------------------------------------------------------------------------
# IceCoreLoader
# ---------------------------------------------------------------------------

class IceCoreLoader:
    """
    Loads ice core data and returns per-year observation node feature arrays.
    """

    def __init__(self, data_dir: str, embeddings_dir: str):
        """
        Parameters
        ----------
        data_dir : str
            Path to directory containing AccumCoresCoords.csv, AccumCoresData.csv,
            WaterCoresCoords.csv, WaterCoresData.csv
        embeddings_dir : str
            Path to directory containing embedding_clim.npy (shape 11160 × 512).
            Generate it first with compute_clim_embedding.py.
        """
        self.data_dir = Path(data_dir)
        self.embeddings_dir = Path(embeddings_dir)

        print("Loading ice core metadata and data files...")
        self._load_accum()
        self._load_water()
        self._build_site_registry()
        self._load_clim_embedding()
        print(f"Site registry built: {len(self.site_registry)} unique 1° grid nodes occupied")
        print(f"  Accumulation sites: {len(self.accum_coords)} raw → "
              f"{len(self.accum_grid_map)} grid nodes")
        print(f"  Water isotope sites: {len(self.water_coords)} raw → "
              f"{len(self.water_grid_map)} grid nodes")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_accum(self):
        """Load accumulation coordinates and data."""
        coords_path = self.data_dir / 'AccumCoresCoords.csv'
        data_path   = self.data_dir / 'AccumCoresData.csv'

        self.accum_coords = pd.read_csv(
            coords_path, header=None,
            names=['site_id', 'site_name', 'lat', 'lon'],
            engine='python'
        )
        self.accum_coords['site_id'] = self.accum_coords['site_id'].str.strip()
        self.accum_coords['lat'] = pd.to_numeric(self.accum_coords['lat'], errors='coerce')
        self.accum_coords['lon'] = pd.to_numeric(self.accum_coords['lon'], errors='coerce')

        # Data: row 0 = site numbers (skip via header=1), index = year
        self.accum_data = pd.read_csv(
            data_path, header=1, index_col=0, engine='python'
        )
        self.accum_data.columns = self.accum_data.columns.str.strip()
        self.accum_data.index = pd.to_numeric(self.accum_data.index, errors='coerce')
        self.accum_data = self.accum_data[self.accum_data.index.notna()]
        self.accum_data.index = self.accum_data.index.astype(int)

        print(f"  Accum: {len(self.accum_coords)} coord rows, "
              f"{self.accum_data.shape[1]} data columns, "
              f"years {int(self.accum_data.index.min())}–{int(self.accum_data.index.max())}")

    def _load_water(self):
        """Load water isotope coordinates and data, applying name fixes."""
        coords_path = self.data_dir / 'WaterCoresCoords.csv'
        data_path   = self.data_dir / 'WaterCoresData.csv'

        self.water_coords = pd.read_csv(
            coords_path, header=None,
            names=['site_id', 'site_name', 'lat', 'lon'],
            engine='python'
        )
        self.water_coords['site_id'] = self.water_coords['site_id'].str.strip()

        # Apply hardcoded name fixes
        self.water_coords['site_id'] = self.water_coords['site_id'].replace(WATER_COORD_FIXES)

        self.water_coords['lat'] = pd.to_numeric(self.water_coords['lat'], errors='coerce')
        self.water_coords['lon'] = pd.to_numeric(self.water_coords['lon'], errors='coerce')

        self.water_data = pd.read_csv(
            data_path, header=1, index_col=0, engine='python'
        )
        self.water_data.columns = self.water_data.columns.str.strip()
        self.water_data.rename(columns=WATER_DATA_FIXES, inplace=True)
        self.water_data.index = pd.to_numeric(self.water_data.index, errors='coerce')
        self.water_data = self.water_data[self.water_data.index.notna()]
        self.water_data.index = self.water_data.index.astype(int)

        # Identify dD columns (contain 'dD' in name) vs δ18O columns (everything else)
        all_water_cols = set(self.water_data.columns)
        water_ids = set(self.water_coords['site_id'])

        # Columns matched to coords
        matched = all_water_cols & water_ids
        unmatched = all_water_cols - water_ids
        if unmatched:
            print(f"  Water: {len(unmatched)} data columns with no coord match (will be skipped): "
                  f"{sorted(unmatched)}")

        print(f"  Water: {len(self.water_coords)} coord rows, "
              f"{self.water_data.shape[1]} data columns ({len(matched)} matched), "
              f"years {int(self.water_data.index.min())}–{int(self.water_data.index.max())}")

    def _build_site_registry(self):
        """
        For each raw ice core site, snap to 1° grid node.
        Build two maps: grid_idx → list of (site_id, data_type)
          data_type: 'accum', 'iso_d18o', 'iso_dd'
        Sites outside the Antarctic domain (lat > -60) are excluded.
        """
        # accum_grid_map[grid_idx] = list of accum site_ids
        self.accum_grid_map = {}
        self._accum_skipped = []

        for _, row in self.accum_coords.iterrows():
            result = snap_to_grid(row['lat'], row['lon'])
            if result is None:
                self._accum_skipped.append(row['site_id'])
                continue
            _, _, gidx = result
            # Only include if site_id exists as a data column
            if row['site_id'] not in self.accum_data.columns:
                continue
            self.accum_grid_map.setdefault(gidx, []).append(row['site_id'])

        # water_grid_map[grid_idx] = {'d18o': [site_ids], 'dd': [site_ids]}
        self.water_grid_map = {}
        self._water_skipped = []

        water_ids_with_coords = set(self.water_coords['site_id'])

        for _, row in self.water_coords.iterrows():
            site_id = row['site_id']
            result = snap_to_grid(row['lat'], row['lon'])
            if result is None:
                self._water_skipped.append(site_id)
                continue
            if site_id not in self.water_data.columns:
                continue
            _, _, gidx = result

            # Classify as dD or δ18O based on site name
            is_dd = 'dD' in site_id or 'dd' in site_id.lower().replace('add', '')
            self.water_grid_map.setdefault(gidx, {'d18o': [], 'dd': []})
            if is_dd:
                self.water_grid_map[gidx]['dd'].append(site_id)
            else:
                self.water_grid_map[gidx]['d18o'].append(site_id)

        # Combined registry of all grid nodes that have any data
        self.site_registry = sorted(
            set(self.accum_grid_map.keys()) | set(self.water_grid_map.keys())
        )

        if self._accum_skipped:
            print(f"  Skipped {len(self._accum_skipped)} accum sites outside domain: "
                  f"{self._accum_skipped}")
        if self._water_skipped:
            print(f"  Skipped {len(self._water_skipped)} water sites outside domain: "
                  f"{self._water_skipped}")

    # ------------------------------------------------------------------
    # Embedding loading
    # ------------------------------------------------------------------

    def _load_clim_embedding(self):
        """Load the precomputed climatological mean embedding. Shape: (11160, 512)."""
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
        print(f"  Climatological embedding loaded: shape {self.clim_embedding.shape}, "
              f"mean {self.clim_embedding.mean():.4f}, std {self.clim_embedding.std():.4f}")

    # ------------------------------------------------------------------
    # Per-year value extraction
    # ------------------------------------------------------------------

    def _get_accum_value(self, grid_idx: int, year: int):
        """
        Return averaged accumulation value (mm/yr) for a grid node and year.
        Returns (value, available): (float, True) or (0.0, False).
        """
        site_ids = self.accum_grid_map.get(grid_idx, [])
        if not site_ids:
            return 0.0, False

        vals = []
        for sid in site_ids:
            if year in self.accum_data.index:
                v = self.accum_data.loc[year, sid]
                if pd.notna(v):
                    vals.append(float(v))

        if not vals:
            return 0.0, False
        return float(np.mean(vals)), True

    def _get_iso_value(self, grid_idx: int, year: int):
        """
        Return averaged δ18O value for a grid node and year.
        Priority: δ18O > dD (converted via δ18O ≈ δD/8).
        If both available, average all (d18o + converted dd).
        Returns (value, available): (float, True) or (0.0, False).
        """
        entry = self.water_grid_map.get(grid_idx, None)
        if entry is None:
            return 0.0, False

        all_vals = []

        # Collect δ18O values
        for sid in entry['d18o']:
            if year in self.water_data.index:
                v = self.water_data.loc[year, sid]
                if pd.notna(v):
                    all_vals.append(float(v))

        # Collect dD values, convert to δ18O
        for sid in entry['dd']:
            if year in self.water_data.index:
                v = self.water_data.loc[year, sid]
                if pd.notna(v):
                    all_vals.append(float(v) * DD_TO_D18O)

        # If we have δ18O values, use only those (prefer δ18O)
        # Recheck: only use d18o if available, else fall back to converted dd
        d18o_vals = []
        dd_converted_vals = []

        for sid in entry['d18o']:
            if year in self.water_data.index:
                v = self.water_data.loc[year, sid]
                if pd.notna(v):
                    d18o_vals.append(float(v))

        for sid in entry['dd']:
            if year in self.water_data.index:
                v = self.water_data.loc[year, sid]
                if pd.notna(v):
                    dd_converted_vals.append(float(v) * DD_TO_D18O)

        if d18o_vals:
            # Prefer δ18O; if dD also available, average all together
            combined = d18o_vals + dd_converted_vals
            return float(np.mean(combined)), True
        elif dd_converted_vals:
            return float(np.mean(dd_converted_vals)), True
        else:
            return 0.0, False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_year(self, year: int) -> dict:
        """
        Build observation node feature array for a given data year.

        Uses the preloaded climatological mean embedding for all years —
        identical format for both ERA5 training (1979–2017) and
        reconstruction (1801–2000).

        Parameters
        ----------
        year : int
            Ice core data year

        Returns
        -------
        dict with keys:
            'features'     : np.ndarray (N_obs, 518), float32
            'grid_indices' : np.ndarray (N_obs,), int32
            'lats'         : np.ndarray (N_obs,), float32
            'lons'         : np.ndarray (N_obs,), float32
            'n_obs'        : int
            'year'         : int
        """
        features_list = []
        grid_indices_list = []
        lats_list = []
        lons_list = []

        for grid_idx in self.site_registry:
            accum_val, accum_avail = self._get_accum_value(grid_idx, year)
            iso_val,   iso_avail   = self._get_iso_value(grid_idx, year)

            # Drop node if neither measurement is available for this year
            if not accum_avail and not iso_avail:
                continue

            snapped_lat, snapped_lon = grid_index_to_latlon(grid_idx)
            emb = self.clim_embedding[grid_idx]  # (512,)

            feat = np.array(
                [iso_val, float(iso_avail),
                 accum_val, float(accum_avail),
                 snapped_lat, snapped_lon],
                dtype=np.float32
            )
            feat = np.concatenate([feat, emb])  # (516,)

            features_list.append(feat)
            grid_indices_list.append(grid_idx)
            lats_list.append(snapped_lat)
            lons_list.append(snapped_lon)

        if not features_list:
            raise ValueError(f"No observation nodes found for year {year}. "
                             f"Check data coverage.")

        return {
            'features':     np.stack(features_list, axis=0).astype(np.float32),
            'grid_indices': np.array(grid_indices_list, dtype=np.int32),
            'lats':         np.array(lats_list, dtype=np.float32),
            'lons':         np.array(lons_list, dtype=np.float32),
            'n_obs':        len(features_list),
            'year':         year,
        }

    def get_available_years(self, require_both: bool = False) -> list:
        """
        Return sorted list of years where at least one observation node is available.

        Parameters
        ----------
        require_both : bool
            If True, only return years where at least one site has BOTH
            accumulation and isotope data.
        """
        accum_years = set(self.accum_data.index)
        water_years = set(self.water_data.index)

        if require_both:
            return sorted(accum_years & water_years)
        return sorted(accum_years | water_years)

    def summary(self, year: int = None):
        """Print a summary of data coverage."""
        print(f"\n{'='*60}")
        print(f"IceCoreLoader summary")
        print(f"{'='*60}")
        print(f"Accum sites (raw):   {len(self.accum_coords)}")
        print(f"Water sites (raw):   {len(self.water_coords)}")
        print(f"Accum grid nodes:    {len(self.accum_grid_map)}")
        print(f"Water grid nodes:    {len(self.water_grid_map)}")
        print(f"Total grid nodes:    {len(self.site_registry)}")
        print(f"Accum year range:    {int(self.accum_data.index.min())}–"
              f"{int(self.accum_data.index.max())}")
        print(f"Water year range:    {int(self.water_data.index.min())}–"
              f"{int(self.water_data.index.max())}")

        if year is not None:
            n_accum = sum(
                1 for gidx in self.accum_grid_map
                if self._get_accum_value(gidx, year)[1]
            )
            n_water = sum(
                1 for gidx in self.water_grid_map
                if self._get_iso_value(gidx, year)[1]
            )
            print(f"\nCoverage for year {year}:")
            print(f"  Accum nodes with data: {n_accum}")
            print(f"  Water nodes with data: {n_water}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    DATA_DIR       = '/glade/derecho/scratch/advike/graphcast_recon/data'
    EMBEDDINGS_DIR = '/glade/derecho/scratch/advike/graphcast_recon/cache/embeddings'

    print("=== IceCoreLoader test ===\n")
    loader = IceCoreLoader(DATA_DIR, EMBEDDINGS_DIR)
    loader.summary(year=1990)

    print("Testing get_year(1990)...")
    obs = loader.get_year(1990)

    print(f"\nResult for year 1990:")
    print(f"  n_obs:           {obs['n_obs']}")
    print(f"  features shape:  {obs['features'].shape}")
    print(f"  grid_indices:    {obs['grid_indices'][:5]} ...")
    print(f"  lats:            {obs['lats'][:5]} ...")
    print(f"  lons:            {obs['lons'][:5]} ...")

    # Sanity checks
    assert obs['features'].shape == (obs['n_obs'], 518), "Feature dim mismatch"
    assert obs['features'].dtype == np.float32, "Wrong dtype"
    assert not np.any(np.isnan(obs['features'])), "NaNs in features!"
    assert np.all(obs['lats'] <= -60.0), "Node outside Antarctic domain!"
    assert np.all(obs['grid_indices'] < N_NODES), "Grid index out of range!"

    # Check availability flags are binary
    iso_avail   = obs['features'][:, 1]
    accum_avail = obs['features'][:, 3]
    assert np.all(np.isin(iso_avail,   [0.0, 1.0])), "iso_avail not binary!"
    assert np.all(np.isin(accum_avail, [0.0, 1.0])), "accum_avail not binary!"

    n_both  = int(np.sum((iso_avail == 1.0) & (accum_avail == 1.0)))
    n_iso   = int(np.sum((iso_avail == 1.0) & (accum_avail == 0.0)))
    n_accum = int(np.sum((iso_avail == 0.0) & (accum_avail == 1.0)))

    print(f"\n  Nodes with both iso+accum: {n_both}")
    print(f"  Nodes with iso only:       {n_iso}")
    print(f"  Nodes with accum only:     {n_accum}")
    print(f"\n  iso_value   range: [{obs['features'][:, 0].min():.2f}, "
          f"{obs['features'][:, 0].max():.2f}]")
    print(f"  accum_value range: [{obs['features'][:, 2].min():.2f}, "
          f"{obs['features'][:, 2].max():.2f}]")
    print(f"  embedding   range: [{obs['features'][:, 6:].min():.3f}, "
          f"{obs['features'][:, 6:].max():.3f}]")

    print("\n=== All checks passed ===")
