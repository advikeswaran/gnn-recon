"""
era5_loader.py
Loads ERA5 data from RDA d633000 for GraphCast input preparation.
"""

import os
import glob
import numpy as np
import xarray as xr
import netCDF4 as nc
from datetime import datetime, timedelta
from functools import lru_cache

# ── Paths ──────────────────────────────────────────────────────────────────────
RDA_ROOT   = "/glade/campaign/collections/rda/data/d633000"
ZARR_2T    = f"{RDA_ROOT}/e5.oper.an.sfc.zarr/e5.oper.an.sfc.2t.zarr"
ZARR_MSL   = f"{RDA_ROOT}/e5.oper.an.sfc.zarr/e5.oper.an.sfc.msl.zarr"
ZARR_10U   = f"{RDA_ROOT}/e5.oper.an.sfc.zarr/e5.oper.an.sfc.10u.zarr"
ZARR_10V   = f"{RDA_ROOT}/e5.oper.an.sfc.zarr/e5.oper.an.sfc.10v.zarr"
ACCUMU_DIR = f"{RDA_ROOT}/e5.oper.fc.sfc.accumu"
PL_DIR     = f"{RDA_ROOT}/e5.oper.an.pl"

# ── GraphCast grid constants ───────────────────────────────────────────────────
GC_NLAT, GC_NLON = 181, 360
GC_LAT    = np.linspace(90, -90, GC_NLAT)
GC_LON    = np.linspace(0, 359, GC_NLON)
GC_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

ERA5_LAT = np.linspace(90, -90, 721)
ERA5_LON = np.linspace(0, 359.75, 1440)


class ERA5Loader:
    """Load and regrid ERA5 fields to GraphCast 1deg grid."""

    def __init__(self, verbose=True):
        self.verbose   = verbose
        self._zarr_2t  = None
        self._zarr_msl = None
        self._zarr_10u = None
        self._zarr_10v = None

    # ── Public interface ───────────────────────────────────────────────────────

    def load_snapshot(self, dt: datetime) -> dict:
        """
        Load all GraphCast input fields for a single datetime snapshot.
        Returns dict with keys: 2t, msl, u10, v10, tp, z, t, u, v, q, w,
                                lat, lon, levels, datetime
        All arrays float32. Pressure-level fields shape (13,181,360), surface (181,360).
        """
        self._log(f"Loading snapshot: {dt}")
        out = {}
        out['2t']       = self.load_2t(dt)
        out['msl']      = self.load_msl(dt)
        out['u10']      = self.load_10u(dt)
        out['v10']      = self.load_10v(dt)
        out['tp']       = self.load_tp(dt)
        pl              = self.load_upper_air(dt)
        out.update(pl)
        out['lat']      = GC_LAT
        out['lon']      = GC_LON
        out['levels']   = np.array(GC_LEVELS, dtype=np.float32)
        out['datetime'] = dt
        self._log("Snapshot complete.")
        return out

    # ── 2m Temperature ─────────────────────────────────────────────────────────

    def load_2t(self, dt: datetime) -> np.ndarray:
        """Load 2m temperature at dt, regrid to 1deg. Returns (181,360) float32."""
        self._log(f"  Loading 2T at {dt}")
        if self._zarr_2t is None:
            self._log("    Opening Zarr store (first call)...")
            self._zarr_2t = xr.open_zarr(ZARR_2T)
        t2     = self._zarr_2t['VAR_2T'].sel(time=dt, method='nearest')
        arr    = np.array(t2.values)
        assert arr.shape == (721, 1440), f"Unexpected 2T shape: {arr.shape}"
        result = self._regrid_0p25_to_1deg(arr).astype(np.float32)
        self._log(f"    2T range: {result.min():.2f} - {result.max():.2f} K")
        return result

    # ── Mean Sea Level Pressure ────────────────────────────────────────────────

    def load_msl(self, dt: datetime) -> np.ndarray:
        """Load mean sea level pressure at dt, regrid to 1deg. Returns (181,360) float32."""
        self._log(f"  Loading MSL at {dt}")
        if self._zarr_msl is None:
            self._log("    Opening Zarr store (first call)...")
            self._zarr_msl = xr.open_zarr(ZARR_MSL)
        arr    = np.array(self._zarr_msl['MSL'].sel(time=dt, method='nearest').values)
        assert arr.shape == (721, 1440), f"Unexpected MSL shape: {arr.shape}"
        result = self._regrid_0p25_to_1deg(arr).astype(np.float32)
        self._log(f"    MSL range: {result.min():.2f} - {result.max():.2f} Pa")
        return result

    # ── 10m Winds ─────────────────────────────────────────────────────────────

    def load_10u(self, dt: datetime) -> np.ndarray:
        """Load 10m U wind at dt, regrid to 1deg. Returns (181,360) float32."""
        self._log(f"  Loading 10U at {dt}")
        if self._zarr_10u is None:
            self._log("    Opening Zarr store (first call)...")
            self._zarr_10u = xr.open_zarr(ZARR_10U)
        arr    = np.array(self._zarr_10u['VAR_10U'].sel(time=dt, method='nearest').values)
        assert arr.shape == (721, 1440), f"Unexpected 10U shape: {arr.shape}"
        result = self._regrid_0p25_to_1deg(arr).astype(np.float32)
        self._log(f"    10U range: {result.min():.2f} - {result.max():.2f} m/s")
        return result

    def load_10v(self, dt: datetime) -> np.ndarray:
        """Load 10m V wind at dt, regrid to 1deg. Returns (181,360) float32."""
        self._log(f"  Loading 10V at {dt}")
        if self._zarr_10v is None:
            self._log("    Opening Zarr store (first call)...")
            self._zarr_10v = xr.open_zarr(ZARR_10V)
        arr    = np.array(self._zarr_10v['VAR_10V'].sel(time=dt, method='nearest').values)
        assert arr.shape == (721, 1440), f"Unexpected 10V shape: {arr.shape}"
        result = self._regrid_0p25_to_1deg(arr).astype(np.float32)
        self._log(f"    10V range: {result.min():.2f} - {result.max():.2f} m/s")
        return result

    # ── Total Precipitation ────────────────────────────────────────────────────

    def load_tp(self, dt: datetime) -> np.ndarray:
        """
        Load daily total precipitation (LSP + CP) for the calendar date of dt.
        Sums forecast hours from 06Z and 18Z init times covering the date.
        Returns (181,360) float32 in meters.
        """
        date = dt.date()
        self._log(f"  Loading TP for {date}")
        lsp    = self._load_accumu_var('128_142_lsp', 'LSP', date)
        cp     = self._load_accumu_var('128_143_cp',  'CP',  date)
        tp     = lsp + cp
        result = self._regrid_0p25_to_1deg(tp).astype(np.float32)
        self._log(f"    TP range: {result.min():.6f} - {result.max():.6f} m")
        return result

    def _load_accumu_var(self, var_code, var_name, date) -> np.ndarray:
        """
        Sum one accumulation variable across all forecast hours for a calendar date.
        Finds the 06Z and 18Z init times on date, sums their 12 forecast hours each.
        Returns (721,1440) float32.
        """
        yyyymm   = date.strftime('%Y%m')
        pattern  = f"{ACCUMU_DIR}/{yyyymm}/*{var_code}*.nc"
        files    = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No accumu files found: {pattern}")

        date_int         = int(date.strftime('%Y%m%d'))
        daily_total      = None
        init_times_found = []

        for fpath in files:
            ds        = nc.Dataset(fpath)
            utc_dates = ds.variables['utc_date'][:]
            data      = ds.variables[var_name][:]

            for i, utcd in enumerate(utc_dates):
                utcd_int  = int(utcd)
                init_date = utcd_int // 100
                init_hour = utcd_int % 100

                if init_date == date_int and init_hour in (6, 18):
                    chunk       = data[i].sum(axis=0).astype(np.float32)
                    daily_total = chunk if daily_total is None else daily_total + chunk
                    init_times_found.append(f"{init_date}_{init_hour:02d}Z")

            ds.close()

        if daily_total is None:
            raise ValueError(
                f"No 06Z or 18Z init times found for {date} in {var_code}.\n"
                f"Files checked: {[os.path.basename(f) for f in files]}"
            )

        self._log(f"    {var_name}: summed init times {init_times_found}")
        return daily_total

    # ── Upper-air Pressure Levels ──────────────────────────────────────────────

    def load_upper_air(self, dt: datetime) -> dict:
        """
        Load Z, T, U, V, Q, W at GraphCast's 13 pressure levels for dt.
        Returns dict of float32 arrays each shape (13, 181, 360).
        """
        self._log(f"  Loading upper-air at {dt}")
        yyyymm   = dt.strftime('%Y%m')
        date_str = dt.strftime('%Y%m%d')

        var_specs = {
            'z': ('128_129_z', 'Z', 'll025sc'),
            't': ('128_130_t', 'T', 'll025sc'),
            'u': ('128_131_u', 'U', 'll025uv'),
            'v': ('128_132_v', 'V', 'll025uv'),
            'q': ('128_133_q', 'Q', 'll025sc'),
            'w': ('128_135_w', 'W', 'll025sc'),
        }

        out = {}
        for key, (code, varname, suffix) in var_specs.items():
            pattern = f"{PL_DIR}/{yyyymm}/*{code}*{suffix}*{date_str}*.nc"
            files   = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"No upper-air file for {key} on {date_str}: {pattern}"
                )

            fpath  = files[0]
            self._log(f"    {key}: {os.path.basename(fpath)}")

            ds         = nc.Dataset(fpath)
            levels     = tuple(ds.variables['level'][:].tolist())
            times      = ds.variables['time'][:]
            data       = ds.variables[varname][:]
            time_idx   = self._find_time_index(times, dt)
            level_idxs = self._get_level_indices(levels)
            arr        = data[time_idx][level_idxs]
            ds.close()

            out[key] = np.stack(
                [self._regrid_0p25_to_1deg(arr[k]).astype(np.float32)
                 for k in range(len(GC_LEVELS))],
                axis=0
            )  # (13, 181, 360)
            self._log(f"      {key} shape={out[key].shape}  "
                      f"range=[{out[key].min():.3f}, {out[key].max():.3f}]")

        return out

    # ── Regridding ─────────────────────────────────────────────────────────────

    def _regrid_0p25_to_1deg(self, arr: np.ndarray) -> np.ndarray:
        """
        Regrid (721, 1440) 0.25deg to (181, 360) 1deg by subsampling at exact
        coincident grid nodes (every 4th point). Returns (181, 360).
        """
        assert arr.shape == (721, 1440), f"Expected (721,1440), got {arr.shape}"
        return arr[::4, ::4]

    # ── Helpers ────────────────────────────────────────────────────────────────

    @lru_cache(maxsize=None)
    def _get_level_indices(self, levels_tuple: tuple) -> list:
        """Return indices into ERA5's 37-level array for GraphCast's 13 levels."""
        levels  = np.array(levels_tuple)
        indices = []
        for gc_lev in GC_LEVELS:
            matches = np.where(levels == gc_lev)[0]
            if len(matches) == 0:
                raise ValueError(
                    f"Level {gc_lev} hPa not in ERA5 levels: {list(levels_tuple)}"
                )
            indices.append(int(matches[0]))
        return indices

    def _find_time_index(self, times_hrs, dt: datetime) -> int:
        """Find index in hours-since-1900 time array matching dt."""
        origin = datetime(1900, 1, 1)
        target = (dt - origin).total_seconds() / 3600.0
        diffs  = np.abs(np.array(times_hrs) - target)
        idx    = int(np.argmin(diffs))
        if diffs[idx] > 1.0:
            raise ValueError(
                f"Nearest time is {diffs[idx]:.1f}h from {dt} — "
                f"file may not cover this dt"
            )
        return idx

    def _log(self, msg):
        if self.verbose:
            print(msg)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import traceback

    print("=" * 60)
    print("ERA5Loader quick test — single snapshot: 1979-03-15 00Z")
    print("=" * 60)

    loader  = ERA5Loader(verbose=True)
    dt_test = datetime(1979, 3, 15, 0)

    try:
        data = loader.load_snapshot(dt_test)
        print("\n── Results ──────────────────────────────────────")
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(f"  {k:8s}: shape={str(v.shape):16s} "
                      f"dtype={v.dtype}  [{v.min():.4f}, {v.max():.4f}]")
            else:
                print(f"  {k:8s}: {v}")
        print("\nPASSED")
    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
