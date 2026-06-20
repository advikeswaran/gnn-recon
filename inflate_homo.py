"""
inflate_homo.py -- Apply per-node variance inflation to recon_homo/.
Computes inflation_factor[i] = std(ERA5_anom[1979-2000, i]) / std(recon_anom[1979-2000, i])
for T and P separately, then scales all 200 years of anomalies by that factor.
Output: cache/recon_homo_inflated/
"""
import numpy as np
from pathlib import Path

RECON_DIR = Path('/glade/derecho/scratch/advike/graphcast_recon')
TGT_DIR   = RECON_DIR / 'cache' / 'era5_targets'
IN_DIR    = RECON_DIR / 'cache' / 'recon_homo'
OUT_DIR   = RECON_DIR / 'cache' / 'recon_homo_inflated'
OUT_DIR.mkdir(parents=True, exist_ok=True)

clim_mean = np.load(TGT_DIR / 'clim_mean_1979_2000.npy')  # (11160, 2)
overlap_years = list(range(1979, 2001))
recon_years   = list(range(1801, 2001))

print("Loading overlap years (1979-2000)...")
era5_anom  = np.zeros((len(overlap_years), 11160, 2), dtype=np.float32)
recon_anom = np.zeros((len(overlap_years), 11160, 2), dtype=np.float32)

for yi, yr in enumerate(overlap_years):
    era5  = np.load(TGT_DIR / f'targets_{yr}.npy')
    recon = np.load(IN_DIR   / f'recon_{yr}.npy')
    era5_anom[yi]  = era5  - clim_mean
    recon_anom[yi] = recon - clim_mean

# per-node std over 1979-2000
era5_std  = era5_anom.std(axis=0)   # (11160, 2)
recon_std = recon_anom.std(axis=0)  # (11160, 2)

# inflation factor, floored to avoid div-by-zero
MIN_STD = 1e-6
inflation = era5_std / np.maximum(recon_std, MIN_STD)  # (11160, 2)

# cap extreme factors (e.g. coastal nodes with near-zero recon variance)
MAX_FACTOR = 10.0
inflation = np.clip(inflation, 0.0, MAX_FACTOR)

print(f"Inflation factor T: mean={inflation[:,0].mean():.2f} median={np.median(inflation[:,0]):.2f} "
      f"min={inflation[:,0].min():.2f} max={inflation[:,0].max():.2f}")
print(f"Inflation factor P: mean={inflation[:,1].mean():.2f} median={np.median(inflation[:,1]):.2f} "
      f"min={inflation[:,1].min():.2f} max={inflation[:,1].max():.2f}")

np.save(str(OUT_DIR / 'inflation_factors.npy'), inflation)

print(f"Applying inflation to {len(recon_years)} years...")
for yr in recon_years:
    recon = np.load(IN_DIR / f'recon_{yr}.npy')   # (11160, 2)
    anom  = recon - clim_mean
    anom_inflated = anom * inflation               # broadcast over nodes
    out   = (anom_inflated + clim_mean).astype(np.float32)
    np.save(str(OUT_DIR / f'recon_{yr}.npy'), out)

print(f"Done. Saved to {OUT_DIR}")

# quick sanity check on overlap
print("\nSanity check (mean T anom over grounded ice, 1979-2000):")
mask = np.load(str(RECON_DIR / 'cache' / 'grounded_ice_mask.npy'))
for yr in [1979, 1990, 2000]:
    era5  = np.load(TGT_DIR / f'targets_{yr}.npy')
    infl  = np.load(OUT_DIR  / f'recon_{yr}.npy')
    orig  = np.load(IN_DIR   / f'recon_{yr}.npy')
    e_a = (era5[mask,0]  - clim_mean[mask,0]).mean()
    o_a = (orig[mask,0]  - clim_mean[mask,0]).mean()
    i_a = (infl[mask,0]  - clim_mean[mask,0]).mean()
    print(f"  {yr}: ERA5={e_a:+.3f}K  orig={o_a:+.3f}K  inflated={i_a:+.3f}K")
