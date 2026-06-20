"""
inflate_homo_T_only.py -- Apply per-node variance inflation to T only.
P is taken directly from recon_homo/ unchanged.
Output: cache/recon_homo_T_inflated/
"""
import numpy as np
from pathlib import Path

RECON_DIR = Path('/glade/derecho/scratch/advike/graphcast_recon')
TGT_DIR   = RECON_DIR / 'cache' / 'era5_targets'
IN_DIR    = RECON_DIR / 'cache' / 'recon_homo'
OUT_DIR   = RECON_DIR / 'cache' / 'recon_homo_T_inflated'
OUT_DIR.mkdir(parents=True, exist_ok=True)

clim_mean     = np.load(TGT_DIR / 'clim_mean_1979_2000.npy')  # (11160, 2)
overlap_years = list(range(1979, 2001))
recon_years   = list(range(1801, 2001))

print("Loading overlap years (1979-2000) for T inflation factor...")
era5_T_anom  = np.zeros((len(overlap_years), 11160), dtype=np.float32)
recon_T_anom = np.zeros((len(overlap_years), 11160), dtype=np.float32)

for yi, yr in enumerate(overlap_years):
    era5  = np.load(TGT_DIR / f'targets_{yr}.npy')
    recon = np.load(IN_DIR   / f'recon_{yr}.npy')
    era5_T_anom[yi]  = era5[:,0]  - clim_mean[:,0]
    recon_T_anom[yi] = recon[:,0] - clim_mean[:,0]

era5_T_std  = era5_T_anom.std(axis=0)   # (11160,)
recon_T_std = recon_T_anom.std(axis=0)  # (11160,)

MIN_STD    = 1e-6
MAX_FACTOR = 10.0
inflation_T = np.clip(era5_T_std / np.maximum(recon_T_std, MIN_STD), 0.0, MAX_FACTOR)

print(f"T inflation factor: mean={inflation_T.mean():.2f} median={np.median(inflation_T):.2f} "
      f"min={inflation_T.min():.2f} max={inflation_T.max():.2f}")

np.save(str(OUT_DIR / 'inflation_T_factors.npy'), inflation_T)

print(f"Applying T inflation + copying P unchanged for {len(recon_years)} years...")
for yr in recon_years:
    recon = np.load(IN_DIR / f'recon_{yr}.npy')  # (11160, 2)
    T_anom_inflated = (recon[:,0] - clim_mean[:,0]) * inflation_T + clim_mean[:,0]
    P_nonneg = np.maximum(recon[:,1], 0.0)
    out = np.stack([T_anom_inflated, P_nonneg], axis=1).astype(np.float32)
    np.save(str(OUT_DIR / f'recon_{yr}.npy'), out)

print(f"Done. Saved to {OUT_DIR}")

print("\nSanity check (grounded ice mean anomalies, selected years):")
mask = np.load(str(RECON_DIR / 'cache' / 'grounded_ice_mask.npy'))
for yr in [1979, 1990, 2000]:
    era5  = np.load(TGT_DIR / f'targets_{yr}.npy')
    orig  = np.load(IN_DIR   / f'recon_{yr}.npy')
    out   = np.load(OUT_DIR  / f'recon_{yr}.npy')
    e_T = (era5[mask,0]  - clim_mean[mask,0]).mean()
    o_T = (orig[mask,0]  - clim_mean[mask,0]).mean()
    i_T = (out[mask,0]   - clim_mean[mask,0]).mean()
    e_P = (era5[mask,1]  - clim_mean[mask,1]).mean()
    o_P =  orig[mask,1].mean()
    i_P =  out[mask,1].mean()
    print(f"  {yr}: T  ERA5={e_T:+.3f}K  orig={o_T:+.3f}K  inflated={i_T:+.3f}K")
    print(f"       P  ERA5={e_P:+.4f}m/yr  orig={o_P:.4f}m/yr  out={i_P:.4f}m/yr  (P unchanged)")
