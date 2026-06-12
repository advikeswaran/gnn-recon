import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RECON_DIR = Path('/glade/derecho/scratch/advike/graphcast_recon')
TGT_DIR   = RECON_DIR / 'cache' / 'era5_targets'
RECON_OUT = RECON_DIR / 'cache' / 'recon'

clim_mean = np.load(TGT_DIR/'clim_mean_1979_2000.npy')
anom_std  = np.load(TGT_DIR/'anom_std_1979_2000.npz')
t_std     = float(anom_std['temp_anom_std'])

# load reconstruction
years = list(range(1801, 2001))
t_mean = []
p_mean = []
for yr in years:
    r = np.load(str(RECON_OUT/f'recon_{yr}.npy'))
    t_mean.append(r[:,0].mean())
    p_mean.append(r[:,1].mean())
t_mean = np.array(t_mean)
p_mean = np.array(p_mean)
t_anom = t_mean - clim_mean[:,0].mean()

# load ERA5 for overlap
era5_years = list(range(1979, 2001))
era5_t = []
for yr in era5_years:
    e = np.load(str(TGT_DIR/f'targets_{yr}.npy'))
    era5_t.append(e[:,0].mean())
era5_t = np.array(era5_t)
era5_anom = era5_t - clim_mean[:,0].mean()

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# T anomaly time series
ax = axes[0]
ax.plot(years, t_anom, 'b-', lw=1.2, label='Reconstruction')
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Year'); ax.set_ylabel('T anomaly (K)')
ax.set_title('Antarctic Mean Temperature Anomaly 1801-2000')
ax.legend()

# overlap comparison
ax = axes[1]
overlap_idx = [i for i,y in enumerate(years) if y in era5_years]
recon_overlap = t_anom[overlap_idx]
ax.plot(era5_years, era5_anom, 'r-', lw=1.5, label='ERA5')
ax.plot(era5_years, recon_overlap, 'b-', lw=1.5, label='Reconstruction')
r = float(np.corrcoef(era5_anom, recon_overlap)[0,1])
rmse = float(np.sqrt(np.mean((era5_anom - recon_overlap)**2)))
ax.set_title(f'Overlap 1979-2000: r={r:.3f}, RMSE={rmse:.3f}K')
ax.set_xlabel('Year'); ax.set_ylabel('T anomaly (K)')
ax.legend()

# P time series
ax = axes[2]
p_anom = p_mean - clim_mean[:,1].mean()
ax.plot(years, p_anom, 'g-', lw=1.2)
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Year'); ax.set_ylabel('P anomaly (m/yr)')
ax.set_title('Antarctic Mean Precipitation Anomaly 1801-2000')

plt.tight_layout()
plt.savefig(str(RECON_DIR/'cache'/'recon'/'timeseries.png'), dpi=150)
print(f"Saved timeseries.png")
print(f"Overlap r={r:.3f}  RMSE={rmse:.3f}K")
print(f"T anom range: {t_anom.min():.3f} to {t_anom.max():.3f} K")
