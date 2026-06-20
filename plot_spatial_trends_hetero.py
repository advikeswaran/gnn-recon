"""
plot_spatial_trends_hetero.py -- Spatial trend maps for heteroscedastic model reconstruction.
Reads from cache/recon_hetero/. Outputs to cache/plots/hetero/
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from scipy import stats

RECON_DIR = Path('/glade/derecho/scratch/advike/graphcast_recon')
TGT_DIR   = RECON_DIR / 'cache' / 'era5_targets'
PLOT_DIR  = RECON_DIR / 'cache' / 'plots' / 'hetero'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
clim_mean = np.load(TGT_DIR/'clim_mean_1979_2000.npy')

LAT_VALS = np.arange(-60, -91, -1, dtype=np.float64)
LON_VALS = np.arange(0, 360, 1, dtype=np.float64)
LONS, LATS = np.meshgrid(LON_VALS, LAT_VALS)
LONS_180 = np.where(LONS > 180, LONS - 360, LONS)

periods = [
    (1801, 2000, '1801-2000'),
    (1901, 2000, '1901-2000'),
    (1957, 2000, '1957-2000'),
    (1979, 2000, '1979-2000'),
]

print("Loading reconstruction...")
recon_years = np.array(range(1801, 2001))
T_all = np.zeros((len(recon_years), 11160), dtype=np.float32)
P_all = np.zeros((len(recon_years), 11160), dtype=np.float32)
for yi, yr in enumerate(recon_years):
    r = np.load(str(RECON_DIR/'cache'/'recon_hetero'/f'recon_{yr}.npy'))
    T_all[yi] = r[:,0] - clim_mean[:,0]
    P_all[yi] = (r[:,1] - clim_mean[:,1]) * 1000

def compute_trend_map(years_arr, data_arr, y1, y2):
    mask = (years_arr >= y1) & (years_arr <= y2)
    yrs  = years_arr[mask].astype(float)
    dat  = data_arr[mask]
    trends = np.zeros(11160, dtype=np.float32)
    for i in range(11160):
        sl, _, _, _, _ = stats.linregress(yrs, dat[:,i])
        trends[i] = sl * 10
    return trends.reshape(31, 360)

proj = ccrs.SouthPolarStereo()
fig, axes = plt.subplots(2, 4, figsize=(20, 12),
                          subplot_kw={'projection': proj})
fig.suptitle('Antarctic Climate Reconstruction — Spatial Trend Maps\n(Kernel GNN — Heteroscedastic Model)',
             fontsize=13, fontweight='bold')

for col, (y1, y2, label) in enumerate(periods):
    print(f"Computing trends {label}...")
    T_trend = compute_trend_map(recon_years, T_all, y1, y2)
    P_trend = compute_trend_map(recon_years, P_all, y1, y2)
    for row, (trend, cmap, vmin, vmax, unit) in enumerate([
        (T_trend, 'RdBu_r', -1.0,  1.0,  'K/dec'),
        (P_trend, 'BrBG',  -20.0, 20.0,  'mm/yr/dec'),
    ]):
        ax = axes[row, col]
        ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
        ax.gridlines(linewidth=0.3, color='gray', alpha=0.5)
        im = ax.pcolormesh(LONS_180, LATS, trend,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=4)
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label=unit, shrink=0.85)
        varname = 'T' if row==0 else 'SMB'
        ax.set_title(f'{varname} trend {label}', fontsize=10)

plt.tight_layout()
plt.savefig(str(PLOT_DIR/'spatial_trends_hetero.png'), dpi=150, bbox_inches='tight')
print('Saved spatial_trends_hetero.png')
