"""
plot_summary.py -- Generate summary plots for Antarctic reconstruction.
Reads from cache/recon_corrected/ (T variance inflation + P offset corrected).
Outputs to cache/recon/plots/
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

RECON_DIR = Path('/glade/derecho/scratch/advike/graphcast_recon')
TGT_DIR   = RECON_DIR / 'cache' / 'era5_targets'
PLOT_DIR  = RECON_DIR / 'cache' / 'recon' / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
clim_mean = np.load(TGT_DIR/'clim_mean_1979_2000.npy')
mask      = np.load(str(RECON_DIR/'cache'/'grounded_ice_mask.npy'))

LAT_VALS = np.arange(-60, -91, -1, dtype=np.float64)
LON_VALS = np.arange(0, 360, 1, dtype=np.float64)
R_EARTH  = 6.371e6
cell_areas = np.array([R_EARTH**2 * np.cos(np.radians(lat)) * np.radians(1.) * np.radians(1.)
                        for lat in LAT_VALS for _ in LON_VALS])
water_density = 1000.0
mm_to_Gt = cell_areas[mask].sum() * 0.001 * water_density / 1e12

recon_years = np.array(range(1801, 2001))
t_anom = []; p_Gt = []
for yr in recon_years:
    r = np.load(str(RECON_DIR/'cache'/'recon_corrected'/f'recon_{yr}.npy'))
    t_anom.append((r[mask,0] - clim_mean[mask,0]).mean())
    p_Gt.append(((r[mask,1] - clim_mean[mask,1]) * cell_areas[mask] * water_density / 1e12).sum())
t_anom = np.array(t_anom)
p_Gt   = np.array(p_Gt)

era5_years = np.array(range(1979, 2001))
era5_t = []; era5_p_Gt = []
for yr in era5_years:
    e = np.load(str(TGT_DIR/f'targets_{yr}.npy'))
    era5_t.append((e[mask,0] - clim_mean[mask,0]).mean())
    era5_p_Gt.append(((e[mask,1] - clim_mean[mask,1]) * cell_areas[mask] * water_density / 1e12).sum())
era5_t    = np.array(era5_t)
era5_p_Gt = np.array(era5_p_Gt)

overlap_mask = (recon_years >= 1979) & (recon_years <= 2000)
r_T    = float(np.corrcoef(era5_t, t_anom[overlap_mask])[0,1])
r_P    = float(np.corrcoef(era5_p_Gt, p_Gt[overlap_mask])[0,1])
rmse_T = float(np.sqrt(np.mean((era5_t - t_anom[overlap_mask])**2)))

def smooth11(x):
    s = np.convolve(x, np.ones(11)/11, mode='same')
    s[:5] = np.nan; s[-5:] = np.nan
    return s

def trend(yrs, vals):
    sl, ic, r, p, _ = stats.linregress(yrs.astype(float), vals.astype(float))
    pstar = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '(ns)'
    return sl, ic, pstar

periods = [
    (1801, 2000, 'red',    '1801-2000'),
    (1901, 2000, 'orange', '1901-2000'),
    (1957, 2000, 'green',  '1957-2000'),
    (1979, 2000, 'purple', '1979-2000'),
]

fig = plt.figure(figsize=(16, 14))
fig.suptitle('Antarctic Grounded Ice Sheet Climate Reconstruction 1801-2000\n(Kernel GNN + Variance Inflation + P Offset Correction)',
             fontsize=13, fontweight='bold', y=0.98)

ax1 = fig.add_subplot(3, 2, (1,2))
ax1.fill_between(recon_years, t_anom, alpha=0.2, color='steelblue')
ax1.plot(recon_years, t_anom, color='steelblue', lw=0.8, alpha=0.5, label='Reconstruction')
ax1.plot(recon_years, smooth11(t_anom), color='steelblue', lw=2, zorder=5)
ax1.plot(era5_years, era5_t, color='red', lw=1.5, alpha=0.8, label='ERA5 1979-2000')
for y1,y2,col,label in periods:
    m = (recon_years>=y1)&(recon_years<=y2)
    sl,ic,ps = trend(recon_years[m], t_anom[m])
    ax1.plot(recon_years[m], sl*recon_years[m]+ic, '--', color=col, lw=1.8,
            label=f'Recon {label}: {sl*10:+.3f}K/dec {ps}')
sl,ic,ps = trend(era5_years, era5_t)
ax1.plot(era5_years, sl*era5_years+ic, '-', color='darkred', lw=2.5,
        label=f'ERA5 trend: {sl*10:+.3f}K/dec {ps}')
ax1.axhline(0, color='k', lw=0.5, ls=':')
ax1.set_ylabel('T anomaly (K)', fontsize=11)
ax1.set_title('Grounded Ice Sheet Mean Temperature Anomaly (rel. 1979-2000)', fontsize=11)
ax1.legend(fontsize=8, loc='upper left', ncol=2); ax1.set_xlim(1801, 2000)

ax2 = fig.add_subplot(3, 2, (3,4))
ax2.fill_between(recon_years, p_Gt, alpha=0.2, color='forestgreen')
ax2.plot(recon_years, p_Gt, color='forestgreen', lw=0.8, alpha=0.5, label='Reconstruction')
ax2.plot(recon_years, smooth11(p_Gt), color='forestgreen', lw=2, zorder=5)
ax2.plot(era5_years, era5_p_Gt, color='red', lw=1.5, alpha=0.8, label='ERA5 1979-2000')
for y1,y2,col,label in periods:
    m = (recon_years>=y1)&(recon_years<=y2)
    sl,ic,ps = trend(recon_years[m], p_Gt[m])
    ax2.plot(recon_years[m], sl*recon_years[m]+ic, '--', color=col, lw=1.8,
            label=f'Recon {label}: {sl:+.1f}Gt/yr/yr {ps}')
sl,ic,ps = trend(era5_years, era5_p_Gt)
ax2.plot(era5_years, sl*era5_years+ic, '-', color='darkred', lw=2.5,
        label=f'ERA5 trend: {sl:+.1f}Gt/yr/yr {ps}')
ax2.axhline(0, color='k', lw=0.5, ls=':')
ax2.set_ylabel('SMB anomaly (Gt/yr)', fontsize=11)
ax2.set_title('Grounded Ice Sheet SMB Anomaly (rel. 1979-2000)', fontsize=11)
ax2.legend(fontsize=8, loc='upper left', ncol=2); ax2.set_xlim(1801, 2000)
ax2b = ax2.twinx()
ax2b.set_ylim(ax2.get_ylim()[0]/mm_to_Gt, ax2.get_ylim()[1]/mm_to_Gt)
ax2b.set_ylabel('mm w.e./yr', fontsize=10, color='gray')
ax2b.tick_params(axis='y', colors='gray')

ax3 = fig.add_subplot(3, 2, 5)
ax3.plot(era5_years, era5_t, 'r-o', lw=1.5, ms=4, label='ERA5')
ax3.plot(recon_years[overlap_mask], t_anom[overlap_mask], 'b-o', lw=1.5, ms=4, label='Reconstruction')
ax3.axhline(0, color='k', lw=0.5, ls=':')
ax3.set_title(f'T Overlap 1979-2000\nr={r_T:.3f}, RMSE={rmse_T:.3f}K', fontsize=11)
ax3.set_ylabel('T anomaly (K)', fontsize=11); ax3.legend(fontsize=9)

ax4 = fig.add_subplot(3, 2, 6)
ax4.plot(era5_years, era5_p_Gt, 'r-o', lw=1.5, ms=4, label='ERA5')
ax4.plot(recon_years[overlap_mask], p_Gt[overlap_mask], 'g-o', lw=1.5, ms=4, label='Reconstruction')
ax4.axhline(0, color='k', lw=0.5, ls=':')
ax4.set_title(f'SMB Overlap 1979-2000\nr={r_P:.3f}', fontsize=11)
ax4.set_ylabel('SMB anomaly (Gt/yr)', fontsize=11); ax4.legend(fontsize=9)

plt.tight_layout()
plt.savefig(str(PLOT_DIR/'summary_corrected.png'), dpi=150, bbox_inches='tight')
print(f'Saved summary_corrected.png')
print(f'T overlap: r={r_T:.3f} RMSE={rmse_T:.3f}K')
print(f'P overlap: r={r_P:.3f}')
print('\nT trends:')
for y1,y2,col,label in periods:
    m = (recon_years>=y1)&(recon_years<=y2)
    sl,ic,ps = trend(recon_years[m], t_anom[m])
    print(f'  {label}: {sl*10:+.3f}K/dec {ps}')
print('\nSMB trends:')
for y1,y2,col,label in periods:
    m = (recon_years>=y1)&(recon_years<=y2)
    sl,ic,ps = trend(recon_years[m], p_Gt[m])
    print(f'  {label}: {sl:+.2f}Gt/yr/yr {ps}')
