"""
plot_trends.py — Trend maps of reconstructed Antarctic T and P
Periods: 1801-2000, 1901-2000, 1957-2000
Units: K/decade (temperature), mm/yr/decade (precipitation)
Run in PlotEnv: source /glade/u/home/advike/PlotEnv/bin/activate
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

RECON_DIR  = "/glade/derecho/scratch/advike/graphcast_recon"
CACHE_DIR  = os.path.join(RECON_DIR, "cache")
RECON_DIR_ = os.path.join(CACHE_DIR, "recon")
PLOTS_DIR  = os.path.join(CACHE_DIR, "plots")

LATS = list(range(-60, -91, -1))
LONS = list(range(0, 360))

PERIODS = [
    (1801, 2000, "1801–2000"),
    (1901, 2000, "1901–2000"),
    (1957, 2000, "1957–2000"),
]

# Fixed physically sensible color limits
T_LIM = 0.5    # K/decade
P_LIM = 15.0   # mm/yr/decade

os.makedirs(PLOTS_DIR, exist_ok=True)


def load_recon(start, end):
    years = list(range(start, end + 1))
    data  = []
    for yr in years:
        path = os.path.join(RECON_DIR_, f"recon_{yr}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        data.append(np.load(path))
    return np.stack(data), np.array(years)


def ols_trend(years, data):
    """OLS trend per grid node. Returns trend in units/year, shape (11160,)"""
    x  = years.astype(np.float64)
    x  = x - x.mean()
    ss = (x ** 2).sum()
    y  = data.astype(np.float64)
    return ((x[:, None] * y).sum(axis=0) / ss).astype(np.float32)


def to_grid(vec):
    return vec.reshape(len(LATS), len(LONS))


def polar_panel(ax, data_grid, lons, lats, cmap, vmin, vmax, title, unit):
    DATA_CRS = ccrs.PlateCarree()
    ax.set_extent([-180, 180, -90, -60], crs=DATA_CRS)
    ax.add_feature(cfeature.LAND,  facecolor="0.85", zorder=2)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=1)
    ax.coastlines(resolution="110m", linewidth=0.6, zorder=3)
    gl = ax.gridlines(crs=DATA_CRS, draw_labels=False,
                      linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
    gl.ylocator = mticker.FixedLocator([-90, -80, -70, -60])
    lon2d, lat2d = np.meshgrid(lons, lats)
    cf = ax.pcolormesh(lon2d, lat2d, data_grid, transform=DATA_CRS,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="auto", zorder=2)
    plt.colorbar(cf, ax=ax, orientation="horizontal",
                 pad=0.04, fraction=0.046, label=unit, extend="both")
    ax.set_title(title, fontsize=9, pad=4)


def main():
    lats_1d = np.array(LATS, dtype=np.float32)
    lons_1d = np.array(LONS, dtype=np.float32)

    print("Computing trends...")
    temp_trends = []
    prec_trends = []

    for start, end, label in PERIODS:
        print(f"  {label} ...")
        data, years = load_recon(start, end)
        # K/decade
        t_trend = ols_trend(years, data[:, :, 0]) * 10
        # mm/yr/decade (convert from m/yr by *1000, then *10 for per decade)
        p_trend = ols_trend(years, data[:, :, 1]) * 10 * 1000
        temp_trends.append(t_trend)
        prec_trends.append(p_trend)
        print(f"    T trend: mean={t_trend.mean():.4f} K/dec "
              f"min={t_trend.min():.4f} max={t_trend.max():.4f}")
        print(f"    P trend: mean={p_trend.mean():.2f} mm/yr/dec "
              f"min={p_trend.min():.2f} max={p_trend.max():.2f}")

    PROJ = ccrs.SouthPolarStereo()
    fig, axes = plt.subplots(
        3, 2, figsize=(11, 14),
        subplot_kw={"projection": PROJ},
    )
    fig.suptitle(
        "Antarctic Reconstruction — Linear Trends\n"
        "GNN reconstruction from ice core proxies",
        fontsize=12,
    )

    for row, (start, end, label) in enumerate(PERIODS):
        polar_panel(
            axes[row, 0],
            to_grid(temp_trends[row]), lons_1d, lats_1d,
            "RdBu_r", -T_LIM, T_LIM,
            f"T2m trend {label}", "K/decade",
        )
        polar_panel(
            axes[row, 1],
            to_grid(prec_trends[row]), lons_1d, lats_1d,
            "BrBG", -P_LIM, P_LIM,
            f"Precip trend {label}", "mm/yr/decade",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(PLOTS_DIR, "trend_maps.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
