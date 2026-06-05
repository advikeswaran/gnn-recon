"""
plot_timeseries.py — Area-weighted domain-mean time series of reconstructed T and P 1801-2000
Run in PlotEnv: source /glade/u/home/advike/PlotEnv/bin/activate
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RECON_DIR  = "/glade/derecho/scratch/advike/graphcast_recon"
CACHE_DIR  = os.path.join(RECON_DIR, "cache")
RECON_DIR_ = os.path.join(CACHE_DIR, "recon")
PLOTS_DIR  = os.path.join(CACHE_DIR, "plots")

YEARS = list(range(1801, 2001))
LATS  = list(range(-60, -91, -1))   # -60 … -90  (31 values)

os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    # area weights: cos(lat) per grid node, shape (11160,)
    # each lat row has 360 lon cells all with the same weight
    weights = np.cos(np.radians(np.array(LATS, dtype=np.float64))).repeat(360)
    weights /= weights.sum()   # normalise so weights sum to 1

    print("Loading reconstruction...")
    temp_mean = []
    prec_mean = []

    for yr in YEARS:
        path = os.path.join(RECON_DIR_, f"recon_{yr}.npy")
        data = np.load(path)   # (11160, 2)
        temp_mean.append(float(np.sum(data[:, 0] * weights)))
        prec_mean.append(float(np.sum(data[:, 1] * weights) * 1000))  # m/yr -> mm/yr

    temp_mean = np.array(temp_mean)
    prec_mean = np.array(prec_mean)
    years     = np.array(YEARS)

    print(f"T: mean={temp_mean.mean():.3f}K  std={temp_mean.std():.4f}K  "
          f"min={temp_mean.min():.3f}  max={temp_mean.max():.3f}")
    print(f"P: mean={prec_mean.mean():.2f}mm/yr  std={prec_mean.std():.4f}  "
          f"min={prec_mean.min():.2f}  max={prec_mean.max():.2f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        "Antarctic Reconstruction — Area-weighted Domain-mean Time Series 1801–2000\n"
        "GNN reconstruction from ice core proxies",
        fontsize=11,
    )

    ax1.plot(years, temp_mean, linewidth=0.8, color="tomato", alpha=0.7)
    ax1.axhline(temp_mean.mean(), color="darkred", linewidth=1.2,
                linestyle="--", label=f"mean={temp_mean.mean():.3f}K")
    ax1.set_ylabel("T2m (K)", fontsize=9)
    ax1.set_title("Area-weighted Domain-mean 2m Temperature", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, linewidth=0.4, alpha=0.5)
    ax1.tick_params(labelsize=8)

    ax2.plot(years, prec_mean, linewidth=0.8, color="steelblue", alpha=0.7)
    ax2.axhline(prec_mean.mean(), color="navy", linewidth=1.2,
                linestyle="--", label=f"mean={prec_mean.mean():.2f}mm/yr")
    ax2.set_ylabel("Precipitation (mm/yr)", fontsize=9)
    ax2.set_title("Area-weighted Domain-mean Total Precipitation", fontsize=9)
    ax2.set_xlabel("Year", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, linewidth=0.4, alpha=0.5)
    ax2.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(PLOTS_DIR, "timeseries.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
