"""
plot_icecores.py — Plot raw calibrated ice core time series 1801-2000
Shows domain-mean and individual site anomalies relative to 1979-2000 mean.
Run in PlotEnv: source /glade/u/home/advike/PlotEnv/bin/activate
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RECON_DIR  = "/glade/derecho/scratch/advike/graphcast_recon"
CACHE_DIR  = os.path.join(RECON_DIR, "cache")
CALIB_DIR  = os.path.join(CACHE_DIR, "calibration")
PLOTS_DIR  = os.path.join(CACHE_DIR, "plots")

OVERLAP_START_IDX = 178   # index of 1979
OVERLAP_END_IDX   = 200   # exclusive

os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    # load calibrated data
    calib_iso   = np.load(os.path.join(CALIB_DIR, "calibrated_iso.npy"))    # (80, 200)
    calib_accum = np.load(os.path.join(CALIB_DIR, "calibrated_accum.npy"))  # (84, 200)
    meta        = np.load(os.path.join(CALIB_DIR, "calibration_meta.npz"))
    years       = meta["recon_years"]   # 1801-2000

    # compute per-site 1979-2000 means
    overlap = slice(OVERLAP_START_IDX, OVERLAP_END_IDX)
    iso_means   = np.nanmean(calib_iso[:,   overlap], axis=1, keepdims=True)  # (80,1)
    accum_means = np.nanmean(calib_accum[:, overlap], axis=1, keepdims=True)  # (84,1)

    # anomalies
    iso_anom   = calib_iso   - iso_means    # (80, 200)
    accum_anom = calib_accum - accum_means  # (84, 200)

    # domain means (nanmean across sites for each year)
    iso_domain   = np.nanmean(iso_anom,   axis=0)   # (200,)  K
    accum_domain = np.nanmean(accum_anom, axis=0)   # (200,)  m/yr -> mm/yr

    # count valid sites per year
    iso_n_valid   = np.sum(np.isfinite(iso_anom),   axis=0)  # (200,)
    accum_n_valid = np.sum(np.isfinite(accum_anom), axis=0)  # (200,)

    print(f"Iso sites:   {calib_iso.shape[0]}  "
          f"T anom range: [{iso_domain.min():.2f}, {iso_domain.max():.2f}] K")
    print(f"Accum sites: {calib_accum.shape[0]}  "
          f"P anom range: [{accum_domain.min():.4f}, {accum_domain.max():.4f}] m/yr")
    print(f"Valid iso sites per year: min={iso_n_valid.min()} max={iso_n_valid.max()}")
    print(f"Valid accum sites per year: min={accum_n_valid.min()} max={accum_n_valid.max()}")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Raw Calibrated Ice Core Anomalies (ref: 1979–2000)\n"
        "1801–2000",
        fontsize=11,
    )

    # --- panel 1: individual iso site anomalies (light) + domain mean (bold) ---
    ax = axes[0]
    for i in range(calib_iso.shape[0]):
        row = iso_anom[i]
        valid = np.isfinite(row)
        if valid.sum() > 10:
            ax.plot(years[valid], row[valid], linewidth=0.3,
                    color="tomato", alpha=0.2)
    ax.plot(years, iso_domain, linewidth=1.5, color="darkred",
            label="domain mean")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvspan(1979, 2000, alpha=0.1, color="blue", label="ERA5 overlap")
    ax.set_ylabel("T anomaly (K)", fontsize=9)
    ax.set_title("Isotope sites — T2m anomaly (individual sites + mean)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # --- panel 2: individual accum site anomalies + domain mean ---
    ax = axes[1]
    for i in range(calib_accum.shape[0]):
        row = accum_anom[i] * 1000   # m/yr -> mm/yr
        valid = np.isfinite(row)
        if valid.sum() > 10:
            ax.plot(years[valid], row[valid], linewidth=0.3,
                    color="steelblue", alpha=0.2)
    ax.plot(years, accum_domain * 1000, linewidth=1.5, color="navy",
            label="domain mean")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvspan(1979, 2000, alpha=0.1, color="blue", label="ERA5 overlap")
    ax.set_ylabel("P anomaly (mm/yr)", fontsize=9)
    ax.set_title("Accumulation sites — Precip anomaly (individual sites + mean)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # --- panel 3: number of valid sites per year ---
    ax = axes[2]
    ax.fill_between(years, iso_n_valid,   alpha=0.6, color="tomato",
                    label="iso sites")
    ax.fill_between(years, accum_n_valid, alpha=0.6, color="steelblue",
                    label="accum sites")
    ax.axvspan(1979, 2000, alpha=0.1, color="blue")
    ax.set_ylabel("N sites", fontsize=9)
    ax.set_title("Number of sites with valid data per year", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # --- panel 4: 20-year running mean of domain means ---
    ax = axes[3]
    def running_mean(x, w=20):
        out = np.full_like(x, np.nan)
        for i in range(w-1, len(x)):
            window = x[i-w+1:i+1]
            if np.sum(np.isfinite(window)) >= w // 2:
                out[i] = np.nanmean(window)
        return out

    ax.plot(years, running_mean(iso_domain),         linewidth=1.5,
            color="darkred",  label="T (20-yr mean)")
    ax2 = ax.twinx()
    ax2.plot(years, running_mean(accum_domain*1000), linewidth=1.5,
             color="navy", linestyle="--", label="P (20-yr mean)")
    ax.axhline(0,  color="gray", linewidth=0.8, linestyle="--")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvspan(1979, 2000, alpha=0.1, color="blue")
    ax.set_ylabel("T anomaly (K)", fontsize=9, color="darkred")
    ax2.set_ylabel("P anomaly (mm/yr)", fontsize=9, color="navy")
    ax.set_title("20-year running mean of domain-mean anomalies", fontsize=9)
    ax.set_xlabel("Year", fontsize=9)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="upper left")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(PLOTS_DIR, "icecores_timeseries.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
