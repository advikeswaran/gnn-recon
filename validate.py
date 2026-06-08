"""
validate.py — Evaluate trained reconstruction GNN on held-out ERA5 years (2001–2005).

Anomaly mode: both obs inputs and targets are anomalies relative to 1979-2000 mean.

Two-phase design (avoids cartopy/JAX conflicts):

  Phase 1 — inference  (graphcast conda env):
      python validate.py --phase infer

  Phase 2 — plot/metrics  (PlotEnv):
      python validate.py --phase plot
"""

import argparse
import os
import sys

RECON_DIR  = "/glade/derecho/scratch/advike/graphcast_recon"
CACHE_DIR  = os.path.join(RECON_DIR, "cache")
VAL_YEARS  = list(range(2001, 2006))
HIDDEN         = 128
T2T_ROUNDS     = 6
OBS_TO_TGT_DEG = 45.0
TGT_TO_TGT_DEG = 2.0
N_OUTPUT       = 2

OUT_DIR = os.path.join(CACHE_DIR, "validate")


def run_infer():
    import numpy as np
    import jax
    import jax.numpy as jnp
    import haiku as hk

    sys.path.insert(0, RECON_DIR)
    from train_head import (
        build_obs_to_target_edges,
        build_target_features,
        make_forward_fn,
        TARGET_LATS,
        TARGET_LONS,
    )
    from ice_core_loader import IceCoreLoader, grid_index_to_latlon

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── norm stats ────────────────────────────────────────────────────────────
    ns = np.load(os.path.join(CACHE_DIR, "era5_targets", "norm_stats.npz"))
    temp_mean = float(ns["temp_mean"])
    temp_std  = float(ns["temp_std"])
    prec_mean = float(ns["prec_mean"])
    prec_std  = float(ns["prec_std"])

    # ── clim mean (1979-2000) for anomaly computation ─────────────────────────
    clim_mean = np.load(
        os.path.join(CACHE_DIR, "era5_targets", "clim_mean_1979_2000.npy")
    )  # (11160, 2)  col0=T(K) col1=P(m/yr)
    print(f"[infer] clim mean: T={clim_mean[:,0].mean():.2f}K  "
          f"P={clim_mean[:,1].mean():.4f}m/yr")

    # ── ERA5 anomaly stds for obs normalisation (must match training)
    _era5_anom = np.load(os.path.join(CACHE_DIR, "era5_targets", "anom_std_1979_2000.npz"))
    era5_temp_anom_std = float(_era5_anom["temp_anom_std"])
    era5_prec_anom_std = float(_era5_anom["prec_anom_std"])

    # ── loader ────────────────────────────────────────────────────────────────
    loader = IceCoreLoader(
        data_dir        = os.path.join(RECON_DIR, "data"),
        embeddings_dir  = os.path.join(CACHE_DIR, "embeddings"),
        calibration_dir = os.path.join(CACHE_DIR, "calibration"),
        temp_mean=temp_mean, temp_std=temp_std,
        prec_mean=prec_mean, prec_std=prec_std,
    )

    # ── target features (static) ──────────────────────────────────────────────
    tgt_feats = jnp.array(build_target_features(loader.clim_embedding))
    print(f"[infer] tgt_feats: {tgt_feats.shape}")

    # ── t2t edges ─────────────────────────────────────────────────────────────
    t2t_path = os.path.join(CACHE_DIR, "t2t_edges.npz")
    t2t      = np.load(t2t_path)
    t2t_s    = jnp.array(t2t["senders"])
    t2t_r    = jnp.array(t2t["receivers"])
    print(f"[infer] t2t edges: {t2t_s.shape[0]:,}")

    # ── obs grid indices and coordinates ──────────────────────────────────────
    iso_grid_indices   = sorted(loader.iso_grid_map.keys())
    accum_grid_indices = sorted(loader.accum_grid_map.keys())
    n_iso   = len(iso_grid_indices)
    n_accum = len(accum_grid_indices)
    n_obs   = n_iso + n_accum
    print(f"[infer] obs nodes: {n_iso} iso + {n_accum} accum = {n_obs} total")

    obs_grid_indices = iso_grid_indices + accum_grid_indices
    obs_lats = np.array([grid_index_to_latlon(g)[0] for g in obs_grid_indices],
                        dtype=np.float32)
    obs_lons = np.array([grid_index_to_latlon(g)[1] for g in obs_grid_indices],
                        dtype=np.float32)

    # ── obs->target edges (static) ────────────────────────────────────────────
    print("[infer] building obs->target edges ...")
    o2t_s_np, o2t_r_np = build_obs_to_target_edges(
        obs_lats, obs_lons, TARGET_LATS, TARGET_LONS, OBS_TO_TGT_DEG
    )
    o2t_s = jnp.array(o2t_s_np)
    o2t_r = jnp.array(o2t_r_np)
    print(f"[infer] o2t edges: {o2t_s.shape[0]:,}")

    # ── embedding similarity edge features ───────────────────────────────────
    from train_head import compute_o2t_edge_sim
    o2t_sim = jnp.array(compute_o2t_edge_sim(
        np.array(obs_grid_indices), o2t_s_np, o2t_r_np, loader.clim_embedding
    ))

    # ── model + weights ───────────────────────────────────────────────────────
    forward = make_forward_fn(HIDDEN, T2T_ROUNDS)
    dummy_obs = jnp.zeros((n_obs, 518), dtype=jnp.float32)
    dummy_sim = jnp.zeros((o2t_s.shape[0], 1), dtype=jnp.float32)
    params    = forward.init(jax.random.PRNGKey(0), dummy_obs, tgt_feats,
                             o2t_s, o2t_r, dummy_sim, t2t_s, t2t_r)
    raw    = np.load(os.path.join(RECON_DIR, "weights", "weights_final.npz"))
    leaves = [raw[str(i)] for i in range(len(raw.files))]
    params = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(params), leaves
    )
    print(f"[infer] weights loaded: {len(leaves)} leaves")

    # ── inference loop ────────────────────────────────────────────────────────
    for yr in VAL_YEARS:
        out_path = os.path.join(OUT_DIR, f"pred_{yr}.npy")
        if os.path.exists(out_path):
            print(f"[infer] {yr}: already exists, skipping.")
            continue

        emb_path = os.path.join(CACHE_DIR, "embeddings", f"window_{yr}.npy")
        tgt_path = os.path.join(CACHE_DIR, "era5_targets", f"targets_{yr}.npy")

        if not os.path.exists(emb_path):
            print(f"[infer] WARNING: embedding for {yr} not found, skipping.")
            continue
        if not os.path.exists(tgt_path):
            print(f"[infer] WARNING: ERA5 targets for {yr} not found, skipping.")
            continue

        # sample ERA5 anomalies at obs grid nodes
        # anomaly = ERA5_value - clim_mean, normalised by std only
        era5_yr = np.load(tgt_path)  # (11160, 2)

        iso_obs_norm = np.array(
            [(era5_yr[g, 0] - clim_mean[g, 0]) / era5_temp_anom_std
             for g in iso_grid_indices],
            dtype=np.float32)
        accum_obs_norm = np.array(
            [(era5_yr[g, 1] - clim_mean[g, 1]) / era5_prec_anom_std
             for g in accum_grid_indices],
            dtype=np.float32)

        print(f"[infer] {yr}: obs T anom norm [{iso_obs_norm.min():.3f}, "
              f"{iso_obs_norm.max():.3f}]  "
              f"P anom norm [{accum_obs_norm.min():.3f}, "
              f"{accum_obs_norm.max():.3f}]")

        # assemble 518-dim obs features matching training layout
        obs_feats_list = []
        for k, g in enumerate(iso_grid_indices):
            feat = np.concatenate([
                np.array([iso_obs_norm[k], 1.0, 0.0, 0.0,
                          obs_lats[k], obs_lons[k]], dtype=np.float32),
                loader.clim_embedding[g],
            ])
            obs_feats_list.append(feat)
        for k, g in enumerate(accum_grid_indices):
            feat = np.concatenate([
                np.array([0.0, 0.0, accum_obs_norm[k], 1.0,
                          obs_lats[n_iso + k], obs_lons[n_iso + k]], dtype=np.float32),
                loader.clim_embedding[g],
            ])
            obs_feats_list.append(feat)

        obs_feats_np  = np.stack(obs_feats_list).astype(np.float32)
        obs_feats_jnp = jnp.array(obs_feats_np)

        print(f"[infer] {yr}: running forward pass ...")
        pred_norm = forward.apply(params, None, obs_feats_jnp, tgt_feats,
                                  o2t_s, o2t_r, o2t_sim, t2t_s, t2t_r)
        pred_norm = np.array(pred_norm)  # (11160, 2) normalised anomalies

        tmp = out_path.replace(".npy", ".tmp.npy")
        np.save(tmp, pred_norm)
        os.rename(tmp, out_path)
        print(f"[infer] {yr}: saved -> {out_path}  shape={pred_norm.shape}")

    print("[infer] done.")


def run_plot():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    os.makedirs(OUT_DIR, exist_ok=True)

    ns = np.load(os.path.join(CACHE_DIR, "era5_targets", "norm_stats.npz"))
    temp_std  = float(ns["temp_std"])
    prec_std  = float(ns["prec_std"])

    # load anomaly stds
    anom_stds          = np.load(os.path.join(CACHE_DIR, "calibration", "anom_stds.npz"))
    iso_anom_std       = float(anom_stds["iso_anom_std"])
    accum_anom_std     = float(anom_stds["accum_anom_std"])
    era5_anom          = np.load(os.path.join(CACHE_DIR, "era5_targets", "anom_std_1979_2000.npz"))
    era5_temp_anom_std = float(era5_anom["temp_anom_std"])
    era5_prec_anom_std = float(era5_anom["prec_anom_std"])

    # clim mean for converting targets to anomalies
    clim_mean = np.load(
        os.path.join(CACHE_DIR, "era5_targets", "clim_mean_1979_2000.npy")
    )  # (11160, 2)

    LATS    = list(range(-60, -91, -1))
    LONS    = list(range(0, 360))
    lats_1d = np.array(LATS, dtype=np.float32)
    lons_1d = np.array(LONS, dtype=np.float32)

    def to_grid(vec):
        return vec.reshape(len(LATS), len(LONS))

    pred_temp_list, pred_prec_list = [], []
    true_temp_list, true_prec_list = [], []
    years_found = []

    for yr in VAL_YEARS:
        pred_path = os.path.join(OUT_DIR, f"pred_{yr}.npy")
        tgt_path  = os.path.join(CACHE_DIR, "era5_targets", f"targets_{yr}.npy")
        if not os.path.exists(pred_path):
            print(f"[plot] WARNING: prediction for {yr} missing, skipping.")
            continue
        if not os.path.exists(tgt_path):
            print(f"[plot] WARNING: ERA5 target for {yr} missing, skipping.")
            continue

        pred_norm = np.load(pred_path)   # (11160, 2) normalised anomalies
        tgt       = np.load(tgt_path)    # (11160, 2) physical

        # denormalise predictions: model outputs normalised by ERA5 anomaly std
        pred_temp_anom = pred_norm[:, 0] * era5_temp_anom_std   # K anomaly
        pred_prec_anom = pred_norm[:, 1] * era5_prec_anom_std   # m/yr anomaly

        # convert targets to anomalies
        true_temp_anom = tgt[:, 0] - clim_mean[:, 0]   # K anomaly
        true_prec_anom = tgt[:, 1] - clim_mean[:, 1]   # m/yr anomaly

        pred_temp_list.append(pred_temp_anom)
        pred_prec_list.append(pred_prec_anom)
        true_temp_list.append(true_temp_anom)
        true_prec_list.append(true_prec_anom)
        years_found.append(yr)

    if not years_found:
        print("[plot] No valid year pairs found. Run --phase infer first.")
        sys.exit(1)

    pred_temp_arr = np.stack(pred_temp_list)
    pred_prec_arr = np.stack(pred_prec_list)
    true_temp_arr = np.stack(true_temp_list)
    true_prec_arr = np.stack(true_prec_list)
    N_yr = len(years_found)
    print(f"[plot] evaluating {N_yr} years: {years_found}")

    def rmse(pred, true):
        return float(np.sqrt(np.mean((pred - true) ** 2)))

    def bias(pred, true):
        return float(np.mean(pred - true))

    def spatial_r(pred, true):
        p = pred - pred.mean()
        t = true - true.mean()
        denom = np.sqrt((p ** 2).sum() * (t ** 2).sum())
        return float(np.dot(p, t) / denom) if denom > 0 else float("nan")

    metrics = {
        "years":     np.array(years_found),
        "temp_rmse": np.zeros(N_yr),
        "temp_bias": np.zeros(N_yr),
        "temp_r":    np.zeros(N_yr),
        "prec_rmse": np.zeros(N_yr),
        "prec_bias": np.zeros(N_yr),
        "prec_r":    np.zeros(N_yr),
    }

    for i in range(N_yr):
        metrics["temp_rmse"][i] = rmse(pred_temp_arr[i], true_temp_arr[i])
        metrics["temp_bias"][i] = bias(pred_temp_arr[i], true_temp_arr[i])
        metrics["temp_r"][i]    = spatial_r(pred_temp_arr[i], true_temp_arr[i])
        metrics["prec_rmse"][i] = rmse(pred_prec_arr[i], true_prec_arr[i])
        metrics["prec_bias"][i] = bias(pred_prec_arr[i], true_prec_arr[i])
        metrics["prec_r"][i]    = spatial_r(pred_prec_arr[i], true_prec_arr[i])

    metrics["temp_rmse_mean"] = float(metrics["temp_rmse"].mean())
    metrics["temp_bias_mean"] = float(metrics["temp_bias"].mean())
    metrics["temp_r_mean"]    = float(metrics["temp_r"].mean())
    metrics["prec_rmse_mean"] = float(metrics["prec_rmse"].mean())
    metrics["prec_bias_mean"] = float(metrics["prec_bias"].mean())
    metrics["prec_r_mean"]    = float(metrics["prec_r"].mean())

    print("\n-- Validation metrics anomaly mode (2001-2005) ----------------------")
    print(f"{'Year':>6}  {'T RMSE(K)':>10} {'T Bias(K)':>10} {'T r':>8}  "
          f"{'P RMSE(m/yr)':>13} {'P Bias(m/yr)':>13} {'P r':>8}")
    for i, yr in enumerate(years_found):
        print(f"{yr:>6}  {metrics['temp_rmse'][i]:>10.3f} "
              f"{metrics['temp_bias'][i]:>10.3f} "
              f"{metrics['temp_r'][i]:>8.4f}  "
              f"{metrics['prec_rmse'][i]:>13.4f} "
              f"{metrics['prec_bias'][i]:>13.4f} "
              f"{metrics['prec_r'][i]:>8.4f}")
    print(f"{'MEAN':>6}  {metrics['temp_rmse_mean']:>10.3f} "
          f"{metrics['temp_bias_mean']:>10.3f} "
          f"{metrics['temp_r_mean']:>8.4f}  "
          f"{metrics['prec_rmse_mean']:>13.4f} "
          f"{metrics['prec_bias_mean']:>13.4f} "
          f"{metrics['prec_r_mean']:>8.4f}")
    print("---------------------------------------------------------------------\n")

    metrics_path = os.path.join(OUT_DIR, "val_metrics.npz")
    tmp = metrics_path.replace(".npz", ".tmp.npz")
    np.savez(tmp, **metrics)
    os.rename(tmp, metrics_path)
    print(f"[plot] metrics saved -> {metrics_path}")

    temp_rmse_map = to_grid(np.sqrt(np.mean((pred_temp_arr - true_temp_arr) ** 2, axis=0)))
    temp_bias_map = to_grid(np.mean(pred_temp_arr - true_temp_arr, axis=0))
    prec_rmse_map = to_grid(np.sqrt(np.mean((pred_prec_arr - true_prec_arr) ** 2, axis=0)))
    prec_bias_map = to_grid(np.mean(pred_prec_arr - true_prec_arr, axis=0))

    def pixelwise_r(pred, true):
        p = pred - pred.mean(axis=0, keepdims=True)
        t = true - true.mean(axis=0, keepdims=True)
        num   = (p * t).sum(axis=0)
        denom = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0))
        with np.errstate(invalid="ignore"):
            return np.where(denom > 0, num / denom, np.nan)

    temp_r_map = to_grid(pixelwise_r(pred_temp_arr, true_temp_arr))
    prec_r_map = to_grid(pixelwise_r(pred_prec_arr, true_prec_arr))

    PROJ     = ccrs.SouthPolarStereo()
    DATA_CRS = ccrs.PlateCarree()

    def polar_panel(ax, data_grid, lons, lats, cmap, vmin, vmax,
                    title, unit, extend="both"):
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
                     pad=0.04, fraction=0.046, label=unit, extend=extend)
        ax.set_title(title, fontsize=9, pad=4)

    # figure 1: spatial error maps
    fig1, axes = plt.subplots(2, 3, figsize=(15, 9),
                               subplot_kw={"projection": PROJ})
    fig1.suptitle(
        f"Antarctic Reconstruction -- Spatial Error Maps (anomaly mode)\n"
        f"Held-out ERA5 years {years_found[0]}-{years_found[-1]} "
        f"(ERA5-sampled obs, mean over {N_yr} years)", fontsize=11)
    polar_panel(axes[0,0], temp_rmse_map, lons_1d, lats_1d,
                "YlOrRd", 0, None, "T2m RMSE (K)", "K", extend="max")
    polar_panel(axes[0,1], temp_bias_map, lons_1d, lats_1d,
                "RdBu_r", -5, 5, "T2m Bias (K)", "K")
    polar_panel(axes[0,2], temp_r_map,    lons_1d, lats_1d,
                "RdYlGn", -1, 1, "T2m Inter-annual r", "r")
    polar_panel(axes[1,0], prec_rmse_map, lons_1d, lats_1d,
                "YlOrRd", 0, None, "Precip RMSE (m/yr)", "m/yr", extend="max")
    polar_panel(axes[1,1], prec_bias_map, lons_1d, lats_1d,
                "BrBG", -0.2, 0.2, "Precip Bias (m/yr)", "m/yr")
    polar_panel(axes[1,2], prec_r_map,    lons_1d, lats_1d,
                "RdYlGn", -1, 1, "Precip Inter-annual r", "r")
    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    map_path = os.path.join(OUT_DIR, "val_spatial_errors.png")
    fig1.savefig(map_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"[plot] spatial error maps -> {map_path}")

    # figure 2: time series
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    fig2.suptitle(
        "Antarctic Reconstruction -- Annual Validation Metrics (anomaly mode)\n"
        f"Held-out ERA5 years {years_found[0]}-{years_found[-1]}",
        fontsize=11)
    yrs = np.array(years_found)
    _panels = [
        (axes2[0,0], metrics["temp_rmse"], "K",    "T2m RMSE",         None),
        (axes2[0,1], metrics["temp_bias"], "K",    "T2m Bias",         0.0),
        (axes2[0,2], metrics["temp_r"],    "r",    "T2m Spatial r",    None),
        (axes2[1,0], metrics["prec_rmse"], "m/yr", "Precip RMSE",      None),
        (axes2[1,1], metrics["prec_bias"], "m/yr", "Precip Bias",      0.0),
        (axes2[1,2], metrics["prec_r"],    "r",    "Precip Spatial r", None),
    ]
    for ax, ydata, ylabel, title, hline in _panels:
        ax.plot(yrs, ydata, "o-", linewidth=1.8, markersize=6)
        if hline is not None:
            ax.axhline(hline, color="gray", linewidth=0.8, linestyle="--")
        ax.axhline(ydata.mean(), color="tomato", linewidth=1.2,
                   linestyle=":", label=f"mean={ydata.mean():.3f}")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(yrs)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, linewidth=0.4, alpha=0.5)
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    ts_path = os.path.join(OUT_DIR, "val_timeseries.png")
    fig2.savefig(ts_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[plot] time series -> {ts_path}")

    # figure 3: example year side-by-side (anomalies)
    ex_idx = N_yr // 2
    ex_yr  = years_found[ex_idx]
    fig3, axes3 = plt.subplots(2, 2, figsize=(13, 9),
                                subplot_kw={"projection": PROJ})
    fig3.suptitle(
        f"Antarctic Reconstruction vs ERA5 Anomaly -- Example Year {ex_yr}",
        fontsize=11)
    t_abs = max(abs(true_temp_arr[ex_idx]).max(),
                abs(pred_temp_arr[ex_idx]).max())
    p_abs = max(abs(true_prec_arr[ex_idx]).max(),
                abs(pred_prec_arr[ex_idx]).max())
    polar_panel(axes3[0,0], to_grid(true_temp_arr[ex_idx]), lons_1d, lats_1d,
                "RdBu_r", -t_abs, t_abs, f"ERA5 T2m anomaly {ex_yr} (K)", "K")
    polar_panel(axes3[0,1], to_grid(pred_temp_arr[ex_idx]), lons_1d, lats_1d,
                "RdBu_r", -t_abs, t_abs, f"Predicted T2m anomaly {ex_yr} (K)", "K")
    polar_panel(axes3[1,0], to_grid(true_prec_arr[ex_idx]), lons_1d, lats_1d,
                "BrBG", -p_abs, p_abs, f"ERA5 Precip anomaly {ex_yr} (m/yr)", "m/yr")
    polar_panel(axes3[1,1], to_grid(pred_prec_arr[ex_idx]), lons_1d, lats_1d,
                "BrBG", -p_abs, p_abs, f"Predicted Precip anomaly {ex_yr} (m/yr)", "m/yr")
    fig3.tight_layout(rect=[0, 0, 1, 0.94])
    ex_path = os.path.join(OUT_DIR, f"val_example_{ex_yr}.png")
    fig3.savefig(ex_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"[plot] example year map -> {ex_path}")

    print("\n[plot] all outputs written to:", OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Antarctic reconstruction GNN (anomaly mode).")
    parser.add_argument("--phase", choices=["infer", "plot"], required=True)
    args = parser.parse_args()
    if args.phase == "infer":
        run_infer()
    else:
        run_plot()
