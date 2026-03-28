"""
apply.py — Run trained reconstruction GNN on ice core observations 1801-2000.

For each year:
  - Loads calibrated ice core obs via IceCoreLoader.get_year()
  - Uses year-specific GraphCast embedding (1979-2000) or climatological mean (1801-1978)
  - Runs N_ENSEMBLE=50 forward passes with per-site calibration noise perturbations
  - Saves mean and std of ensemble predictions in physical units

Outputs (cache/recon/):
  recon_YYYY.npy      (11160, 2) float32  mean prediction  [T(K), P(m/yr)]
  recon_YYYY_std.npy  (11160, 2) float32  ensemble std     [T(K), P(m/yr)]

Usage:
  python apply.py [--year YYYY]   # single year (for testing)
  python apply.py [--all]         # full 1801-2000 reconstruction

Run in graphcast conda env via batch job.
"""

import os
import sys
import argparse
import numpy as np

RECON_DIR  = "/glade/derecho/scratch/advike/graphcast_recon"
CACHE_DIR  = os.path.join(RECON_DIR, "cache")
OUT_DIR    = os.path.join(CACHE_DIR, "recon")

RECON_YEARS    = list(range(1801, 2001))   # 200 years
EMB_YEARS      = set(range(1979, 2018))    # years with window embeddings
N_ENSEMBLE     = 50
HIDDEN         = 128
T2T_ROUNDS     = 6
OBS_TO_TGT_DEG = 9.0
TGT_TO_TGT_DEG = 2.0


def compute_residual_stds():
    """
    Compute per-site calibration residual std in physical units.
    residual_std = std(ERA5_at_node_overlap) * sqrt(1 - R2)

    For ERA5 std we use the full target distribution (all 38 years, all nodes)
    as a proxy — cheaper than loading per-node overlap series.
    More precisely we use the per-variable global std from norm_stats, scaled
    by sqrt(1-R2) per site. This is slightly conservative but avoids re-loading
    all ERA5 targets here.

    Returns:
      iso_resid_std   (n_iso,)   in K
      accum_resid_std (n_accum,) in m/yr
    """
    ns = np.load(os.path.join(CACHE_DIR, "era5_targets", "norm_stats.npz"))
    temp_std = float(ns["temp_std"])
    prec_std = float(ns["prec_std"])

    coeffs = np.load(os.path.join(CACHE_DIR, "calibration", "coefficients.npz"))
    iso_r2   = coeffs["iso_r2"].astype(np.float32)
    accum_r2 = coeffs["accum_r2"].astype(np.float32)

    # Clamp R2 to [0, 1] — a few sites have negative R2 (worse than mean)
    iso_r2   = np.clip(iso_r2,   0.0, 1.0)
    accum_r2 = np.clip(accum_r2, 0.0, 1.0)

    iso_resid_std   = temp_std * np.sqrt(1.0 - iso_r2)    # (n_iso,)  K
    accum_resid_std = prec_std * np.sqrt(1.0 - accum_r2)  # (n_accum,) m/yr

    return iso_resid_std, accum_resid_std


def run_year(yr, forward, params, loader, tgt_feats, clim_emb,
             t2t_s, t2t_r, iso_resid_std, accum_resid_std,
             build_obs_to_target_edges, TARGET_LATS, TARGET_LONS,
             temp_mean, temp_std, prec_mean, prec_std, rng):
    import jax.numpy as jnp

    out_path     = os.path.join(OUT_DIR, f"recon_{yr}.npy")
    out_std_path = os.path.join(OUT_DIR, f"recon_{yr}_std.npy")

    if os.path.exists(out_path) and os.path.exists(out_std_path):
        print(f"[apply] {yr}: already exists, skipping.")
        return

    # -- load obs via IceCoreLoader -------------------------------------------
    try:
        obs = loader.get_year(yr)
    except ValueError as e:
        print(f"[apply] {yr}: no obs ({e}), skipping.")
        return

    obs_feats_base = obs["features"].copy()   # (n_obs, 518)
    obs_lats       = obs["lats"]
    obs_lons       = obs["lons"]
    n_obs          = obs["n_obs"]

    # (build_target_features uses clim_emb, not year_emb, matching training)

    # -- obs->target edges (varies by year as available sites change) ----------
    o2t_s_np, o2t_r_np = build_obs_to_target_edges(
        obs_lats, obs_lons, TARGET_LATS, TARGET_LONS, OBS_TO_TGT_DEG
    )
    o2t_s = jnp.array(o2t_s_np)
    o2t_r = jnp.array(o2t_r_np)

    # -- identify which obs nodes are iso vs accum ----------------------------
    # features layout: [iso_val, iso_avail, accum_val, accum_avail, lat, lon, emb]
    iso_avail_mask   = obs_feats_base[:, 1] == 1.0   # (n_obs,) bool
    accum_avail_mask = obs_feats_base[:, 3] == 1.0   # (n_obs,) bool

    # Map per-site residual stds to obs nodes.
    # iso_grid_map / accum_grid_map give us which grid nodes have obs, but
    # IceCoreLoader may merge multiple sites per node. We use the mean residual
    # std across sites sharing a node, mapped to the obs node ordering.
    iso_grid_indices   = sorted(loader.iso_grid_map.keys())
    accum_grid_indices = sorted(loader.accum_grid_map.keys())

    # Build node-level residual std arrays aligned to obs ordering in get_year()
    # get_year() iterates site_registry (union of iso+accum grid indices, sorted)
    # and skips nodes with no available data for that year.
    # We match by grid index stored in obs["grid_indices"].
    obs_grid_indices = obs["grid_indices"]   # (n_obs,) int32

    iso_node_std   = np.zeros(n_obs, dtype=np.float32)
    accum_node_std = np.zeros(n_obs, dtype=np.float32)

    iso_idx_map   = {g: i for i, g in enumerate(iso_grid_indices)}
    accum_idx_map = {g: i for i, g in enumerate(accum_grid_indices)}

    for k, g in enumerate(obs_grid_indices):
        if iso_avail_mask[k] and g in iso_idx_map:
            # mean residual std across sites sharing this grid node
            site_indices = [loader.iso_site_to_row[sid]
                            for sid in loader.iso_grid_map[g]]
            iso_node_std[k] = float(np.mean(iso_resid_std[site_indices]))
        if accum_avail_mask[k] and g in accum_idx_map:
            site_indices = [loader.accum_site_to_row[sid]
                            for sid in loader.accum_grid_map[g]]
            accum_node_std[k] = float(np.mean(accum_resid_std[site_indices]))

    # Convert residual stds to normalised space for perturbation
    iso_node_std_norm   = iso_node_std   / temp_std   # (n_obs,)
    accum_node_std_norm = accum_node_std / prec_std   # (n_obs,)

    # -- ensemble forward passes ----------------------------------------------
    ensemble_preds = np.zeros((N_ENSEMBLE, 11160, 2), dtype=np.float32)

    for e in range(N_ENSEMBLE):
        rng, rng_iso, rng_accum = np.random.default_rng(rng.integers(2**31)), \
                                   np.random.default_rng(rng.integers(2**31)), \
                                   np.random.default_rng(rng.integers(2**31))

        obs_feats_perturbed = obs_feats_base.copy()

        # perturb iso values (feature col 0) where iso_avail==1
        noise_iso = rng_iso.normal(0.0, iso_node_std_norm).astype(np.float32)
        obs_feats_perturbed[:, 0] += noise_iso * iso_avail_mask

        # perturb accum values (feature col 2) where accum_avail==1
        noise_accum = rng_accum.normal(0.0, accum_node_std_norm).astype(np.float32)
        obs_feats_perturbed[:, 2] += noise_accum * accum_avail_mask

        pred_norm = forward.apply(
            params, None,
            jnp.array(obs_feats_perturbed), tgt_feats,
            o2t_s, o2t_r, t2t_s, t2t_r,
        )
        ensemble_preds[e] = np.array(pred_norm)

    # -- compute mean and std, denormalise ------------------------------------
    pred_mean_norm = ensemble_preds.mean(axis=0)   # (11160, 2)
    pred_std_norm  = ensemble_preds.std(axis=0)    # (11160, 2)

    pred_mean = np.stack([
        pred_mean_norm[:, 0] * temp_std + temp_mean,
        pred_mean_norm[:, 1] * prec_std + prec_mean,
    ], axis=1).astype(np.float32)

    pred_std = np.stack([
        pred_std_norm[:, 0] * temp_std,
        pred_std_norm[:, 1] * prec_std,
    ], axis=1).astype(np.float32)

    # -- atomic save ----------------------------------------------------------
    tmp = out_path.replace(".npy", ".tmp.npy")
    np.save(tmp, pred_mean)
    os.rename(tmp, out_path)

    tmp_std = out_std_path.replace(".npy", ".tmp.npy")
    np.save(tmp_std, pred_std)
    os.rename(tmp_std, out_std_path)

    print(f"[apply] {yr}: saved  mean={pred_mean[:,0].mean():.2f}K "
          f"std={pred_std[:,0].mean():.3f}K  "
          f"P_mean={pred_mean[:,1].mean():.4f}m/yr")


def main():
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
    from ice_core_loader import IceCoreLoader

    parser = argparse.ArgumentParser(description="Apply reconstruction GNN 1801-2000.")
    parser.add_argument("--year", type=int, default=None,
                        help="Single year to process (for testing)")
    parser.add_argument("--all", action="store_true",
                        help="Process all years 1801-2000")
    args = parser.parse_args()

    if args.year is None and not args.all:
        parser.print_help()
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    # -- norm stats -----------------------------------------------------------
    ns = np.load(os.path.join(CACHE_DIR, "era5_targets", "norm_stats.npz"))
    temp_mean = float(ns["temp_mean"])
    temp_std  = float(ns["temp_std"])
    prec_mean = float(ns["prec_mean"])
    prec_std  = float(ns["prec_std"])
    print(f"[apply] norm stats: T μ={temp_mean:.2f} σ={temp_std:.2f} | "
          f"P μ={prec_mean:.4f} σ={prec_std:.4f}")

    # -- residual stds --------------------------------------------------------
    iso_resid_std, accum_resid_std = compute_residual_stds()
    print(f"[apply] residual stds: "
          f"T mean={iso_resid_std.mean():.3f}K max={iso_resid_std.max():.3f}K | "
          f"P mean={accum_resid_std.mean():.4f} max={accum_resid_std.max():.4f} m/yr")

    # -- loader ---------------------------------------------------------------
    loader = IceCoreLoader(
        data_dir        = os.path.join(RECON_DIR, "data"),
        embeddings_dir  = os.path.join(CACHE_DIR, "embeddings"),
        calibration_dir = os.path.join(CACHE_DIR, "calibration"),
        temp_mean=temp_mean, temp_std=temp_std,
        prec_mean=prec_mean, prec_std=prec_std,
    )

    # -- target features (static) ---------------------------------------------
    tgt_feats = jnp.array(build_target_features(loader.clim_embedding))
    print(f"[apply] tgt_feats: {tgt_feats.shape}")

    # -- clim embedding (used for pre-ERA5 years) -----------------------------
    clim_emb = loader.clim_embedding   # (11160, 512)

    # -- t2t edges (static, cached) -------------------------------------------
    t2t_path = os.path.join(CACHE_DIR, "t2t_edges.npz")
    t2t      = np.load(t2t_path)
    t2t_s    = jnp.array(t2t["senders"])
    t2t_r    = jnp.array(t2t["receivers"])
    print(f"[apply] t2t edges: {t2t_s.shape[0]:,}")

    # -- model + weights ------------------------------------------------------
    forward = make_forward_fn(HIDDEN, T2T_ROUNDS)

    # init with dummy inputs to get pytree structure
    dummy_obs = jnp.zeros((135, 518), dtype=jnp.float32)   # 135 = typical n_obs
    dummy_o2t_s = jnp.zeros(1, dtype=jnp.int32)
    dummy_o2t_r = jnp.zeros(1, dtype=jnp.int32)
    params = forward.init(jax.random.PRNGKey(0), dummy_obs, tgt_feats,
                          dummy_o2t_s, dummy_o2t_r, t2t_s, t2t_r)

    weights_path = os.path.join(RECON_DIR, "weights", "weights_final.npz")
    raw    = np.load(weights_path)
    leaves = [raw[str(i)] for i in range(len(raw.files))]
    params = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(params), leaves
    )
    print(f"[apply] weights loaded: {len(leaves)} leaves")

    # -- RNG for ensemble perturbations ---------------------------------------
    rng = np.random.default_rng(42)

    # -- run ------------------------------------------------------------------
    years = [args.year] if args.year else RECON_YEARS

    for yr in years:
        run_year(
            yr, forward, params, loader, tgt_feats, clim_emb,
            t2t_s, t2t_r, iso_resid_std, accum_resid_std,
            build_obs_to_target_edges, TARGET_LATS, TARGET_LONS,
            temp_mean, temp_std, prec_mean, prec_std, rng,
        )

    print(f"[apply] done. outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
