"""
apply_kernel.py -- Run ice core proxies through kernel GNN to reconstruct
Antarctic T and P fields for 1801-2000.
"""
import sys, os, numpy as np, jax, jax.numpy as jnp
from pathlib import Path
sys.path.insert(0, '/glade/derecho/scratch/advike/graphcast_recon')
from recon_gnn_kernel import (build_icosahedral_mesh, haversine_deg,
                               make_forward, norm_pos,
                               TARGET_LATS, TARGET_LONS)
from ice_core_loader import IceCoreLoader

RECON_DIR  = Path('/glade/derecho/scratch/advike/graphcast_recon')
TGT_DIR    = RECON_DIR / 'cache' / 'era5_targets'
CALIB_DIR  = RECON_DIR / 'cache' / 'calibration'
OUT_DIR    = RECON_DIR / 'cache' / 'recon'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ns        = np.load(TGT_DIR/'norm_stats.npz')
anom_std  = np.load(TGT_DIR/'anom_std_1979_2000.npz')
clim_mean = np.load(TGT_DIR/'clim_mean_1979_2000.npy')
t_std     = float(anom_std['temp_anom_std'])
p_std     = float(anom_std['prec_anom_std'])

loader = IceCoreLoader(
    data_dir=str(RECON_DIR/'data'),
    embeddings_dir=str(RECON_DIR/'cache'/'embeddings'),
    calibration_dir=str(CALIB_DIR),
    temp_mean=float(ns['temp_mean']), temp_std=float(ns['temp_std']),
    prec_mean=float(ns['prec_mean']), prec_std=float(ns['prec_std']),
)
iso_grid   = sorted(loader.iso_grid_map.keys())
accum_grid = sorted(loader.accum_grid_map.keys())
n_iso      = len(iso_grid)
n_accum    = len(accum_grid)
obs_lats   = np.array([float(TARGET_LATS[g]) for g in iso_grid+accum_grid], dtype=np.float32)
obs_lons   = np.array([float(TARGET_LONS[g]) for g in iso_grid+accum_grid], dtype=np.float32)

# build mesh
mesh_lats, mesh_lons, ms_np, mr_np = build_icosahedral_mesh(4)
mesh_pos_np  = norm_pos(mesh_lats, mesh_lons)
obs_mesh_nn  = np.array([np.argmin(haversine_deg(la,lo,mesh_lats,mesh_lons))
    for la,lo in zip(obs_lats,obs_lons)], dtype=np.int32)
grid_mesh_nn = np.array([np.argmin(haversine_deg(la,lo,mesh_lats,mesh_lons))
    for la,lo in zip(TARGET_LATS,TARGET_LONS)], dtype=np.int32)
obs_mesh_nn_iso   = obs_mesh_nn[:n_iso]
obs_mesh_nn_accum = obs_mesh_nn[n_iso:]
ms = jnp.array(ms_np); mr = jnp.array(mr_np)

# load model (kernel_dim=32, matches training)
fwd_T = make_forward(32, 4, 32, 0.3, mesh_pos_np, mesh_lats, mesh_lons,
                     obs_mesh_nn_iso, grid_mesh_nn)
fwd_P = make_forward(32, 4, 32, 0.3, mesh_pos_np, mesh_lats, mesh_lons,
                     obs_mesh_nn_accum, grid_mesh_nn)
dummy_iso   = jnp.zeros(n_iso)
dummy_accum = jnp.zeros(n_accum)
params_T = fwd_T.init(jax.random.PRNGKey(0), dummy_iso,   ms, mr)
params_P = fwd_P.init(jax.random.PRNGKey(1), dummy_accum, ms, mr)
raw_T = np.load(str(RECON_DIR/'weights_kernel'/'best_T.npz'))
raw_P = np.load(str(RECON_DIR/'weights_kernel'/'best_P.npz'))
params_T = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params_T),
    [raw_T[str(i)] for i in range(len(raw_T.files))])
params_P = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params_P),
    [raw_P[str(i)] for i in range(len(raw_P.files))])
print(f"Model loaded. n_iso={n_iso} n_accum={n_accum} n_grid={len(TARGET_LATS)}")

# load calibrated proxy data
calib_iso   = np.load(str(CALIB_DIR/'calibrated_iso.npy'))    # (80, 200)
calib_accum = np.load(str(CALIB_DIR/'calibrated_accum.npy'))  # (84, 200)
anom_stds   = np.load(str(CALIB_DIR/'anom_stds.npz'))
iso_anom_std   = float(anom_stds['iso_anom_std'])
accum_anom_std = float(anom_stds['accum_anom_std'])
print(f"Calibrated: iso={calib_iso.shape} accum={calib_accum.shape}")

# use loader.get_year() to get properly aggregated obs for each year
# this handles the site->grid_node mapping correctly
recon_years = list(range(1801, 2001))
print(f"Reconstructing {len(recon_years)} years...")

for yi, yr in enumerate(recon_years):
    obs = loader.get_year(yr)  # uses calibrated ice core data
    feats = obs['features']    # (n_obs, 518)
    grid_indices = obs['grid_indices']  # (n_obs,) grid node indices

    # build t_obs_norm and p_obs_norm aligned to iso_grid and accum_grid
    t_obs_norm = np.zeros(n_iso,   dtype=np.float32)
    p_obs_norm = np.zeros(n_accum, dtype=np.float32)

    for k, g in enumerate(grid_indices):
        # col 0 = T val (proxy), col 1 = T avail
        # col 2 = P val (proxy), col 3 = P avail
        t_avail = feats[k, 1] > 0
        p_avail = feats[k, 3] > 0

        if t_avail and g in iso_grid:
            ki = iso_grid.index(g)
            # feats[:,0] is proxy T anom normalised by iso_anom_std
            # we need to renormalise to era5_t_std
            t_obs_norm[ki] = feats[k, 0] * iso_anom_std / t_std

        if p_avail and g in accum_grid:
            ki = accum_grid.index(g)
            p_obs_norm[ki] = feats[k, 2] * accum_anom_std / p_std

    # forward pass
    pred_T_norm = np.array(fwd_T.apply(params_T, None, jnp.array(t_obs_norm), ms, mr))
    pred_P_norm = np.array(fwd_P.apply(params_P, None, jnp.array(p_obs_norm), ms, mr))

    # denormalise
    pred_T_K   = pred_T_norm * t_std + clim_mean[:, 0]
    pred_P_myr = pred_P_norm * p_std + clim_mean[:, 1]

    out = np.stack([pred_T_K, pred_P_myr], axis=1).astype(np.float32)
    np.save(str(OUT_DIR/f'recon_{yr}.npy'), out)

    if yi % 20 == 0:
        n_t = int((t_obs_norm != 0).sum())
        n_p = int((p_obs_norm != 0).sum())
        print(f"  {yr}: T_obs={n_t}/{n_iso} P_obs={n_p}/{n_accum} "
              f"T={pred_T_K.mean():.2f}K P={pred_P_myr.mean():.4f}m/yr")

print(f"Done. {len(recon_years)} files saved to {OUT_DIR}")
