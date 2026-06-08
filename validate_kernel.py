import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/glade/derecho/scratch/advike/graphcast_recon')
from recon_gnn_kernel import (build_icosahedral_mesh, haversine_deg,
                               make_forward, norm_pos,
                               TARGET_LATS, TARGET_LONS)
from ice_core_loader import IceCoreLoader

RECON_DIR = '/glade/derecho/scratch/advike/graphcast_recon'
TGT_DIR   = f'{RECON_DIR}/cache/era5_targets'

ns       = np.load(f'{TGT_DIR}/norm_stats.npz')
anom_std = np.load(f'{TGT_DIR}/anom_std_1979_2000.npz')
clim_mean= np.load(f'{TGT_DIR}/clim_mean_1979_2000.npy')
t_std    = float(anom_std['temp_anom_std'])
p_std    = float(anom_std['prec_anom_std'])
loader   = IceCoreLoader(
    data_dir=f'{RECON_DIR}/data',
    embeddings_dir=f'{RECON_DIR}/cache/embeddings',
    calibration_dir=f'{RECON_DIR}/cache/calibration',
    temp_mean=float(ns['temp_mean']), temp_std=float(ns['temp_std']),
    prec_mean=float(ns['prec_mean']), prec_std=float(ns['prec_std']),
)
iso_grid   = sorted(loader.iso_grid_map.keys())
accum_grid = sorted(loader.accum_grid_map.keys())
all_grid   = iso_grid + accum_grid
n_iso      = len(iso_grid)
obs_grid   = np.array(all_grid, dtype=np.int32)
obs_lats   = np.array([float(TARGET_LATS[g]) for g in all_grid], dtype=np.float32)
obs_lons   = np.array([float(TARGET_LONS[g]) for g in all_grid], dtype=np.float32)

mesh_lats, mesh_lons, ms_np, mr_np = build_icosahedral_mesh(4)
mesh_pos_np = norm_pos(mesh_lats, mesh_lons)
obs_mesh_nn = np.array([np.argmin(haversine_deg(la,lo,mesh_lats,mesh_lons))
    for la,lo in zip(obs_lats,obs_lons)], dtype=np.int32)
grid_mesh_nn = np.array([np.argmin(haversine_deg(la,lo,mesh_lats,mesh_lons))
    for la,lo in zip(TARGET_LATS,TARGET_LONS)], dtype=np.int32)
obs_mesh_nn_iso   = obs_mesh_nn[:n_iso]
obs_mesh_nn_accum = obs_mesh_nn[n_iso:]
ms = jnp.array(ms_np); mr = jnp.array(mr_np)

fwd_T = make_forward(32, 4, 32, 0.1, mesh_pos_np, mesh_lats, mesh_lons,
                     obs_mesh_nn_iso, grid_mesh_nn)
fwd_P = make_forward(32, 4, 32, 0.1, mesh_pos_np, mesh_lats, mesh_lons,
                     obs_mesh_nn_accum, grid_mesh_nn)

# init with dummy
dummy_iso   = jnp.zeros(n_iso)
dummy_accum = jnp.zeros(len(all_grid)-n_iso)
params_T = fwd_T.init(jax.random.PRNGKey(0), dummy_iso,   ms, mr)
params_P = fwd_P.init(jax.random.PRNGKey(1), dummy_accum, ms, mr)

raw_T = np.load(f'{RECON_DIR}/weights_kernel/best_T.npz')
raw_P = np.load(f'{RECON_DIR}/weights_kernel/best_P.npz')
params_T = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params_T),
    [raw_T[str(i)] for i in range(len(raw_T.files))])
params_P = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params_P),
    [raw_P[str(i)] for i in range(len(raw_P.files))])

print(f"{'Year':>6} {'T_MSE':>8} {'T_r':>7} {'P_MSE':>8} {'P_r':>7} {'T_RMSE_K':>10}")
for yr in range(2001, 2006):
    era5  = np.load(f'{TGT_DIR}/targets_{yr}.npy')
    anom  = era5 - clim_mean
    t_obs = (anom[obs_grid[:n_iso],  0]/t_std).astype(np.float32)
    p_obs = (anom[obs_grid[n_iso:],  1]/p_std).astype(np.float32)
    t_tgt = (anom[:,0]/t_std).astype(np.float32)
    p_tgt = (anom[:,1]/p_std).astype(np.float32)
    pred_T = np.array(fwd_T.apply(params_T, None, jnp.array(t_obs), ms, mr))
    pred_P = np.array(fwd_P.apply(params_P, None, jnp.array(p_obs), ms, mr))
    t_mse = float(np.mean((pred_T-t_tgt)**2))
    p_mse = float(np.mean((pred_P-p_tgt)**2))
    t_r   = float(np.corrcoef(pred_T, t_tgt)[0,1])
    p_r   = float(np.corrcoef(pred_P, p_tgt)[0,1])
    t_rmse_k = float(np.sqrt(np.mean((pred_T*t_std - t_tgt*t_std)**2)))
    print(f"{yr:>6} {t_mse:>8.4f} {t_r:>7.4f} {p_mse:>8.4f} {p_r:>7.4f} {t_rmse_k:>10.4f}K")

# also check a training year
era5  = np.load(f'{TGT_DIR}/targets_1990.npy')
anom  = era5 - clim_mean
t_obs = (anom[obs_grid[:n_iso],  0]/t_std).astype(np.float32)
p_obs = (anom[obs_grid[n_iso:],  1]/p_std).astype(np.float32)
t_tgt = (anom[:,0]/t_std).astype(np.float32)
pred_T = np.array(fwd_T.apply(params_T, None, jnp.array(t_obs), ms, mr))
t_mse = float(np.mean((pred_T-t_tgt)**2))
t_r   = float(np.corrcoef(pred_T, t_tgt)[0,1])
print(f"\n{'1990':>6} {t_mse:>8.4f} {t_r:>7.4f}  (training year)")
