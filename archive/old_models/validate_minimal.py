import sys, numpy as np, jax, jax.numpy as jnp, os
sys.path.insert(0, '/glade/derecho/scratch/advike/graphcast_recon')
from recon_gnn_minimal import (build_icosahedral_mesh, build_edges,
                                haversine_deg, TARGET_LATS, TARGET_LONS)
from ice_core_loader import IceCoreLoader
import haiku as hk

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
obs_lats   = np.array([float(TARGET_LATS[g]) for g in all_grid], dtype=np.float32)
obs_lons   = np.array([float(TARGET_LONS[g]) for g in all_grid], dtype=np.float32)
obs_grid   = np.array(all_grid, dtype=np.int32)

mesh_lats, mesh_lons, ms_np, mr_np = build_icosahedral_mesh(4)
n_mesh = len(mesh_lats)
mesh_grid = np.array([
    np.argmin(haversine_deg(ml, mlo, TARGET_LATS, TARGET_LONS))
    for ml, mlo in zip(mesh_lats, mesh_lons)
], dtype=np.int32)
o2m_s_np, o2m_r_np = build_edges(obs_lats, obs_lons, mesh_lats, mesh_lons, 45.0)
o2m_s = jnp.array(o2m_s_np); o2m_r = jnp.array(o2m_r_np)
ms    = jnp.array(ms_np);    mr    = jnp.array(mr_np)

# must exactly match training model_fn
mesh_lats_j = jnp.array(mesh_lats)
mesh_lons_j = jnp.array(mesh_lons)
H = 64; n_rounds = 4

def model_fn(obs_f, o2m_s, o2m_r, ms, mr):
    obs_h = hk.Linear(H, name="obs_enc")(obs_f)
    mesh_pos = jnp.stack([
        (mesh_lats_j - (-75.)) / 15.,
        jnp.sin(jnp.radians(mesh_lons_j)),
        jnp.cos(jnp.radians(mesh_lons_j)),
    ], axis=1)
    mesh_h = hk.Linear(H, name="mesh_enc")(mesh_pos)
    o2m_msg = hk.Linear(H, name="o2m")(jnp.concatenate([obs_h[o2m_s], mesh_h[o2m_r]], axis=-1))
    o2m_sum = jax.ops.segment_sum(o2m_msg, o2m_r, num_segments=n_mesh)
    o2m_cnt = jax.ops.segment_sum(jnp.ones(o2m_r.shape[0]), o2m_r, num_segments=n_mesh)
    mesh_h  = jax.nn.relu(mesh_h + o2m_sum / jnp.maximum(o2m_cnt[:,None], 1.))
    for i in range(n_rounds):
        msg  = hk.Linear(H, name=f"m2m_{i}")(jnp.concatenate([mesh_h[ms], mesh_h[mr]], axis=-1))
        agg  = jax.ops.segment_sum(msg, mr, num_segments=n_mesh)
        cnt  = jax.ops.segment_sum(jnp.ones(mr.shape[0]), mr, num_segments=n_mesh)
        mesh_h = jax.nn.relu(mesh_h + agg / jnp.maximum(cnt[:,None], 1.))
    return hk.Linear(2, name="out")(mesh_h)

forward = hk.transform(model_fn)

# build a real sample to init (not zeros -- mesh_lats_j is a closure)
era5_init = np.load(f'{TGT_DIR}/targets_1990.npy')
anom_init = era5_init - clim_mean
t_a = (anom_init[obs_grid,0]/t_std).astype(np.float32)
p_a = (anom_init[obs_grid,1]/p_std).astype(np.float32)
t_av = np.zeros(len(all_grid), dtype=np.float32); t_av[:n_iso] = 1.
p_av = np.zeros(len(all_grid), dtype=np.float32); p_av[n_iso:] = 1.
init_obs = jnp.array(np.stack([t_a, t_av, p_a, p_av], axis=1).astype(np.float32))

params = forward.init(jax.random.PRNGKey(0), init_obs, o2m_s, o2m_r, ms, mr)
print(f"Model leaves at init: {len(jax.tree_util.tree_leaves(params))}")

raw    = np.load(f'{RECON_DIR}/weights_minimal/weights_best.npz')
leaves = [raw[str(i)] for i in range(len(raw.files))]
print(f"Saved leaves: {len(leaves)}")
params = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), leaves)

print("\n-- Validation 2001-2005 (233 mesh nodes) --")
for yr in range(2001, 2006):
    era5  = np.load(f'{TGT_DIR}/targets_{yr}.npy')
    anom  = era5 - clim_mean
    tgt   = np.stack([anom[mesh_grid,0]/t_std, anom[mesh_grid,1]/p_std], axis=1)
    t_a   = (anom[obs_grid,0]/t_std).astype(np.float32)
    p_a   = (anom[obs_grid,1]/p_std).astype(np.float32)
    t_av  = np.zeros(len(all_grid), dtype=np.float32); t_av[:n_iso] = 1.
    p_av  = np.zeros(len(all_grid), dtype=np.float32); p_av[n_iso:] = 1.
    obs_f = jnp.array(np.stack([t_a, t_av, p_a, p_av], axis=1).astype(np.float32))
    pred  = np.array(forward.apply(params, None, obs_f, o2m_s, o2m_r, ms, mr))
    mse   = float(np.mean((pred-tgt)**2))
    r_T   = float(np.corrcoef(pred[:,0], tgt[:,0])[0,1])
    r_P   = float(np.corrcoef(pred[:,1], tgt[:,1])[0,1])
    diff_zero = float(abs(pred[:,0] - 0).mean())
    print(f"  {yr}: MSE={mse:.4f}  T_r={r_T:.4f}  P_r={r_P:.4f}  pred_std={pred[:,0].std():.4f}")

# also check training year to confirm it's not just predicting zero
print("\n-- Training year check (1990) --")
pred_train = np.array(forward.apply(params, None, init_obs, o2m_s, o2m_r, ms, mr))
tgt_train  = np.stack([anom_init[mesh_grid,0]/t_std, anom_init[mesh_grid,1]/p_std], axis=1)
print(f"  1990 train: MSE={float(np.mean((pred_train-tgt_train)**2)):.4f}  T_r={float(np.corrcoef(pred_train[:,0],tgt_train[:,0])[0,1]):.4f}  pred_std={pred_train[:,0].std():.4f}")
