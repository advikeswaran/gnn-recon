import sys, numpy as np, jax, jax.numpy as jnp, os
sys.path.insert(0, '/glade/derecho/scratch/advike/graphcast_recon')
from train_head import (build_obs_to_target_edges, build_target_features,
                        make_forward_fn, compute_o2t_edge_sim, mse_loss,
                        TARGET_LATS, TARGET_LONS)
from ice_core_loader import IceCoreLoader, grid_index_to_latlon
import optax

ns = np.load('/glade/derecho/scratch/advike/graphcast_recon/cache/era5_targets/norm_stats.npz')
loader = IceCoreLoader(
    data_dir='/glade/derecho/scratch/advike/graphcast_recon/data',
    embeddings_dir='/glade/derecho/scratch/advike/graphcast_recon/cache/embeddings',
    calibration_dir='/glade/derecho/scratch/advike/graphcast_recon/cache/calibration',
    temp_mean=float(ns['temp_mean']), temp_std=float(ns['temp_std']),
    prec_mean=float(ns['prec_mean']), prec_std=float(ns['prec_std']),
)
clim_mean = np.load('/glade/derecho/scratch/advike/graphcast_recon/cache/era5_targets/clim_mean_1979_2000.npy')
anom_std  = np.load('/glade/derecho/scratch/advike/graphcast_recon/cache/era5_targets/anom_std_1979_2000.npz')
t_std = float(anom_std['temp_anom_std'])
p_std = float(anom_std['prec_anom_std'])

era5 = np.load('/glade/derecho/scratch/advike/graphcast_recon/cache/era5_targets/targets_1990.npy')
targets_anom = era5 - clim_mean
targets_norm = np.stack([targets_anom[:,0]/t_std, targets_anom[:,1]/p_std], axis=1).astype(np.float32)

iso_grid   = sorted(loader.iso_grid_map.keys())
accum_grid = sorted(loader.accum_grid_map.keys())
all_grid   = iso_grid + accum_grid
n_iso = len(iso_grid)
obs_lats = np.array([grid_index_to_latlon(g)[0] for g in all_grid], dtype=np.float32)
obs_lons = np.array([grid_index_to_latlon(g)[1] for g in all_grid], dtype=np.float32)

obs_feats = np.zeros((len(all_grid), 518), dtype=np.float32)
for k, g in enumerate(all_grid):
    is_iso = k < n_iso
    obs_feats[k, 0] = targets_anom[g, 0] / t_std if is_iso else 0.0
    obs_feats[k, 1] = 1.0 if is_iso else 0.0
    obs_feats[k, 2] = targets_anom[g, 1] / p_std if not is_iso else 0.0
    obs_feats[k, 3] = 0.0 if is_iso else 1.0
    obs_feats[k, 4] = obs_lats[k]
    obs_feats[k, 5] = obs_lons[k]
    obs_feats[k, 6:] = loader.clim_embedding[g]

tgt_feats = jnp.array(build_target_features(loader.clim_embedding))
t2t = np.load('/glade/derecho/scratch/advike/graphcast_recon/cache/t2t_edges.npz')
t2t_s = jnp.array(t2t['senders']); t2t_r = jnp.array(t2t['receivers'])
o2t_s_np, o2t_r_np = build_obs_to_target_edges(obs_lats, obs_lons, TARGET_LATS, TARGET_LONS, 45.0)
o2t_sim = jnp.array(compute_o2t_edge_sim(np.array(all_grid), o2t_s_np, o2t_r_np, loader.clim_embedding))
o2t_s = jnp.array(o2t_s_np); o2t_r = jnp.array(o2t_r_np)
print(f'o2t edges: {len(o2t_s_np):,}  n_obs: {len(all_grid)}')

obs_jnp     = jnp.array(obs_feats)
targets_jnp = jnp.array(targets_norm)

forward = make_forward_fn(128, 6)
params  = forward.init(jax.random.PRNGKey(42), obs_jnp, tgt_feats, o2t_s, o2t_r, o2t_sim, t2t_s, t2t_r)
pred0   = np.array(forward.apply(params, None, obs_jnp, tgt_feats, o2t_s, o2t_r, o2t_sim, t2t_s, t2t_r))
print(f'Init MSE: {np.mean((pred0-targets_norm)**2):.4f}  zero_MSE: {np.mean(targets_norm**2):.4f}')

optimiser = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-2))
opt_state = optimiser.init(params)

@jax.jit
def step(params, opt_state):
    def loss_fn(p):
        return mse_loss(forward.apply(p, None, obs_jnp, tgt_feats,
                                      o2t_s, o2t_r, o2t_sim, t2t_s, t2t_r),
                        targets_jnp)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt = optimiser.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), new_opt, loss

print("Training on single sample 1990, 200 epochs, lr=0.01")
for epoch in range(1, 201):
    params, opt_state, loss = step(params, opt_state)
    if epoch % 20 == 0:
        pred = np.array(forward.apply(params, None, obs_jnp, tgt_feats,
                                      o2t_s, o2t_r, o2t_sim, t2t_s, t2t_r))
        print(f'Epoch {epoch:3d}: loss={float(loss):.5f}  pred_std={pred[:,0].std():.4f}  range=[{pred[:,0].min():.3f},{pred[:,0].max():.3f}]')
print("Done")
