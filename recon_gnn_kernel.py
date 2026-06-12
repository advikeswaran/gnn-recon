"""
recon_gnn_kernel.py -- Physics-constrained kernel regression for Antarctic reconstruction.

Instead of arbitrary message passing, learn a positive-definite covariance kernel:
    pred(j) = K(j, obs) @ inv(K(obs,obs) + noise*I) @ obs_vals

Where K(i,j) = phi(i) . phi(j) / norm  (learned feature map -> guaranteed PSD kernel)

phi is computed by a GNN on the icosahedral mesh, giving each node a learned
representation that captures atmospheric teleconnections.

This is equivalent to Gaussian Process regression with a learned kernel,
which is the mathematically optimal interpolation method.
"""
import os, sys, time, argparse, logging
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import haiku as hk
import optax

RECON_DIR  = Path("/glade/derecho/scratch/advike/graphcast_recon")
CACHE_DIR  = RECON_DIR / "cache"
TGT_DIR    = CACHE_DIR / "era5_targets"
EMB_DIR    = CACHE_DIR / "embeddings"
CALIB_DIR  = CACHE_DIR / "calibration"
WEIGHTS_DIR = RECON_DIR / "weights_kernel"
LOG_DIR    = RECON_DIR / "logs"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RECON_DIR))

LAT_VALS = np.arange(-60, -91, -1, dtype=np.float32)
LON_VALS = np.arange(0, 360, 1, dtype=np.float32)
TARGET_LATS, TARGET_LONS = np.meshgrid(LAT_VALS, LON_VALS, indexing='ij')
TARGET_LATS = TARGET_LATS.reshape(-1)
TARGET_LONS = TARGET_LONS.reshape(-1)
N_GRID = len(TARGET_LATS)

TRAIN_YEARS = list(range(1979, 2001)) + list(range(2006, 2018))
VAL_YEARS   = list(range(2001, 2006))

def haversine_deg(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon = (lon2 - lon1 + 180) % 360 - 180
    return np.sqrt(dlat**2 + (dlon * np.cos(np.radians((lat1+lat2)/2)))**2)

def build_icosahedral_mesh(refinements=4):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,phi,0],[1,phi,0],[-1,-phi,0],[1,-phi,0],
        [0,-1,phi],[0,1,phi],[0,-1,-phi],[0,1,-phi],
        [phi,0,-1],[phi,0,1],[-phi,0,-1],[-phi,0,1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int32)
    for _ in range(refinements):
        cache, vlist, new_faces = {}, list(verts), []
        def mid(i,j):
            k=(min(i,j),max(i,j))
            if k not in cache:
                m=(vlist[i]+vlist[j])/2; m/=np.linalg.norm(m)
                cache[k]=len(vlist); vlist.append(m)
            return cache[k]
        for f in faces:
            a,b,c=f; ab,bc,ca=mid(a,b),mid(b,c),mid(c,a)
            new_faces.extend([[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]])
        verts=np.array(vlist); faces=np.array(new_faces,dtype=np.int32)
    x,y,z=verts[:,0],verts[:,1],verts[:,2]
    lats=np.degrees(np.arcsin(np.clip(z,-1,1))).astype(np.float32)
    lons=(np.degrees(np.arctan2(y,x))%360).astype(np.float32)
    mask=lats<=-55.; idx=np.where(mask)[0]
    lats,lons=lats[idx],lons[idx]
    o2n={int(o):n for n,o in enumerate(idx)}
    edges=set()
    for f in faces:
        for i in range(3):
            a,b=int(f[i]),int(f[(i+1)%3])
            if a in o2n and b in o2n:
                u,v=o2n[a],o2n[b]
                if u!=v: edges.add((min(u,v),max(u,v)))
    s=np.array([e[0] for e in edges]+[e[1] for e in edges],dtype=np.int32)
    r=np.array([e[1] for e in edges]+[e[0] for e in edges],dtype=np.int32)
    return lats, lons, s, r

def build_edges(slats, slons, tlats, tlons, radius):
    senders, receivers = [], []
    for i,(sl,slo) in enumerate(zip(slats,slons)):
        d=haversine_deg(sl,slo,tlats,tlons)
        t=np.where(d<=radius)[0]
        senders.append(np.full(len(t),i,dtype=np.int32))
        receivers.append(t.astype(np.int32))
    if senders:
        return np.concatenate(senders), np.concatenate(receivers)
    return np.array([],dtype=np.int32), np.array([],dtype=np.int32)

def norm_pos(lats, lons):
    return np.stack([
        (lats-(-75.))/15.,
        np.sin(np.radians(lons)),
        np.cos(np.radians(lons)),
    ], axis=1).astype(np.float32)

def kernel_gnn(
    all_pos,      # (n_all, 3) positional features for ALL nodes (obs + grid)
    mesh_pos,     # (n_mesh, 3) mesh node positions
    obs_idx,      # (n_obs,) indices into all_pos for obs nodes
    grid_idx,     # (n_grid,) indices into all_pos for grid nodes  
    obs_vals,     # (n_obs,) observed T or P anomaly values
    o2m_s, o2m_r, # obs->mesh edges
    ms, mr,       # mesh->mesh edges
    m2g_s, m2g_r, # mesh->grid edges
    hidden, n_rounds, kernel_dim, noise_var
):
    n_obs  = obs_idx.shape[0]
    n_mesh = mesh_pos.shape[0]
    n_grid = grid_idx.shape[0]

    # 1. Learn feature map phi: position -> kernel_dim embedding
    # phi(i) . phi(j) defines our PSD kernel K(i,j)
    def phi_net(x):
        h = hk.Linear(hidden, name="phi_l1")(x)
        h = jax.nn.relu(h)
        h = hk.Linear(hidden, name="phi_l2")(h)
        h = jax.nn.relu(h)
        h = hk.Linear(kernel_dim, name="phi_out")(h)
        # L2 normalize to bound the kernel values
        return h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)

    # compute phi for all nodes via mesh message passing
    # first get base features for mesh nodes
    mesh_h = phi_net(mesh_pos)  # (n_mesh, kernel_dim)

    # propagate information through mesh (captures teleconnections)
    for i in range(n_rounds):
        msg    = jax.ops.segment_sum(mesh_h[ms], mr, num_segments=n_mesh)
        cnt    = jax.ops.segment_sum(jnp.ones(mr.shape[0]), mr, num_segments=n_mesh)
        mesh_h = mesh_h + hk.Linear(kernel_dim, name=f"m2m_{i}")(
            jnp.concatenate([mesh_h, msg / jnp.maximum(cnt[:,None], 1.)], axis=-1))
        mesh_h = mesh_h / (jnp.linalg.norm(mesh_h, axis=-1, keepdims=True) + 1e-8)

    # interpolate mesh features to obs and grid nodes
    def mesh_to_nodes(node_lats, node_lons):
        # find nearest mesh node for each target node
        nn = np.array([
            np.argmin(haversine_deg(la, lo,
                      np.array(mesh_pos[:,0]*15.-75.), # approx, good enough
                      np.array(mesh_pos[:,1])))
            for la, lo in zip(node_lats, node_lons)
        ], dtype=np.int32)
        return mesh_h[nn]

    # get phi for obs and grid nodes from nearest mesh node
    obs_lats_np  = np.array(all_pos[obs_idx, 0]) * 15. - 75.  # denorm
    obs_lons_np  = np.degrees(np.arctan2(
        np.array(all_pos[obs_idx, 1]),
        np.array(all_pos[obs_idx, 2]))) % 360
    grid_lats_np = np.array(TARGET_LATS)
    grid_lons_np = np.array(TARGET_LONS)

    # snap obs to nearest mesh node
    obs_mesh_nn = np.array([
        np.argmin(haversine_deg(la, lo, np.array(mesh_lats_g), np.array(mesh_lons_g)))
        for la, lo in zip(obs_lats_np, obs_lons_np)
    ], dtype=np.int32)

    # snap grid to nearest mesh node  
    grid_mesh_nn = np.array([
        np.argmin(haversine_deg(la, lo, np.array(mesh_lats_g), np.array(mesh_lons_g)))
        for la, lo in zip(grid_lats_np[:n_grid], grid_lons_np[:n_grid])
    ], dtype=np.int32)

    phi_obs  = mesh_h[jnp.array(obs_mesh_nn)]   # (n_obs, kernel_dim)
    phi_grid = mesh_h[jnp.array(grid_mesh_nn)]  # (n_grid, kernel_dim)

    # 2. Kernel regression: pred = K(grid,obs) @ (K(obs,obs) + noise*I)^-1 @ obs_vals
    # K(i,j) = phi(i) . phi(j)  -- guaranteed PSD
    K_oo = phi_obs  @ phi_obs.T   # (n_obs, n_obs) -- PSD
    K_go = phi_grid @ phi_obs.T   # (n_grid, n_obs)

    # add noise for numerical stability and regularization
    log_noise = hk.get_parameter("log_noise", [], init=lambda *_: jnp.log(jnp.array(noise_var)))
    noise = jnp.exp(log_noise)
    K_oo_reg = K_oo + noise * jnp.eye(n_obs)

    # solve (K_oo + noise*I) alpha = obs_vals
    alpha = jnp.linalg.solve(K_oo_reg, obs_vals)  # (n_obs,)

    # predict
    pred = K_go @ alpha  # (n_grid,)
    return pred


def make_forward(hidden, n_rounds, kernel_dim, noise_var,
                 mesh_pos_np, mesh_lats, mesh_lons,
                 obs_mesh_nn_np, grid_mesh_nn_np,
                 obs_node_noise_np=None):
    """Create forward function for T or P separately.
    obs_node_noise_np: (n_obs,) per-node noise from calibration R2.
                       If None, uses learned scalar noise (homoscedastic).
    """
    mesh_pos_j  = jnp.array(mesh_pos_np)
    obs_nn_j    = jnp.array(obs_mesh_nn_np)
    grid_nn_j   = jnp.array(grid_mesh_nn_np)
    n_mesh = len(mesh_lats)
    # heteroscedastic: fixed per-node noise scaled by learned global factor
    use_hetero = obs_node_noise_np is not None
    if use_hetero:
        obs_noise_j = jnp.array(obs_node_noise_np.astype(np.float32))

    def fwd(obs_vals, ms, mr):
        # compute phi for all mesh nodes via message passing
        mesh_h = hk.Linear(kernel_dim, name="phi_enc")(mesh_pos_j)
        mesh_h = jax.nn.relu(mesh_h)

        for i in range(n_rounds):
            msg    = jax.ops.segment_sum(mesh_h[ms], mr, num_segments=n_mesh)
            cnt    = jax.ops.segment_sum(jnp.ones(mr.shape[0]), mr, num_segments=n_mesh)
            upd    = hk.Linear(kernel_dim, name=f"m2m_{i}")(
                jnp.concatenate([mesh_h, msg / jnp.maximum(cnt[:,None], 1.)], axis=-1))
            mesh_h = mesh_h + jax.nn.relu(upd)
            mesh_h = mesh_h / (jnp.linalg.norm(mesh_h, axis=-1, keepdims=True) + 1e-8)

        phi_obs  = mesh_h[obs_nn_j]   # (n_obs, kernel_dim)
        phi_grid = mesh_h[grid_nn_j]  # (n_grid, kernel_dim)

        K_oo = phi_obs  @ phi_obs.T
        K_go = phi_grid @ phi_obs.T

        log_noise = hk.get_parameter("log_noise", [],
                        init=lambda *_: jnp.log(jnp.array(noise_var)))
        noise_scale = jnp.exp(log_noise)

        if use_hetero:
            # heteroscedastic: diagonal noise proportional to (1-R2)/R2
            # noise_scale is a global multiplier, obs_noise_j is per-node shape
            noise_diag = noise_scale * obs_noise_j
            K_oo_reg = K_oo + jnp.diag(noise_diag)
        else:
            K_oo_reg = K_oo + noise_scale * jnp.eye(obs_vals.shape[0])

        alpha = jnp.linalg.solve(K_oo_reg, obs_vals)
        return K_go @ alpha

    return hk.transform(fwd)


def train(args, logger):
    from ice_core_loader import IceCoreLoader
    ns       = np.load(TGT_DIR/"norm_stats.npz")
    anom_std = np.load(TGT_DIR/"anom_std_1979_2000.npz")
    clim_mean= np.load(TGT_DIR/"clim_mean_1979_2000.npy")
    t_std    = float(anom_std['temp_anom_std'])
    p_std    = float(anom_std['prec_anom_std'])
    loader   = IceCoreLoader(
        data_dir=str(RECON_DIR/"data"),
        embeddings_dir=str(EMB_DIR),
        calibration_dir=str(CALIB_DIR),
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

    logger.info("Building mesh...")
    mesh_lats, mesh_lons, ms_np, mr_np = build_icosahedral_mesh(args.refinements)
    n_mesh = len(mesh_lats)
    mesh_pos_np = norm_pos(mesh_lats, mesh_lons)
    logger.info(f"  {n_mesh} nodes")

    # snap obs and grid nodes to nearest mesh node
    logger.info("Snapping nodes to mesh...")
    obs_mesh_nn = np.array([
        np.argmin(haversine_deg(la, lo, mesh_lats, mesh_lons))
        for la, lo in zip(obs_lats, obs_lons)
    ], dtype=np.int32)
    grid_mesh_nn = np.array([
        np.argmin(haversine_deg(la, lo, mesh_lats, mesh_lons))
        for la, lo in zip(TARGET_LATS, TARGET_LONS)
    ], dtype=np.int32)
    logger.info(f"  obs->mesh: {len(np.unique(obs_mesh_nn))} unique mesh nodes used by obs")
    logger.info(f"  grid->mesh: {len(np.unique(grid_mesh_nn))} unique mesh nodes used by grid")

    ms = jnp.array(ms_np); mr = jnp.array(mr_np)

    # build dataset
    logger.info("Building dataset...")
    samples = []
    years = TRAIN_YEARS[:3] if args.dry_run else TRAIN_YEARS
    for yr in years:
        p = TGT_DIR/f"targets_{yr}.npy"
        if not p.exists(): continue
        era5 = np.load(p)
        anom = era5 - clim_mean
        t_obs = (anom[obs_grid,0]/t_std).astype(np.float32)
        p_obs = (anom[obs_grid,1]/p_std).astype(np.float32)
        t_tgt = (anom[:,0]/t_std).astype(np.float32)
        p_tgt = (anom[:,1]/p_std).astype(np.float32)
        t_obs_iso   = t_obs[:n_iso]
        p_obs_accum = p_obs[n_iso:]
        samples.append((t_obs_iso, p_obs_accum, t_tgt, p_tgt))
    logger.info(f"  {len(samples)} samples")

    obs_mesh_nn_iso   = obs_mesh_nn[:n_iso]
    obs_mesh_nn_accum = obs_mesh_nn[n_iso:]

    # load per-node R2 noise
    iso_node_noise   = np.load(str(CACHE_DIR/'iso_node_noise.npy'))
    accum_node_noise = np.load(str(CACHE_DIR/'accum_node_noise.npy'))
    logger.info(f"  iso noise: mean={iso_node_noise.mean():.1f} max={iso_node_noise.max():.1f}")
    logger.info(f"  accum noise: mean={accum_node_noise.mean():.1f} max={accum_node_noise.max():.1f}")

    fwd_T = make_forward(args.hidden, args.mesh_rounds, args.kernel_dim,
                         args.noise_var, mesh_pos_np, mesh_lats, mesh_lons,
                         obs_mesh_nn_iso, grid_mesh_nn,
                         obs_node_noise_np=iso_node_noise)
    fwd_P = make_forward(args.hidden, args.mesh_rounds, args.kernel_dim,
                         args.noise_var, mesh_pos_np, mesh_lats, mesh_lons,
                         obs_mesh_nn_accum, grid_mesh_nn,
                         obs_node_noise_np=accum_node_noise)

    # init
    rng = jax.random.PRNGKey(42)
    rng, r1, r2 = jax.random.split(rng, 3)
    params_T = fwd_T.init(r1, jnp.array(samples[0][0]), ms, mr)
    params_P = fwd_P.init(r2, jnp.array(samples[0][1]), ms, mr)
    n_p = (sum(x.size for x in jax.tree_util.tree_leaves(params_T)) +
           sum(x.size for x in jax.tree_util.tree_leaves(params_P)))
    logger.info(f"Parameters: {n_p:,} (T: {sum(x.size for x in jax.tree_util.tree_leaves(params_T)):,} + P: {sum(x.size for x in jax.tree_util.tree_leaves(params_P)):,})")

    sched = optax.warmup_cosine_decay_schedule(
        0., args.lr, len(samples)*2, len(samples)*args.epochs, args.lr*0.01)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_T = opt.init(params_T)
    opt_P = opt.init(params_P)

    @jax.jit
    def step_T(params, opt_state, t_obs, t_tgt):
        def loss_fn(p):
            pred = fwd_T.apply(p, None, t_obs, ms, mr)
            return jnp.mean((pred - t_tgt)**2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss

    @jax.jit
    def step_P(params, opt_state, p_obs, p_tgt):
        def loss_fn(p):
            pred = fwd_P.apply(p, None, p_obs, ms, mr)
            return jnp.mean((pred - p_tgt)**2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss

    # load all 5 validation years
    val_samples = []
    for val_yr in [2001, 2002, 2003, 2004, 2005]:
        ve  = np.load(TGT_DIR/f"targets_{val_yr}.npy")
        va  = ve - clim_mean
        val_samples.append((
            (va[obs_grid[:n_iso],  0]/t_std).astype(np.float32),
            (va[obs_grid[n_iso:],  1]/p_std).astype(np.float32),
            (va[:,0]/t_std).astype(np.float32),
            (va[:,1]/p_std).astype(np.float32),
        ))
    logger.info(f"  {len(val_samples)} validation years: 2001-2005")

    best_val = float('inf')

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        rng, sh = jax.random.split(rng)
        idx = jax.random.permutation(sh, len(samples)).tolist()
        tl = pl = 0.
        for i, si in enumerate(idx):
            t_obs_i, p_obs_i, t_tgt_i, p_tgt_i = samples[si]
            params_T, opt_T, lt = step_T(params_T, opt_T,
                jnp.array(t_obs_i), jnp.array(t_tgt_i))
            params_P, opt_P, lp = step_P(params_P, opt_P,
                jnp.array(p_obs_i), jnp.array(p_tgt_i))
            tl += float(lt); pl += float(lp)
            if args.dry_run and i >= 1: break
        avg_T = tl/len(idx); avg_P = pl/len(idx)

        # validate every epoch
        vl_T_all = vl_P_all = r_T_all = 0.
        for vt_obs, vp_obs, vt_tgt, vp_tgt in val_samples:
            vp_T = np.array(fwd_T.apply(params_T, None, jnp.array(vt_obs), ms, mr))
            vp_P = np.array(fwd_P.apply(params_P, None, jnp.array(vp_obs), ms, mr))
            vl_T_all += float(np.mean((vp_T - vt_tgt)**2))
            vl_P_all += float(np.mean((vp_P - vp_tgt)**2))
            r_T_all  += float(np.corrcoef(vp_T, vt_tgt)[0,1])
        vl_T = vl_T_all / len(val_samples)
        vl_P = vl_P_all / len(val_samples)
        r_T  = r_T_all  / len(val_samples)
        vl   = (vl_T + vl_P) / 2

        if epoch % 10 == 0 or epoch <= 5:
            logger.info(f"Epoch {epoch:4d} | train T={avg_T:.4f} P={avg_P:.4f} | "
                       f"val T={vl_T:.4f} P={vl_P:.4f} | T_r={r_T:.3f} | "
                       f"t={time.time()-t0:.1f}s")

        if vl < best_val:
            best_val = vl
            flat_T = {str(i):v for i,v in enumerate(jax.tree_util.tree_leaves(jax.device_get(params_T)))}
            flat_P = {str(i):v for i,v in enumerate(jax.tree_util.tree_leaves(jax.device_get(params_P)))}
            np.savez_compressed(str(WEIGHTS_DIR/"best_T.npz"), **flat_T)
            np.savez_compressed(str(WEIGHTS_DIR/"best_P.npz"), **flat_P)
            logger.info(f"  Best val: {best_val:.4f} (epoch {epoch})")

    logger.info(f"Done. Best val: {best_val:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--hidden",      type=int,   default=32)
    p.add_argument("--kernel-dim",  type=int,   default=16)
    p.add_argument("--mesh-rounds", type=int,   default=3)
    p.add_argument("--noise-var",   type=float, default=0.1)
    p.add_argument("--refinements", type=int,   default=4)
    p.add_argument("--dry-run",     action="store_true")
    args = p.parse_args()

    logger = logging.getLogger("kernel_gnn")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(LOG_DIR/f"kernel_{time.strftime('%Y%m%d_%H%M%S')}.log")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)

    logger.info("="*60)
    logger.info("Antarctic Reconstruction -- Kernel GNN")
    logger.info(f"Args: {vars(args)}")
    logger.info("="*60)
    train(args, logger)

if __name__ == "__main__":
    main()
