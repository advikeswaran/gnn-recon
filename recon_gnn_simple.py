"""
recon_gnn_simple.py -- Bare bones GNN for Antarctic spatial interpolation.

No embeddings. No attention. Just:
1. Obs nodes (135) with T/P anomaly values + lat/lon
2. Obs -> Icosahedral mesh (radius-based, uniform aggregation)
3. Mesh -> Mesh (message passing, multiple rounds)
4. Mesh -> Target grid

If this works, we add complexity back gradually.
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
WEIGHTS_DIR = RECON_DIR / "weights3"
LOG_DIR    = RECON_DIR / "logs"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RECON_DIR))

LAT_VALS = np.arange(-60, -91, -1, dtype=np.float32)
LON_VALS = np.arange(0, 360, 1, dtype=np.float32)
TARGET_LATS, TARGET_LONS = np.meshgrid(LAT_VALS, LON_VALS, indexing='ij')
TARGET_LATS = TARGET_LATS.reshape(-1)
TARGET_LONS = TARGET_LONS.reshape(-1)
N_GRID = len(TARGET_LATS)
N_OUT  = 2

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
    """Normalised position features: [lat_norm, sin_lon, cos_lon]"""
    return np.stack([
        (lats - (-75.)) / 15.,
        np.sin(np.radians(lons)),
        np.cos(np.radians(lons)),
    ], axis=1).astype(np.float32)

def mlp(sizes, name):
    layers = []
    for i,s in enumerate(sizes):
        layers.append(hk.Linear(s, name=f"{name}_l{i}"))
        if i < len(sizes)-1:
            layers.append(jax.nn.relu)
    return hk.Sequential(layers, name=name)

def gnn(obs_f, grid_f, mesh_f,
        o2m_s, o2m_r, ms, mr, m2g_s, m2g_r,
        hidden, n_rounds):
    n_mesh = mesh_f.shape[0]
    n_grid = grid_f.shape[0]

    # encode obs with LINEAR projection only (no nonlinearity)
    # this ensures gradient flows directly from loss to obs anomaly values
    # without being killed by ReLU dead neurons
    obs_h  = hk.Linear(hidden, name="obs_enc")(obs_f)
    mesh_h = mlp([hidden, hidden], "mesh_enc")(mesh_f)
    grid_h = mlp([hidden, hidden], "grid_enc")(grid_f)

    # obs -> mesh (mean aggregation to prevent dilution)
    o2m_msg = mlp([hidden, hidden], "o2m_edge")(
        jnp.concatenate([obs_h[o2m_s], mesh_h[o2m_r]], axis=-1))
    # segment_mean: normalize by number of senders per receiver
    o2m_sum   = jax.ops.segment_sum(o2m_msg, o2m_r, num_segments=n_mesh)
    o2m_count = jax.ops.segment_sum(jnp.ones(o2m_r.shape[0]), o2m_r,
                                     num_segments=n_mesh)
    o2m_agg   = o2m_sum / jnp.maximum(o2m_count[:, None], 1.0)
    mesh_h  = mlp([hidden, hidden], "o2m_upd")(
        jnp.concatenate([mesh_h, o2m_agg], axis=-1)) + mesh_h

    # mesh -> mesh (residual rounds)
    for i in range(n_rounds):
        msg  = mlp([hidden, hidden], f"m2m_{i}")(
            jnp.concatenate([mesh_h[ms], mesh_h[mr]], axis=-1))
        agg  = jax.ops.segment_sum(msg, mr, num_segments=n_mesh)
        mesh_h = mlp([hidden, hidden], f"m2m_upd_{i}")(
            jnp.concatenate([mesh_h, agg], axis=-1)) + mesh_h

    # mesh -> grid
    m2g_msg = mlp([hidden, hidden], "m2g_edge")(
        jnp.concatenate([mesh_h[m2g_s], grid_h[m2g_r]], axis=-1))
    m2g_agg = jax.ops.segment_sum(m2g_msg, m2g_r, num_segments=n_grid)
    grid_h  = mlp([hidden, hidden], "m2g_upd")(
        jnp.concatenate([grid_h, m2g_agg], axis=-1)) + grid_h

    # decode with near-zero init
    return hk.Linear(N_OUT, name="out",
                     w_init=hk.initializers.TruncatedNormal(0.1),
                     b_init=jnp.zeros)(grid_h)

def make_fwd(hidden, n_rounds):
    def fwd(obs_f, grid_f, mesh_f, o2m_s, o2m_r, ms, mr, m2g_s, m2g_r):
        return gnn(obs_f, grid_f, mesh_f, o2m_s, o2m_r, ms, mr,
                   m2g_s, m2g_r, hidden, n_rounds)
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

    # build mesh
    logger.info("Building icosahedral mesh...")
    mesh_lats, mesh_lons, ms_np, mr_np = build_icosahedral_mesh(4)
    logger.info(f"  {len(mesh_lats)} nodes, {len(ms_np)} edges")

    # edges
    logger.info("Building edges...")
    o2m_s_np, o2m_r_np = build_edges(obs_lats, obs_lons, mesh_lats, mesh_lons, 45.0)
    m2g_s_np, m2g_r_np = build_edges(mesh_lats, mesh_lons, TARGET_LATS, TARGET_LONS, 5.0)
    logger.info(f"  obs->mesh: {len(o2m_s_np):,}  mesh->grid: {len(m2g_s_np):,}")

    # static node features (position only, no embeddings)
    obs_pos  = norm_pos(obs_lats, obs_lons)   # (135, 3)
    grid_pos = norm_pos(TARGET_LATS, TARGET_LONS)  # (11160, 3)
    mesh_pos = norm_pos(mesh_lats, mesh_lons)      # (n_mesh, 3)

    grid_f = jnp.array(grid_pos)
    mesh_f = jnp.array(mesh_pos)
    o2m_s  = jnp.array(o2m_s_np); o2m_r = jnp.array(o2m_r_np)
    mss    = jnp.array(ms_np);    msr   = jnp.array(mr_np)
    m2g_s  = jnp.array(m2g_s_np); m2g_r = jnp.array(m2g_r_np)

    # dataset
    logger.info("Building dataset...")
    samples = []
    years = TRAIN_YEARS[:3] if args.dry_run else TRAIN_YEARS
    for yr in years:
        p = TGT_DIR/f"targets_{yr}.npy"
        if not p.exists(): continue
        era5 = np.load(p)
        anom = era5 - clim_mean
        tgt  = np.stack([anom[:,0]/t_std, anom[:,1]/p_std], axis=1).astype(np.float32)
        # obs features: [T_val, T_avail, P_val, P_avail, lat_norm, sin_lon, cos_lon]
        t_a = (anom[obs_grid,0]/t_std).astype(np.float32)
        p_a = (anom[obs_grid,1]/p_std).astype(np.float32)
        t_av = np.zeros(len(all_grid), dtype=np.float32); t_av[:n_iso] = 1.
        p_av = np.zeros(len(all_grid), dtype=np.float32); p_av[n_iso:] = 1.
        # obs features: anomaly values + availability only (no position)
        # position is already encoded in graph structure (edges built from lat/lon)
        # including position gives model a stable fallback that bypasses anomaly values
        obs_f = np.concatenate([
            t_a[:,None], t_av[:,None], p_a[:,None], p_av[:,None]
        ], axis=1).astype(np.float32)  # (135, 4)
        samples.append((obs_f, tgt))
    logger.info(f"  {len(samples)} samples")

    # model
    forward = make_fwd(args.hidden, args.mesh_rounds)
    rng = jax.random.PRNGKey(42)
    s0  = jnp.array(samples[0][0])
    params = forward.init(rng, s0, grid_f, mesh_f,
                          o2m_s, o2m_r, mss, msr, m2g_s, m2g_r)
    n_p = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"Parameters: {n_p:,}")

    sched = optax.warmup_cosine_decay_schedule(
        0., args.lr, len(samples)*2, len(samples)*args.epochs, args.lr*0.01)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, obs_f, tgt):
        def loss_fn(p):
            pred = forward.apply(p, None, obs_f, grid_f, mesh_f,
                                 o2m_s, o2m_r, mss, msr, m2g_s, m2g_r)
            return jnp.mean((pred-tgt)**2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss

    best = float('inf')
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        rng, sh = jax.random.split(rng)
        idx = jax.random.permutation(sh, len(samples)).tolist()
        el = 0.
        for i, si in enumerate(idx):
            params, opt_state, loss = step(params, opt_state,
                jnp.array(samples[si][0]), jnp.array(samples[si][1]))
            el += float(loss)
            if args.dry_run and i >= 1:
                logger.info("  [DRY RUN] stop"); break
        avg = el/len(idx)
        logger.info(f"Epoch {epoch:4d}/{args.epochs} | loss={avg:.5f} | time={time.time()-t0:.1f}s")
        if avg < best:
            best = avg
            flat = {str(i):v for i,v in enumerate(jax.tree_util.tree_leaves(jax.device_get(params)))}
            tmp = str(WEIGHTS_DIR/"weights_best.tmp.npz")
            np.savez_compressed(tmp, **flat)
            os.rename(tmp, str(WEIGHTS_DIR/"weights_best.npz"))
            logger.info(f"  Best: {best:.5f}")
        elif epoch % 10 == 0:
            flat = {str(i):v for i,v in enumerate(jax.tree_util.tree_leaves(jax.device_get(params)))}
            tmp = str(WEIGHTS_DIR/f"ckpt_{epoch:04d}.tmp.npz")
            np.savez_compressed(tmp, **flat)
            os.rename(tmp, str(WEIGHTS_DIR/f"ckpt_{epoch:04d}.npz"))
    logger.info(f"Done. Best: {best:.5f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--hidden",      type=int,   default=128)
    p.add_argument("--mesh-rounds", type=int,   default=6)
    p.add_argument("--dry-run",     action="store_true")
    args = p.parse_args()
    logger = logging.getLogger("simple_gnn")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_DIR/f"simple_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fh.setFormatter(fmt); sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    logger.info("Simple GNN (no embeddings)"); logger.info(f"Args: {vars(args)}")
    train(args, logger)

if __name__ == "__main__":
    main()
