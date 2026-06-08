import os, sys, time, argparse, logging
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

RECON_DIR  = Path("/glade/derecho/scratch/advike/graphcast_recon")
CACHE_DIR  = RECON_DIR / "cache"
TGT_DIR    = CACHE_DIR / "era5_targets"
EMB_DIR    = CACHE_DIR / "embeddings"
CALIB_DIR  = CACHE_DIR / "calibration"
WEIGHTS_DIR = RECON_DIR / "weights2"
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
    dlon = lon2 - lon1
    dlon = (dlon + 180) % 360 - 180
    return np.sqrt(dlat**2 + (dlon * np.cos(np.radians((lat1+lat2)/2)))**2)

def build_icosahedral_mesh(refinements=4):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,phi,0],[1,phi,0],[-1,-phi,0],[1,-phi,0],
        [0,-1,phi],[0,1,phi],[0,-1,-phi],[0,1,-phi],
        [phi,0,-1],[phi,0,1],[-phi,0,-1],[-phi,0,1],
    ], dtype=np.float64)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int32)
    for _ in range(refinements):
        new_faces = []
        cache = {}
        vlist = list(verts)
        def midpt(i, j):
            k = (min(i,j), max(i,j))
            if k not in cache:
                m = (vlist[i] + vlist[j]) / 2
                m = m / np.linalg.norm(m)
                cache[k] = len(vlist)
                vlist.append(m)
            return cache[k]
        for f in faces:
            a,b,c = f
            ab,bc,ca = midpt(a,b), midpt(b,c), midpt(c,a)
            new_faces.extend([[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]])
        verts = np.array(vlist)
        faces = np.array(new_faces, dtype=np.int32)
    x,y,z = verts[:,0], verts[:,1], verts[:,2]
    lats = np.degrees(np.arcsin(np.clip(z,-1,1))).astype(np.float32)
    lons = (np.degrees(np.arctan2(y,x)) % 360).astype(np.float32)
    mask = lats <= -55.0
    idx  = np.where(mask)[0]
    lats, lons = lats[idx], lons[idx]
    old2new = {int(old): new for new, old in enumerate(idx)}
    edges = set()
    for f in faces:
        for i in range(3):
            a,b = int(f[i]), int(f[(i+1)%3])
            if a in old2new and b in old2new:
                u,v = old2new[a], old2new[b]
                if u != v: edges.add((min(u,v), max(u,v)))
    s = np.array([e[0] for e in edges]+[e[1] for e in edges], dtype=np.int32)
    r = np.array([e[1] for e in edges]+[e[0] for e in edges], dtype=np.int32)
    return lats, lons, s, r

def build_bipartite_edges(slats, slons, tlats, tlons, radius):
    senders, receivers = [], []
    for i,(sl,slo) in enumerate(zip(slats,slons)):
        dists = haversine_deg(sl,slo,tlats,tlons)
        tgts  = np.where(dists<=radius)[0]
        senders.append(np.full(len(tgts),i,dtype=np.int32))
        receivers.append(tgts.astype(np.int32))
    if senders:
        return np.concatenate(senders), np.concatenate(receivers)
    return np.array([],dtype=np.int32), np.array([],dtype=np.int32)

def compute_edge_sim(src_grid_idx, s, r, emb):
    se = emb[src_grid_idx[s]]
    te = emb[r]
    sn = se / (np.linalg.norm(se,axis=1,keepdims=True)+1e-8)
    tn = te / (np.linalg.norm(te,axis=1,keepdims=True)+1e-8)
    return (sn*tn).sum(axis=1,keepdims=True).astype(np.float32)

def mlp(sizes, name, act_final=False):
    layers = []
    for i,s in enumerate(sizes):
        layers.append(hk.Linear(s, name=f"{name}_l{i}"))
        if i < len(sizes)-1 or act_final:
            layers.append(jax.nn.silu)
    return hk.Sequential(layers, name=name)

def recon_gnn(obs_f, grid_f, mesh_f, o2m_s, o2m_r, o2m_sim,
              ms, mr, m2g_s, m2g_r, hidden, n_rounds):
    n_mesh = mesh_f.shape[0]
    n_grid = grid_f.shape[0]
    obs_h  = mlp([hidden,hidden], "obs_enc",  act_final=True)(obs_f)
    mesh_h = mlp([hidden,hidden], "mesh_enc", act_final=True)(mesh_f)
    grid_h = mlp([hidden,hidden], "grid_enc", act_final=True)(grid_f)
    log_t  = hk.get_parameter("log_temp", [], init=jnp.zeros)
    temp   = jnp.exp(log_t) + 0.01
    o2m_in = jnp.concatenate([obs_h[o2m_s], mesh_h[o2m_r], o2m_sim], axis=-1)
    o2m_msg= mlp([hidden,hidden], "o2m_edge", act_final=True)(o2m_in)
    rw     = o2m_sim[:,0] / temp
    mxw    = jax.ops.segment_max(rw, o2m_r, num_segments=n_mesh)
    ew     = jnp.exp(rw - mxw[o2m_r])
    sw     = jax.ops.segment_sum(ew, o2m_r, num_segments=n_mesh)
    nw     = ew / (sw[o2m_r] + 1e-8)
    o2m_agg= jax.ops.segment_sum(o2m_msg * nw[:,None], o2m_r, num_segments=n_mesh)
    mesh_h = mlp([hidden,hidden], "o2m_agg", act_final=True)(
        jnp.concatenate([mesh_h, o2m_agg], axis=-1)) + mesh_h
    for i in range(n_rounds):
        ein  = jnp.concatenate([mesh_h[ms], mesh_h[mr]], axis=-1)
        msg  = mlp([hidden,hidden], f"m2m_{i}", act_final=True)(ein)
        agg  = jax.ops.segment_sum(msg, mr, num_segments=n_mesh)
        mesh_h = mlp([hidden,hidden], f"m2m_agg_{i}", act_final=True)(
            jnp.concatenate([mesh_h, agg], axis=-1)) + mesh_h
    m2g_in = jnp.concatenate([mesh_h[m2g_s], grid_h[m2g_r]], axis=-1)
    m2g_msg= mlp([hidden,hidden], "m2g_edge", act_final=True)(m2g_in)
    m2g_agg= jax.ops.segment_sum(m2g_msg, m2g_r, num_segments=n_grid)
    grid_h = mlp([hidden,hidden], "m2g_agg", act_final=True)(
        jnp.concatenate([grid_h, m2g_agg], axis=-1)) + grid_h
    return hk.Linear(N_OUT, name="decoder",
                     w_init=hk.initializers.TruncatedNormal(0.01),
                     b_init=jnp.zeros)(grid_h)

def make_forward_fn(hidden, n_rounds):
    def fwd(obs_f, grid_f, mesh_f, o2m_s, o2m_r, o2m_sim, ms, mr, m2g_s, m2g_r):
        return recon_gnn(obs_f, grid_f, mesh_f, o2m_s, o2m_r, o2m_sim,
                         ms, mr, m2g_s, m2g_r, hidden, n_rounds)
    return hk.transform(fwd)

def grid_idx_to_latlon(g):
    return float(TARGET_LATS[g]), float(TARGET_LONS[g])

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
    clim_emb   = loader.clim_embedding
    iso_grid   = sorted(loader.iso_grid_map.keys())
    accum_grid = sorted(loader.accum_grid_map.keys())
    all_grid   = iso_grid + accum_grid
    n_iso      = len(iso_grid)
    obs_lats   = np.array([grid_idx_to_latlon(g)[0] for g in all_grid], dtype=np.float32)
    obs_lons   = np.array([grid_idx_to_latlon(g)[1] for g in all_grid], dtype=np.float32)
    obs_grid   = np.array(all_grid, dtype=np.int32)

    logger.info("Building icosahedral mesh (refinements=4)...")
    mesh_lats, mesh_lons, mesh_s_np, mesh_r_np = build_icosahedral_mesh(4)
    logger.info(f"  Mesh: {len(mesh_lats)} nodes, {len(mesh_s_np)} edges")
    mesh_grid_map = np.array([
        np.argmin(haversine_deg(ml,mlo,TARGET_LATS,TARGET_LONS))
        for ml,mlo in zip(mesh_lats,mesh_lons)
    ], dtype=np.int32)

    logger.info("Building obs->mesh edges...")
    o2m_s_np, o2m_r_np = build_bipartite_edges(obs_lats, obs_lons,
                                                mesh_lats, mesh_lons, 45.0)
    o2m_sim_np = compute_edge_sim(obs_grid, o2m_s_np, o2m_r_np, clim_emb)
    logger.info(f"  obs->mesh: {len(o2m_s_np):,} edges")

    logger.info("Building mesh->grid edges...")
    m2g_s_np, m2g_r_np = build_bipartite_edges(mesh_lats, mesh_lons,
                                                TARGET_LATS, TARGET_LONS, 5.0)
    logger.info(f"  mesh->grid: {len(m2g_s_np):,} edges")

    # grid features
    lat_n = (TARGET_LATS-(-75.))/15.
    grid_feats_np = np.concatenate([
        lat_n[:,None],
        np.sin(np.radians(TARGET_LONS))[:,None],
        np.cos(np.radians(TARGET_LONS))[:,None],
        clim_emb,
    ], axis=1).astype(np.float32)

    # mesh features
    mlat_n = (mesh_lats-(-75.))/15.
    mesh_feats_np = np.concatenate([
        mlat_n[:,None],
        np.sin(np.radians(mesh_lons))[:,None],
        np.cos(np.radians(mesh_lons))[:,None],
        clim_emb[mesh_grid_map],
    ], axis=1).astype(np.float32)

    grid_feats = jnp.array(grid_feats_np)
    mesh_feats = jnp.array(mesh_feats_np)
    mesh_s = jnp.array(mesh_s_np); mesh_r = jnp.array(mesh_r_np)
    o2m_s  = jnp.array(o2m_s_np);  o2m_r  = jnp.array(o2m_r_np)
    o2m_sim= jnp.array(o2m_sim_np)
    m2g_s  = jnp.array(m2g_s_np);  m2g_r  = jnp.array(m2g_r_np)

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
        t_a  = (anom[obs_grid,0]/t_std).astype(np.float32)
        p_a  = (anom[obs_grid,1]/p_std).astype(np.float32)
        t_av = np.zeros(len(all_grid), dtype=np.float32)
        p_av = np.zeros(len(all_grid), dtype=np.float32)
        t_av[:n_iso] = 1.0; p_av[n_iso:] = 1.0
        obs_f = np.concatenate([
            t_a[:,None], t_av[:,None], p_a[:,None], p_av[:,None],
            ((obs_lats-(-75.))/15.)[:,None],
            np.sin(np.radians(obs_lons))[:,None],
            np.cos(np.radians(obs_lons))[:,None],
            clim_emb[obs_grid],
        ], axis=1).astype(np.float32)
        samples.append((obs_f, tgt))
    logger.info(f"  {len(samples)} samples loaded")

    # model
    forward = make_forward_fn(args.hidden, args.mesh_rounds)
    rng = jax.random.PRNGKey(42)
    params = forward.init(rng, jnp.array(samples[0][0]), grid_feats, mesh_feats,
                          o2m_s, o2m_r, o2m_sim, mesh_s, mesh_r, m2g_s, m2g_r)
    n_p = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"Parameters: {n_p:,}")

    sched = optax.warmup_cosine_decay_schedule(0., args.lr,
                len(samples)*2, len(samples)*args.epochs, args.lr*0.01)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, obs_f, tgt):
        def loss_fn(p):
            pred = forward.apply(p, None, obs_f, grid_feats, mesh_feats,
                                 o2m_s, o2m_r, o2m_sim, mesh_s, mesh_r, m2g_s, m2g_r)
            return jnp.mean((pred-tgt)**2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss

    best = float('inf')
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        rng, sh = jax.random.split(rng)
        idx = jax.random.permutation(sh, len(samples)).tolist()
        el = 0.0
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
            tmp = str(WEIGHTS_DIR/f"checkpoint_{epoch:04d}.tmp.npz")
            np.savez_compressed(tmp, **flat)
            os.rename(tmp, str(WEIGHTS_DIR/f"checkpoint_{epoch:04d}.npz"))

    logger.info(f"Done. Best loss: {best:.5f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--hidden",      type=int,   default=128)
    p.add_argument("--mesh-rounds", type=int,   default=6)
    p.add_argument("--dry-run",     action="store_true")
    args = p.parse_args()

    logger = logging.getLogger("recon_gnn")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_DIR/f"recon_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)

    logger.info("="*60)
    logger.info("Antarctic Reconstruction GNN v2 (icosahedral mesh)")
    logger.info(f"Args: {vars(args)}")
    logger.info("="*60)
    train(args, logger)

if __name__ == "__main__":
    main()
