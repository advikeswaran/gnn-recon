"""
Absolute minimal GNN test: predict ERA5 at 233 mesh nodes from 135 obs nodes.
No mesh->grid step. If this can't learn, the architecture is fundamentally broken.
"""
import os, sys, time, argparse, logging
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import haiku as hk
import optax

RECON_DIR = Path("/glade/derecho/scratch/advike/graphcast_recon")
CACHE_DIR  = RECON_DIR / "cache"
TGT_DIR    = CACHE_DIR / "era5_targets"
LOG_DIR    = RECON_DIR / "logs"
sys.path.insert(0, str(RECON_DIR))

LAT_VALS = np.arange(-60, -91, -1, dtype=np.float32)
LON_VALS = np.arange(0, 360, 1, dtype=np.float32)
TARGET_LATS, TARGET_LONS = np.meshgrid(LAT_VALS, LON_VALS, indexing='ij')
TARGET_LATS = TARGET_LATS.reshape(-1)
TARGET_LONS = TARGET_LONS.reshape(-1)

TRAIN_YEARS = list(range(1979, 2001)) + list(range(2006, 2018))

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

def train(args, logger):
    from ice_core_loader import IceCoreLoader
    ns       = np.load(TGT_DIR/"norm_stats.npz")
    anom_std = np.load(TGT_DIR/"anom_std_1979_2000.npz")
    clim_mean= np.load(TGT_DIR/"clim_mean_1979_2000.npy")
    t_std    = float(anom_std['temp_anom_std'])
    p_std    = float(anom_std['prec_anom_std'])
    loader   = IceCoreLoader(
        data_dir=str(RECON_DIR/"data"),
        embeddings_dir=str(RECON_DIR/"cache/embeddings"),
        calibration_dir=str(RECON_DIR/"cache/calibration"),
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
    mesh_lats, mesh_lons, ms_np, mr_np = build_icosahedral_mesh(4)
    n_mesh = len(mesh_lats)
    logger.info(f"  {n_mesh} mesh nodes")

    # snap mesh nodes to nearest grid node for ERA5 lookup
    mesh_grid = np.array([
        np.argmin(haversine_deg(ml, mlo, TARGET_LATS, TARGET_LONS))
        for ml, mlo in zip(mesh_lats, mesh_lons)
    ], dtype=np.int32)

    logger.info("Building edges...")
    o2m_s_np, o2m_r_np = build_edges(obs_lats, obs_lons, mesh_lats, mesh_lons, 45.0)
    logger.info(f"  obs->mesh: {len(o2m_s_np):,}")

    o2m_s = jnp.array(o2m_s_np); o2m_r = jnp.array(o2m_r_np)
    ms    = jnp.array(ms_np);    mr    = jnp.array(mr_np)

    # dataset: predict ERA5 at mesh nodes from ERA5 at obs nodes
    logger.info("Building dataset...")
    samples = []
    years = TRAIN_YEARS[:3] if args.dry_run else TRAIN_YEARS
    for yr in years:
        p = TGT_DIR/f"targets_{yr}.npy"
        if not p.exists(): continue
        era5 = np.load(p)
        anom = era5 - clim_mean
        # targets: ERA5 at mesh nodes only (233 values, not 11160)
        tgt = np.stack([
            anom[mesh_grid, 0] / t_std,
            anom[mesh_grid, 1] / p_std,
        ], axis=1).astype(np.float32)  # (233, 2)
        # obs: ERA5 at obs nodes
        t_a  = (anom[obs_grid, 0] / t_std).astype(np.float32)
        p_a  = (anom[obs_grid, 1] / p_std).astype(np.float32)
        t_av = np.zeros(len(all_grid), dtype=np.float32); t_av[:n_iso] = 1.
        p_av = np.zeros(len(all_grid), dtype=np.float32); p_av[n_iso:] = 1.
        obs_f = np.stack([t_a, t_av, p_a, p_av], axis=1).astype(np.float32)  # (135, 4)
        samples.append((obs_f, tgt))
    logger.info(f"  {len(samples)} samples, targets shape: {samples[0][1].shape}")

    # build validation sample (2001) for early stopping
    val_era5 = np.load(TGT_DIR/f"targets_2001.npy")
    val_anom = val_era5 - clim_mean
    val_tgt  = np.stack([val_anom[mesh_grid,0]/t_std, val_anom[mesh_grid,1]/p_std], axis=1).astype(np.float32)
    val_t_a  = (val_anom[obs_grid,0]/t_std).astype(np.float32)
    val_p_a  = (val_anom[obs_grid,1]/p_std).astype(np.float32)
    val_t_av = np.zeros(len(all_grid), dtype=np.float32); val_t_av[:n_iso] = 1.
    val_p_av = np.zeros(len(all_grid), dtype=np.float32); val_p_av[n_iso:] = 1.
    val_obs  = jnp.array(np.stack([val_t_a, val_t_av, val_p_a, val_p_av], axis=1).astype(np.float32))
    val_tgt_j = jnp.array(val_tgt)
    logger.info(f"  Validation year: 2001")

    # minimal model: linear obs enc -> o2m -> m2m -> linear decode
    def model_fn(obs_f, o2m_s, o2m_r, ms, mr):
        H = args.hidden
        # linear obs encoding
        obs_h = hk.Linear(H, name="obs_enc")(obs_f)
        # mesh init from positional features
        mesh_pos = jnp.stack([
            (jnp.array(mesh_lats) - (-75.)) / 15.,
            jnp.sin(jnp.radians(jnp.array(mesh_lons))),
            jnp.cos(jnp.radians(jnp.array(mesh_lons))),
        ], axis=1)
        mesh_h = hk.Linear(H, name="mesh_enc")(mesh_pos)

        # obs -> mesh (mean aggregation)
        o2m_msg = hk.Linear(H, name="o2m")(jnp.concatenate([obs_h[o2m_s], mesh_h[o2m_r]], axis=-1))
        o2m_sum = jax.ops.segment_sum(o2m_msg, o2m_r, num_segments=n_mesh)
        o2m_cnt = jax.ops.segment_sum(jnp.ones(o2m_r.shape[0]), o2m_r, num_segments=n_mesh)
        o2m_agg = o2m_sum / jnp.maximum(o2m_cnt[:, None], 1.)
        mesh_h  = jax.nn.relu(mesh_h + o2m_agg)

        # mesh -> mesh (n rounds)
        for i in range(args.mesh_rounds):
            msg    = hk.Linear(H, name=f"m2m_{i}")(jnp.concatenate([mesh_h[ms], mesh_h[mr]], axis=-1))
            agg    = jax.ops.segment_sum(msg, mr, num_segments=n_mesh)
            cnt    = jax.ops.segment_sum(jnp.ones(mr.shape[0]), mr, num_segments=n_mesh)
            mesh_h = jax.nn.relu(mesh_h + agg / jnp.maximum(cnt[:, None], 1.))

        # decode
        return hk.Linear(2, name="out")(mesh_h)

    forward = hk.transform(model_fn)
    rng = jax.random.PRNGKey(42)
    s0  = jnp.array(samples[0][0])
    params = forward.init(rng, s0, o2m_s, o2m_r, ms, mr)
    n_p = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"Parameters: {n_p:,}")

    sched = optax.warmup_cosine_decay_schedule(
        0., args.lr, len(samples)*2, len(samples)*args.epochs, args.lr*0.01)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(sched, weight_decay=1e-2),
    )
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, obs_f, tgt):
        def loss_fn(p):
            pred = forward.apply(p, None, obs_f, o2m_s, o2m_r, ms, mr)
            return jnp.mean((pred - tgt)**2)
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
            if args.dry_run and i >= 1: break
        avg = el/len(idx)
        if epoch % 50 == 0 or epoch <= 10:
            logger.info(f"Epoch {epoch:4d} | loss={avg:.5f} | time={time.time()-t0:.2f}s")
        # compute validation loss every 50 epochs
        if epoch % 50 == 0:
            val_pred = forward.apply(params, None, val_obs, o2m_s, o2m_r, ms, mr)
            val_loss = float(jnp.mean((val_pred - val_tgt_j)**2))
            logger.info(f"  Val loss: {val_loss:.5f}")

        if avg < best:
            best = avg
            logger.info(f"  Best train: {best:.5f}")
            save_dir = Path("/glade/derecho/scratch/advike/graphcast_recon/weights_minimal")
            save_dir.mkdir(parents=True, exist_ok=True)
            flat = {str(i):v for i,v in enumerate(jax.tree_util.tree_leaves(jax.device_get(params)))}
            tmp = str(save_dir/"weights_best.tmp.npz")
            np.savez_compressed(tmp, **flat)
            os.rename(tmp, str(save_dir/"weights_best.npz"))
        if epoch % 100 == 0:
            save_dir = Path("/glade/derecho/scratch/advike/graphcast_recon/weights_minimal")
            flat = {str(i):v for i,v in enumerate(jax.tree_util.tree_leaves(jax.device_get(params)))}
            tmp = str(save_dir/f"ckpt_{epoch:04d}.tmp.npz")
            np.savez_compressed(tmp, **flat)
            os.rename(tmp, str(save_dir/f"ckpt_{epoch:04d}.npz"))
    logger.info(f"Done. Best: {best:.5f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--hidden",      type=int,   default=64)
    p.add_argument("--mesh-rounds", type=int,   default=4)
    p.add_argument("--dry-run",     action="store_true")
    args = p.parse_args()
    logger = logging.getLogger("minimal_gnn")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
        logger.addHandler(sh)
    logger.info(f"Minimal GNN | Args: {vars(args)}")
    train(args, logger)

if __name__ == "__main__":
    main()

def validate():
    """Quick validation on 2001-2005."""
    from ice_core_loader import IceCoreLoader
    ns       = np.load(TGT_DIR/"norm_stats.npz")
    anom_std = np.load(TGT_DIR/"anom_std_1979_2000.npz")
    clim_mean= np.load(TGT_DIR/"clim_mean_1979_2000.npy")
    t_std    = float(anom_std['temp_anom_std'])
    p_std    = float(anom_std['prec_anom_std'])
    loader   = IceCoreLoader(
        data_dir=str(RECON_DIR/"data"),
        embeddings_dir=str(RECON_DIR/"cache/embeddings"),
        calibration_dir=str(RECON_DIR/"cache/calibration"),
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

    # load weights
    mesh_lats_j = jnp.array(mesh_lats); mesh_lons_j = jnp.array(mesh_lons)
    hidden = 64; n_rounds = 4

    def model_fn(obs_f, o2m_s, o2m_r, ms, mr):
        H = hidden
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
        mesh_h  = jax.nn.relu(mesh_h + o2m_sum / jnp.maximum(o2m_cnt[:, None], 1.))
        for i in range(n_rounds):
            msg    = hk.Linear(H, name=f"m2m_{i}")(jnp.concatenate([mesh_h[ms], mesh_h[mr]], axis=-1))
            agg    = jax.ops.segment_sum(msg, mr, num_segments=n_mesh)
            cnt    = jax.ops.segment_sum(jnp.ones(mr.shape[0]), mr, num_segments=n_mesh)
            mesh_h = jax.nn.relu(mesh_h + agg / jnp.maximum(cnt[:, None], 1.))
        return hk.Linear(2, name="out")(mesh_h)

    forward = hk.transform(model_fn)
    dummy = jnp.zeros((len(all_grid), 4))
    params = forward.init(jax.random.PRNGKey(0), dummy, o2m_s, o2m_r, ms, mr)
    raw = np.load(RECON_DIR/"weights3"/"weights_best.npz")
    leaves = [raw[str(i)] for i in range(len(raw.files))]
    params = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), leaves)

    print("\n-- Validation 2001-2005 (mesh nodes only) --")
    for yr in range(2001, 2006):
        era5 = np.load(TGT_DIR/f"targets_{yr}.npy")
        anom = era5 - clim_mean
        tgt  = np.stack([anom[mesh_grid,0]/t_std, anom[mesh_grid,1]/p_std], axis=1)
        t_a  = (anom[obs_grid,0]/t_std).astype(np.float32)
        p_a  = (anom[obs_grid,1]/p_std).astype(np.float32)
        t_av = np.zeros(len(all_grid), dtype=np.float32); t_av[:n_iso] = 1.
        p_av = np.zeros(len(all_grid), dtype=np.float32); p_av[n_iso:] = 1.
        obs_f = jnp.array(np.stack([t_a, t_av, p_a, p_av], axis=1).astype(np.float32))
        pred  = np.array(forward.apply(params, None, obs_f, o2m_s, o2m_r, ms, mr))
        mse   = float(np.mean((pred - tgt)**2))
        r_T   = float(np.corrcoef(pred[:,0], tgt[:,0])[0,1])
        print(f"  {yr}: MSE={mse:.4f}  T_r={r_T:.4f}  pred_std={pred[:,0].std():.4f}")

if __name__ == "__main__":
    import sys
    if "--validate" in sys.argv:
        validate()
    else:
        main()
