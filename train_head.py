"""
train_head.py -- Reconstruction GNN Training Loop
Antarctic Climate Reconstruction from Ice Core Data
Approach 4: GraphCast frozen feature extractor + trainable reconstruction GNN head

Architecture:
  - Bipartite graph: observation nodes (ice cores) -> target nodes (Antarctic grid)
  - Phase 1: obs->target message passing (radius-based, ~1000km)
  - Phase 2: target->target message passing (6 rounds, ~500km radius)
  - Output: 2D prediction (temp + precip) at all 11,160 target nodes

Usage (batch job recommended):
  python train_head.py [--epochs 200] [--lr 1e-3] [--hidden 256] [--dry-run]
"""

import os
import sys
import time
import argparse
import logging
import resource
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import jraph

# ------------------------------------
# Paths
# ------------------------------------
RECON_DIR   = Path("/glade/derecho/scratch/advike/graphcast_recon")
CACHE_DIR   = RECON_DIR / "cache"
EMB_DIR     = CACHE_DIR / "embeddings"
TGT_DIR     = CACHE_DIR / "era5_targets"
CALIB_DIR   = CACHE_DIR / "calibration"
WEIGHTS_DIR = RECON_DIR / "weights"
LOG_DIR     = RECON_DIR / "logs"

NORM_STATS_PATH = TGT_DIR / "norm_stats.npz"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(RECON_DIR))

# ------------------------------------
# Constants
# ------------------------------------
TRAIN_YEARS    = list(range(1979, 2001))
N_TARGET_NODES = 11_160
N_OUTPUT       = 2

LAT_VALS = np.arange(-60, -91, -1, dtype=np.float32)
LON_VALS = np.arange(0,   360,  1, dtype=np.float32)
TARGET_LATS, TARGET_LONS = np.meshgrid(LAT_VALS, LON_VALS, indexing='ij')
TARGET_LATS = TARGET_LATS.reshape(-1)
TARGET_LONS = TARGET_LONS.reshape(-1)

OBS_TO_TGT_RADIUS_DEG = 9.0
TGT_TO_TGT_RADIUS_DEG = 2.0

# ------------------------------------
# Logging
# ------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("train_head")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def log_memory(logger: logging.Logger, tag: str = ""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    mb = usage.ru_maxrss / 1024
    logger.info(f"[MEM{' '+tag if tag else ''}] peak RSS = {mb:.0f} MB")


# ------------------------------------
# Graph construction helpers
# ------------------------------------
def great_circle_distance_deg(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    dlon = (dlon + 180) % 360 - 180
    return np.sqrt(dlat**2 + (dlon * np.cos(np.radians((lat1 + lat2) / 2)))**2)


def build_obs_to_target_edges(
    obs_lats: np.ndarray,
    obs_lons: np.ndarray,
    tgt_lats: np.ndarray,
    tgt_lons: np.ndarray,
    radius_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    senders, receivers = [], []
    for i, (olat, olon) in enumerate(zip(obs_lats, obs_lons)):
        dists = great_circle_distance_deg(olat, olon, tgt_lats, tgt_lons)
        targets = np.where(dists <= radius_deg)[0]
        senders.append(np.full(len(targets), i, dtype=np.int32))
        receivers.append(targets.astype(np.int32))
    if senders:
        return np.concatenate(senders), np.concatenate(receivers)
    else:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)


def build_target_to_target_edges(
    tgt_lats: np.ndarray,
    tgt_lons: np.ndarray,
    radius_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(tgt_lats)
    chunk = 500
    senders_list, receivers_list = [], []
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        lats_chunk = tgt_lats[start:end, None]
        lons_chunk = tgt_lons[start:end, None]
        dlat = tgt_lats[None, :] - lats_chunk
        dlon = tgt_lons[None, :] - lons_chunk
        dlon = (dlon + 180) % 360 - 180
        mean_lat = (tgt_lats[None, :] + lats_chunk) / 2
        dist = np.sqrt(dlat**2 + (dlon * np.cos(np.radians(mean_lat)))**2)
        local_s, local_r = np.where((dist <= radius_deg) & (dist > 0))
        senders_list.append((local_s + start).astype(np.int32))
        receivers_list.append(local_r.astype(np.int32))
    return np.concatenate(senders_list), np.concatenate(receivers_list)


# ------------------------------------
# Target feature assembly
# ------------------------------------
def build_target_features(clim_emb: np.ndarray) -> np.ndarray:
    """
    Build 514-dim target node features: [lat_norm, lon_norm, emb_512]
    clim_emb: (11160, 512) full-grid climatological embedding
    """
    lat_norm = (TARGET_LATS - (-75.0)) / 15.0
    lon_norm = (TARGET_LONS - 180.0) / 180.0
    pos = np.stack([lat_norm, lon_norm], axis=1)
    return np.concatenate([pos, clim_emb], axis=1).astype(np.float32)


# ------------------------------------
# Model definition
# ------------------------------------
def make_mlp(output_size: int, hidden_size: int, name: str):
    return hk.Sequential([
        hk.Linear(hidden_size, name=f"{name}_l1"), jax.nn.relu,
        hk.Linear(hidden_size, name=f"{name}_l2"), jax.nn.relu,
        hk.Linear(output_size, name=f"{name}_out"),
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                     name=f"{name}_ln"),
    ], name=name)


def reconstruction_gnn(
    obs_feats: jnp.ndarray,
    tgt_feats: jnp.ndarray,
    o2t_senders: jnp.ndarray,
    o2t_receivers: jnp.ndarray,
    t2t_senders: jnp.ndarray,
    t2t_receivers: jnp.ndarray,
    hidden_size: int,
    n_t2t_rounds: int,
) -> jnp.ndarray:
    # 1. Encode
    obs_enc = make_mlp(hidden_size, hidden_size * 2, "obs_encoder")(obs_feats)
    tgt_enc = make_mlp(hidden_size, hidden_size * 2, "tgt_encoder")(tgt_feats)

    # 2. Obs -> Target
    o2t_edge_mlp = make_mlp(hidden_size, hidden_size * 2, "o2t_edge")
    o2t_agg_mlp  = make_mlp(hidden_size, hidden_size * 2, "o2t_agg")

    def o2t_message_pass(obs_h, tgt_h):
        edge_in  = jnp.concatenate([obs_h[o2t_senders], tgt_h[o2t_receivers]], axis=-1)
        messages = o2t_edge_mlp(edge_in)
        agg      = jax.ops.segment_sum(messages, o2t_receivers,
                                       num_segments=tgt_h.shape[0])
        return o2t_agg_mlp(jnp.concatenate([tgt_h, agg], axis=-1))

    tgt_h = o2t_message_pass(obs_enc, tgt_enc)

    # 3. Target -> Target (n rounds, residual)
    t2t_edge_mlp = make_mlp(hidden_size, hidden_size * 2, "t2t_edge")
    t2t_agg_mlp  = make_mlp(hidden_size, hidden_size * 2, "t2t_agg")

    def t2t_round(tgt_h):
        edge_in  = jnp.concatenate([tgt_h[t2t_senders], tgt_h[t2t_receivers]], axis=-1)
        messages = t2t_edge_mlp(edge_in)
        agg      = jax.ops.segment_sum(messages, t2t_receivers,
                                       num_segments=tgt_h.shape[0])
        return t2t_agg_mlp(jnp.concatenate([tgt_h, agg], axis=-1)) + tgt_h

    for _ in range(n_t2t_rounds):
        tgt_h = t2t_round(tgt_h)

    # 4. Decode
    return hk.Linear(N_OUTPUT, name="decoder")(tgt_h)


# ------------------------------------
# Dataset
# ------------------------------------
class ReconDataset:
    def __init__(self, years: list[int], loader, logger: logging.Logger):
        self.samples = []
        self.logger  = logger
        logger.info(f"Loading dataset for {len(years)} years...")

        norm = np.load(NORM_STATS_PATH)
        self.temp_mean   = float(norm['temp_mean'])
        self.temp_std    = float(norm['temp_std'])
        self.precip_mean = float(norm['prec_mean'])
        self.precip_std  = float(norm['prec_std'])
        logger.info(f"  Norm stats -- T: {self.temp_mean:.1f}+/-{self.temp_std:.1f}K  "
                    f"P: {self.precip_mean:.3f}+/-{self.precip_std:.3f} m/yr")

        skipped = 0
        for yr in years:
            tgt_path = TGT_DIR / f"targets_{yr}.npy"
            if not tgt_path.exists():
                logger.warning(f"  Year {yr}: ERA5 targets missing, skipping")
                skipped += 1
                continue

            try:
                obs = loader.get_year(yr)
            except ValueError as e:
                logger.warning(f"  Year {yr}: {e}, skipping")
                skipped += 1
                continue

            if obs['n_obs'] == 0:
                logger.warning(f"  Year {yr}: no obs nodes, skipping")
                skipped += 1
                continue

            targets_raw  = np.load(tgt_path)
            targets_norm = np.stack([
                (targets_raw[:, 0] - self.temp_mean)   / self.temp_std,
                (targets_raw[:, 1] - self.precip_mean) / self.precip_std,
            ], axis=1).astype(np.float32)

            self.samples.append((obs, targets_norm))

        logger.info(f"  Dataset: {len(self.samples)} samples loaded, {skipped} skipped")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ------------------------------------
# Precompute t2t edges
# ------------------------------------
def precompute_t2t_edges(logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    cache_path = CACHE_DIR / "t2t_edges.npz"
    if cache_path.exists():
        logger.info("Loading cached t2t edges...")
        data = np.load(cache_path)
        return data['senders'], data['receivers']

    logger.info("Building target->target edges (first run, ~30s)...")
    t0 = time.time()
    s, r = build_target_to_target_edges(TARGET_LATS, TARGET_LONS,
                                         TGT_TO_TGT_RADIUS_DEG)
    logger.info(f"  t2t edges: {len(s):,} in {time.time()-t0:.1f}s")
    tmp = str(cache_path).replace('.npz', '.tmp.npz')
    np.savez_compressed(tmp, senders=s, receivers=r)
    os.rename(tmp, cache_path)
    logger.info(f"  Cached to {cache_path}")
    return s, r


# ------------------------------------
# Loss
# ------------------------------------
def mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((predictions - targets) ** 2)


def make_forward_fn(hidden_size: int, n_t2t_rounds: int):
    def _forward(obs_feats, tgt_feats, o2t_s, o2t_r, t2t_s, t2t_r):
        return reconstruction_gnn(obs_feats, tgt_feats, o2t_s, o2t_r,
                                   t2t_s, t2t_r, hidden_size, n_t2t_rounds)
    return hk.transform(_forward)


# ------------------------------------
# Training loop
# ------------------------------------
def train(args, logger: logging.Logger):
    rng = jax.random.PRNGKey(args.seed)

    # Initialise ice core loader
    logger.info("Initialising IceCoreLoader...")
    from ice_core_loader import IceCoreLoader
    loader = IceCoreLoader(
        temp_mean=float(np.load(NORM_STATS_PATH)['temp_mean']),
        temp_std=float(np.load(NORM_STATS_PATH)['temp_std']),
        prec_mean=float(np.load(NORM_STATS_PATH)['prec_mean']),
        prec_std=float(np.load(NORM_STATS_PATH)['prec_std']),
        data_dir        = str(RECON_DIR / "data"),
        embeddings_dir  = str(EMB_DIR),
        calibration_dir = str(CALIB_DIR),
    )

    # Build target features using full-grid clim embedding from loader
    logger.info("Building target node features (static)...")
    tgt_feats_np = build_target_features(loader.clim_embedding)  # (11160, 514)
    tgt_feats    = jnp.array(tgt_feats_np)
    logger.info(f"  tgt_feats: {tgt_feats.shape}")

    # t2t edges (static, cached)
    t2t_s_np, t2t_r_np = precompute_t2t_edges(logger)
    t2t_s = jnp.array(t2t_s_np)
    t2t_r = jnp.array(t2t_r_np)
    logger.info(f"  t2t edges on device: {t2t_s.shape[0]:,}")

    # Dataset
    years   = TRAIN_YEARS[:3] if args.dry_run else TRAIN_YEARS
    dataset = ReconDataset(years, loader, logger)
    if len(dataset) == 0:
        logger.error("No training samples -- aborting.")
        return

    # Initialise model
    logger.info(f"Initialising model (hidden={args.hidden}, t2t_rounds={args.t2t_rounds})...")
    forward = make_forward_fn(args.hidden, args.t2t_rounds)

    obs0, _ = dataset[0]
    obs_feats0 = jnp.array(obs0['features'])
    o2t_s0_np, o2t_r0_np = build_obs_to_target_edges(
        obs0['lats'], obs0['lons'], TARGET_LATS, TARGET_LONS, OBS_TO_TGT_RADIUS_DEG,
    )
    o2t_s0 = jnp.array(o2t_s0_np)
    o2t_r0 = jnp.array(o2t_r0_np)

    rng, init_rng = jax.random.split(rng)
    params = forward.init(init_rng, obs_feats0, tgt_feats,
                          o2t_s0, o2t_r0, t2t_s, t2t_r)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"  Model parameters: {n_params:,}")

    # Optimiser
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=min(len(dataset) * 2, max(1, len(dataset) * args.epochs - 1)),
        decay_steps=len(dataset) * args.epochs,
        end_value=args.lr * 0.01,
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule),
    )
    opt_state = optimiser.init(params)

    @jax.jit
    def train_step(params, opt_state, obs_feats, tgt_targets, o2t_s, o2t_r):
        def loss_fn(p):
            preds = forward.apply(p, None, obs_feats, tgt_feats,
                                  o2t_s, o2t_r, t2t_s, t2t_r)
            return mse_loss(preds, tgt_targets)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimiser.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    def save_checkpoint(params, epoch: int, loss: float) -> Path:
        path = WEIGHTS_DIR / f"checkpoint_epoch{epoch:04d}.npz"
        tmp  = str(path).replace('.npz', '.tmp.npz')
        flat = {str(i): v for i, v in
                enumerate(jax.tree_util.tree_leaves(jax.device_get(params)))}
        np.savez_compressed(tmp, **flat)
        os.rename(tmp, path)
        logger.info(f"  Checkpoint saved: {path.name}")
        return path

    best_loss    = float('inf')
    best_ckpt    = None
    epoch_losses = []

    logger.info(f"Starting training: {args.epochs} epochs x {len(dataset)} samples")
    log_memory(logger, "pre-train")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        rng, shuf_rng = jax.random.split(rng)
        indices = jax.random.permutation(shuf_rng, len(dataset)).tolist()

        epoch_loss = 0.0
        for step, idx in enumerate(indices):
            obs, targets_norm = dataset[idx]

            # Features come directly from loader -- no build_obs_features needed
            obs_feats_np = obs['features']   # (N_obs, 518)
            o2t_s_np, o2t_r_np = build_obs_to_target_edges(
                obs['lats'], obs['lons'],
                TARGET_LATS, TARGET_LONS, OBS_TO_TGT_RADIUS_DEG,
            )

            obs_feats   = jnp.array(obs_feats_np)
            tgt_targets = jnp.array(targets_norm)
            o2t_s       = jnp.array(o2t_s_np)
            o2t_r       = jnp.array(o2t_r_np)

            params, opt_state, loss = train_step(
                params, opt_state, obs_feats, tgt_targets, o2t_s, o2t_r
            )
            epoch_loss += float(loss)

            if args.dry_run and step >= 99:
                logger.info("  [DRY RUN] stopping after 2 steps")
                break

        avg_loss = epoch_loss / len(indices)
        epoch_losses.append(avg_loss)
        elapsed  = time.time() - epoch_start

        logger.info(f"Epoch {epoch:4d}/{args.epochs} | "
                    f"loss={avg_loss:.5f} | time={elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = save_checkpoint(params, epoch, avg_loss)
            logger.info(f"  New best loss: {best_loss:.5f}")
        elif epoch % 10 == 0:
            save_checkpoint(params, epoch, avg_loss)


    # Save final weights
    final_path = WEIGHTS_DIR / "weights_final.npz"
    flat = {str(i): v for i, v in
            enumerate(jax.tree_util.tree_leaves(jax.device_get(params)))}
    tmp = str(final_path).replace('.npz', '.tmp.npz')
    np.savez_compressed(tmp, **flat)
    os.rename(tmp, final_path)
    logger.info(f"Final weights saved: {final_path}")

    # Save loss curve
    loss_path = LOG_DIR / "loss_curve.npy"
    tmp = str(loss_path).replace('.npy', '.tmp.npy')
    np.save(tmp, np.array(epoch_losses))
    os.rename(tmp, loss_path)
    logger.info(f"Loss curve saved: {loss_path}")

    log_memory(logger, "post-train")
    logger.info(f"Training complete. Best loss: {best_loss:.5f} @ {best_ckpt}")


# ------------------------------------
# Entry point
# ------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Antarctic reconstruction GNN head")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden",     type=int,   default=256)
    parser.add_argument("--t2t-rounds", type=int,   default=6)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    log_path = LOG_DIR / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("Antarctic Reconstruction GNN -- Training")
    logger.info(f"  JAX devices: {jax.devices()}")
    logger.info(f"  Args: {vars(args)}")
    logger.info("=" * 60)

    for path, desc in [
        (NORM_STATS_PATH,          "normalisation stats"),
        (EMB_DIR / "embedding_clim.npy", "climatological embedding"),
        (CALIB_DIR / "calibrated_iso.npy",   "calibrated isotope data"),
        (CALIB_DIR / "calibrated_accum.npy", "calibrated accum data"),
    ]:
        if not Path(path).exists():
            logger.error(f"Required file missing: {path} ({desc})")
            sys.exit(1)

    train(args, logger)


if __name__ == "__main__":
    main()
