"""
compute_clim_embedding.py
-------------------------
Computes the climatological mean GraphCast embedding by averaging all available
yearly embedding files in cache/embeddings/window_YYYY.npy.

Handles partial availability gracefully — averages over whatever years are
present. Re-run once all years are extracted to get the full 1979–2017 mean.

Output: cache/embeddings/embedding_clim.npy  shape (11160, 512), float32

Usage:
  python3 compute_clim_embedding.py
"""

import numpy as np
from pathlib import Path

EMBEDDINGS_DIR = Path('/glade/derecho/scratch/advike/graphcast_recon/cache/embeddings')
OUTPUT_PATH    = EMBEDDINGS_DIR / 'embedding_clim.npy'

EXPECTED_YEARS = list(range(1979, 2018))  # 1979–2017 inclusive
N_NODES        = 11160
EMB_DIM        = 512


def main():
    print("=== compute_clim_embedding.py ===\n")

    # Find which years are actually available
    available = []
    missing   = []
    for year in EXPECTED_YEARS:
        path = EMBEDDINGS_DIR / f'window_{year}.npy'
        if path.exists():
            available.append(year)
        else:
            missing.append(year)

    if not available:
        raise RuntimeError(
            f"No embedding files found in {EMBEDDINGS_DIR}. "
            f"Run extract_embeddings.py first."
        )

    print(f"Found {len(available)}/{len(EXPECTED_YEARS)} yearly embeddings")
    print(f"  Available: {available[0]}–{available[-1]}" if len(available) > 1
          else f"  Available: {available[0]}")
    if missing:
        print(f"  Missing ({len(missing)} years): {missing[0]}–{missing[-1]}"
              if len(missing) > 1 else f"  Missing: {missing[0]}")
    if len(available) < len(EXPECTED_YEARS):
        print(f"\n  NOTE: Climatological mean will be computed over {len(available)} year(s).")
        print(f"  Re-run this script once all years are extracted for the full 38-year mean.\n")

    # Accumulate sum in float64 to avoid precision loss, then convert at end
    running_sum = np.zeros((N_NODES, EMB_DIM), dtype=np.float64)

    for i, year in enumerate(available):
        path = EMBEDDINGS_DIR / f'window_{year}.npy'
        emb  = np.load(path).astype(np.float64)

        assert emb.shape == (N_NODES, EMB_DIM), \
            f"Unexpected shape {emb.shape} for window_{year}.npy, expected ({N_NODES}, {EMB_DIM})"
        assert not np.any(np.isnan(emb)), f"NaNs found in window_{year}.npy"

        running_sum += emb

        if (i + 1) % 5 == 0 or (i + 1) == len(available):
            print(f"  Loaded {i+1}/{len(available)}: window_{year}.npy")

    clim = (running_sum / len(available)).astype(np.float32)

    # Sanity checks
    assert clim.shape == (N_NODES, EMB_DIM)
    assert clim.dtype == np.float32
    assert not np.any(np.isnan(clim)), "NaNs in climatological mean — check input files"

    print(f"\nClimatological mean stats:")
    print(f"  Shape:  {clim.shape}")
    print(f"  Mean:   {clim.mean():.4f}")
    print(f"  Std:    {clim.std():.4f}")
    print(f"  Min:    {clim.min():.4f}")
    print(f"  Max:    {clim.max():.4f}")
    print(f"  Years averaged: {len(available)}")

    np.save(OUTPUT_PATH, clim)
    print(f"\nSaved to {OUTPUT_PATH}")
    print("=== Done ===")


if __name__ == '__main__':
    main()
