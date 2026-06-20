#!/bin/bash
# submit_pipeline.sh -- Submit full reconstruction pipeline with PBS dependencies.
# Usage: bash submit_pipeline.sh
# Chain: apply (GPU) -> inflate T only (CPU) -> plot (CPU)

set -e
cd /glade/derecho/scratch/advike/graphcast_recon

echo "Submitting apply job..."
APPLY_JOB=$(qsub run_apply_kernel.pbs)
echo "  apply: $APPLY_JOB"

echo "Submitting inflate job (depends on apply)..."
INFLATE_JOB=$(qsub -W depend=afterok:$APPLY_JOB run_inflate_homo_T_only.pbs)
echo "  inflate: $INFLATE_JOB"

echo "Submitting plot job (depends on inflate)..."
PLOT_JOB=$(qsub -W depend=afterok:$INFLATE_JOB run_plot_homo_T_inflated.pbs)
echo "  plot: $PLOT_JOB"

echo ""
echo "Pipeline submitted:"
echo "  1. $APPLY_JOB  -> cache/recon_homo/"
echo "  2. $INFLATE_JOB -> cache/recon_homo_T_inflated/"
echo "  3. $PLOT_JOB   -> cache/plots/homo_T_inflated/"
echo ""
echo "Monitor with: qstat -u $USER"
echo "Logs: logs/apply_kernel.out | logs/inflate_T_only.out | logs/plot_homo_T_inflated.out"
