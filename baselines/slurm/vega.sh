#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=ewi-insy-prb,general
#SBATCH --qos=medium
#SBATCH --time=24:00:00      # 5 seeds x 2 GMTs x 3 arms x 300 epochs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --mail-type=END
#SBATCH --output=slurm/out/%j_vega.out
#SBATCH --error=slurm/out/%j_vega.out
#SBATCH --gres=gpu
#
# VEGA baseline: all seeds, both gene-set databases, all three graph arms.
#
# Run baselines/make_splits.py first; both baselines read the same splits.

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

module use /opt/insy/modulefiles
module load cuda/12.4

apptainer run --nv --writable-tmpfs --containall \
    --bind "$PWD":/workspace \
    baselines/environments/vega.sif \
        --expr-data data/TCGA_complete_bp_top1k.csv \
        --split-dir out/baselines/splits \
        --out-dir out/baselines/vega \
        --gmt-files data/reactomes_uniprot.gmt data/hallmark_v2026_1_Hs_uniprot.gmt \
        --seeds 2 3 4 5 6 \
        --n-epochs 300 --batch-size 128 --train-size 0.82
