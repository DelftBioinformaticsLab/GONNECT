#!/bin/sh
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=ewi-insy-prb,general
#SBATCH --qos=long
#SBATCH --time=72:00:00      # 5 seeds x 3 arms x 300 epochs, plus a ~20 min ontology build
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB           # ~11GB resident; headroom for the unpickled ontology
#SBATCH --mail-type=END
#SBATCH --output=slurm/out/%j_ontovae.out
#SBATCH --error=slurm/out/%j_ontovae.out
#SBATCH --gres=gpu
#
# OntoVAE baseline, all seeds and all three GO-graph arms.
#
# Needs a 16GB+ GPU: the decoder's final layer carries ~275M weights. Below that
# the run script switches AdamW to its slower single-tensor path automatically.
#
# Run baselines/make_splits.py first; both baselines read the same splits.

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

module use /opt/insy/modulefiles
module load cuda/12.4

apptainer run --nv --writable-tmpfs --containall \
    --bind "$PWD":/workspace \
    baselines/environments/ontovae.sif \
        --expr-data data/TCGA_complete_bp_top1k.csv \
        --split-dir out/baselines/splits \
        --out-dir out/baselines/ontovae \
        --obo data/go-basic.obo \
        --gene-annot data/gene_annot_ontovae.txt \
        --seeds 2 3 4 5 6 \
        --top-thresh 1000 --bottom-thresh 30 \
        --epochs 300 --batch-size 128 --lr 1e-4 --kl-coeff 1e-4
