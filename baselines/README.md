# baselines/ — VEGA and OntoVAE reproducibility

## Inputs

| File | Used by | Notes |
|---|---|---|
| `data/TCGA_complete_bp_top1k.csv` | both | 9797 samples, 1000 genes, 5 leading metadata columns |
| `data/go-basic.obo` | OntoVAE | GO DAG |
| `data/gene_annot_ontovae.txt` | OntoVAE | headerless `Gene<TAB>GO_ID` TSV, UniProt accessions |
| `data/reactomes_uniprot.gmt` | VEGA | 674 gene sets |
| `data/hallmark_v2026_1_Hs_uniprot.gmt` | VEGA | 50 gene sets |


## Running

Split data (used by both baselines):

```
python baselines/make_splits.py --verify
```

`--verify` asserts the split is identical to what was used in the GONNECT
experiments. It has to be run from GONNECT's own environment, since it imports
`gonnect`; the baseline runs then use their own environments.

Then each baseline, via its container (see `slurm/` for the cluster form):

```
apptainer run --nv -B $PWD:/workspace baselines/environments/ontovae.sif \
    --expr-data data/TCGA_complete_bp_top1k.csv \
    --split-dir out/baselines/splits \
    --obo data/go-basic.obo --gene-annot data/gene_annot_ontovae.txt \
    --seeds 2 3 4 5 6 --top-thresh 1000 --bottom-thresh 30 \
    --epochs 300 --batch-size 128 --lr 1e-4 --kl-coeff 1e-4

apptainer run --nv -B $PWD:/workspace baselines/environments/vega.sif \
    --expr-data data/TCGA_complete_bp_top1k.csv \
    --split-dir out/baselines/splits \
    --gmt-files data/reactomes_uniprot.gmt data/hallmark_v2026_1_Hs_uniprot.gmt \
    --seeds 2 3 4 5 6 --n-epochs 300 --batch-size 128 --train-size 0.82
```

Finally:

```
python baselines/collect_metrics.py
```

This writes `ontovae_rand.txt` and `vega_rand.txt` to `out/baselines/metrics/`,
in the shape of `figures/data/metrics/test_split/`, which the figures read.
Both runners score the test split, and since the revision they take the
silhouette against the true cancer types, as GONNECT's is
(`--silhouette-against labels`, the default). `--silhouette-against kmeans`
reproduces the runs behind preprint v3, which took it against the k-means clusters; each
run prints both. The deposited baseline numbers were rescored from the latents of the
runs behind preprint v3 rather than retrained (`figures/src/prepare/rescore_baselines.py`),
because those runs left model initialisation unseeded.

## Untrained controls

The structural baseline for Figure 4's panels d and e: the same statistic on
models that were built and never trained. Both scripts reuse their runner's
data loading, and they need the splits above and each baseline's own
environment. They export, per split seed, 10 initializations: VEGA's test
latents on each of its three graphs (`--arms`, with the randomized masks drawn
as `run_vega.py` draws them for that seed), and OntoVAE's test pathway
activities on the true graph, in the trained runs' shapes. VEGA masks only its
decoder, so its untrained latents are the same on every graph. OntoVAE reads
the cached ontology (`out/baselines/ontovae/cached_ontology.pkl`), whose terms,
order and layers match the deposited activations. Both take the latent at its
posterior mean rather than a sample, since an untrained model's posterior
variance would otherwise bury the wiring's signal in noise.

```
python baselines/vega/untrained_vega.py
python baselines/ontovae/untrained_ontovae.py --gene-annot data/gene_annot_ontovae.txt
```

They write `out/baselines/untrained/{vega,ontovae}/run_seed-<seed>/init-<k>/`,
which `figures/src/prepare/untrained_baselines.py` scores. Each needs well
under half an hour on a laptop.

## Environments

For both baselines we used pinned Github commits.

| | Pinned at | 
|---|---|
| OntoVAE | `hdsu-bioquant/onto-vae@a000755` | 
| VEGA | `LucasESBS/vega@146c6c8` |

The rest of the environment is defined separately from the GONNECT environment, in `baselines/environments/ontovae.yml` and `baselines/environments/vega.yml`. 

`ontovae/ontovae_compat.py` carries the small adaptations OntoVAE needs on top
of upstream, each documented in place. `mask_randomization.py` is shared by both
baselines, and mirrors `gonnect/data_processing/generate_masks.py` so the
randomized arms are built the same way as GONNECT's own AE_2.2 arm.
