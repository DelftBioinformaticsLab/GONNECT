# prepare/ — rebuilding the derived inputs

`../fig*.py` read `data/` and draw. Nothing in this folder is needed to
reproduce a published figure: `data/` already ships every input, and the
figures reproduce from it bit-for-bit.

These are the steps that *produced* the derived parts of `data/`, kept so the
chain from raw TCGA expression to the figure inputs is inspectable rather than
taken on trust.

## The chain

```
TCGA_complete_bp_top1k.csv.gz
  └─ plot_deg_volcano.py ──────────────→ deg_results.csv
       ├─ + hard_links.csv
       │    ├─ run_gsea.py ────────────→ gsea_gonnect_layers/
       │    └─ run_gsea_bottleneck.py ─→ gsea_gonnect_bottleneck/
       └─ + ontologies/*.gmt
            └─ run_gsea_baselines.py ──→ gsea_baselines/

go_term_activations/ + hard_links.csv + gsea_gonnect_layers/
  └─ plot_perm_nulls_layers.py ────────→ perm_nulls_gonnect/

go_term_activations/
  └─ activation_preservation.py ───────→ activation_preservation_per_node.csv

baseline_activations/ + gsea_baselines/
  └─ compute_baseline_nulls.py ────────→ perm_nulls_baselines/
```

Support modules, imported by the steps above rather than run directly:
`plot_activation_vs_gsea.py`, `plot_activation_vs_gsea_baselines.py`,
`plot_perm_nulls.py`, `gonnect_names.py`, `_paths.py`.

`extract_alpha_sweep_weights.py` and `alpha_sweep_numbers.py` sit apart from
that chain, and from its no-writing-into-`data/` rule: both are deterministic,
so a rerun cannot quietly move a figure the way the GSEA and permutation steps
could.

```
out/trained_models/AE_3.*/          (training output, not deposited)
  └─ extract_alpha_sweep_weights.py ──→ data/alpha_sweep_weights/
```

`extract_alpha_sweep_weights.py` copies out the four weight matrices of the
module each α-sweep run constrains, discarding the unconstrained counterpart
module that nothing reads — 592 MB of checkpoints become 296 MB of extracts with
no value changed. It writes into `data/` directly, because that is where figS8
reads from and because copying tensors is not a computation that can drift.

`alpha_sweep_numbers.py` sits apart from that chain. It reads `loss_traces/` and
the `AE_3.x` checkpoints under `model_checkpoints/` and writes one table,
`out/prepare/alpha_sweep/alpha_sweep_numbers.tsv`, holding every number figures
S7 and S8 are quoted for: final MSE, epochs to plateau, active soft links at
|w| > 0.1, and |w| percentiles for GO edges against soft links. Unlike the rest
of this folder it is deterministic and reads only shipped inputs, so a rerun
reproduces the table exactly.

## Running

Defaults come from `_paths.py` and are resolved from this file's location, so a
bare command works from any directory:

```
pixi run python figures/src/prepare/run_gsea.py
```

**Output goes to `figures/out/prepare/<step>/`, not into `data/`.** The shipped
`data/` is what reproduces the published figures exactly, and the GSEA and
permutation steps are stochastic — a rerun landing on top of it would move the
figures without anyone noticing. Compare first; copy over the shipped file only
if you mean to replace it.

`run_gsea_baselines.py` takes one `--gmt` and one `--output-csv` per run, so it
is invoked once per gene-set database:

```
pixi run python figures/src/prepare/run_gsea_baselines.py \
    --gmt        figures/data/ontologies/hallmark_v2026_1_Hs_uniprot.gmt \
    --output-csv figures/out/prepare/gsea_baselines/gsea_results_hallmark.csv
```

and likewise for `reactomes_uniprot.gmt` and `ontovae_go_uniprot.gmt`.

## The baseline activations

`compute_baseline_nulls.py` reads per-sample baseline activations from
`data/baseline_activations/`:

```
baseline_activations/vega/run_seed-<2..6>/z_test_<arm>_<db>.csv
baseline_activations/ontovae/run_seed-<2..6>/pathway_activities_test_<arm>.parquet
```

where `<arm>` is `true`, `random` or `degree_preserving`, and `<db>` is
`hallmark_v2026_1_Hs_uniprot` or `reactomes_uniprot`. 45 files, about 1.5 GB.
`fig4.baseline_dbs()` reads the same tree for its per-seed dots.

## These are not the figure scripts

They come from the exploratory tree the paper grew out of and are held to a
lower standard than `../fig*.py`: they write extra diagnostic plots and CSVs
that nothing downstream reads, and their output layout is theirs rather than
`data/`'s. What was changed on the way in was only the plumbing — path
defaults, and the `sys.path` juggling that let them find each other in the old
directory tree. The analysis code is untouched.
