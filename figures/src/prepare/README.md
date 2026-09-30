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

## The untrained control

`untrained_control.py` asks how much of Figure 4's agreement between a node's
activation and the GSEA enrichment of its term the wiring gives before any
training. A GONNECT encoder node only sees its own term's genes, so random
weights already tie its activation to that gene set, and the column-shuffle test,
which breaks exactly that pairing, cannot tell the wiring apart from learning. The step
builds GONNECT and GONNECT-SL from the masks in the repository's `out/masks/`,
leaves them at their initialization (50 seeded initializations of each), and
scores them with Figure 4's per-seed statistic next to the five shipped seeds.

```
out/masks/ (repository) + TCGA_complete_bp_top1k.csv.gz + hard_links.csv
  + gsea_gonnect_layers/ + go_term_activations/ + latent_embeddings/
  └─ untrained_control.py ─────────────→ out/prepare/untrained_control/

out/masks/ (_random8-12, local only) + TCGA_complete_bp_top1k.csv.gz + hard_links.csv
  + gsea_gonnect_bottleneck/ + latent_embeddings/AE_2.2/
  └─ untrained_dpr.py ─────────────────→ out/prepare/untrained_dpr/

out/baselines/untrained/ (baselines/*/untrained_*.py) + baseline_activations/ + gsea_baselines/
  └─ untrained_baselines.py ───────────→ out/prepare/untrained_baselines/

out/trained_models/AE_2.{0,1}/*_decoder_model.pt (cluster) + go_term_activations/
  └─ extract_decoder_activations.py ───→ out/prepare/decoder_reextracted/go_term_activations/
       ├─ plot_perm_nulls_layers.py ───→ out/prepare/decoder_reextracted/perm_nulls_gonnect/
       └─ activation_preservation.py ──→ out/prepare/decoder_reextracted/activation_preservation/

all of the above
  └─ fig4_decoder_comparison.py ───────→ out/prepare/fig4_decoder_comparison/
```

Like `alpha_sweep_numbers.py` it is deterministic and feeds no figure. It
writes one row per (model, layer, state, seed), a summary per layer -- trained
against untrained, with a two-sided Mann-Whitney p and its Benjamini-Hochberg q
-- and a diagnostic plot. It needs no GPU and takes about fifteen minutes on a
CPU. It also writes the untrained reference in the shape `fig4.py --untrained`
reads, twice: with the decoder labelled correctly, and labelled the way the
shipped files are.

`untrained_dpr.py` does the same for panel c, GONNECT-DPR. Each trained split
seed of AE_2.2 has its own degree-preserving mask, `_random8` to `_random12`
under `out/masks/`, shared by the encoder, decoder and both models; the step
checks that against the checkpoints when they are present. It builds 10
initializations on each of the five masks, 50 in all, and scores their
bottleneck as panel c does. Two of its results hold by construction. The
`decoder` model's bottleneck comes from a plain dense encoder, which no mask
touches. And the `encoder` and `both` models build the same masked encoder from
the same seed, so their references are equal. It takes a few minutes.

`untrained_baselines.py` does the same for panels d and e: OntoVAE's
true-graph violins, and all six of VEGA's. It scores the untrained models that
`baselines/ontovae/untrained_ontovae.py` and `baselines/vega/untrained_vega.py`
export, 10 initializations on each split seed (see `baselines/README.md`),
against the five shipped seeds. VEGA masks only its decoder, so its untrained
latent, and with it the reference, is the same on the true, DPR and FR graphs.
It writes the reference `fig4.py --untrained` takes for those panels. It runs
in a few minutes; the exports run in the baselines' own environments.

**The shipped decoder activations are labelled one layer off.** They were
written by a version of `gonnect.analysis.activations.activations_per_term` that
named every decoder layer's outputs after the terms of its input layer, so each
decoder column holds the node at the same position one layer closer to the
genes, and Figures 4, 5 and S6 read them under those names. The function is
fixed (see `tests/test_activations.py`).

`extract_decoder_activations.py` rebuilds the decoder files from the
decoder-only checkpoints of the cluster runs, which were not deposited:
`out/trained_models/AE_2.{0,1}/AE_2.{0,1}.{2..6}_decoder_model.pt`, placed at
the same path here. Before it writes anything, each checkpoint has to reproduce
its shipped file under the old labelling, and its saved latent. That shows the
shipped files came from these very models. Where the encoder checkpoints are
present, the shipped encoder files must come back unchanged too. All ten of each
held to ~1e-15. The output is a complete drop-in for `go_term_activations/`:
the shipped headers with corrected values, and the encoder files hard-linked in.

Without the checkpoints, `relabel_decoder_activations.py` renames what the
shipped files hold instead, taking the bottleneck from `latent_embeddings/`. It
agrees with the re-extraction on every column it recovers. But decoder layer 6
was only written as far as the bottleneck is wide, so it keeps 51 of that
layer's 92 GSEA-covered terms. `untrained_control.py` relabels the same way on
its own, unless `--activations-dir` points it at corrected files. The `as
shipped` states always come from the shipped files.

`plot_perm_nulls_layers.py --activations-dir` rebuilds the GONNECT nulls from
the corrected files. Run on the shipped files, it reproduces the shipped nulls
exactly, so the rebuilt decoder nulls differ by the labels alone.
`activation_preservation.py --activations-dir` rebuilds Figure 5's per-node
table the same way. Run on the shipped files, it matches the deposited table to
1e-12, apart from nodes constant in the fixed-link model, whose correlation
comes out NaN or +-1e-17 depending on summation order.

`fig4_decoder_comparison.py` then draws every figure that reads decoder
activations twice, into `as_published/` and `decoder_fixed/`. That is Figure 4
(with the untrained reference in both), Figure S6 and Figure 5. Next to them it
writes a table per figure:
- Figure 4: every panel a and b violin under both labellings.
- Figure S6: per GO term, which node its shipped column held, and how well the
  three instances agree on the signed and the absolute per-cancer-type means.
- Figure 5: the activation-preservation summary per layer.

`--figures` picks a subset. In order:

```
pixi run python figures/src/prepare/extract_decoder_activations.py
pixi run python figures/src/prepare/untrained_control.py \
    --activations-dir figures/out/prepare/decoder_reextracted/go_term_activations
pixi run python figures/src/prepare/plot_perm_nulls_layers.py --metrics auc \
    --activations-dir figures/out/prepare/decoder_reextracted/go_term_activations \
    --output-dir figures/out/prepare/decoder_reextracted/perm_nulls_gonnect
pixi run python figures/src/prepare/activation_preservation.py
pixi run python figures/src/prepare/activation_preservation.py \
    --activations-dir figures/out/prepare/decoder_reextracted/go_term_activations \
    --output-dir figures/out/prepare/decoder_reextracted/activation_preservation
pixi run python figures/src/prepare/untrained_dpr.py
pixi run python figures/src/prepare/untrained_baselines.py   # after the baseline exports
pixi run python figures/src/prepare/fig4_decoder_comparison.py
```

Without the checkpoints, run `relabel_decoder_activations.py` in place of the
first command, use `decoder_relabelled` for `decoder_reextracted` throughout,
and pass `--fixed-dir figures/out/prepare/decoder_relabelled` to the last one.

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
