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
       │    ├─ run_gsea.py ────────────→ gsea_gonnect_receptive_fields/
       │    │    (--gene-sets build_gene_sets → gsea_gonnect_layers/, as published)
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

### Figure 4 on held-out samples

As published, the two sides of Figure 4 score different samples: panels a-c
(and `untrained_control.py`, `untrained_dpr.py`) take every primary tumour,
about 70% of them training samples, while panels d-e (and
`untrained_baselines.py`) take each seed's test split, of every sample type.
`fig4.py --eval-set test --pool seeds` scores every panel on the primary
tumours of each seed's own test split (`split_data` at that seed; AE_2.2.22-26
on splits 2-6). The baselines' test files are not exactly that split: their
runners saved it as patient ids and reloaded every row of those patients, which
adds 40-55 rows per seed that `split_data` put in train or validation. Under
`test`, fig4 checks that each file holds the whole split and nothing but rows of
its patients, and scores only the split's own rows. The three untrained steps take the same `--eval-set`, so each box sees its
violin's samples. `eval_set_report.py` tabulates the two figures side by side,
with the per-cancer-type test-split counts, into `out/prepare/eval_set/`.

`test` is now the default of `fig4.py` and of the three untrained steps; `all`
is what the published figure did. With every default, the revised Figure 4 and
its untrained references rebuild as follows (`data/untrained_reference/` holds
copies of the three references, renamed `fig4_untrained_{gonnect,dpr,baselines}.tsv`):

```
pixi run python figures/src/prepare/untrained_control.py     # ~40 min
pixi run python figures/src/prepare/untrained_dpr.py
pixi run python figures/src/prepare/untrained_baselines.py
pixi run python figures/src/fig4.py
```

## Figure 4's gene sets

Panels a and b of Figure 4 read `gsea_gonnect_layers/`, panel c reads
`gsea_gonnect_bottleneck/`, and the two were built from different gene-set
definitions: `run_gsea.build_gene_sets` takes a term's direct annotations, read
from encoder layers 0 and 1 only, and `run_gsea_bottleneck.build_receptive_fields`
takes every gene that reaches the node. Reading only two layers also loses
genes: a gene whose leaf-proxy chain enters its term at layer 2 or 3 is dropped.

```
hard_links.csv
  └─ gsea_gene_sets.py ───────────────→ out/prepare/gsea_consistency/gene_sets_per_term.tsv
                                         (+ build_gene_sets_audit.tsv)
hard_links.csv + deg_results.csv + go_term_activations (corrected) + the GSEA above
  └─ gsea_consistency.py ─────────────→ out/prepare/gsea_direct_traced/
                                        out/prepare/gsea_receptive_fields/
                                        out/prepare/gsea_consistency/ (stats per layer)
```

`gsea_gene_sets.py` traces both definitions through every encoder layer, and
writes one row per (definition, term): `build_gene_sets` (unchanged), `direct`
(the term's own annotations, through proxies of any length) and
`receptive_field`. `gsea_consistency.py` reruns GSEA on all three with
`run_gsea.py`'s settings, checks that the unchanged sets reproduce
`gsea_gonnect_layers/` and that the receptive fields reproduce
`gsea_gonnect_bottleneck/` (both exactly), and scores every GONNECT layer
against every reference with Figure 4's `--pool seeds` statistic. GSEA takes
under a minute, the AUCs about fifteen. `untrained_control.py --gsea-csv` scores
the untrained reference against one of the new files.

**Chosen: receptive fields, for every panel**, which is what the Methods
describe. `run_gsea.py` now builds them by default (`--gene-sets
receptive_field`, through `gsea_gene_sets.py`) into
`out/prepare/gsea_gonnect_receptive_fields/`, which reproduces
`gsea_consistency.py`'s `gsea_receptive_fields/` exactly and adds the term
names; `data/gsea_gonnect_receptive_fields/` is a copy of it. `fig4.py`,
`untrained_control.py` and `untrained_dpr.py` read it by default.
`--gene-sets build_gene_sets` rebuilds the published `gsea_gonnect_layers/`.
`out/fig4_old.pdf`, `fig4_fixed_rf.pdf` and `fig4_fixed_direct.pdf` compare the
three definitions on the test split (see `out/prepare/fig4_gene_sets/README.md`).

## The test-split clustering metrics

As first published, Figure 2's clustering metrics were not on one footing. The
GONNECT workbooks took SS, ARI and NMI (panels b–d) and the per-cancer-type SS
and purity (j, k) over all 9,797 samples, training data included. Its MSE
panels (a, i) and every baseline metric came from the test split. The deposited
OntoVAE and VEGA metrics also took the silhouette against the k-means clusters,
where GONNECT's SS takes it against the true cancer types, which put the
baselines on a higher scale. Two steps make every metric panel score held-out
samples, with the label silhouette:

```
latent_embeddings/ + metric_data_TCGA_1000_30_new.xlsx + TCGA_complete_bp_top1k.csv.gz
  └─ test_split_metrics.py ────────────→ out/prepare/test_split_metrics/

baseline_activations/ + metrics/*_rand.txt + TCGA_complete_bp_top1k.csv.gz
  └─ rescore_baselines.py ─────────────→ out/prepare/rescore_baselines/
```

`test_split_metrics.py` scores the MLP and GONNECT embeddings on the rows
`gonnect.train.train.split_data` held out at each seed. `gonnect_clustering.csv`
holds b–d. `per_type_ss.csv` holds j, the per-sample silhouette over the same test
split averaged per cancer type, so it is the per-type breakdown of b.
`per_type_purity_k30.csv` holds k. There the test samples' 30 neighbours come from
the training split, not the test split: at 15% of the data, half the cancer
types have fewer than 30 test samples in some seed, and a search among those
would cap their purity at their size. CHOL is the one type with fewer than 30
training samples (23–28); none of its held-out samples reaches that limit. A
`Random` column holds chance level, the type's share of the training split. The
step covers all ten models with embeddings, so S3 and Figure 3 read it too. The
randomized AE_2.2 arm is numbered 22–26 but trained on splits 2–6, so it is
scored on those. Each run's logged test loss reproduces on its split and no
other. The purity tables exist for k = 10, 20 and 30.

`rescore_baselines.py` handles the baselines. The runners in `baselines/` now
default to the label definition, but retraining cannot reproduce the deposited
models, so it scores the test-split latents they already produced: NMI, ARI and
the label silhouette, for every seed and graph arm. MSE needs reconstructions,
which were not deposited, and is carried over. It writes `ontovae_rand.txt` and
`vega_rand.txt` in the shape of their `data/metrics/` namesakes (test split
only), plus a table setting each rescored value beside the deposited one. That
table also carries the k-means silhouette of the same latents, which lands
within ~0.01 of the deposited SS on every model and arm. That is the check that
the deposited values used that definition.

Two sets of runs have no deposited embeddings, and their checkpoints stay on the
cluster. One is Supplementary Figure S1's ct=5, ct=10 and 2,000-gene runs
(AE_4.x, 5.x, 6.x, ~60 GB). The other is Figure 3's fully random and randomized
soft-link arms (AE_10.2, AE_11.1, AE_10.1, ~2.4 GB). Two more steps cover them,
each with a preset per set, `s1` and `fig3`:

```
(cluster) out/trained_models/AE_{4,5,6}.{0,1}/, AE_{10.1,10.2,11.1}/ + data/TCGA_complete_bp_top{1,2}k.csv.gz
  └─ embed_cluster_runs.py <preset> ───→ latent_embeddings/cluster_test_split/

latent_embeddings/cluster_test_split/ + the preset's metric workbooks
  └─ cluster_test_metrics.py <preset> ─→ out/prepare/cluster_test_metrics/
```

`embed_cluster_runs.py` runs on the cluster, next to the checkpoints. It rebuilds
nothing: it applies each checkpoint's weights directly. It keeps a run only after
two checks: the run's reconstruction MSE on its held-out rows must equal its
training log, and that log must equal the S1 workbook's value for that run. It
writes the held-out rows' embeddings plus a report on both checks. Later runs'
file names carry a slurm job id; where a rerun left two checkpoints, the one
whose log matches the workbook is kept. `cluster_test_metrics.py` scores them
into `sweep_metrics.csv` (s1) or `randomized_metrics.csv` (fig3), with each
run's MSE from its log. For Figure 3, every workbook value matches that
convention. That corrects two S1 bars: for ct=10 SL-enc and SL-dec the workbook
had the test loss including the soft-link penalty. The ct=30 and
1,000-gene references are Figure 2's own runs, so figS1 takes those from
`gonnect_clustering.csv`.

All steps use the same k-means (k = number of classes, seed 42, n_init=10), and
all are deterministic. Figures 2, 3, S1, S3 and S4 read their output files from
`data/metrics/test_split/`, kept apart from the deposited all-sample files they
replace. Copy them there after a rebuild.

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
