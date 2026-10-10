# GONNECT - figure reproducibility scripts

Of the scripts in this folder, each directly outputs a figure in the paper. Each writes both a PNG and a PDF into `out/`,
named after the figure number. Only the PDFs are committed; the PNGs are
regenerated on the first run.

```
figures/
  src/     one figN.py per figure, plus the shared _common.py
  data/    every input the scripts need (not in git -- see below)
  out/     generated figures (fig2.pdf committed, fig2.png regenerated)
```

The files required in the `data/` folder are too large to include in the
repository. They are deposited at 4TU.ResearchData:
[10.4121/0d78788b-6bd7-4941-a942-245309107b6d](https://doi.org/10.4121/0d78788b-6bd7-4941-a942-245309107b6d).

Download the archive from there and unpack it so that its `data/` directory
lands at `figures/data/`, i.e. `figures/data/hard_links.csv`,
`figures/data/latent_embeddings/`, and so on. The scripts resolve every path
relative to this file's location, so nothing else needs configuring.


### `data/` contents


The GONNECT model versions are `AE_2.0` (fixed link), `AE_2.1` (soft link), `AE_2.2` (randomized GO graph) and `AE_9.1` (10 % of GO edges held out); the trailing number in `AE_2.1.4` is the seed, and seeds run 2–6.

`AE_3.x` is the soft-link α sweep, and numbers differently: `AE_3.0` to `AE_3.3` are α = 10² to 10⁵ and `AE_3.-1` holds the two references (fixed links, and fully connected), while the trailing number is a re-run label rather than a seed — every sweep run is a single instance at data-split seed 1. See *Figures S7 and S8*.

| Folder | Holds |
|---|---|
| `metrics/` | Every model metric. Workbooks (`.xlsx`) for the GONNECT family; the two `*_rand.txt` baseline files; and the flat per-method baselines of the published deposit (`ontovae.txt`, `vega_hallmark.txt`, `vega_reactome674.txt`), which no figure reads any more. `metrics/test_split/` holds the held-out metrics figures 2, 3, S1, S3 and S4 read: every model on its test split, silhouette against the true cancer types. See below. |
| `latent_embeddings/` | Per-model sample embeddings, `.pt`, one folder per model version. Drives every t-SNE panel and, through `src/prepare/test_split_metrics.py`, every held-out clustering metric. `cluster_test_split/` holds the held-out embeddings of runs whose checkpoints stayed on the cluster. |
| `model_checkpoints/` | Trained model weights, `.pt`. Only `AE_2.0` and `AE_2.1` are present. |
| `loss_traces/` | Per-epoch train / validation / test loss, one tab-separated `.txt` per `AE_3.x` α-sweep run. 11 files, 0.8 MB. Read by figS7. |
| `alpha_sweep_weights/` | The weight matrices of the α-sweep runs, one `.pt` per run and module, holding only the module that run constrains. 11 files, 296 MB. Read by figS8. Extracted from the training checkpoints — see *Figures S7 and S8*. |
| `go_term_activations/` | Per-GO-node activations per sample, `.csv.gz`, one file per seed and module. Its decoder files are labelled one layer off (see *fig4.py*). |
| `go_term_activations_corrected/` | The same with corrected decoder files (re-extracted from the checkpoints) and the deposited encoder files. Read by fig4, figS5 and figS6 by default (`--preprint`: `go_term_activations/`). Local only until the next deposit version (see `4TU_TODO.md`). |
| `soft_link_weights/` | Learned soft-link weight matrices, `.csv.gz`, one file per seed and module (~3M rows each). |
| `gsea_gonnect_receptive_fields/` | GSEA enrichment for GONNECT, all layers, on receptive-field gene sets (a term's own and its descendants' annotations). Every panel of the revised Figure 4 reads it. Local only until the next deposit version. |
| `gsea_gonnect_layers/` | GSEA enrichment for GONNECT, all layers, on direct annotations read from layers 0-1 only. Panels a-b of preprint v3's Figure 4 (`fig4.py --preprint`). |
| `gsea_gonnect_bottleneck/` | GSEA enrichment for GONNECT, bottleneck layer only, on receptive fields. Panel c of preprint v3's Figure 4. |
| `untrained_reference/` | The untrained boxes of the revised Figure 4: `fig4_untrained_{gonnect,dpr,baselines}.tsv`, from `prepare/untrained_{control,dpr,baselines}.py`. Local only until the next deposit version. |
| `gsea_baselines/` | GSEA enrichment for OntoVAE and the two VEGA gene sets. |
| `perm_nulls_gonnect/` | Permutation nulls for GONNECT, one `.npz` per layer. Derived — see *Where the derived inputs come from*. |
| `perm_nulls_baselines/` | Permutation nulls for OntoVAE and VEGA, one `.npz` per model and arm. Derived — see *Where the derived inputs come from*. |
| `baseline_activations/` | Per-sample OntoVAE and VEGA latent activations, `vega/run_seed-<2..6>/z_test_<arm>_<db>.csv` and `ontovae/run_seed-<2..6>/pathway_activities_test_<arm>.parquet`. Read by fig4 for its per-seed dots, and by the baseline permutation step. |
| `ontologies/` | The gene-set databases (`.gmt`): Hallmark, Reactome and the OntoVAE GO set, each with its `_rand_degree` and `_rand_size` counterparts for the randomized arms. Input to the GSEA step, not read by any `figures/` script. |
| `cache/` | Regenerable intermediates, built on first run. Not shipped. See *Caches*. |

Loose files at the top of `data/`:

| File | Holds |
|---|---|
| `hard_links.csv` | The GO graph itself: every fixed edge, with each node's layer and bottleneck flag. |
| `TCGA_complete_bp_top1k.csv.gz` | The expression matrix — top 1000 genes, with the cancer-type label per sample. |
| `activation_preservation_per_node_corrected.csv` | Derived table for fig5 panels b and c, from `go_term_activations_corrected/` (see *fig5.py*). Local only until the next deposit version. |
| `activation_preservation_per_node.csv` | The same from `go_term_activations/`: preprint v3's fig5 (`fig5.py --preprint`). |
| `AE_9.1_{encoder,decoder}_10perc_removed.csv.gz` | Which GO edges were held out for the recovery experiment. |
| `deg_results.csv` | Per-cancer-type differential expression (one-vs-rest). Input to the GSEA step, not read by any `figures/` script. |

### Where the derived inputs come from

The `gsea_*` and `perm_nulls_*` folders are analysis intermediates rather than
measurements. They ship ready-made — together about 15 MB, nothing next to the
rest of `data/` — so a figure never waits on a GSEA run. `ontologies/`,
`deg_results.csv` and `baseline_activations/` are the raw material those steps
read; of the four only `baseline_activations/` is opened by a figure script.

The steps that build them are in [`src/prepare/`](src/prepare/), with their own
README:

```
TCGA_complete_bp_top1k.csv.gz
  └─ prepare/plot_deg_volcano.py ─────────→ deg_results.csv
       ├─ + hard_links.csv
       │    ├─ prepare/run_gsea.py ───────→ gsea_gonnect_receptive_fields/
       │    │    (--gene-sets build_gene_sets → gsea_gonnect_layers/, as in preprint v3)
       │    └─ prepare/run_gsea_bottleneck.py → gsea_gonnect_bottleneck/
       └─ + ontologies/*.gmt
            └─ prepare/run_gsea_baselines.py → gsea_baselines/

go_term_activations/ + hard_links.csv + gsea_gonnect_layers/
  └─ prepare/plot_perm_nulls_layers.py ───→ perm_nulls_gonnect/

go_term_activations_corrected/
  └─ prepare/activation_preservation.py ──→ activation_preservation_per_node_corrected.csv
       (on go_term_activations/ → activation_preservation_per_node.csv, as in preprint v3)

baseline_activations/ + gsea_baselines/
  └─ prepare/compute_baseline_nulls.py ───→ perm_nulls_baselines/
```

**Every input in `data/` is rebuildable from the raw TCGA expression matrix and
the GO graph.** Nothing here depends on a file that is not shipped. Regeneration
writes to `out/prepare/` rather than into `data/`, so a stochastic rerun cannot
quietly move a committed figure — see `src/prepare/README.md`.

#### The baseline `.txt` files in `metrics/`

| | Shape | Read by |
|---|---|---|
| `ontovae_rand.txt`, `vega_rand.txt` | `run-N: {arm: {split: ...}}` for OntoVAE and `run-N: {annotation: {arm: {split: ...}}}` for VEGA, nested by graph arm, carrying `true`, `random` and `degree_preserving` | `read_baseline_ontovae` / `read_baseline_vega`, with an explicit `graph_types` |
| `ontovae.txt`, `vega_hallmark.txt`, `vega_reactome674.txt` | `run-N: {split: {metric: val}}`, flat, one method per file | nothing any more |

The flat files are a separate training run from the `true` arm of the
`*_rand.txt` pair. Figure 3 used to take its true-graph bars from them, so
Figures 2 and 3 showed different numbers for the same baseline. Every figure now
reads the `*_rand.txt` pair in `metrics/test_split/`: Figure 2 its `true` arm,
Figure 3 every arm.

The `test_split/` pair has the same shape, but its NMI, ARI and silhouette were
rescored from `baseline_activations/`. The deposited ones took the silhouette
against the k-means clusters rather than the true cancer types. Its MSE is carried
over unchanged. Beside them, `gonnect_clustering.csv` holds SS, ARI and NMI for the
MLP and the GONNECT family on each seed's test split, in long form (`metric`,
`method`, `repeat`, `value`). `per_type_ss.csv` and `per_type_purity_within_test_k10.csv`
hold the per-cancer-type SS and purity of Figure 2j–k, cancer types × methods, averaged
over seeds; the purity file adds a `Random` column and is blank (NaN) for the types
too small to score. `per_type_purity_within_test_k{20,30}.csv` add figS3's other two
k values, on the same definition. The workbooks took all of these over every
sample, so Figure 2 now takes only MSE from them. See `src/prepare/README.md`, *The
test-split clustering metrics*.

## Running

Everything runs under the repository's pixi environment, from the repository
root:

```
pixi run python figures/src/fig2.py
pixi run python figures/src/figS12.py
```

Each script defaults to `figures/data` for input and `figures/out` for output, so
no arguments are needed. Both are overridable:

```
pixi run python figures/src/fig2.py --data-dir some/other/data --out-dir /tmp
```

To build everything:

```
pixi run python figures/src/run_all.py
```

## Figures

| Figure | Script | Came from |
|---|---|---|
| 2 | `fig2.py` | `fig_main/plot_main.py` |
| 3 | `fig3.py` | `fig_main/fig3.py` (`--output-main`) |
| 4 | `fig4.py` | `fig_deg/biotalk/plot_violin_4row.py` |
| 5 | `fig5.py` | `fig_main/weight_activation_combined.py` |
| S1 | `figS1.py` | `fig_main/fig2_supp.py` |
| S2 | `figS2.py` | `fig2/plot_embeddings.py --version 2.1` |
| S3 | `figS3.py` | newly written, from `fig_main/knn_purity.py` — see *Per-cancer-type metrics* below |
| S4 | `figS4.py` | `fig_main/fig3.py` (`--output-supp`) |
| S5 | `figS5.py` | newly written — see *Figures S5 and S6* below |
| S6 | `figS6.py` | newly written |
| S7 | `figS7.py` | newly written — see *Figures S7 and S8* below |
| S8 | `figS8.py` | newly written |
| S9 | `figS9.py` | `fig_main/fig_sl_stability.py` |
| S10 | `figS10.py` | `fig_main/fig_sl_recovery_trajectory.py` (encoder) |
| S11 | `figS11.py` | same script, decoder |
| S12 | `figS12.py` | `fig_main/fig_sl_edits.py` |

Figures 1 and 6 are hand-drawn schematics with no generating code — see
*Not here*.

`figS4.py` imports `main()` from `fig3.py`, and `figS11.py` from `figS10.py`,
because each pair is the same analysis over a different metric or module.
Both remain independently runnable.

## Which files go into which script

All paths are relative to `figures/data`.

### fig2.py — performance metrics, t-SNE embeddings, per-cancer-type heatmaps

```
metrics/metric_data_TCGA_1000_30_new.xlsx     GONNECT family, MSE only
metrics/test_split/gonnect_clustering.csv     GONNECT family, SS / ARI / NMI
metrics/test_split/ontovae_rand.txt           OntoVAE baseline ('true' arm, rescored)
metrics/test_split/vega_rand.txt              VEGA Hallmark + Reactome ('true' arm, rescored)
metrics/mse_per_cluster_TCGA_1000_30.xlsx     panel i (MSE per c.t.; repaired on read)
metrics/test_split/per_type_ss.csv            panel j (SS per c.t.)
metrics/test_split/per_type_purity_within_test_k10.csv   panel k (purity per c.t., plus chance)
TCGA_complete_bp_top1k.csv.gz                     cancer-type labels, abundance (panel l)
latent_embeddings/AE_2.0/AE_2.0.2_{none,decoder,encoder,both}_full_dataset.pt   panels e-h
cache/tsne/                                       t-SNE coordinates (regenerable)
```

Panels: a–d metric bars, e–h t-SNE, i–k the per-cancer-type heatmaps
(MSE, SS, k-NN purity at k=10), l the abundance bar. Every metric panel scores
held-out samples only; the t-SNE panels show every sample. Panels i and j are the
per-type breakdowns of a and b, on the same test split. Panel k stays inside the
test split too: each test sample's 10 nearest neighbours are other test samples.
A type with fewer than 10 test samples in some seed cannot fill 10 neighbours with
its own kind, so its row is left blank: CHOL, DLBC, KICH, MESO, UCS and UVM. k = 10
rather than preprint v3's 30 is what keeps the rest; at k = 30, half the types
would be blank. Its last column, `Random`, is chance level: the cancer type's
share of the test split, the sample itself excluded. The randomized arm is deliberately absent, since it is Figure 3's
subject and appears per cancer type in figS3.

### fig3.py and figS4.py — GO-graph randomization

```
metrics/metric_data_TCGA_1000_30_new.xlsx              MSE of the true-graph and DPR GONNECT arms
metrics/test_split/gonnect_clustering.csv              their SS / ARI / NMI
metrics/test_split/randomized_metrics.csv              all four metrics of the FR, DPR-SL and FR-SL arms
metrics/test_split/{ontovae,vega}_rand.txt             every baseline arm, true graph included (rescored)
```

`fig3.py` plots MSE and SS; `figS4.py` plots ARI and NMI from the same inputs.
Every value is on held-out samples, and the true-graph bars are Figure 2's, so
the baselines no longer come from the flat per-method files. Those were a
separate training run from Figure 2's. The FR, DPR-SL and FR-SL arms (AE_10.2,
AE_11.1, AE_10.1) have no deposited embeddings; see `src/prepare/README.md`.

### fig4.py — activation–enrichment agreement

```
TCGA_complete_bp_top1k.csv.gz            cancer-type labels
hard_links.csv                           GO graph: bottleneck index, layer map
gsea_gonnect_receptive_fields/gsea_results.csv   GSEA reference, every panel (preprint v3: the two below)
gsea_gonnect_layers/gsea_results.csv             --preprint: panels a, b
gsea_gonnect_bottleneck/gsea_results.csv         --preprint: panel c
gsea_baselines/gsea_results_hallmark.csv       panel e
gsea_baselines/gsea_results_reactomes.csv      panel e
gsea_baselines/gsea_results_ontovae.csv        panel d
perm_nulls_gonnect/AE_{2.0,2.1}_{encoder,decoder}_auc/perm_layer_{0..8}_arrays.npz      --preprint only
perm_nulls_baselines/OntoVAE_layer_{00..10}_true_auc/perm_nulls_arrays.npz             --preprint only
perm_nulls_baselines/VEGA_{hallmark,reactomes}_{true,degree_preserving,random}_auc/perm_nulls_arrays.npz
latent_embeddings/AE_2.2/AE_2.2.{22..26}_{encoder,decoder,both}_full_dataset.pt   panel c
go_term_activations_corrected/AE_{2.0,2.1}.{2..6}_{encoder,decoder}_activations.csv.gz  per-seed dots
                                         (--preprint: go_term_activations/)
untrained_reference/fig4_untrained_{gonnect,dpr,baselines}.tsv                  untrained boxes
cache/auc_4row_<hash>/                   per-seed AUC matrices (regenerable, slow;
                                         --preprint: cache/auc_4row/)
```

The `perm_nulls_*` `.npz` files are inputs to *this* script rather than caches
it manages: `fig4.py` never writes them, and a missing one drops that violin and
is reported in a summary line at the end of the run. They are built by
`prepare/plot_perm_nulls_layers.py` and `prepare/compute_baseline_nulls.py`,
both expensive to rerun — which is why they ship precomputed.

`--no-dots` skips the per-seed dots and the whole `cache/auc_4row` path — much
faster while iterating on layout. It does not reproduce the preprint v3 figure.

Also writes `out/fig4.csv` with the observed values, permutation p, null mean
and SD, and the five per-seed values per violin.

**Defaults: the revised figure.** A bare `fig4.py` draws the revised Figure 4,
which differs from the preprint v3 one in five ways: `--pool seeds` (see below),
`--eval-set test` (every panel scored on the primary tumours of each seed's test
split, not panels a-c on every primary tumour), the corrected decoder
activations, one gene-set definition (receptive fields) for every panel, and the
untrained boxes. `--preprint` restores all five for any option not given
explicitly; the preprint v3 figure also needs the `perm_nulls_*` files above.
See `src/prepare/README.md`, *Figure 4 on held-out samples* and *Figure 4's gene
sets*.

`--untrained` draws an untrained reference as a box beside a violin, and adds
its quantiles to `fig4.csv`; `--no-untrained` drops the default ones. The references come from `prepare/untrained_control.py` (panels
a and b), `prepare/untrained_dpr.py` (panel c) and
`prepare/untrained_baselines.py` (panel d's true graph, all of panel e).
`--activations-dir`, `--gonnect-nulls-dir` and `--cache-dir` point panels a and
b at other activations and nulls, such as the relabelled decoder files. See
`src/prepare/README.md`, *The untrained control*. `--gsea-layers-csv` and
`--gsea-bottleneck-csv` set the GSEA reference of panels a-b and of panel c.

`--pool` sets the red line. The preprint v3 figure (`activations`) scores one
AUC on seed-averaged activations for panels a–c, and per-seed AUCs averaged
over seeds for d–e. `auc` uses the averaged AUCs everywhere. `seeds` makes every
red line the mean of the five per-seed values. Its null (1,000 permutations,
each seed's GO-term columns shuffled independently, the five statistics
averaged) is computed in the run, so no `perm_nulls_*` file is read.

### fig5.py — weight distributions and activation preservation

```
model_checkpoints/AE_2.0/AE_2.0.{2..6}_both_model.pt     fixed-link checkpoints
model_checkpoints/AE_2.0/AE_2.0.{2..6}_none_model.pt     fully-connected MLP (panel a)
model_checkpoints/AE_2.1/AE_2.1.{2..6}_both_model.pt     soft-link checkpoints
hard_links.csv                                        which weight positions are GO edges
activation_preservation_per_node_corrected.csv        panels b, c
activation_preservation_per_node.csv                  --preprint: panels b, c
```

The per-node table is a derived one, not raw data; here it is treated as an
input. `prepare/activation_preservation.py` builds the default one from
`go_term_activations_corrected/AE_{2.0,2.1}.{2..6}_{encoder,decoder}_activations.csv.gz`.
Preprint v3's table was built by the original
`fig_sl_preserve/activation_preservation.py` from `go_term_activations/`, whose
decoder columns are labelled one layer off. Only panel c differs between the
two: decoder L8's preservation falls from 0.48 to 0.12, because the preprint v3
L8 columns held the reconstructed genes. `--preprint` reads preprint v3's
table; `--preservation-csv` swaps in any other.

### figS1.py — GO-processing hyperparameter sweep

```
metrics/metric_data_TCGA_1000_30_new.xlsx              section 1 GONNECT family, MSE
metrics/test_split/{ontovae,vega}_rand.txt             section 1 baselines (rescored)
metrics/test_split/gonnect_clustering.csv              SS / ARI / NMI: section 1, and the ct=30 / 1k references
metrics/metric_data_TCGA_1000_5.xlsx                   section 2, ct=5, MSE
metrics/metric_data_TCGA_1000_10.xlsx                  section 2, ct=10, MSE
metrics/metric_data_TCGA_1000_30.xlsx                  section 2 ct=30, and section 3 "1k", MSE
metrics/metric_data_TCGA_2000_30.xlsx                  section 3, "2k", MSE
metrics/test_split/sweep_metrics.csv                   all four metrics: ct=5, ct=10, 2k
```

Every value is on held-out samples, as in Figure 2. Section 1 is exactly
Figure 2a–d, and the ct=30 / 1k reference bars are Figure 2's runs. The
workbooks' SS, ARI and NMI spanned all samples, so only their MSE is used, for
the ct=30 / 1k references. The sweep runs' MSE comes from their training logs
with the rest, which corrects ct=10 SL-enc / SL-dec: their workbook MSE included
the soft-link penalty. Note that `metric_data_TCGA_1000_30.xlsx` and
`metric_data_TCGA_1000_30_new.xlsx` are different files: the first drives the
sweep sections, the second the main-text model set. They hold identical values
for every run both contain.

### figS2.py — soft-link t-SNE embeddings

```
TCGA_complete_bp_top1k.csv.gz                              panel a input space, labels
latent_embeddings/AE_2.0/AE_2.0.2_none_full_dataset.pt            panel b (MLP reference)
latent_embeddings/AE_2.1/AE_2.1.2_{decoder,encoder,both}_full_dataset.pt   panels c, d, e
cache/tsne/                                                t-SNE coordinates (regenerable)
```

Panel b always comes from the AE_2.0 `_none` model — only the fixed-link
experiment ships a fully-connected variant — so this figure mixes one AE_2.0
file with three AE_2.1 files.

### figS3.py — per-cancer-type embedding quality, all ten models

```
metrics/test_split/per_type_purity_within_test_k{10,20,30}.csv   panels a-c
metrics/test_split/per_type_ss.csv                   panel d
metrics/mse_per_cluster_TCGA_1000_30.xlsx            panel e (MSE sheet; repaired on read)
TCGA_complete_bp_top1k.csv.gz                        cancer-type labels, abundance (panel f)
```

Every panel scores held-out samples, as in Figure 2j–k: SS and purity are both
taken within the test split. A type with fewer than k test samples in some seed
is blank in that purity panel: 6 types at k = 10 (the panel equal to Figure 2k),
11 at k = 20 and 16 at k = 30. The degree-preserving arm (AE_2.2.22–26) trained
on splits 2–6, like the others. Each purity panel ends in a `Random` column:
chance level, the cancer type's share of the test split, the sample itself
excluded.

Panels a–e are the five heatmaps (purity at k = 10/20/30, SS, MSE) and f the
abundance bar. The figure carries no title — what it shows belongs in the
caption.

Also writes `out/figS3.csv` with every number in the figure, one row per
(cancer type, metric, method).

### figS5.py and figS6.py — GO-term activation heatmaps

```
go_term_activations_corrected/AE_2.0.{2,3,4}_encoder_activations.csv.gz   figS5
go_term_activations_corrected/AE_2.0.{2,3,4}_decoder_activations.csv.gz   figS6
TCGA_complete_bp_top1k.csv.gz                           cancer-type labels
hard_links.csv                                          restricts columns to GO-term nodes
```

`--preprint` reads `go_term_activations/` instead, as the preprint v3 figures
did. Its encoder files are the same, so figS5 does not change; its decoder
files are labelled one layer off, so figS6 does (see *Figures S5 and S6*).
`--activations-dir` reads the activations from anywhere else.

### figS7.py — α-sweep training curves

```
loss_traces/AE_3.-1/AE_3.-1.2_none_results.txt              MLP
loss_traces/AE_3.-1/AE_3.-1.2_{encoder,decoder}_results.txt Fixed links
loss_traces/AE_3.{0,1,2,3}/AE_3.{0,1,2,3}.3_{encoder,decoder}_results.txt   α = 10²–10⁵
```

Plots the last column of each file. For the four α runs that is `MSE loss`, the
unregularized reconstruction MSE on the test split; column 3 for those is the
regularized objective, which reaches ~32 at α = 10⁵ and is not comparable across
α. The three `AE_3.-1.2` baselines predate that column and carry only three —
see *Figures S7 and S8*.

### figS8.py — α-sweep weight distributions

```
alpha_sweep_weights/AE_3.-1.2_none_weights.pt       fully connected
alpha_sweep_weights/AE_3.-1.2_encoder_weights.pt    fixed links
alpha_sweep_weights/AE_3.0.3_encoder_weights.pt     α = 10²
alpha_sweep_weights/AE_3.2.3_encoder_weights.pt     α = 10⁴
```

Encoder only, layers pooled. GO positions are the nonzero weights of the
fixed-link checkpoint — that model pins every non-GO weight at exactly 0 and
every proxy weight at exactly 1, so no mask file is needed and the positions
cannot drift from the model.

### figS9.py — soft-link weight stability across seeds

```
soft_link_weights/AE_2.1.{2..6}_{encoder,decoder}_soft_links.csv.gz
```

Seeds are discovered by globbing, so the figure adapts if more are added.

### figS10.py and figS11.py — held-out GO-edge recovery

```
soft_link_weights/AE_2.1.3_{encoder,decoder}_soft_links.csv.gz    degree baseline (seed 3, fixed)
AE_9.1_{encoder,decoder}_10perc_removed.csv.gz             which edges were held out
soft_link_weights/AE_9.1.{3,4,5,6}_{encoder,decoder}_soft_links.csv.gz   the held-out-edge runs
```

These use seeds 3–6; seed 2 has no AE_9.1 run. Note the `10perc_removed` files
sit at the top of `data/`, not under `soft_link_weights/`.

### figS12.py — soft-link weight vs GO-edit count

```
soft_link_weights/AE_2.1.{seed}_{encoder,decoder}_soft_links.csv.gz   --seed, default 2
cache/sl_edits/                                                per-cell arrays (regenerable)
```

Streaming the ~3M-row weight files takes minutes per module, so the hexbin
cells are cached. Delete `cache/sl_edits/` to force a rebuild; a stale cache is
detected and rebuilt automatically.

## Per-cancer-type metrics

Figure 2j–k and figS3a–d are computed from the embeddings rather than read from a
workbook, so they cover every model with embeddings on disk, including the
randomized AE_2.2 arm. `src/prepare/test_split_metrics.py` computes them and
writes them to `metrics/test_split/`; the figures only lay the tables out.

**k-NN purity**: the fraction of a held-out sample's k nearest neighbours
(Euclidean, full latent space) carrying its own cancer-type label, averaged per
type and then over the five seeds. Purely local: it sees neighbourhood
contamination and nothing else, so a huge diffuse but uncontaminated cluster
scores 1.0 where silhouette would not. Ported from `fig_main/knn_purity.py`.
The neighbours are searched within the test split, and a type with fewer than k
test samples in some seed is left blank. Figure 2 shows k = 10, figS3 k = 10, 20
and 30; both carry a `Random` chance-level column.

**Silhouette per type**: the per-sample silhouette over the test split, averaged
per type.

**NMI per cancer type does not exist.** NMI is a global clustering metric that
compares two partitions of all samples; it has no per-type decomposition the
way the silhouette and purity do.

## A repaired input

`metrics/mse_per_cluster_TCGA_1000_30.xlsx` reaches us damaged, and both
figures that read it (2 and S3) go through `_common.read_per_cluster_workbook`,
which decodes it on the way in.

The workbook was written with every value as text at three decimals, then
opened and saved in a locale where `.` groups thousands. Excel converted each
string that parses as a grouped number — `"1.219"` became the integer 1219 with
number format `#,##0` — and left the rest alone, because `"0.354"` is not valid
grouping (a leading zero group is rejected). The damage is therefore
value-conditional: **exactly the cells whose true value is ≥ 1, multiplied by
1000.** In the released file that is 28 cells of the MSE sheet, all 4-digit
integers, i.e. true values between 1.008 and 1.678. The SS sheet is untouched —
every value there is below 1 in magnitude.

Three independent things confirm the reading:

- **storage types** — coerced cells are integers carrying a grouping format,
  every other cell in the sheet is text;
- **the distribution** — 292 cells ≤ 0.98, 28 cells ≥ 1008, and nothing at all
  in between, which no continuous quantity produces;
- **the arithmetic** — undoing the factor makes the sample-weighted per-type
  mean match the independently reported overall MSE to within 0.5% for all
  seven methods. Before the repair GONNECT-dec was out by a factor of 180.

The digits survive the coercion, so dividing by 1000 restores the original to
its full three decimals: this is a decode, not an estimate. It is done on read
rather than by repairing the file because **opening the workbook in the same
locale re-corrupts it** — a repaired file would silently regress. Cells that
were never coerced pass through untouched, so a properly regenerated workbook
needs no change here. The repair count is printed on every run; if it stops
saying 28, the input changed.

Consequences worth knowing. Those 28 cells used to sit above the `MSE > 10`
threshold and were drawn as grey "x" cells labelled *diverged models* — in
Figure 2 that was 12 cells across the GONNECT-dec and GONNECT-both columns.
Nothing had diverged: all five seeds of both models report a stable overall
MSE (0.645–0.665 and 0.687–0.737), and the silhouette scores for the same
cells are normal or better than MLP's. **No cell trips the threshold any
more**, so the grey marks are gone from both figures and the real values are
plotted. What they show is a genuine, modest effect that the marks had hidden:
seven cancer types (LIHC, PCPG, ACC, PAAD, TGCT, LAML, THYM) reconstruct about
2× worse than typical under the two fixed-link GO-decoder variants, while the
encoder-only and all three soft-link variants are unaffected. The diverged-run
machinery is kept in both scripts as a guard.

Note that the legacy scripts outside `figures/` which read the same workbook
(`fig2/plot_per_cluster.py`, `fig_main/plot_main.py`) do **not** have this
repair and still show the grey cells.

## Caches

The scripts keep three caches under `data/cache/`. **They are not shipped** — the
directory is absent from both the repository and the deposit, so a first run
builds all three from `data/` and every figure pays the rebuild cost once:

| Path | Used by | Cost to build on first run |
|---|---|---|
| `cache/tsne/` | fig2, figS2 | minutes per panel |
| `cache/sl_edits/` | figS12 | minutes per module |
| `cache/auc_4row*/` | fig4 | ~13 min, rereads every activation file |

**All three are safe to delete** at any point; the scripts recompute and rewrite
them from `data/`.

Because `cache/auc_4row/` is absent on a fresh checkout, fig4 reads the OntoVAE
`.parquet` baselines directly, which needs a parquet engine. `pyarrow` is
declared in `pyproject.toml` for exactly this reason.

`cache/auc_4row/` is the one worth a note, because it is the only cache whose
absence is expensive rather than merely slow. Its entries are keyed by model,
seed and evaluation set, not by the files they came from, so fig4 keeps one
cache per activations directory: `auc_4row/` for `go_term_activations/`
(`--preprint`), and `auc_4row_<hash of the directory>/` for any other, such as
the default `go_term_activations_corrected/`. The note below is about
`auc_4row/`. Its 105 entries cover both the
GONNECT models (from `go_term_activations/`) and the baselines (from
`baseline_activations/`). Verified: with the directory removed entirely,
`fig4.py` rebuilds all 105 from source and reproduces `fig4.png` byte for byte,
with every rebuilt AUC matrix identical to the shipped one.

## Not here

**Figures 1 and 6** are schematics (the GONNECT architecture overview and the
GO-processing steps). They were drawn by hand, not generated, so there is no
script to include.

That is all that is missing now. **Figures S7 and S8** used to be listed here
too: they were produced outside this repository, and the inputs they need were
not in `data/`. Both have since been recovered from the cluster and are shipped
— see *Figures S7 and S8* below.

## Figures S5 and S6

These had no generating script either, but unlike S7/S8 the inputs are present,
so they were rewritten from the preprint v3 captions.

The 20 hand-curated GO terms — "processes expected to vary in activity across
cancer types" — exist nowhere in the repository, so they were **read off the x
axis of the preprint v3 figure** and are now hard-coded in `figS5.GO_TERMS`, in
the preprint v3 column order. Every id was validated against `hard_links.csv`:
all 20 are real GO nodes, present in both the encoder and the decoder, and none
is a bottleneck node. Setting `GO_TERMS = None` falls back to the documented
stand-in (the 20 nodes with the highest **η²**, the share of a node's variance
lying between cancer types rather than within them, computed on the first seed
and held fixed across instances) — kept only as a fallback; the real list is
the default now.

The preprint v3 figure labelled its columns with the bare accessions. These
scripts label with **`process name (GO:id)`** — the name makes the panel
readable, the accession keeps it checkable against the graph and the
manuscript; `figS5.LABEL_WITH_NAMES = False` restores bare accessions. Names
come from `hard_links.csv` at run time rather than a second hand-typed copy, so
they cannot drift from the graph. They are long enough set on their side that
only the bottom row carries them — both rows show the same 20 terms in the same
order, and labelling each would push the figure past what an A4 page holds. The
band is measured from the labels themselves (`xtick_band_in`) rather than
tracked as a constant, so it follows the text if the labels change.

**Columns are centred and scaled per GO term** (`normalize_terms`): each column
is centred on its own mean across cancer types, then each term is scaled into
[-1, 1] by one factor shared across the three instances. Both steps earn their
place.

*Scaling*, because one colour scale cannot serve these 20 terms raw:

| | spread across the 20 terms |
|---|---|
| encoder | 0.020 → 0.73 (**37×**) |
| decoder | 0.087 → 170 (**1,957×**) — one term at 170, the next largest under 16 |
| decoder, `--preprint` | 0.004 → 224 (**55,005×**) — one term at 224, fifteen under 2.7 |

Raw, the decoder's limit is set by that single node and every other column is
pale. Scaling asks what the panels are about — which cancer types light a
process up, and whether the instances agree — rather than which process has the
largest activations.

*Centring*, because a node's offset comes from its random initialisation rather
than the biology and is often the largest thing in the column. Scaling without
removing it first turns a constant column into a saturated ±1 stripe that reads
as the figure's strongest result. Columns with no variation at all are drawn
blank and named in the run output. None of the 20 is flat in the default
inputs; under `--preprint` the decoder's `GO:0006631` is, constant to 4e-09
across all 32 cancer types.

The scale factor is shared across instances rather than computed per instance,
so differences between instances survive; normalising each on its own would
flatten exactly the cross-instance agreement panel b exists to show.

`--no-normalize` plots raw means instead, which keeps magnitude comparable
between terms at the cost above. `--vmax` fixes a symmetric limit, which clips:
on the decoder ±4 clips 5.5 % of cells and ±2 clips 10.8 %.

Neither figure carries a title — that belongs in the caption — and the panel
letters are lower case (**a**, **b**) to match the rest of the paper.

What the panels show, measured as the Pearson r between two instances of a
term's 32 centred means (median over the 20 terms and the three pairs of
instances):

| | signed | absolute |
|---|---|---|
| encoder (S5) | +0.08 | 0.50 |
| decoder (S6) | −0.08 | 0.44 |
| decoder, `--preprint` | +0.42 | 0.64 |

The signs are arbitrary across instances, as panel a is meant to show. The
magnitudes agree only moderately, not near-identically as this section used to
say. The preprint v3 decoder input looked more consistent, probably because 11
of its columns were reconstructed genes, which every instance has to reproduce.

**The decoder input was corrected.** Preprint v3's Figure S6 read
`go_term_activations/`, whose decoder columns are labelled one layer off: of
its 20 columns, 11 held reconstructed genes, 3 proxies and 6 other GO terms.
The |mean| 224 term and the constant `GO:0006631` both came from that. The
figure now reads `go_term_activations_corrected/`; `--preprint` restores the
preprint v3 one. See `src/prepare/README.md`, *The untrained control*.

## Figures S7 and S8

These two had no generating script and, until recently, no inputs either. The
training runs behind them survived on the cluster and are now shipped in
`data/`, so both figures reproduce from `data/` like every other one.

### Which runs they are

`experiment_log` lines 158–241 record the sweep. The full-length runs — 1,000
epochs with early stopping disabled (`patience = 10000`) — are the ones the
figures use; lower trailing numbers are earlier attempts that stopped early,
marked *Too low patience* in the log.

| Curve | Run | Trained on |
|---|---|---|
| MLP | `AE_3.-1.2_none` | `mse` |
| Fixed links | `AE_3.-1.2_{encoder,decoder}` | `mse masked` |
| α = 10² | `AE_3.0.3_{encoder,decoder}` | `soft links` |
| α = 10³ | `AE_3.1.3_{encoder,decoder}` | `soft links` |
| α = 10⁴ | `AE_3.2.3_{encoder,decoder}` | `soft links` |
| α = 10⁵ | `AE_3.3.3_{encoder,decoder}` | `soft links` |

`AE_3.1.4` is a separate *Proxyless* variant (loss `soft links proxy`, which
adds a 100× penalty on soft links pointing at proxy terms) and is not part of
this figure. The later `AE_3.-1.{3..6}_none` runs are an L1-regularized MLP
experiment, not the S7 baselines.

Every run is a single instance at data-split seed 1, unlike the main text's
seeds 2–6, and no `torch.manual_seed` is set anywhere in `src/`, so weight
initialization was never seeded and these traces are not reproducible by
re-running. That is why the recovered runs are shipped rather than regenerated.

### The fixed-links curves are a masked MSE

The two `AE_3.-1.2_{encoder,decoder}` baselines predate the commit that added
plain-MSE tracking (`dc3c6ea`), so their files carry three columns and the
plotted test loss is the objective they trained on, `MSE_Masked`. That zeroes
the residuals of the 29-of-1000 genes with no GO annotation while still dividing
by 1000, so it sits below a plain MSE on the same weights. Measured from the
saved checkpoints:

| Run | Plotted (masked) | Plain MSE |
|---|---|---|
| `AE_3.-1.2_none` | 0.21175 | 0.21175 |
| `AE_3.-1.2_encoder` | 0.23071 | **0.29414** |
| `AE_3.-1.2_decoder` | 0.67214 | 0.70147 |

`AE_3.-1.2_none` trained on plain `mse`, so it is unaffected. The other two are
plotted as in preprint v3 and carry a footnote in the figure saying what they are;
recovering a plain-MSE *trace* would need a re-run, since the per-epoch history
cannot be recomputed from a final checkpoint.

### What changed against the preprint v3 version

Both figures keep their data and panels. What changed is the house style —
`FIG_WIDTH_IN`, the type scale, Type 42 fonts — plus two deliberate choices:

- **Colour.** The preprint v3 pair used matplotlib's `tab10` defaults, whose red
  and green sit at deuteranopic ΔE 0.7, i.e. one colour for a red-green
  colourblind reader. Both now use an Okabe-Ito set verified against the
  all-pairs CVD comparison, with dash patterns as a secondary encoding. Figure
  5a's own four fail the same check (ΔE 3.9) and are worth revisiting.
- **figS8's x axis.** The preprint v3 version binned the signed weight on a linear
  axis, which crushes eleven decades into a spike at zero. It now uses
  `log₁₀|w|`, matching Figure 5a — which is what the S8 caption points the
  reader at for the early-stopped counterpart of panel a. Set
  `figS8.SIGNED_LINEAR_X = True` to restore the preprint v3 rendering.

Panel letters are lowercase here, as in every other regenerated figure; the
preprint v3 S7/S8 captions use `A)`/`B)` and need the one-line change.

### What is deposited, and what is not

The α-sweep training runs save a whole autoencoder per checkpoint in float64,
53.8 MB each. In every run only one module is biologically informed; the other
is a plain dense counterpart that nothing here reads. So `data/` carries the
constrained module's four weight matrices instead of the checkpoint —
`prepare/extract_alpha_sweep_weights.py` cuts them out, halving 592 MB to 296 MB
without changing a value. That is the same trade `soft_link_weights/` already
makes.

What each consumer reads, verified by tracing `open()` and `torch.load`:

| | `loss_traces/` | `alpha_sweep_weights/` |
|---|---|---|
| `figS7.py` | 11 | — |
| `figS8.py` | — | 4 |
| `prepare/alpha_sweep_numbers.py` | 11 | 10 |

The union is exactly what ships: 11 traces (0.8 MB) and 11 weight files
(296 MB). Dropping `alpha_sweep_numbers.py` would take the deposit to 4 weight
files and 216 MB, at the cost of making the sweep's numbers unreproducible from
it — so all 11 are included.

The checkpoints themselves stay in `out/trained_models/AE_3.*`, which is
training output rather than repository content and is not deposited. Regenerating
the extracts needs them present; `extract_alpha_sweep_weights.py` says so if
they are missing.

### The numbers behind the figures

`prepare/alpha_sweep_numbers.py` writes every value the sweep is quoted for —
final MSE, epochs to plateau, active soft links at |w| > 0.1, and |w|
percentiles for GO edges against soft links — to
`out/prepare/alpha_sweep/alpha_sweep_numbers.tsv`. Unlike the rest of
`prepare/`, it is deterministic and reads only shipped inputs.


## Figure width and type

Every figure is saved at **exactly 16.5 in wide**. That single property is what
makes the type consistent across the paper: LaTeX scales a figure by
(placed width / saved width), so if the saved widths are equal, one point size
means one rendered size everywhere. Place them all at the same width — the
scripts assume 180 mm — and the labels match.

Getting this wrong is easy and invisible. The old set was saved at anything
from 11.97 to 23.61 in, a 1.97× spread, so the same 8 pt tick label came out
between 2.40 and 4.74 pt on the page depending on which figure it was in. All
of them were below what a reader can comfortably read, and no two agreed.

Because of that scaling, **the point sizes in the code are not what the reader
sees**. At 16.5 in placed on 180 mm the scale factor is 0.4295, so 16.3 pt in
code renders as 7 pt on the page. Always go through `_common.pt()`, which takes
the size you want *on the page*:

```python
from _common import PT_BODY, figsize, pt

fig = plt.figure(figsize=figsize(9.4))     # width is fixed, you pick the height
ax.set_xlabel("Abundance [n]", fontsize=PT_BODY)   # 7 pt on the page
ax.set_title("MSE", fontsize=pt(8))                # 8 pt on the page
```

The shared type scale, in rendered points: `PT_TINY` 4 (in-cell marks),
`PT_SMALL` 6 (tick labels, legend entries), `PT_BODY` 7 (axis labels,
significance stars), `PT_TITLE` 8 (panel titles), `PT_PANEL` and `PT_SUPTITLE`
10 (bold panel letters, figure titles). Editing those constants restyles every
figure at once. Line and marker weights are in points too, so they go through
`pt()` as well — otherwise they print as hairlines once the figure is scaled.

`save_figure` enforces the width: it keeps the **full canvas width** and trims
only vertically, and raises if a script hands it a canvas that is not
`FIG_WIDTH_IN`. Cropping horizontally would silently undo the whole scheme, and
rescaling to fix a wrong width would silently change the type size, so neither
is done quietly. The practical consequence for anyone editing a figure is that
**the layout must fill all 16.5 in** — dead margin at the left or right edge is
no longer cropped away, it becomes white space in the paper.

Heights are free, and several figures use a lot of it. S5, S6 and S12 come out
roughly 19–20 in tall, which at 180 mm is a full supplementary page each; that
is the honest cost of setting 32 cancer types and 20 ten-character GO ids at a
legible size.

`_common.report_text_overlaps(fig)` reports text artists whose drawn boxes
collide. Every script calls it, so a layout regression shows up in the run
output. Treat it as a signal rather than a verdict — it cannot see a `***` that
merges with its neighbour into a run of six asterisks, since those boxes only
graze. Look at the figure too.

## Verification

**Before** the width-and-type standardisation, six figures were regenerated
from the real data and compared pixel-for-pixel against their original scripts,
run on the same inputs in the same environment. That established the port had
not changed any numbers:

| Figure | Result |
|---|---|
| 2 | 100.000000 % identical pixels |
| 3 | 100.000000 % identical |
| 4 | 100.000000 % identical, and `fig4.csv` matches exactly |
| S1 | 100.000000 % identical |
| S2 | panels a, c, d, e and the legend 100 % identical; panel b differs — see below |
| S4 | 100.000000 % identical |

Figure S2's panel b is the MLP reference. The original recomputed its t-SNE on
every run; this version reads it from `cache/tsne/`, the same entry Figure 2
uses. Both compute t-SNE with `random_state=42` and identical parameters, so on
one machine they agree exactly — and sharing the entry now guarantees that the
MLP panel is the *same* embedding in both figures. The difference seen during
verification came from the cached coordinates having been computed with a
different scikit-learn version than the one recomputing them.

The remaining scripts were verified the same way against synthetic fixtures
(fig5, figS9, figS10, figS11, figS12 — all pixel-identical), but their real
inputs are the multi-GB soft-link and checkpoint files, which could not be
moved into the verification environment. Run them under pixi to confirm.

The restyle then deliberately changed how the figures *look*, so they are no
longer pixel-comparable to the originals — that is the point of it. What was
checked afterwards, on the real data:

- every figure saves at 16.5 in, spread 0.000000 in across all fourteen;
- `report_text_overlaps` returns zero for all fourteen;
- each figure was inspected as a rendered image, not just by the checker;
- `fig4.csv`'s six original columns are unchanged (the run now also writes five
  `seed_*` columns, because it was previously exported with `--no-dots`).

Figures 5, S9, S10, S11 and S12 were rendered against **synthetic fixtures**
built to match the real data's shape — same seeds, layer names, category
labels and label lengths, since those are what drive layout — not against real
data. Their geometry and overlap checks are therefore sound but not conclusive;
give them a look after a local `pixi run`. Two specific things to re-check
there: figS9's bar-total labels are sized for 4-digit counts and would crowd at
6 digits, and figS10's left margin assumes 4-digit "# removed" ticks.

Regenerating a figure on a different machine will produce small
antialiasing-level differences against the committed PDFs if the matplotlib
version differs — about 1.6 % of pixels for Figure 2, none of them structural.

## Notes on the refactor

- Shared code lives in `src/_common.py`: paths and figure saving, the model
  display names (formerly `gonnect_names.py`), the metric readers for the
  performance workbooks, the BH-FDR and paired-t-test helpers, and the t-SNE
  cache. Figure-specific plotting stays in the individual scripts.
- Every script takes `--data-dir` and `--out-dir` and derives all paths from
  them. No script contains a hardcoded path.
- Dead code from abandoned experiments was removed: the cluster mode of the
  stability figure, the violin mode of Figure 5, unused flags, and several
  vestigial constants. `fig4.py` inlines the handful of functions it used from
  four sibling modules so it stands alone.
- Stale documentation was corrected. Most notably, Figure 2's code called its
  embedding panels "UMAP" throughout while computing t-SNE — the paper says
  t-SNE, and t-SNE is what it always did.
- `data/` was populated using hard links, so the ~11 GB it reports costs no
  extra disk. The files are real directory entries and survive if the original
  `data/` is removed. Copying `figures/` elsewhere turns them into ordinary
  copies.

## A note on the PDF output

The scatter panels are **rasterized**, in the PDF as well as the PNG, and that
is deliberate: Figure 2's four t-SNE panels hold ~39k points and Figure S2's
five hold ~49k, and one vector circle per point is what makes a PDF viewer
crawl on every redraw. Marking them `rasterized=True` collapses each panel to a
single embedded image:

| | draw operations | PDF size |
|---|---|---|
| Figure 2 | 40,377 → **378** | 0.73 → 0.49 MB |
| Figure S2 | 49,194 → **97** | 0.82 → 0.68 MB |

Smaller *and* two orders of magnitude less work to draw. At the default dpi the
rasters land at ~465 dpi once the figure is placed at 180 mm, so nothing
visible is given up. The PNGs are unaffected either way — raster backends
ignore the flag.

This was not always safe. Saving a vector format with a cropping bbox used to
**misplace rasterized artists**: they were drawn through a MixedModeRenderer
built from the figure's original bbox and only then cropped, so the raster
landed at the wrong offset and the wrong scale, with the error growing with
dpi. In Figure 2 that threw all four t-SNE panels into a single blob outside
their axes and left the colourbar fills floating across the page, while the
vector frames, ticks and labels stayed exactly where they belonged. For a
while `save_figure` worked around it by clearing `rasterized` for the PDF pass
— which is precisely what made these two figures slow.

That bug is fixed in the matplotlib pins in `pyproject.toml` (`>=3.10.8,<3.11`), so the
workaround is gone. Verified two ways: an embedded image sits at the same
position relative to its axes whether the save crops 0.0 or 0.5 in (the
residual 0.7 pt is matplotlib's own image padding, present at both), and in the
real Figure 2 the four panel images come out as an exact 2×2 grid with the
three heatmaps and their colourbars aligned above. If the figures ever regress
to a blob of points outside their axes, this is the bug returning and the pin
is the first thing to check.

Figure S12 is the one remaining heavy PDF (~23k draw operations): its 64
hexbin panels are still vector. Adding `rasterized=True` to the `hexbin` call
would give it the same treatment.
