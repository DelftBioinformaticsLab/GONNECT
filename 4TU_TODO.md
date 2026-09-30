# Pending updates to the 4TU.ResearchData deposit

The figure inputs in `figures/data/` are deposited at 4TU.ResearchData
([10.4121/0d78788b-6bd7-4941-a942-245309107b6d](https://doi.org/10.4121/0d78788b-6bd7-4941-a942-245309107b6d)).
This file lists what the next version of that deposit has to change. Add an item
whenever a change touches what `figures/data/` holds, and tick it off once the
new version is published.

## Corrected decoder activations (revision, reviewer comment 1)

The deposited decoder activations are labelled one layer off: each column holds
the node at the same position one layer closer to the genes. See
`figures/src/prepare/README.md`, *The untrained control*. The corrected
replacements are in `figures/out/prepare/decoder_reextracted/`. They were
re-extracted from the cluster checkpoints, each of which first reproduced its
deposited file to ~1e-15.

- [ ] Replace `go_term_activations/AE_2.{0,1}.{2..6}_decoder_activations.csv.gz`
      (10 files) with those in `decoder_reextracted/go_term_activations/`.
      The encoder files there are the deposited ones, hard-linked, and stay as
      they are. The replacements were written at gzip level 1, so recompress
      them first if size matters.
- [ ] Make `--pool seeds` the default in `fig4.py`. Every red line is now the
      mean of the five per-seed values, and fig4 computes that null itself.
      No figure then reads `perm_nulls_gonnect/` or `perm_nulls_baselines/`:
      drop both, or keep them as the published figure's inputs. Either way,
      they need no rebuilt replacement. That includes the
      `AE_2.{0,1}_decoder_mean_abs/` nulls, which were built from the
      mislabelled files and are read by nothing.
      (`decoder_reextracted/perm_nulls_gonnect{,_meanauc}/` are the rebuilt
      nulls for the two earlier definitions; they are needed only if one of
      those is chosen instead.)
- [ ] Replace `activation_preservation_per_node.csv` with
      `decoder_reextracted/activation_preservation/activation_preservation_per_node.csv`.
- [ ] Decide whether to deposit the decoder-only checkpoints,
      `out/trained_models/AE_2.{0,1}/AE_2.{0,1}.{2..6}_decoder_model.pt`
      (10 files, ~540 MB). `extract_decoder_activations.py` needs them to
      rebuild the corrected files; `model_checkpoints/` holds only the `_both`
      and `_none` models.
- [ ] If Figure 4 keeps the untrained reference (`fig4.py --untrained`),
      deposit its three inputs:
      - `figures/out/prepare/untrained_control/fig4_untrained_corrected.tsv`
        (GONNECT, panels a and b);
      - `figures/out/prepare/untrained_dpr/fig4_untrained_dpr.tsv`
        (GONNECT-DPR, panel c);
      - `figures/out/prepare/untrained_baselines/fig4_untrained_baselines.tsv`
        (OntoVAE and VEGA, panels d and e).

- [ ] Deposit, or track in git, the degree-preserving masks the GONNECT-DPR
      runs were trained with:
      `out/masks/{encoder,decoder}/(1, 30, 50)/TCGA_complete_bp_top1k_dense_edge_masks_random{8..12}.pt`
      (10 files, ~34 MB). `untrained_dpr.py` builds on them, and they exist
      only locally: `.gitignore` ignores `out/`, and only the true masks are
      excepted. Each matches its AE_2.2 checkpoints' weight pattern exactly.

      Then have `fig4.py` read them from `data/` by default. The untrained
      baseline exports behind the second file
      (`out/baselines/untrained/`, ~2.2 GB) can be regenerated in their
      environments (see `baselines/README.md`), so they need not be deposited.
- [ ] Locally, delete `figures/data/cache/auc_4row/AE_2.*_decoder_*` after
      replacing the files. `fig4.py` does not check a cache entry against its
      source, so a stale one would be reused silently. The cache itself is not
      deposited.

Once the corrected files are in `figures/data/`, clean up what only served the
mislabelled ones. None of it can go before then.

- [ ] Rewrite the text that describes the old decoder data. That is the figS6
      docstring (the |mean| 224 and constant columns), the `GO:0006631` example
      in `figS5.normalize_terms`, and `figures/README.md`, *Figures S5 and S6*
      (the 55,005x spread and the claim that magnitudes are reproducible).
- [ ] Retire the shifted-layout code:
      - `relabel_decoder_activations.py`;
      - the relabelling and the `as shipped` states in `untrained_control.py`;
      - `fig4_decoder_comparison.py`, whose `as_published` side needs the old
        files.

      `extract_decoder_activations.py` checks each checkpoint against the
      deposited files under the old labelling, so switch it to the corrected
      labelling.

## Publishing the new version

- [ ] Regenerate the figures that read replaced inputs (4, 5 and S6), from
      the updated `figures/data/`, and commit their PDFs.
- [ ] Repackage `figures/data/` for upload. `.gitignore` reserves
      `figures/source_data_for_figures.zip` for that copy.
- [ ] If the DOI changes, update it in `README.md` (twice), `figures/README.md`,
      `CITATION.cff` and `pyproject.toml` (`[project.urls] Dataset`).
