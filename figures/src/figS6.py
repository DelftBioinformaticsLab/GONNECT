"""Supplementary Figure S6: mean GO-term activations per cancer type, decoder.

The decoder counterpart of Supplementary Figure S5: identical figure, identical
code, run over the decoder GO-term nodes (layers 5-8) instead of the encoder
ones. Rows are the 32 TCGA cancer types, columns 20 GO terms, and each panel is
one heatmap per model instance (seeds 2, 3, 4):

  a  signed mean activation per (cancer type, GO term)
  b  the absolute value of the numbers in panel a

As in S5, panel a shows that the sign of the mean activation is arbitrary
across instances while panel b shows that the magnitudes are reproducible.

The implementation lives in figS5.py; this script only fixes the module and the
output name, so it plots the same 20 published GO terms (figS5.GO_TERMS).

The decoder is why the columns are centred and scaled per GO term by default:
one of the 20 terms (positive regulation of neuron projection regeneration)
reaches |mean| 224 while 16 of the 20 stay under 2.7 -- a 55,005x spread,
against 37x in the encoder -- so on one raw scale that node sets the limit and
every other column is pale. A second of the 20 (fatty acid metabolic process)
is constant across all 32 cancer types and is drawn blank rather than scaled
up. See figS5's "Colour scale"; --no-normalize returns to raw means.

Inputs (relative to --data-dir)
------------------------------
    go_term_activations/AE_2.0.<seed>_decoder_activations.csv.gz
    TCGA_complete_bp_top1k.csv.gz
    hard_links.csv

Usage
-----
    python figS6.py [--data-dir figures/data] [--out-dir figures/out]
                    [--seeds 2 3 4] [--vmax V]
"""

from figS5 import main

if __name__ == "__main__":
    main(module="decoder", out_name="figS6")
