"""Supplementary Figure S6: mean GO-term activations per cancer type, decoder.

The decoder counterpart of Supplementary Figure S5: identical figure, identical
code, run over the decoder GO-term nodes (layers 5-8) instead of the encoder
ones. Rows are the 32 TCGA cancer types, columns 20 GO terms, and each panel is
one heatmap per model instance (seeds 2, 3, 4):

  a  signed mean activation per (cancer type, GO term)
  b  the absolute value of the numbers in panel a

As in S5, panel a asks whether the sign of the mean activation agrees across
instances and panel b whether the magnitudes do.

The implementation lives in figS5.py; this script only fixes the module and the
output name, so it plots the same 20 GO terms as preprint v3 (figS5.GO_TERMS).

The decoder is why the columns are centred and scaled per GO term by default:
one of the 20 terms (negative regulation of blood coagulation) reaches |mean|
170 while the next largest stays under 16 -- a 1,957x spread, against 37x in
the encoder -- so on one raw scale that node sets the limit and every other
column is pale. See figS5's "Colour scale"; --no-normalize returns to raw means.

The activations are the corrected decoder files. --preprint reads the
preprint v3 figure's, which are labelled one layer off: 11 of its 20 columns
held reconstructed genes, 3 proxies and 6 other GO terms, which is where its
|mean| 224 term and its constant column (fatty acid metabolic process) came
from. See prepare/README.md, *The untrained control*.

Inputs (relative to --data-dir)
------------------------------
    go_term_activations_corrected/AE_2.0.<seed>_decoder_activations.csv.gz
        (--preprint: go_term_activations/)
    TCGA_complete_bp_top1k.csv.gz
    hard_links.csv

Usage
-----
    python figS6.py [--data-dir figures/data] [--out-dir figures/out]
                    [--seeds 2 3 4] [--vmax V] [--preprint]
"""

from figS5 import main

if __name__ == "__main__":
    main(module="decoder", out_name="figS6")
