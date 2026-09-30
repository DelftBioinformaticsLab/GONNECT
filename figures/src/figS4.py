"""Supplementary Figure S4: effect of GO graph randomization on clustering quality.

The ARI / NMI counterpart of Figure 3 — same models, same grouping by
randomization level (true / degree-preserving / fully random), same statistics,
only the two clustering metrics instead of MSE and SS:
  a  ARI [up]   b  NMI [up]

All data loading, statistics and plotting live in fig3.py; this script only
picks the metrics and the output name, so the two figures cannot drift apart.
Inputs are exactly the files fig3.py reads: the held-out metrics under
metrics/test_split/, and the MSE of metrics/metric_data_TCGA_1000_30_new.xlsx
(which this figure does not show).

Run: python figS4.py   ->   out/figS4.png, out/figS4.pdf
"""

from fig3 import TITLE_SUPP, main

if __name__ == "__main__":
    main(metrics=("ARI", "NMI"), out_name="figS4", title=TITLE_SUPP)
