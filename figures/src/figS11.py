"""Supplementary Figure S11: soft-link recovery trajectory, decoder module.

The decoder counterpart of Supplementary Figure S10: identical figure, identical
code, run over decoder layers 5-8 instead of encoder layers 0-3. Per layer, the
top row shows the mean log10 median |w| per GO hard-link-degree bin for soft
links on removed GO edges vs. all other soft links (half-violins behind,
Stouffer-combined Mann-Whitney stars); the bottom row shows how many removed
edges fall in each bin.

The implementation lives in figS10.py; this script only fixes the side and the
output name.

Inputs (relative to --data-dir)
------------------------------
    soft_link_weights/AE_2.1.3_decoder_soft_links.csv.gz
        unperturbed baseline, seed 3, used only for the GO hard-link degree
    AE_9.1_decoder_10perc_removed.csv.gz
        the GO edges that were removed (data-dir root, not soft_link_weights/)
    soft_link_weights/AE_9.1.<seed>_decoder_soft_links.csv.gz  for seeds 3, 4, 5, 6

Usage
-----
    python figS11.py [--data-dir figures/data] [--out-dir figures/out]
"""

from figS10 import main

if __name__ == "__main__":
    main(side="decoder", out_name="figS11")
