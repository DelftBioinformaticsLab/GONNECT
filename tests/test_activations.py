"""activations_per_term must label every column with the node whose value it holds.

A linear layer's output belongs to the GO layer it feeds into. The decoder's
first GO layer is not the output of any of its layers: it is the decoder's
input, the latent representation. An earlier decoder branch labelled each
layer's outputs with the terms of its input layer, so every decoder column held
the value of a different node.
"""
from unittest import SkipTest, TestCase

import numpy as np
import pandas as pd
import torch
from goatools.obo_parser import GOTerm

from gonnect.data_processing.GeneTerm import GeneTerm
from gonnect.data_processing.ProxyTerm import ProxyTerm
from gonnect.model.Autoencoder import Autoencoder
from gonnect.model.Decoder import DenseBIDecoder
from gonnect.model.Encoder import DenseBIEncoder

try:
    # gonnect.analysis is left out of the distributed package (see pyproject.toml), so a
    # run from an unpacked sdist skips here rather than erroring during collection.
    from gonnect.analysis.activations import activations_per_term
except ImportError as error:
    raise SkipTest(f"gonnect.analysis is not importable: {error}")


def go_term(item_id):
    term = GOTerm()
    term.item_id = item_id
    return term


class ActivationsPerTermTest(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # By ascending depth: the bottleneck, a GO layer that also holds a proxy, then the genes
        self.go_layers = [
            [go_term("GO:0000001")],
            [go_term("GO:0000002"), ProxyTerm("Proxy:1_g0", set(), set()), go_term("GO:0000003")],
            [GeneTerm(gene, set()) for gene in ("g0", "g1", "g2")],
        ]
        proxy = torch.tensor([[False], [True], [False]])
        encoder = DenseBIEncoder(self.go_layers, torch.nn.ReLU, torch.float64, masks=(
            [torch.ones(3, 3, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)],
            [proxy, torch.zeros(1, 1, dtype=torch.bool)]))
        decoder = DenseBIDecoder(self.go_layers, torch.nn.ReLU, torch.float64, masks=(
            [torch.ones(3, 1, dtype=torch.bool), torch.ones(3, 3, dtype=torch.bool)],
            [proxy, torch.zeros(3, 1, dtype=torch.bool)]))
        self.model = Autoencoder(encoder, decoder)
        self.data = pd.DataFrame(np.random.default_rng(0).normal(size=(16, 3)), columns=["g0", "g1", "g2"])

        with torch.no_grad():
            x = torch.tensor(self.data.values)
            self.z = self.model.encoder(x).numpy()
            self.encoder_hidden = self.model.encoder.net_layers[0](x).numpy()
            self.decoder_hidden = self.model.decoder.net_layers[0](torch.tensor(self.z)).numpy()

    def test_encoder_columns_hold_their_own_nodes(self):
        activations = activations_per_term(self.model, self.go_layers, self.data, "encoder")
        self.assertEqual(list(activations.columns), ["GO:0000002", "GO:0000003", "GO:0000001"])
        np.testing.assert_allclose(activations["GO:0000002"], self.encoder_hidden[:, 0])
        np.testing.assert_allclose(activations["GO:0000003"], self.encoder_hidden[:, 2])
        np.testing.assert_allclose(activations["GO:0000001"], self.z[:, 0])

    def test_decoder_columns_hold_their_own_nodes(self):
        activations = activations_per_term(self.model, self.go_layers, self.data, "decoder")
        self.assertEqual(list(activations.columns), ["GO:0000001", "GO:0000002", "GO:0000003"])
        # The bottleneck is the decoder's input
        np.testing.assert_allclose(activations["GO:0000001"], self.z[:, 0])
        np.testing.assert_allclose(activations["GO:0000002"], self.decoder_hidden[:, 0])
        np.testing.assert_allclose(activations["GO:0000003"], self.decoder_hidden[:, 2])
