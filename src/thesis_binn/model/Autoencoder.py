import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.name = f"{encoder.name}:{decoder.name}"
        if hasattr(encoder, "soft_links"):
            if encoder.soft_links:
                self.name += " (SL)"
        elif hasattr(decoder, "soft_links"):
            if decoder.soft_links:
                self.name += " (SL)"
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

    def mask_weights(self):
        self.encoder.mask_weights()
        self.decoder.mask_weights()

    def masks_to(self, device):
        self.encoder.masks_to(device)
        self.decoder.masks_to(device)
