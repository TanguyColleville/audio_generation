import torch
import torch.nn as nn


class AE(nn.Module):

    def __init__(self, input_size: int, latent_dim: int):
        super(AE, self).__init__()

        self.layer_1 = nn.Linear(input_size, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, latent_dim)
        self.relu = nn.ReLU()
        self.layer_4 = nn.Linear(latent_dim, 128)
        self.layer_5 = nn.Linear(128, 256)
        self.layer_6 = nn.Linear(256, input_size)

        self.encoder = nn.Sequential(self.layer_1, self.relu,
                                     self.layer_2, self.relu,
                                     self.layer_3
                                     )
        self.decoder = nn.Sequential(self.layer_4, self.relu,
                                     self.layer_5, self.relu,
                                     self.layer_6
                                     )
    # TODO add a method create encoder, create decoder and change attribut

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        latent = self.encoder(x)
        return self.decoder(latent)
