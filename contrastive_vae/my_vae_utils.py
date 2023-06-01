import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


class Sampler(nn.Module):
    def forward(self, inputs):
        z_mean, z_log_var = inputs
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = Variable(torch.randn(batch, dim))
        return z_mean + torch.exp(z_log_var / 2) * epsilon


class DenseLayers(nn.Module):
    def __init__(self, dims, activation_fn=nn.ReLU()):
        super(DenseLayers, self).__init__()
        # Create a list of layers e.g. dims=[2, 3, 4] -> [nn.Linear(2, 3), nn.Linear(3, 4)]
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims, dims[1:])])
        self.activation_fn = activation_fn

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim):
        super(Encoder, self).__init__()
        # hidden layers
        self.z_h_layers = DenseLayers([input_dim] + intermediate_dim) #line143 in vae_utils.py
        self.z_mean_layer = nn.Linear(intermediate_dim[-1], latent_dim)
        self.z_log_var_layer = nn.Linear(intermediate_dim[-1], latent_dim)
        self.sampler = Sampler()

    def forward(self, x):
        z_h = self.z_h_layers(x)
        z_mean = self.z_mean_layer(z_h)
        z_log_var = self.z_log_var_layer(z_h)
        z = self.sampler((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    def __init__(self, latent_dim, intermediate_dim, output_dim):
        super(Decoder, self).__init__()
        self.layers = DenseLayers([latent_dim] + intermediate_dim + [output_dim])

    def forward(self, x):
        return self.layers(x)


class ContrastiveVAE(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim):
        super(ContrastiveVAE, self).__init__()
        self.z_encoder = Encoder(input_dim, intermediate_dim, latent_dim)
        self.s_encoder = Encoder(input_dim, intermediate_dim, latent_dim)
        self.decoder = Decoder(2 * latent_dim, intermediate_dim, input_dim)

    def forward(self, tg_inputs, bg_inputs):
        tg_z_mean, tg_z_log_var, tg_z = self.z_encoder(tg_inputs)
        tg_s_mean, tg_s_log_var, tg_s = self.s_encoder(tg_inputs)
        bg_s_mean, bg_s_log_var, bg_s = self.s_encoder(bg_inputs)

        tg_outputs = self.decoder(torch.cat((tg_z, tg_s), dim=-1))
        bg_outputs = self.decoder(torch.cat((torch.zeros_like(tg_z), bg_s), dim=-1))

        return tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_s_mean, bg_s_log_var


