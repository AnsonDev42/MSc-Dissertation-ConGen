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
        self.z_h_layers = DenseLayers(input_dim + intermediate_dim)  # line143 in vae_utils.py
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
    def __init__(self, input_dim, intermediate_dim, latent_dim, beta=1, disentangle=False, gamma=0):
        super(ContrastiveVAE, self).__init__()

        if isinstance(intermediate_dim, int):
            intermediate_dim = [intermediate_dim]
        self.z_encoder = Encoder(input_dim, intermediate_dim, latent_dim)
        self.s_encoder = Encoder(input_dim, intermediate_dim, latent_dim)
        self.decoder = Decoder(2 * latent_dim, intermediate_dim, input_dim)
        self.disentangle = disentangle
        self.beta = beta
        self.gamma = gamma

        if self.disentangle:  # line 209 in vae_utils.py
            self.discriminator = nn.Linear(2 * latent_dim, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, tg_inputs, bg_inputs):
        tg_z_mean, tg_z_log_var, tg_z = self.z_encoder(tg_inputs)
        tg_s_mean, tg_s_log_var, tg_s = self.s_encoder(tg_inputs)
        bg_z_mean, bg_z_log_var, bg_z = self.z_encoder(
            bg_inputs)  # s->z:conflict with example code but match with paper

        tg_outputs = self.decoder(torch.cat((tg_z, tg_s), dim=-1))  # line 196 in vae_utils.py
        bg_outputs = self.decoder(torch.cat((torch.zeros_like(tg_z), bg_z), dim=-1))
        fg_outputs = self.decoder(torch.cat((tg_z, torch.zeros_like(tg_z)), dim=-1))

        if self.disentangle:
            batch_size = tg_inputs.size(0)
            z1 = tg_z[:batch_size // 2, :]
            z2 = tg_z[batch_size // 2:, :]
            s1 = tg_s[:batch_size // 2, :]
            s2 = tg_s[batch_size // 2:, :]
            q_bar = torch.cat(
                [torch.cat([s1, z2], dim=1),
                 torch.cat([s2, z1], dim=1)],
                dim=0)
            q = torch.cat(  # line 219 in vae_utils.py
                [torch.cat([s1, z1], dim=1),
                 torch.cat([s2, z2], dim=1)],
                dim=0)

            q_bar_score = self.sigmoid(self.discriminator(q_bar))
            q_score = self.sigmoid(self.discriminator(q))
            tc_loss = torch.log(q_score / (1 - q_score))
            discriminator_loss = -torch.log(q_score) - torch.log(1 - q_bar_score)
            # the reconstruction_loss is not complete, need to add when training
            # REMEMBER to replace bg_s to bg_z from the example code
            return (
                tg_outputs, bg_outputs,
                tg_z_mean, tg_z_log_var,
                tg_s_mean, tg_s_log_var,
                bg_z_mean, bg_z_log_var,
                tc_loss, discriminator_loss)
        else:
            return tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var


def cvae_loss(original_dim,
              tg_inputs, bg_inputs, tg_outputs, bg_outputs, 
             tg_z_mean, tg_z_log_var, 
             tg_s_mean, tg_s_log_var, 
             bg_z_mean, bg_z_log_var, 
            beta=1.0, disentangle=False, gamma=0.0):

    # Reconstruction Loss
    reconstruction_loss = nn.MSELoss(reduction="none")
    tg_reconstruction_loss = reconstruction_loss(tg_inputs, tg_outputs).view(tg_inputs.size(0), -1).sum(dim=1)
    bg_reconstruction_loss = reconstruction_loss(bg_inputs, bg_outputs).view(bg_inputs.size(0), -1).sum(dim=1)
    reconstruction_loss = (tg_reconstruction_loss + bg_reconstruction_loss).mean() * original_dim

    # KL Loss
    kl_loss = 1 + tg_z_log_var - tg_z_mean.pow(2) - tg_z_log_var.exp()
    kl_loss += 1 + tg_s_log_var - tg_s_mean.pow(2) - tg_s_log_var.exp()
    kl_loss += 1 + bg_z_log_var - bg_z_mean.pow(2) - bg_z_log_var.exp()
    kl_loss = kl_loss.sum(dim=-1) * -0.5
    if disentangle:
        cvae_loss = reconstruction_loss.mean() + beta * kl_loss.mean() + gamma * tc_loss + discriminator_loss
    else:
        cvae_loss = reconstruction_loss.mean() + beta * kl_loss.mean()

    return cvae_loss