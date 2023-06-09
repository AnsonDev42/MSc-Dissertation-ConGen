import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class Sampler(nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()

    def forward(self, inputs):
        z_mean, z_log_var = inputs
        print(f"shape of z_mean is {z_mean.size()}") # shape of z_mean is torch.Size([128, 2]
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = Variable(torch.randn(batch, dim))
        return z_mean + torch.exp(z_log_var / 2) * epsilon


class DenseLayers(nn.Module):
    def __init__(self, dims, activation_fn=nn.ReLU()):
        super(DenseLayers, self).__init__()
        # Create a list of layers e.g. dims=[2, 3, 4] -> [nn.Linear(2, 3), nn.Linear(3, 4)]
        print(dims)
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims, dims[1:])])
        self.activation_fn = activation_fn

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return x


class Encoder(nn.Module):
    def __init__(self, input_shape, intermediate_dim, latent_dim,filters, kernel_size, bias=True):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        # since the input shape would be 1*160*192*160, where 1 is the input channel number 
        self.z_conv1 = nn.Conv3d(input_shape[0], filters * 2, kernel_size, stride=2, padding=(1,1,1), bias=bias)
        self.z_conv2 = nn.Conv3d(filters * 2, filters * 4, kernel_size, stride=2, padding=(1,1,1), bias=bias)

        # hidden layers
        # self.z_h_layers = DenseLayers([input_shape, intermediate_dim])  # line143 in vae_utils.py
        self.z_h_layers= nn.Linear(10543232, intermediate_dim) 
        self.z_mean_layer = nn.Linear(intermediate_dim, latent_dim)
        self.z_log_var_layer = nn.Linear(intermediate_dim, latent_dim)
        self.sampler = Sampler()

    def forward(self, x):
        x = F.relu(self.z_conv1(x))
        x = F.relu(self.z_conv2(x))
        
        z_shape = list(x.size())
        print(f"shape of x is {z_shape}") # shape of x is [ 128, 40, 48, 40]
        x = x.flatten(start_dim=1)
        z_h = self.z_h_layers(x)
        z_mean = self.z_mean_layer(z_h)
        z_log_var = self.z_log_var_layer(z_h)
        z = self.sampler((z_mean, z_log_var))
        assert z.shape[1:] == (self.latent_dim,)
        return z_mean, z_log_var, z , z_shape


# tg_inputs = torch.randn(1,1, 160, 192, 160)
# encoder= Encoder(input_shape=(1,160,192,160), intermediate_dim=128, latent_dim=2,filters=32, kernel_size=3, bias=True)
# res = encoder(tg_inputs)
# print(f"z_shape {res[3]}") # z_shape [1, 128, 40, 48, 40]

class Decoder(nn.Module):
    """
         input of decoders: torch.Size([1, 4])
    """
    def __init__(self, latent_dim, intermediate_dim,z_shape, output_dim,filters,kernel_size,nlayers, bias=True):
        super(Decoder, self).__init__()
        self.z_shape =  [1, 128, 40, 48, 40]
        self.filters = filters
        self.kernel_size = kernel_size
        self.linear1 = nn.Linear(latent_dim * 2, intermediate_dim, bias=bias)
        self.linear2 = nn.Linear(intermediate_dim,self.z_shape[1]*self.z_shape[2]* self.z_shape[3]*self.z_shape[4], bias=bias) # based on z_shape
        
        layers = []
        layers.append(nn.ConvTranspose3d(in_channels=self.z_shape[1],
                                         out_channels=self.filters,
                                         kernel_size=self.kernel_size,
                                         stride=2,
                                         padding=0,
                                         output_padding=1,
                                         bias=bias))
        for i in range(1, nlayers):
            layers.append(nn.ConvTranspose3d(in_channels=self.filters,
                                             out_channels=self.filters//4,
                                             kernel_size=self.kernel_size,
                                             stride=2,
                                             padding=0,
                                             output_padding=1,
                                             bias=bias))
            layers.append(nn.ReLU(inplace=True))
            self.filters //= 4
        self.convlayers = nn.Sequential(*layers)
        self.output_3dT = nn.ConvTranspose3d(in_channels=self.filters,
                                         out_channels=1,
                                         kernel_size=self.kernel_size,
                                         stride=1,
                                         padding=2,
                                         bias=bias
                                         )
        self.sigmoid = nn.Sigmoid() 
        
  
    def forward(self, x):
        x = x.flatten(start_dim=0)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = x.view(-1, *self.z_shape[1:])
        # print(f"decoder: shape of x is {x.shape}") ([1, 128, 40, 48, 40])
        x = self.convlayers(x)
        x = torch.sigmoid(self.output_3dT(x))
        return x

# tg_inputs = torch.randn(4,1)
# decoder= Decoder(latent_dim=2, intermediate_dim=128,z_shape=[1, 128, 40, 48, 40], output_dim=9830400,filters=32,kernel_size=2,nlayers=2, bias=True)
# res = decoder(tg_inputs)
# print(f"decoder result: shape of x is {res.shape}") # decoder: shape of x is torch.Size([4, 1, 160, 192, 160])

class ContrastiveVAE(nn.Module):
    def __init__(self, input_shape=(1,160,192,160), intermediate_dim=128, latent_dim=2, beta=1, disentangle=False, gamma=1, bias=True, batch_size=64):
        super(ContrastiveVAE, self).__init__()
        # image_size, _, _, channels = input_shape
        kernel_size = 2 # 3 to 2
        filters = 32
        nlayers = 2
        self.z_encoder = Encoder(input_shape, intermediate_dim, latent_dim,filters, kernel_size, bias=bias)
        self.s_encoder = Encoder(input_shape, intermediate_dim, latent_dim,filters, kernel_size, bias=bias)
        
        if input_shape == (1,160,192,160):
            self.z_shape =  [128, 40, 48, 40]
        else:
            raise NotImplementedError( "for now encoder is using hardcoded product of z_shape")
            # res = self.z_encoder(torch.randn(*input_shape))
            # self.z_shape = res[3]
            # print(f"z_shape set to unverified {self.z_shape}")
        self.decoder = Decoder(latent_dim, intermediate_dim,self.z_shape,output_dim=input_shape,filters=filters,kernel_size= kernel_size, nlayers = nlayers, bias=bias)
        self.disentangle = disentangle
        self.beta = beta
        self.gamma = gamma

        if self.disentangle:  # line 209 in vae_utils.py
            self.discriminator = nn.Linear(2 * latent_dim, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, tg_inputs, bg_inputs):
        tg_z_mean, tg_z_log_var, tg_z, shape_z = self.z_encoder(tg_inputs)
        print("tg_z shape",tg_z.shape)
        tg_s_mean, tg_s_log_var, tg_s, shape_s = self.s_encoder(tg_inputs)
        bg_z_mean, bg_z_log_var, bg_z, _ = self.z_encoder(
            bg_inputs)  # s->z:conflict with example code but match with paper
        print(f"input of decoders: {torch.cat((tg_z, tg_s), dim=-1).shape}")
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
    def get_z_shape(self):
        tg_inputs = torch.randn(1,1, 160, 192, 160)
        encoder= Encoder(input_shape=(1,160,192,160), intermediate_dim=128, latent_dim=2,filters=32, kernel_size=3, bias=True)
        res = encoder(tg_inputs)
        print(f"z_shape {res[3]}") # z_shape [1, 128, 40, 48, 40]
        self.z_shape = res[3]
        return self.z_shape

def cvae_loss(original_dim,
              tg_inputs, bg_inputs, tg_outputs, bg_outputs, 
             tg_z_mean, tg_z_log_var, 
             tg_s_mean, tg_s_log_var, 
             bg_z_mean, bg_z_log_var, 
            beta=1.0, disentangle=False, gamma=0.0,tc_loss=None, discriminator_loss=None):

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

if __name__ == '__main__':
    tg_inputs = torch.randn(1,1, 160, 192, 160)
    bg_inputs = torch.randn(1,1, 160, 192, 160)
    cvae = ContrastiveVAE()
    tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var = cvae(tg_inputs, bg_inputs)
    print(f"tg_outputs shape {tg_outputs.shape}")
    print(f"bg_outputs shape {bg_outputs.shape}")
    print(f"tg_z_mean shape {tg_z_mean.shape}")
    print(f"tg_z_log_var shape {tg_z_log_var.shape}")
    print(f"tg_s_mean shape {tg_s_mean.shape}")
    print(f"tg_s_log_var shape {tg_s_log_var.shape}")
    print(f"bg_z_mean shape {bg_z_mean.shape}")
