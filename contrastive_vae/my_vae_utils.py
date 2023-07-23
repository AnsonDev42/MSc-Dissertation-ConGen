import datetime
import os
import pickle

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import sys
from sklearn.metrics import silhouette_score

sys.path.append('/afs/inf.ed.ac.uk/user/s23/s2341683/pycharm_remote_tmp/ConGeLe')
from dataloader import DataStoreDataset, filter_depressed, filter_healthy, custom_collate_fn


class Sampler(nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()

    def forward(self, inputs):
        z_mean, z_log_var = inputs
        # print(f"shape of z_mean is {z_mean.size()}")  # shape of z_mean is torch.Size([128, 2]
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = Variable(torch.randn(batch, dim))
        if z_mean.is_cuda:
            epsilon = epsilon.to(z_mean.device)
        return z_mean + torch.exp(z_log_var / 2) * epsilon


class DenseLayers(nn.Module):
    def __init__(self, dims, activation_fn=nn.ReLU()):
        super(DenseLayers, self).__init__()
        # Create a list of layers e.g. dims=[2, 3, 4] -> [nn.Linear(2, 3), nn.Linear(3, 4)]
        # print(dims)
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims, dims[1:])])
        self.activation_fn = activation_fn

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return x


class Encoder(nn.Module):
    def __init__(self, input_shape, intermediate_dim, latent_dim, filters, kernel_size, bias=True):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        # print(f"input_shape is {input_shape}")
        # since the input shape would be 1*160*192*160, where 1 is the input channel number
        # stride =2
        self.z_conv1 = nn.Conv3d(input_shape[0], filters * 2, kernel_size=3, stride=2, padding=(1, 1, 1), bias=bias)
        self.z_conv2 = nn.Conv3d(filters * 2, filters * 4, kernel_size=3, stride=2, padding=(1, 1, 1), bias=bias)
        # hidden layers
        # self.z_h_layers = DenseLayers([input_shape, intermediate_dim])  # line143 in vae_utils.py
        self.z_h_layers = nn.Linear(524288, intermediate_dim)
        self.z_mean_layer = nn.Linear(intermediate_dim, latent_dim)
        self.z_log_var_layer = nn.Linear(intermediate_dim, latent_dim)
        self.sampler = Sampler()
        # print each layers parameters
        # for name, param in self.named_parameters():
        #     print(f"Encoder layer name: {name}, size: {param.size()}")
        # self.bn1 = nn.BatchNorm3d(filters * 2)
        # self.bn2 = nn.BatchNorm3d(filters * 4)
        self.z_h_layers_bn = nn.BatchNorm1d(intermediate_dim)
        self.z_mean_layer_bn = nn.BatchNorm1d(latent_dim)
        self.z_log_var_layer_bn = nn.BatchNorm1d(latent_dim)
        # self.bn3 = nn.BatchNorm3d(filters * 8)

    def forward(self, x):
        x = self.z_conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.z_conv2(x)
        # x = self.bn2(x)
        # x = F.relu(self.z_conv2(x))

        z_shape = list(x.size())
        # print(f"shape of x is {z_shape}")  # shape of x is [ 128, 40, 48, 40]
        z_h = x.flatten(start_dim=1)
        # print(f"shape of x is {z_h.size()}")  # shape of x is 314,572,800
        z_h = self.z_h_layers(z_h)
        z_h = self.z_h_layers_bn(z_h)
        z_mean = self.z_mean_layer(z_h)
        z_mean = self.z_mean_layer_bn(z_mean)
        z_log_var = self.z_log_var_layer(z_h)
        z_log_var = self.z_log_var_layer_bn(z_log_var)
        z = self.sampler((z_mean, z_log_var))
        assert z.shape[1:] == (self.latent_dim,)
        return z_mean, z_log_var, z, z_shape


#
# tg_inputs = torch.randn(1, 1, 160, 192, 160)
# encoder = Encoder(input_shape=(1, 160, 192, 160), intermediate_dim=128, latent_dim=2, filters=32, kernel_size=3,
#                   bias=True)
# res = encoder(tg_inputs)
# print(f"!!!!z_shape {res[3]}")  # z_shape [1, 128, 40, 48, 40]
#

class Decoder(nn.Module):
    """
         input of decoders: torch.Size([1, 4])
    """

    def __init__(self, latent_dim, intermediate_dim, z_shape, output_dim, filters, kernel_size, nlayers, bias=True):
        super(Decoder, self).__init__()
        self.z_shape = [1, 64, 16, 16, 16]  # [1, 128, 40, 48, 40] for 160*192*160
        self.filters = filters
        self.kernel_size = kernel_size
        self.linear1 = nn.Linear(latent_dim * 2, intermediate_dim, bias=bias)
        self.linear2 = nn.Linear(intermediate_dim,
                                 self.z_shape[1] * self.z_shape[2] * self.z_shape[3] * self.z_shape[4],
                                 bias=bias)  # based on z_shape

        layers = []
        layers.append(nn.ConvTranspose3d(in_channels=self.z_shape[1],
                                         out_channels=self.filters * 2,
                                         kernel_size=self.kernel_size,
                                         stride=2,
                                         padding=1,
                                         output_padding=1,
                                         bias=bias))
        for i in range(1, nlayers):
            layers.append(nn.ConvTranspose3d(in_channels=self.filters * 2,
                                             out_channels=self.filters * 1,
                                             kernel_size=self.kernel_size,
                                             stride=2,
                                             padding=1,
                                             output_padding=1,
                                             bias=bias))
            layers.append(nn.ReLU(inplace=True))
            # self.filters //= 4s
        self.convlayers = nn.Sequential(*layers)
        self.output_3dT = nn.ConvTranspose3d(in_channels=self.filters * 1,
                                             out_channels=1,
                                             kernel_size=self.kernel_size,
                                             stride=1,
                                             padding=1,
                                             output_padding=0,
                                             bias=bias
                                             )
        self.sigmoid = nn.Sigmoid()
        # self.bn1 = nn.BatchNorm1d(self.filters)
        # self.bn2 = nn.BatchNorm3d(self.filters // 4)
        # self.bn3 = nn.BatchNorm3d(self.filters)
        # self.sigmoid = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # x = x.flatten(start_dim=0)
        x = self.linear1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.linear2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = x.view(-1, *self.z_shape[1:])
        # print(f"decoder: shape of x is {x.shape}")  # ([1, 128, 40, 48, 40])
        x = self.convlayers(x)
        # x = self.bn3(x)
        x = self.sigmoid(self.output_3dT(x))
        return x


# tg_inputs = torch.randn(4,1)
# decoder= Decoder(latent_dim=2, intermediate_dim=128,z_shape=[1, 128, 40, 48, 40], output_dim=9830400,filters=32,kernel_size=2,nlayers=2, bias=True)
# res = decoder(tg_inputs)
# print(f"decoder result: shape of x is {res.shape}") # decoder: shape of x is torch.Size([4, 1, 160, 192, 160])

class ContrastiveVAE(nn.Module):
    def __init__(self, input_shape=(1, 160, 192, 160), intermediate_dim=128, latent_dim=16, beta=1, disentangle=False,
                 gamma=1, bias=True, batch_size=64, device=None):
        super(ContrastiveVAE, self).__init__()
        # image_size, _, _, channels = input_shape
        kernel_size = 3  # 3 to 2
        filters = 32  # 32
        nlayers = 2
        self.z_encoder = Encoder(input_shape, intermediate_dim, latent_dim, filters, kernel_size, bias=bias)
        self.s_encoder = Encoder(input_shape, intermediate_dim, latent_dim, filters, kernel_size, bias=bias)

        if input_shape == (1, 160, 192, 160):
            self.z_shape = [128, 40, 48, 40]
        elif input_shape == (1, 64, 64, 64):
            self.z_shape = [64, 16, 16, 16]
        else:
            res = self.z_encoder(torch.randn(*input_shape))
            self.z_shape = res[3]
            print(f"z_shape set to unverified {self.z_shape}")
            raise NotImplementedError("for now encoder is using hardcoded product of z_shape")
        self.decoder = Decoder(latent_dim, intermediate_dim, self.z_shape, output_dim=input_shape, filters=filters,
                               kernel_size=kernel_size, nlayers=nlayers, bias=bias)
        self.disentangle = disentangle
        self.beta = beta
        self.gamma = gamma

        if self.disentangle:  # line 209 in vae_utils.py
            self.discriminator = nn.Linear(2 * latent_dim, 1)
            self.sigmoid = nn.Sigmoid()
            # self.sigmoid = nn.Tanh()

    def forward(self, tg_inputs, bg_inputs):
        tg_z_mean, tg_z_log_var, tg_z, shape_z = self.z_encoder(tg_inputs)
        # print("tg_z shape", tg_z.shape)
        tg_s_mean, tg_s_log_var, tg_s, shape_s = self.s_encoder(tg_inputs)
        bg_z_mean, bg_z_log_var, bg_z, _ = self.z_encoder(
            bg_inputs)  # s->z:conflict with example code but match with paper
        # print(f"input of decoders: {torch.cat((tg_z, tg_s), dim=-1).shape}")
        tg_outputs = self.decoder(torch.cat((tg_z, tg_s), dim=-1))  # line 196 in vae_utils.py
        bg_outputs = self.decoder(torch.cat((torch.zeros_like(tg_z), bg_z), dim=-1))
        # fg_outputs = self.decoder(torch.cat((tg_z, torch.zeros_like(tg_z)), dim=-1))

        # if self.disentangle:
        batch_size = tg_inputs.size(0)
        # print(f"tg_s.size(), bg_z.size() {tg_s.size(), bg_z.size()}")
        z1 = tg_z[:batch_size // 2, :]
        z2 = tg_z[batch_size // 2:, :]
        s1 = tg_s[:batch_size // 2, :]
        s2 = tg_s[batch_size // 2:, :]
        # print(f"s1.size(), z2.size() {s1.size(), z2.size()}")
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
        return tg_outputs, bg_outputs, \
            tg_z_mean, tg_z_log_var, \
            tg_s_mean, tg_s_log_var, \
            bg_z_mean, bg_z_log_var, \
            tc_loss, discriminator_loss, tg_z, bg_z
        # else:
        #     return tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var

    # def get_z_shape(self):
    #     tg_inputs = torch.randn(1, 1, 160, 192, 160)
    #     encoder = Encoder(input_shape=(1, 160, 192, 160), intermediate_dim=128, latent_dim=2, filters=32, kernel_size=3,
    #                       bias=True)
    #     res = encoder(tg_inputs)
    #     print(f"z_shape {res[3]}")  # z_shape [1, 128, 40, 48, 40]
    #     self.z_shape = res[3]
    #     return self.z_shape


def cvae_loss(original_dim,
              tg_inputs, bg_inputs, tg_outputs, bg_outputs,
              tg_z_mean, tg_z_log_var,
              tg_s_mean, tg_s_log_var,
              bg_z_mean, bg_z_log_var,
              beta=1.0, disentangle=True, gamma=1.0, tc_loss=None, discriminator_loss=None):
    # Reconstruction Loss
    reconstruction_loss = nn.MSELoss(reduction="none")
    if tg_outputs.is_cuda:
        # print('tg_outputs is cuda, move outputs to device')
        tg_inputs = tg_inputs.to(tg_outputs.device)
        bg_inputs = bg_inputs.to(bg_outputs.device)
    tg_reconstruction_loss = reconstruction_loss(tg_outputs, tg_inputs).view(tg_inputs.size(0), -1).sum(dim=1)
    bg_reconstruction_loss = reconstruction_loss(bg_outputs, bg_inputs).view(bg_inputs.size(0), -1).sum(dim=1)
    reconstruction_loss = 1 * (tg_reconstruction_loss + bg_reconstruction_loss)
    reconstruction_loss /= (original_dim[1] * original_dim[2] * original_dim[3])
    # * (
    # original_dim[1] * original_dim[2] * original_dim[3])  # 0.5*
    # \
    #                       / (original_dim[1] * original_dim[2] * original_dim[3])

    # KL Loss
    kl_loss = (
                      1 + tg_z_log_var - tg_z_mean.pow(2) - tg_z_log_var.exp() \
                      + 1 + tg_s_log_var - tg_s_mean.pow(2) - tg_s_log_var.exp() \
                      + 1 + bg_z_log_var - bg_z_mean.pow(2) - bg_z_log_var.exp()
              ).sum(dim=-1) * -0.5  # * -0.5
    # kl_loss = kl_loss.sum(dim=-1) * -0.5
    if disentangle:
        cvae_loss = reconstruction_loss.mean() + beta * kl_loss.mean() + gamma * tc_loss.mean() + discriminator_loss.mean()  # discriminator_loss
        # print each loss
        print(
            f'cvae_loss {cvae_loss} | reconstruction_loss {reconstruction_loss.mean()} | beta* kl_loss {beta * kl_loss.mean()} | gamma * tc_loss {gamma * tc_loss.mean()} | discriminator_loss {discriminator_loss.mean()}')
    else:
        cvae_loss = reconstruction_loss.mean() + beta * kl_loss.mean()
        # print each loss
        f'cvae_loss {cvae_loss} | reconstruction_loss {reconstruction_loss.mean()} | beta* kl_loss {beta * kl_loss.mean()}'
    return cvae_loss


def plot_latent_space(tg_z_mean_total, bg_z_mean_total, plot=True, name='test_img', epoch=0):
    z_label = np.concatenate((np.ones(tg_z_mean_total.shape[0]), np.zeros(bg_z_mean_total.shape[0])),
                             axis=0)  # create labels
    z = np.concatenate((tg_z_mean_total, bg_z_mean_total), axis=0)
    # print(f"DEBUG: tg_z.shape: {tg_z_mean_total.shape}")
    # print(f"DEBUG: bg_z.shape: {bg_z_mean_total.shape}")
    # print(f"DEBUG: length supposed point: {tg_z_mean_total.shape[0]}")
    # print(f"DEBUG: z_label.shape: {z_label.shape}")
    # print(f"DEBUG: z.shape: {z.shape}")
    ss = round(silhouette_score(z, z_label), 3)

    if not os.path.exists(f'cvae_results/{name}'):
        os.makedirs(f'cvae_results/{name}')
    if plot:
        plt.figure()
        plt.scatter(tg_z_mean_total[:, 0], tg_z_mean_total[:, 1], c='b', label='tg_z(target)')
        plt.scatter(bg_z_mean_total[:, 0], bg_z_mean_total[:, 1], c='r', label='bg_z(background)')
        plt.legend()  # Displays a legend
        plt.title(f'Epoch:{epoch}-{name}, Silhouette score: {str(ss)}')
        plt.savefig(f'cvae_results/{name}/{epoch}.svg')  # Change the path and filename as needed
        plt.savefig(f'cvae_results/{name}/{epoch}.png')  # Change the path and filename as needed
        plt.close()

        # save tg and bg z_mean to pickle
        with open(f'cvae_results/{name}/{epoch}.pkl', 'wb') as f:
            pickle.dump([tg_z_mean_total, bg_z_mean_total, epoch], f)

        # plt.title(name + ', Silhouette score: ' + str(ss))
    return ss


from sklearn.manifold import TSNE


def plot_32d_latent_space(tg_z_mean_total, bg_z_mean_total, plot=True, name='test_img', epoch=0):
    # Create a t-SNE object
    tsne = TSNE(n_components=2)

    # Concatenate all the latent vectors
    z = np.concatenate((tg_z_mean_total, bg_z_mean_total), axis=0)

    # Transform the concatenated latent vectors into 2D
    z_2d = tsne.fit_transform(z)

    # Split the 2D representations
    tg_z_2d, bg_z_2d = np.split(z_2d, [tg_z_mean_total.shape[0]])

    # Calculate the silhouette score
    z_label = np.concatenate((np.ones(tg_z_mean_total.shape[0]), np.zeros(bg_z_mean_total.shape[0])), axis=0)
    ss = round(silhouette_score(z_2d, z_label), 3)

    if not os.path.exists(f'cvae_results/{name}'):
        os.makedirs(f'cvae_results/{name}')

    if plot:
        plt.figure()
        plt.scatter(tg_z_2d[:, 0], tg_z_2d[:, 1], c='b', label='tg_z(target)')
        plt.scatter(bg_z_2d[:, 0], bg_z_2d[:, 1], c='r', label='bg_z(background)')
        plt.legend()  # Displays a legend
        plt.title(f'Epoch:{epoch}-{name}, Silhouette score: {str(ss)}')
        plt.savefig(f'cvae_results/{name}/{epoch}.svg')  # Change the path and filename as needed
        plt.savefig(f'cvae_results/{name}/{epoch}.png')  # Change the path and filename as needed
        plt.close()

        # Save tg and bg z_mean to pickle
        with open(f'cvae_results/{name}/{epoch}.pkl', 'wb') as f:
            pickle.dump([tg_z_mean_total, bg_z_mean_total, epoch], f)

    return ss


# def plot_latent_space(encoder, x, y, plot=True, name='z_mean'):
#     from sklearn.metrics import silhouette_score
#     z_mean, _, _ = encoder.predict(x, batch_size=128)
#     ss = round(silhouette_score(z_mean, y), 3)
#     if plot:
#         plt.figure()
#         plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y, cmap='Accent')
#         plt.title(name + ', Silhouette score: ' + str(ss))
#     return ss


if __name__ == '__main__':
    # current time in ddmm_hhmm format
    now = datetime.datetime.now()
    time_str = now.strftime("%d%m_%H%M")
    torch.cuda.empty_cache()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    device_ids = [3, 4]
    # print(f'Using device: {device} and potential device_ids: {device_ids}')
    # Define hyperparameters
    learning_rate = 0.001
    epochs = 100

    # HOME = os.environ['HOME']
    # root_dir = f'{HOME}/GenScotDepression/data/ukb/imaging/raw/t1_structural_nifti_20252'
    print(f'Using device: {device}')
    # Define hyperparameters
    learning_rate = 0.001
    input_dim = (1, 64, 64, 64)  # 784
    intermediate_dim = 128  # 256
    latent_dim = 16
    beta = 1
    disentangle = True
    gamma = 0
    model = ContrastiveVAE(input_dim, intermediate_dim, latent_dim, beta, disentangle, gamma)
    # model = nn.DataParallel(model, device_ids=[4, 1])  # Wrap the model with DataParallel
    model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")
    # num of parameters in z_encoder
    z_encoder_params = sum(p.numel() for p in model.z_encoder.parameters())
    print(f"Number of parameters in z_encoder: {z_encoder_params}")
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Number of parameters in decoder: {decoder_params}")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {pytorch_total_params}")
    # model = torch.compile(model)
    tg_inputs = torch.randn(4, 1, 64, 64, 64).to(dtype=torch.float32)
    bg_inputs = torch.randn(4, 1, 64, 64, 64).to(dtype=torch.float32)
    tg_inputs = tg_inputs.to(device)
    bg_inputs = bg_inputs.to(device)
    tg_z_total, bg_z_total, tg_z_mean_total, bg_z_mean_total = [], [], [], []

    tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var, \
        tc_loss, discriminator_loss, tg_z, bg_z = model(tg_inputs,
                                                        bg_inputs)

    print(f"tg_outputs shape {tg_outputs.shape}")
    print(f"bg_outputs shape {bg_outputs.shape}")
    print(f"tg_z_mean shape {tg_z_mean.shape}")
    # print(f"tg_z_log_var shape {tg_z_log_var.shape}")
    print(f"tg_s_mean shape {tg_s_mean.shape}")
    # print(f"tg_s_log_var shape {tg_s_log_var.shape}")
    print(f"bg_z_mean shape {bg_z_mean.shape}")
    print(f"tg_z shape {tg_z.shape}")

    # tg_z_mean.cpu(), bg_z_mean.cpu()

    tg_z_total.append(tg_z.detach().cpu().reshape(-1, 2))
    bg_z_total.append(bg_z.detach().cpu().reshape(-1, 2))
    tg_z_mean_total.append(tg_z_mean.detach().cpu().reshape(-1, 2))
    bg_z_mean_total.append(bg_z_mean.detach().cpu().reshape(-1, 2))
    tg_z_mean_total = np.concatenate(tg_z_mean_total, axis=0)
    bg_z_mean_total = np.concatenate(bg_z_mean_total, axis=0)
    print(f"tg_z_mean_total shape {tg_z_mean_total.shape}")
    print(f"bg_z_mean_total shape {bg_z_mean_total.shape}")
    ss = plot_32d_latent_space(tg_z_mean_total=tg_z_mean_total, bg_z_mean_total=bg_z_mean_total, plot=True,
                               name='test_img_32d')
    # ss = plot_latent_space(tg_z_mean_total=tg_z_mean_total, bg_z_mean_total=bg_z_mean_total, plot=True, name='test_img')
    print(f"ss: {ss}")
    loss = cvae_loss(input_dim, tg_inputs, bg_inputs, tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean,
                     tg_s_log_var, bg_z_mean, bg_z_log_var, beta, disentangle, gamma, tc_loss, discriminator_loss)

    loss.backward()
    print(f"loss: {loss}")
