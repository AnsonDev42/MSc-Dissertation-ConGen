import datetime

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import sys

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
        kernel_size = 2
        # stride =2
        self.z_conv1 = nn.Conv3d(input_shape[0], filters * 2, kernel_size=12, stride=11, padding=(1, 1, 1), bias=bias)
        self.z_conv2 = nn.Conv3d(filters * 2, filters * 4, kernel_size=4, stride=3, padding=(1, 1, 1), bias=bias)
        # hidden layers
        # self.z_h_layers = DenseLayers([input_shape, intermediate_dim])  # line143 in vae_utils.py
        self.z_h_layers = nn.Linear(19200, intermediate_dim)
        self.z_mean_layer = nn.Linear(intermediate_dim, latent_dim)
        self.z_log_var_layer = nn.Linear(intermediate_dim, latent_dim)
        self.sampler = Sampler()
        # print each layers parameters
        # for name, param in self.named_parameters():
        #     print(f"Encoder layer name: {name}, size: {param.size()}")

    def forward(self, x):
        x = F.relu(self.z_conv1(x))
        x = F.relu(self.z_conv2(x))
        # x = F.relu(self.z_conv3(x))

        z_shape = list(x.size())
        print(f"shape of x is {z_shape}")  # shape of x is [ 128, 40, 48, 40]
        z_h = x.flatten(start_dim=1)
        print(f"shape of x is {z_h.size()}")  # shape of x is 314,572,800
        z_h = self.z_h_layers(z_h)
        z_mean = self.z_mean_layer(z_h)
        z_log_var = self.z_log_var_layer(z_h)
        z = self.sampler((z_mean, z_log_var))
        assert z.shape[1:] == (self.latent_dim,)
        return z_mean, z_log_var, z, z_shape


tg_inputs = torch.randn(1, 1, 160, 192, 160)
encoder = Encoder(input_shape=(1, 160, 192, 160), intermediate_dim=128, latent_dim=2, filters=32, kernel_size=3,
                  bias=True)
res = encoder(tg_inputs)
print(f"!!!!z_shape {res[3]}")  # z_shape [1, 128, 40, 48, 40]


class Decoder(nn.Module):
    """
         input of decoders: torch.Size([1, 4])
    """

    def __init__(self, latent_dim, intermediate_dim, z_shape, output_dim, filters, kernel_size, nlayers, bias=True):
        super(Decoder, self).__init__()
        self.z_shape = [1, 128, 40, 48, 40]
        self.filters = filters
        self.kernel_size = kernel_size
        self.linear1 = nn.Linear(latent_dim * 2, intermediate_dim, bias=bias)
        self.linear2 = nn.Linear(intermediate_dim,
                                 self.z_shape[1] * self.z_shape[2] * self.z_shape[3] * self.z_shape[4],
                                 bias=bias)  # based on z_shape

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
                                             out_channels=self.filters // 4,
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
                                             padding=4,
                                             output_padding=0,
                                             bias=bias
                                             )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # x = x.flatten(start_dim=0)
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
    def __init__(self, input_shape=(1, 160, 192, 160), intermediate_dim=128, latent_dim=2, beta=1, disentangle=False,
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
        else:
            raise NotImplementedError("for now encoder is using hardcoded product of z_shape")
            # res = self.z_encoder(torch.randn(*input_shape))
            # self.z_shape = res[3]
            # print(f"z_shape set to unverified {self.z_shape}")
        self.decoder = Decoder(latent_dim, intermediate_dim, self.z_shape, output_dim=input_shape, filters=filters,
                               kernel_size=kernel_size, nlayers=nlayers, bias=bias)
        self.disentangle = disentangle
        self.beta = beta
        self.gamma = gamma

        if self.disentangle:  # line 209 in vae_utils.py
            self.discriminator = nn.Linear(2 * latent_dim, 1)
            self.sigmoid = nn.Sigmoid()

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

        if self.disentangle:
            batch_size = tg_inputs.size(0)
            print(f"tg_s.size(), bg_z.size() {tg_s.size(), bg_z.size()}")
            z1 = tg_z[:batch_size // 2, :]
            z2 = tg_z[batch_size // 2:, :]
            s1 = tg_s[:batch_size // 2, :]
            s2 = tg_s[batch_size // 2:, :]
            print(f"s1.size(), z2.size() {s1.size(), z2.size()}")
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
        print('tg_outputs is cuda, move outputs to device')
        tg_inputs = tg_inputs.to(tg_outputs.device)
        bg_inputs = bg_inputs.to(bg_outputs.device)
    tg_reconstruction_loss = reconstruction_loss(tg_outputs, tg_inputs).view(tg_inputs.size(0), -1).sum(dim=1)
    bg_reconstruction_loss = reconstruction_loss(bg_outputs, bg_inputs).view(bg_inputs.size(0), -1).sum(dim=1)
    reconstruction_loss = 0.5 * (tg_reconstruction_loss + bg_reconstruction_loss) \
                          / (original_dim[1] * original_dim[2] * original_dim[3])

    # KL Loss
    kl_loss = (
                      1 + tg_z_log_var - tg_z_mean.pow(2) - tg_z_log_var.exp() \
                      + 1 + tg_s_log_var - tg_s_mean.pow(2) - tg_s_log_var.exp() \
                      + 1 + bg_z_log_var - bg_z_mean.pow(2) - bg_z_log_var.exp()
              ).sum(dim=-1) * -0.5
    # kl_loss = kl_loss.sum(dim=-1) * -0.5
    if disentangle:
        cvae_loss = reconstruction_loss.mean() + beta * kl_loss.mean() + gamma * tc_loss.mean() + discriminator_loss.mean()
        # print each loss
        print(f'cvae_loss {cvae_loss}')
        print(f"reconstruction_loss {reconstruction_loss.mean()}")
        print(f"beta* kl_loss {beta * kl_loss.mean()}")
        print(f"gamma * tc_loss {gamma * tc_loss}")
        print(f"discriminator_loss {discriminator_loss}")
        print(f"tc_loss {tc_loss}")
        print(f"kl_loss {kl_loss.mean()}")
    else:
        cvae_loss = reconstruction_loss.mean() + beta * kl_loss.mean()
        # print each loss
        print(f'cvae_loss {cvae_loss}')
        print(f"reconstruction_loss {reconstruction_loss.mean()}")
        print(f"beta* kl_loss {beta * kl_loss.mean()}")

    return cvae_loss


def plot_latent_space(tg_z_mean, bg_z_mean, y=None, plot=True, name='z_mean'):
    from sklearn.metrics import silhouette_score
    # ss = round(silhouette_score(z_mean, y), 3)
    if plot:
        plt.figure()
        plt.scatter(tg_z_mean[:, 0], tg_z_mean[:, 1], c='b', label='tg_z')
        plt.scatter(bg_z_mean[:, 0], bg_z_mean[:, 1], c='r', label='bg_z')
        plt.legend()  # Displays a legend
        plt.title(name)
        plt.savefig(f'cvae_results/{name}.svg')  # Change the path and filename as needed

        # plt.title(name + ', Silhouette score: ' + str(ss))
    # return ss


if __name__ == '__main__':
    # current time in ddmm_hhmm format
    now = datetime.datetime.now()
    time_str = now.strftime("%d%m_%H%M")
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ids = [0, 3, 4]
    # print(f'Using device: {device} and potential device_ids: {device_ids}')
    # Define hyperparameters
    learning_rate = 0.001
    epochs = 100

    # HOME = os.environ['HOME']
    # root_dir = f'{HOME}/GenScotDepression/data/ukb/imaging/raw/t1_structural_nifti_20252'
    # # csv_file = './../data/filtered_mdd_db_age.csv'
    # csv_file = f'./../brain_age_info_retrained_sfcn_bc_filtered.csv'
    # healthy_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False)
    # healthy_dataset.load_data_info(root_dir, csv_file, filter_func=filter_healthy)
    # mdd_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False)
    # mdd_dataset.load_data_info(root_dir, csv_file, filter_func=filter_depressed)
    #
    # hc_train_size = int(0.8 * len(healthy_dataset))
    # hc_val_size = len(healthy_dataset) - hc_train_size
    # hc_train_dataset, hc_test_dataset = torch.utils.data.random_split(healthy_dataset, [hc_train_size, hc_val_size])
    #
    # mdd_train_size = int(0.8 * len(mdd_dataset))
    # mdd_val_size = len(mdd_dataset) - mdd_train_size
    # mdd_train_dataset, mdd_test_dataset = torch.utils.data.random_split(mdd_dataset, [mdd_train_size, mdd_val_size])
    #
    # tg_train_data = mdd_train_dataset
    # bg_train_data = hc_train_dataset
    #
    # tg_test_data = mdd_test_dataset
    # bg_test_data = hc_test_dataset
    #
    # # Split the data into batches
    # tg_train_loader = torch.utils.data.DataLoader(tg_train_data, batch_size=batch_size, shuffle=True,
    #                                               collate_fn=custom_collate_fn,
    #                                               num_workers=8)
    # bg_train_loader = torch.utils.data.DataLoader(bg_train_data, batch_size=batch_size, shuffle=True,
    #                                               collate_fn=custom_collate_fn,
    #                                               num_workers=8)
    #
    # tg_test_loader = torch.utils.data.DataLoader(tg_test_data, batch_size=batch_size, shuffle=False,
    #                                              collate_fn=custom_collate_fn,
    #                                              num_workers=8)
    # bg_test_loader = torch.utils.data.DataLoader(bg_test_data, batch_size=batch_size, shuffle=False,
    #                                              collate_fn=custom_collate_fn,
    #                                              num_workers=8)

    # best_loss = np.inf  # Initialize the best loss to infinity
    # best_epoch = 0  # Initialize the best epoch to zero
    # epoch_number = 0  # Initialize the epoch number to zero
    # for epoch in range(epochs):
    #     # Loop over the batches of data
    #     assert len(tg_train_loader) == len(bg_train_loader)  # TODO: check if this is true
    #     print('EPOCH {}:'.format(epoch_number + 1))
    #     running_loss = 0.0
    #     last_loss = 0.0
    #     num_batches = 0
    #     for i, batch in enumerate(zip(tg_train_loader, bg_train_loader)):
    #         if batch is None:
    #             continue
    #         tg_inputs, bg_inputs = batch[0]['image_data'].to(dtype=torch.float32), batch[1]['image_data'].to(
    #             dtype=torch.float32)
    #
    #         assert tg_inputs.shape == bg_inputs.shape
    #     break
    # exit(0)
    # device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    # tg_inputs = torch.randn(8, 1, 160, 192, 160).to(device)
    # bg_inputs = torch.randn(8, 1, 160, 192, 160).to(device)
    # cvae = ContrastiveVAE()
    # cvae.to(device)
    # torch.cuda.empty_cache()

    # device_ids = [4, 1]
    print(f'Using device: {device}')
    # Define hyperparameters
    learning_rate = 0.001
    input_dim = (1, 160, 192, 160)  # 784
    intermediate_dim = 128  # 256
    latent_dim = 2
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

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {pytorch_total_params}")
    # model = torch.compile(model)
    tg_inputs = torch.randn(8, 1, 160, 192, 160).to(dtype=torch.float32)
    bg_inputs = torch.randn(8, 1, 160, 192, 160).to(dtype=torch.float32)
    tg_inputs = tg_inputs.to(device)
    bg_inputs = bg_inputs.to(device)
    tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var, \
        tc_loss, discriminator_loss = model(tg_inputs,
                                            bg_inputs)

    print(f"tg_outputs shape {tg_outputs.shape}")
    print(f"bg_outputs shape {bg_outputs.shape}")
    print(f"tg_z_mean shape {tg_z_mean.shape}")
    print(f"tg_z_log_var shape {tg_z_log_var.shape}")
    print(f"tg_s_mean shape {tg_s_mean.shape}")
    print(f"tg_s_log_var shape {tg_s_log_var.shape}")
    print(f"bg_z_mean shape {bg_z_mean.shape}")

    loss = cvae_loss(input_dim, tg_inputs, bg_inputs, tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean,
                     tg_s_log_var, bg_z_mean, bg_z_log_var, beta, disentangle, gamma, tc_loss, discriminator_loss)
    del tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var
    torch.cuda.empty_cache()
    loss.backward()
    print(f"loss: {loss}")
