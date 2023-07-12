import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

from contrastive_vae.vae_utils import plot_latent_space
from my_vae_utils import ContrastiveVAE, cvae_loss
from dataloader import DataStoreDataset, filter_healthy, filter_depressed, custom_collate_fn
from torch.nn.parallel import DataParallel

device = torch.device("cuda")
device_ids = [4, 1]
print(f'Using device: {device}')
# Define hyperparameters
learning_rate = 0.001
num_epochs = 10
batch_size = 8  # from 32

# Instantiate the model
input_dim = (1, 160, 192, 160)  # 784
intermediate_dim = 256

latent_dim = 2
beta = 1
disentangle = True
gamma = 0
model = ContrastiveVAE(input_dim, intermediate_dim, latent_dim, beta, disentangle, gamma)
model = nn.DataParallel(model, device_ids=device_ids)  # Wrap the model with DataParallel

# model.to(device)
model = torch.compile(model)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

HOME = os.environ['HOME']
root_dir = f'{HOME}/GenScotDepression/data/ukb/imaging/raw/t1_structural_nifti_20252'
# csv_file = './../data/filtered_mdd_db_age.csv'
csv_file = f'./../brain_age_info_retrained_sfcn_bc_filtered.csv'
healthy_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False)
healthy_dataset.load_data_info(root_dir, csv_file, filter_func=filter_healthy)
mdd_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False)
mdd_dataset.load_data_info(root_dir, csv_file, filter_func=filter_depressed)

hc_train_size = int(0.1 * len(healthy_dataset))
hc_val_size = len(healthy_dataset) - hc_train_size
hc_train_dataset, hc_test_dataset = torch.utils.data.random_split(healthy_dataset, [hc_train_size, hc_val_size])

mdd_train_size = int(0.1 * len(mdd_dataset))
mdd_val_size = len(mdd_dataset) - mdd_train_size
mdd_train_dataset, mdd_test_dataset = torch.utils.data.random_split(mdd_dataset, [mdd_train_size, mdd_val_size])

tg_train_data = mdd_train_dataset
bg_train_data = hc_train_dataset

tg_test_data = mdd_test_dataset
bg_test_data = hc_test_dataset

# Split the data into batches
tg_train_loader = torch.utils.data.DataLoader(tg_train_data, batch_size=batch_size, shuffle=True,
                                              collate_fn=custom_collate_fn,
                                              num_workers=8)
bg_train_loader = torch.utils.data.DataLoader(bg_train_data, batch_size=batch_size, shuffle=True,
                                              collate_fn=custom_collate_fn,
                                              num_workers=8)
# Loop over the data for the desired number of epochs
for epoch in range(num_epochs):
    # Loop over the batches of data
    assert len(tg_train_loader) == len(bg_train_loader)  # TODO: check if this is true
    model.train()
    for i, batch in enumerate(zip(tg_train_loader, bg_train_loader)):
        optimizer.zero_grad()
        # batch = batch.to(device)
        if batch is None:
            continue
        tg_inputs, bg_inputs = batch
        if 'image_data' not in tg_inputs.keys() or 'image_data' not in bg_inputs.keys():
            raise print('image_data not in batch.keys()')  # never should happen in this stage

        tg_inputs = torch.Tensor(tg_inputs['image_data']).to(dtype=torch.float32)
        bg_inputs = torch.Tensor(bg_inputs['image_data']).to(dtype=torch.float32)

        output = model(tg_inputs, bg_inputs)
        tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var, tc_loss, discriminator_loss = output
        loss = cvae_loss(input_dim, tg_inputs, bg_inputs, tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean,
                         tg_s_log_var, bg_z_mean, bg_z_log_var, beta, disentangle, gamma)

        loss.backward()

        # Update the optimizer parameters using the gradients
        optimizer.step()

        with torch.no_grad():
            model.eval()
            model.disentangle = True
            # batch = batch.to(device)
            if batch is None:
                continue
            tg_inputs, bg_inputs = batch
            if 'image_data' not in tg_inputs.keys() or 'image_data' not in bg_inputs.keys():
                raise print('image_data not in batch.keys()')
            for i, batch in enumerate(zip(tg_test_data, bg_test_data)):
                tg_inputs, bg_inputs = batch
                if 'image_data' not in tg_inputs.keys() or 'image_data' not in bg_inputs.keys():
                    raise print('image_data not in batch.keys()')
                tg_inputs = torch.Tensor(tg_inputs['image_data']).to(dtype=torch.float32)
                bg_inputs = torch.Tensor(bg_inputs['image_data']).to(dtype=torch.float32)
                tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var, \
                    tc_loss, discriminator_loss = model(tg_inputs, bg_inputs)
                plot_latent_space(tg_z_mean, bg_z_mean, f'{num_epochs}')
                loss = cvae_loss(input_dim, tg_inputs, bg_inputs, tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var,
                                 tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var, beta, disentangle, gamma)
                print('Test Loss: {:.4f}'.format(loss.item()))

        model.disentangle = False

    # Print the loss for this epoch
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Save the trained model
torch.save(model.state_dict(), 'contrastive_vae.pth')
