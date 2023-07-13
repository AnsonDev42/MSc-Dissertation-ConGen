import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from my_vae_utils import plot_latent_space
from my_vae_utils import ContrastiveVAE, cvae_loss
from dataloader import DataStoreDataset, filter_healthy, filter_depressed, custom_collate_fn
from torch.nn.parallel import DataParallel

# current time in ddmm_hhmm format
now = datetime.datetime.now()
time_str = now.strftime("%d%m_%H%M")
writer = SummaryWriter(f'runs/sfcn_train_{time_str}')
torch.cuda.empty_cache()
# device = torch.device("cuda:3")
device_ids = [0, 1, 3]
# print(f'Using device: {device} and potential device_ids: {device_ids}')
#
# from accelerate import Accelerator
# from accelerate import infer_auto_device_map
#
# accelerator = Accelerator(device_ids=device_ids)
# accelerator.device_ids = device_ids
# device = accelerator

# Define hyperparameters
learning_rate = 0.001
epochs = 10
batch_size = 4  # from 32

# Instantiate the model
input_dim = (1, 160, 192, 160)  # 784
intermediate_dim = 256

latent_dim = 2
beta = 1
disentangle = True
gamma = 0
model = ContrastiveVAE(input_dim, intermediate_dim, latent_dim, beta, disentangle, gamma)

model = nn.DataParallel(model, device_ids=device_ids)
model.to(device_ids[0])
for name, param in model.named_parameters():
    if not param.device == device_ids[0]:
        print(f"Parameter '{name}' is on device '{param.device}', moving to device '{device_ids[0]}'")
        param.data = param.data.to(device_ids[0])

for name, buf in model.named_buffers():
    if not buf.device == device_ids[0]:
        print(f"Buffer '{name}' is on device '{buf.device}', moving to device '{device_ids[0]}'")
        buf.data = buf.data.to(device_ids[0])

# model = torch.compile(model)

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
# model, optimizer, tg_train_loader, bg_train_loader = accelerator.prepare(model, optimizer, tg_train_loader,
#                                                                          bg_train_loader)

best_loss = np.inf  # Initialize the best loss to infinity
best_epoch = 0  # Initialize the best epoch to zero
epoch_number = 0  # Initialize the epoch number to zero
for epoch in range(epochs):
    # Loop over the batches of data
    assert len(tg_train_loader) == len(bg_train_loader)  # TODO: check if this is true
    model.train()
    print('EPOCH {}:'.format(epoch_number + 1))
    running_loss = 0.0
    last_loss = 0.0
    num_batches = 0
    for i, batch in enumerate(zip(tg_train_loader, bg_train_loader)):
        optimizer.zero_grad()
        if batch is None:
            continue
        tg_inputs, bg_inputs = batch[0]['image_data'], batch[1]['image_data']
        # tg_inputs, bg_inputs = batch[0].to('cuda:2'), batch[1].to('cuda:2')
        # if 'image_data' not in tg_inputs.keys() or 'image_data' not in bg_inputs.keys():
        #     raise print('image_data not in batch.keys()')  # never should happen in this stage

        # tg_inputs = torch.Tensor(tg_inputs['image_data']).to(dtype=torch.float32)
        # bg_inputs = torch.Tensor(bg_inputs['image_data']).to(dtype=torch.float32)
        output = model(tg_inputs, bg_inputs)
        tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var, \
            tc_loss, discriminator_loss = output

        loss = cvae_loss(input_dim, tg_inputs, bg_inputs, tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean,
                         tg_s_log_var, bg_z_mean, bg_z_log_var, beta, disentangle, gamma, tc_loss, discriminator_loss)
        del tg_inputs, bg_inputs, tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var,
        torch.cuda.empty_cache()
        loss.backward()
        # accelerator.backward(loss)
        # Update the optimizer parameters using the gradients
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
        if i % 1 == 0:
            last_loss = running_loss / num_batches
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_number * len(tg_train_loader) + i + 1
            writer.add_scalar('Loss/train', last_loss, tb_x)
            writer.flush()
        # Print statistics
        avg_epoch_loss = running_loss / num_batches
        print(f'Epoch: {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}')
        writer.add_scalar('training loss', avg_epoch_loss, epoch)
        writer.flush()
        running_vloss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            model.eval()
            model.disentangle = False
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
                tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var = model(
                    tg_inputs, bg_inputs)
                plot_latent_space(tg_z_mean, bg_z_mean, f'epoch {epoch}')
                loss = cvae_loss(input_dim, tg_inputs, bg_inputs, tg_outputs, bg_outputs, tg_z_mean, tg_z_log_var,
                                 tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var, beta, disentangle, gamma)
                running_vloss += loss.item()
                num_val_batches += 1
            print('Test Loss: {:.4f}'.format(loss.item()))
        avg_val_loss = running_vloss / num_val_batches  # average validation loss
        writer.add_scalar('validation loss', avg_val_loss, epoch)
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_epoch_loss, 'Validation': avg_val_loss},
                           epoch_number + 1)
        writer.flush()
        # Check if this is the best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_epoch = epoch_number
        torch.save(model.state_dict(), f'best_cvae_model_{time_str}.pth')
    elif epoch_number - best_epoch >= 7:
        print('Early stop for 7 epochs. Stopping training.')
        torch.save(model.state_dict(), f'cvae_model_early_stop_at_{epoch_number}_{time_str}.pth')
        break
    model.train()
    epoch_number += 1
print('Finished Training CVAE')
writer.close()

# # Save the trained model
# torch.save(model.state_dict(), 'contrastive_vae.pth')
