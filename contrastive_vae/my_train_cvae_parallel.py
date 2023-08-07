import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from my_vae_utils import plot_latent_space, plot_32d_latent_space
from my_vae_utils import ContrastiveVAE, cvae_loss
from dataloader import DataStoreDataset, filter_healthy, filter_depressed, custom_collate_fn
from torch.nn.parallel import DataParallel
from ray import tune

HOME = os.environ['HOME']
root_dir = f'{HOME}/GenScotDepression/data/ukb/imaging/raw/t1_structural_nifti_20252'
# csv_file = './../data/filtered_mdd_db_age.csv'
# csv_file = f'./../brain_age_info_retrained_sfcn_bc_filtered.csv'
csv_file = './../brain_age_info_retrained_sfcn_4label_mdd_ac_bc_masked_filtered_1.csv'


# current time in ddmm_hhmm format


def train_single(config):
    now = datetime.datetime.now()
    time_str = now.strftime("%d%m_%H%M")
    fname = f'CVAE_mdd_ac_64_DC-{time_str}'
    writer = SummaryWriter(f'runs/cvae_32d_mdd_ac_NO_TC{time_str}')

    device = torch.device("cuda:1")
    device_ids = [1, 3]
    learning_rate = config['lr']  # 0.0001
    epochs = 100
    batch_size = 64  # from 32 # NOTE: Using the batch size must be divisible by (the number of GPUs * 2) # see

    # Instantiate the model
    input_dim = (1, 64, 64, 64)  # 784 = (1, 160, 192, 160)
    intermediate_dim = 128  # 256
    latent_dim = 32
    beta = config['beta']  # 1  # 1 WITH KL loss
    disentangle = True
    gamma = config['gamma']  # 1.  # 5 WITH TC loss
    model = ContrastiveVAE(input_dim, intermediate_dim, latent_dim, beta, disentangle, gamma)
    print(f'Model configuration: latent_dim={latent_dim}, beta={beta}, gamma={gamma}')
    model = nn.DataParallel(model, device_ids=device_ids)
    model = torch.compile(model)
    model.to(device)
    # min = torch.load('./../min_values.pt').unsqueeze(0).to(dtype=torch.float32, device=device)
    # diff = torch.load('./../diff.pt').unsqueeze(0).to(dtype=torch.float32, device=device)
    # min, max = -189.0, 4647.0
    # diff = max - min
    torch.set_float32_matmul_precision('high')

    for name, param in model.named_parameters():
        if not param.device == device:
            print(f"Parameter '{name}' is on device '{param.device}', moving to device '{device}'")
            param.data = param.data.to(device)
    for name, buf in model.named_buffers():
        if not buf.device == device:
            print(f"Buffer '{name}' is on device '{buf.device}', moving to device '{device}'")
            buf.data = buf.data.to(device)
    # model = torch.compile(model)

    # Define the optimizer
    # optimizer_model = optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer_model = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-07,
        amsgrad=False,
    )
    optimizer_discriminator = torch.optim.Adam(model.module.discriminator.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_model, step_size=8, gamma=0.1, verbose=True)

    healthy_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False, max_min=False)
    healthy_dataset.load_data_info(root_dir, csv_file, filter_func=filter_healthy)
    mdd_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False, max_min=False)
    mdd_dataset.load_data_info(root_dir, csv_file, filter_func=filter_depressed)

    # hc_train_size = int(0.8 * len(healthy_dataset))
    # mdd_train_size = int(0.8 * len(mdd_dataset))
    # hc_val_size = len(healthy_dataset) - hc_train_size
    # mdd_val_size = len(mdd_dataset) - mdd_train_size
    generator = torch.Generator().manual_seed(0)  # 42 for fixing the split for uncontaminated min/max
    # hc_train_dataset, hc_test_dataset = torch.utils.data.random_split(healthy_dataset, [hc_train_size, hc_val_size],
    #                                                                   generator=generator)
    # mdd_train_dataset, mdd_test_dataset = torch.utils.data.random_split(mdd_dataset, [mdd_train_size, mdd_val_size],
    #                                                                     generator=generator)
    hc_train_dataset, hc_test_dataset, hc_true_test_dataset = torch.utils.data.random_split(healthy_dataset,
                                                                                            [0.7, 0.15, 0.15],
                                                                                            generator=generator)
    mdd_train_dataset, mdd_test_dataset, mdd_true_test_dataset = torch.utils.data.random_split(mdd_dataset,
                                                                                               [0.7, 0.15, 0.15],
                                                                                               generator=generator)

    tg_train_data = mdd_train_dataset
    bg_train_data = hc_train_dataset

    tg_test_data = mdd_test_dataset
    bg_test_data = hc_test_dataset

    # Split the data into batches
    tg_train_loader = torch.utils.data.DataLoader(tg_train_data, batch_size=batch_size, shuffle=True,
                                                  collate_fn=custom_collate_fn,
                                                  num_workers=16, drop_last=True)
    bg_train_loader = torch.utils.data.DataLoader(bg_train_data, batch_size=batch_size, shuffle=True,
                                                  collate_fn=custom_collate_fn,
                                                  num_workers=16, drop_last=True)

    tg_test_loader = torch.utils.data.DataLoader(tg_test_data, batch_size=batch_size, shuffle=False,
                                                 collate_fn=custom_collate_fn,
                                                 num_workers=16, drop_last=True)
    bg_test_loader = torch.utils.data.DataLoader(bg_test_data, batch_size=batch_size, shuffle=False,
                                                 collate_fn=custom_collate_fn,
                                                 num_workers=16, drop_last=True)
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
        # model.disentangle = True
        disentangle = True
        print('EPOCH {}:'.format(epoch_number + 1))
        running_loss = 0.0
        # recon, beta_kl, dis, gamma_tc
        running_recon_loss, running_beta_kl_loss, running_dis_loss, running_gamma_tc_loss = 0.0, 0.0, 0.0, 0.0
        last_loss = 0.0
        num_batches = 0
        train_discriminator = epoch % 2 == 1
        # freeze discriminator when training generator and vice versa
        for param in model.module.parameters():
            param.requires_grad = not train_discriminator
        for param in model.module.discriminator.parameters():
            param.requires_grad = train_discriminator
        if train_discriminator:
            optimizer = optimizer_discriminator
        else:
            optimizer = optimizer_model

        for i, batch in enumerate(zip(tg_train_loader, bg_train_loader)):
            optimizer.zero_grad()
            if batch is None:
                continue
            tg_inputs = batch[0]['image_data'].to(dtype=torch.float32)
            bg_inputs = batch[1]['image_data'].to(dtype=torch.float32)
            output = model(tg_inputs, bg_inputs)
            tg_outputs, bg_outputs, \
                tg_z_mean, tg_z_log_var, \
                tg_s_mean, tg_s_log_var, \
                bg_z_mean, bg_z_log_var, \
                tc_loss, discriminator_loss, _, _ = output
            recon, beta_kl, dis, gamma_tc = cvae_loss(input_dim, tg_inputs, bg_inputs, tg_outputs, bg_outputs,
                                                      tg_z_mean, tg_z_log_var,
                                                      tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var, beta,
                                                      disentangle, gamma,
                                                      tc_loss, discriminator_loss)
            loss = recon + beta_kl + dis + gamma_tc
            loss.backward()
            torch.cuda.empty_cache()
            optimizer.step()
            running_loss += loss.item()
            running_recon_loss += recon.item()
            running_beta_kl_loss += beta_kl.item()
            running_dis_loss += dis.item()
            running_gamma_tc_loss += gamma_tc.item()

            num_batches += 1
            if i % 1 == 0:
                last_loss = running_loss / num_batches
                last_recon = running_recon_loss / num_batches
                last_beta_kl = running_beta_kl_loss / num_batches
                last_dis = running_dis_loss / num_batches
                last_gamma_tc = running_gamma_tc_loss / num_batches
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_number * len(tg_train_loader) + i + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                writer.add_scalar('Recon_loss/train', last_recon, tb_x)
                writer.add_scalar('Beta_KL_loss/train', last_beta_kl, tb_x)
                writer.add_scalar('Discrinination/train', last_dis, tb_x)
                writer.add_scalar('Gamma_TC_loss/train', last_gamma_tc, tb_x)
                writer.flush()
        # Print statistics
        avg_epoch_loss = running_loss / num_batches
        avg_epoch_recon = running_recon_loss / num_batches
        avg_epoch_beta_kl = running_beta_kl_loss / num_batches
        avg_epoch_dis = running_dis_loss / num_batches
        avg_epoch_gamma_tc = running_gamma_tc_loss / num_batches
        print(f'Epoch: {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}')
        writer.add_scalar('total training loss', avg_epoch_loss, epoch)
        writer.add_scalar('Recon_loss/train', avg_epoch_recon, epoch)
        writer.add_scalar('Beta_KL_loss/train', avg_epoch_beta_kl, epoch)
        writer.add_scalar('Discrinination/train', avg_epoch_dis, epoch)
        writer.add_scalar('Gamma_TC_loss/train', avg_epoch_gamma_tc, epoch)
        writer.flush()
        running_vloss = 0.0
        running_vrecon, running_vbeta_kl, running_vdis, running_vgamma_tc = 0.0, 0.0, 0.0, 0.0
        num_val_batches = 0
        tg_z_total, bg_z_total, tg_z_mean_total, bg_z_mean_total, tg_labels_total, bg_labels_total = [], [], [], [], [], []
        with torch.no_grad():
            # disentangle = False
            model.eval()
            for i, batch in enumerate(zip(tg_test_loader, bg_test_loader)):
                if batch is None:
                    continue
                tg_inputs = batch[0]['image_data'].to(dtype=torch.float32, device=device)
                bg_inputs = batch[1]['image_data'].to(dtype=torch.float32, device=device)
                tg_labels = batch[0]['mdd_ac_status'].to(dtype=torch.float32, device=device)
                bg_labels = batch[1]['mdd_ac_status'].to(dtype=torch.float32, device=device)
                # assert tg_inputs.device == min.device == device, f'device mismatch{tg_inputs.device}, {min.device}'
                # tg_inputs = (tg_inputs - min) / diff
                # bg_inputs = (bg_inputs - min) / diff
                output = model(tg_inputs, bg_inputs)
                tg_outputs, bg_outputs, \
                    tg_z_mean, tg_z_log_var, \
                    tg_s_mean, tg_s_log_var, \
                    bg_z_mean, bg_z_log_var, \
                    tc_loss, discriminator_loss, _, _ = output
                recon, beta_kl, dis, gamma_tc = cvae_loss(input_dim, tg_inputs, bg_inputs, tg_outputs, bg_outputs,
                                                          tg_z_mean, tg_z_log_var,
                                                          tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var, beta,
                                                          disentangle, gamma,
                                                          tc_loss, discriminator_loss)
                loss = recon + beta_kl + dis + gamma_tc
                running_vloss += loss.item()
                running_vrecon += recon.item()
                running_vbeta_kl += beta_kl.item()
                running_vdis += dis.item()
                running_vgamma_tc += gamma_tc.item()

                num_val_batches += 1
                # tg_z_total.append(tg_z.detach().cpu().reshape(-1, 2))
                # bg_z_total.append(bg_z.detach().cpu().reshape(-1, 2))
                tg_z_mean_total.append(tg_z_mean.cpu().reshape(-1, 32))
                bg_z_mean_total.append(bg_z_mean.cpu().reshape(-1, 32))
                tg_labels_total.append(tg_labels.cpu().reshape(-1, 1))
                bg_labels_total.append(bg_labels.cpu().reshape(-1, 1))
        tg_z_mean_total = np.concatenate(tg_z_mean_total, axis=0)
        bg_z_mean_total = np.concatenate(bg_z_mean_total, axis=0)
        tg_labels_total = np.concatenate(tg_labels_total, axis=0)
        bg_labels_total = np.concatenate(bg_labels_total, axis=0)

        ss = plot_32d_latent_space(tg_z_mean_total, bg_z_mean_total, tg_label=tg_labels_total, bg_label=bg_labels_total,
                                   name=fname, epoch=epoch)

        avg_val_loss = running_vloss / num_val_batches  # average validation loss
        avg_val_recon = running_vrecon / num_val_batches
        avg_val_beta_kl = running_vbeta_kl / num_val_batches
        avg_val_dis = running_vdis / num_val_batches
        avg_val_gamma_tc = running_vgamma_tc / num_val_batches
        print(f'Epoch: {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}')
        writer.add_scalar('total validation loss', avg_val_loss, epoch)
        writer.add_scalar('Recon_loss/val', avg_val_recon, epoch)
        writer.add_scalar('Beta_KL_loss/val', avg_val_beta_kl, epoch)
        writer.add_scalar('Discrinination/val', avg_val_dis, epoch)
        writer.add_scalar('Gamma_TC_loss/val', avg_val_gamma_tc, epoch)
        writer.add_scalar('silhouette score', ss, epoch)
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_epoch_loss, 'Validation': avg_val_loss},
                           epoch_number + 1)
        writer.flush()
        # tune.track.log(mean_accuracy=avg_val_loss)
        scheduler.step()
        # Check if this is the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch_number
            torch.save(model.state_dict(), f'best_cvae_model_{time_str}.pth')
        elif epoch_number - best_epoch >= 15:
            print('Early stop for 15 epochs. Stopping training.')
            torch.save(model.state_dict(), f'cvae_model_early_stop_at_{epoch_number}_{time_str}.pth')
            break
        print(f'Best epoch: {best_epoch + 1}, Best loss: {best_loss:.4f}')
        epoch_number += 1
    print('Finished Training CVAE')
    writer.close()


#
# analysis = tune.run(
#     train_single,
#     config={"lr": tune.grid_search([0.001]),
#             "beta": tune.grid_search([1.0]),
#             "gamma": tune.grid_search([1.0]), }
# )
#
config = {"lr": 0.001,
          "beta": 1,
          "gamma": 1, }
train_single(config)
# print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))
