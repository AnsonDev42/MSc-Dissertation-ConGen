import json
import os
import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from my_vae_utils import plot_latent_space, plot_32d_latent_space
from my_vae_utils import ContrastiveVAE, cvae_loss, vae_loss, VAE
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
    time_str = config['time_str']
    fname = f'VAE_mdd_ac_64_DC-{time_str}'
    writer = SummaryWriter(f'runs/vae_32d_mdd_ac_mega{time_str}')

    device = torch.device("cuda:5")
    device_ids = [5]
    learning_rate = config['lr']  # 0.0001
    epochs = 100
    batch_size = config[
        'batch_size']  # from 32 # NOTE: Using the batch size must be divisible by (the number of GPUs * 2) # see

    # Instantiate the model
    input_dim = (1, 64, 64, 64)  # 784 = (1, 160, 192, 160)
    intermediate_dim = 128  # 256
    latent_dim = 32
    beta = config['beta']  # 1  # 1 WITH KL loss
    disentangle = True
    gamma = config['gamma']  # 1.  # 5 WITH TC loss
    recon_alpha = config['recon_alpha']
    model = VAE(input_dim, intermediate_dim, latent_dim, beta, disentangle, gamma)
    print(
        f'Model configuration: latent_dim={latent_dim}, beta={beta}, gamma={gamma},recon_alpha={recon_alpha}，time_str={time_str}')
    model = nn.DataParallel(model, device_ids=device_ids)
    # model = torch.compile(model)
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
    optimizer_model = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-07,
        amsgrad=False,
    )
    optimizer_discriminator = torch.optim.AdamW(
        model.module.discriminator.parameters(), lr=config['d_lr'],
        betas=(0.9, 0.999),
        eps=1e-07,
        amsgrad=False, )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_model, step_size=10, gamma=0.1, verbose=True)
    # scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_discriminator, step_size=4, gamma=0.1, verbose=True)

    healthy_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False, max_min=False)
    healthy_dataset.load_data_info(root_dir, csv_file, filter_func=filter_healthy)
    mdd_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False, max_min=False)
    mdd_dataset.load_data_info(root_dir, csv_file, filter_func=filter_depressed)

    # hc_train_size = int(0.7 * len(healthy_dataset))
    mdd_train_size = int(0.7 * len(mdd_dataset))
    # hc_val_size = len(healthy_dataset) - hc_train_size
    mdd_val_size = int(0.15 * len(mdd_dataset))
    generator = torch.Generator().manual_seed(0)  # 42 for fixing the split for uncontaminated min/max
    # hc_train_dataset, hc_test_dataset = torch.utils.data.random_split(healthy_dataset, [hc_train_size, hc_val_size],
    #                                                                   generator=generator)
    # mdd_train_dataset, mdd_test_dataset = torch.utils.data.random_split(mdd_dataset, [mdd_train_size, mdd_val_size],
    #                                                                     generator=generator)
    hc_train_dataset, hc_test_dataset, hc_true_test_dataset = torch.utils.data.random_split(healthy_dataset,
                                                                                            [0.7, 0.15, 0.15],
                                                                                            # [mdd_train_size,
                                                                                            #  mdd_val_size,
                                                                                            #  len(healthy_dataset) - mdd_train_size - mdd_val_size],
                                                                                            generator=generator)
    mdd_train_dataset, mdd_test_dataset, mdd_true_test_dataset = torch.utils.data.random_split(mdd_dataset,
                                                                                               [0.7, 0.15, 0.15],
                                                                                               # [mdd_train_size,
                                                                                               #  mdd_val_size,
                                                                                               #  len(mdd_dataset) - mdd_train_size - mdd_val_size],
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
        assert len(tg_train_loader) == len(bg_train_loader)
        model.train()
        # model.disentangle = True
        disentangle = True
        print('EPOCH {}:'.format(epoch_number + 1))
        running_loss = 0.0
        # recon, beta_kl, dis, gamma_tc
        running_recon_loss, running_beta_kl_loss, running_dis_loss, running_gamma_tc_loss = 0.0, 0.0, 0.0, 0.0
        running_q_score, running_q_bar_score = 0.0, 0.0
        running_q, running_q_bar = 0.0, 0.0
        last_loss = 0.0
        num_batches = 0
        optimizer = optimizer_model

        for i, batch in enumerate(zip(tg_train_loader, bg_train_loader)):
            optimizer.zero_grad()
            if batch is None:
                continue
            tg_inputs = batch[0]['image_data'].to(dtype=torch.float32)
            bg_inputs = batch[1]['image_data'].to(dtype=torch.float32)
            # combine tg_inputs with bg_inputs
            tg_inputs = torch.cat((tg_inputs, bg_inputs), dim=0)
            output = model(tg_inputs)
            tg_outputs, bg_outputs, \
                tg_z_mean, tg_z_log_var, \
                tg_z = output

            recon, beta_kl = vae_loss(input_dim,
                                      tg_inputs, bg_inputs, tg_outputs, bg_outputs,
                                      tg_z_mean, tg_z_log_var,
                                      beta=beta,
                                      # tg_s_mean, tg_s_log_var,
                                      # bg_z_mean, bg_z_log_var,beta=1.0,
                                      disentangle=True, recon_alpha=recon_alpha)
            loss = recon + beta_kl
            loss.backward()
            torch.cuda.empty_cache()
            optimizer.step()
            running_loss += loss.item()
            running_recon_loss += recon.item()
            running_beta_kl_loss += beta_kl.item()

            num_batches += 1
            if i % 1 == 0:
                last_loss = running_loss / num_batches
                last_recon = running_recon_loss / num_batches
                last_beta_kl = running_beta_kl_loss / num_batches
                tb_x = epoch_number * len(tg_train_loader) + i + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                writer.add_scalar('Recon_loss/train', last_recon, tb_x)
                writer.add_scalar('Beta_KL_loss/train', last_beta_kl, tb_x)
                writer.flush()
        # Print statistics
        avg_epoch_loss = running_loss / num_batches
        avg_epoch_recon = running_recon_loss / num_batches
        avg_epoch_beta_kl = running_beta_kl_loss / num_batches
        avg_epoch_dis = running_dis_loss / num_batches
        avg_epoch_gamma_tc = running_gamma_tc_loss / num_batches
        print(
            f'Epoch: {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Recon: {avg_epoch_recon:.4f}, KL: {avg_epoch_beta_kl:.4f}')
        writer.add_scalars('total loss', {'train': avg_epoch_loss, }, epoch)
        writer.add_scalars('Recon_loss', {'train': avg_epoch_recon, }, epoch)
        writer.add_scalars('Beta_KL_loss', {'train': avg_epoch_beta_kl, }, epoch)
        writer.flush()
        running_vloss = 0.0
        running_vrecon, running_vbeta_kl, running_vdis, running_vgamma_tc = 0.0, 0.0, 0.0, 0.0
        running_vq_score, running_vq_bar_score = 0.0, 0.0
        zz_total, ss_total = [], []
        running_vq, running_vq_bar = 0.0, 0.0
        running_v, running_v_bar = [], []
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
                tg_inputs = torch.cat((tg_inputs, bg_inputs), dim=0)
                output = model(tg_inputs)
                tg_outputs, bg_outputs, \
                    tg_z_mean, tg_z_log_var, \
                    tg_z = output

                recon, beta_kl = vae_loss(input_dim,
                                          tg_inputs, bg_inputs, tg_outputs, bg_outputs,
                                          tg_z_mean, tg_z_log_var,
                                          beta=beta,
                                          # tg_s_mean, tg_s_log_var,
                                          # bg_z_mean, bg_z_log_var,beta=1.0,
                                          disentangle=True, recon_alpha=recon_alpha)
                loss = recon + beta_kl
                running_vloss += loss.item()
                running_vrecon += recon.item()
                running_vbeta_kl += beta_kl.item()
                # running_vdis += dis.item()
                # running_vgamma_tc += gamma_tc.item()

                num_val_batches += 1
                # tg_z_total.append(tg_z.detach().cpu().reshape(-1, 2))
                # bg_z_total.append(bg_z.detach().cpu().reshape(-1, 2))
                tg_z_mean_total.append(tg_z_mean.cpu().reshape(-1, 32))
                tg_labels_total.append(tg_labels.cpu().reshape(-1, 1))
                bg_labels_total.append(bg_labels.cpu().reshape(-1, 1))
                zz_total.append(tg_z_mean.cpu())
        tg_z_mean_total = np.concatenate(tg_z_mean_total, axis=0)
        tg_labels_total = np.concatenate(tg_labels_total, axis=0)
        bg_labels_total = np.concatenate(bg_labels_total, axis=0)
        zz_total = torch.cat(zz_total)
        # save zz_total and ss_total to file
        # create folder
        if not os.path.exists(f'cvae_d_debug/{time_str}_vae'):
            os.makedirs(f'cvae_d_debug/{time_str}_vae')
        fname = f'cvae_d_debug/{time_str}_vae/epoch_{epoch}_z_s.npz'
        np.savez_compressed(fname, zz=tg_z_mean_total.cpu().numpy(), ss=ss_total.cpu().numpy())

        ss = plot_32d_latent_space(tg_z_mean_total, bg_z_mean_total, tg_label=tg_labels_total, bg_label=bg_labels_total,
                                   name=fname, epoch=epoch, VAE=True)

        avg_val_loss = running_vloss / num_val_batches  # average validation loss
        avg_val_recon = running_vrecon / num_val_batches
        avg_val_beta_kl = running_vbeta_kl / num_val_batches
        print(
            f'Epoch: {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f},Recon Loss: {avg_val_recon:.4f}, KL Loss: {avg_val_beta_kl:.4f}, Discrimination Loss: {avg_val_dis:.4f}, Gamma TC Loss: {avg_val_gamma_tc:.4f}, q_score acc: {avg_val_q_score:.4f}, q_bar_score acc: {avg_val_q_bar_score:.4f}')
        writer.add_scalars('total loss', {'val': avg_val_loss, }, epoch)
        writer.add_scalars('Recon_loss', {'val': avg_val_recon, }, epoch)
        writer.add_scalars('Beta_KL_loss', {'val': avg_val_beta_kl, }, epoch)
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_epoch_loss, 'Validation': avg_val_loss},
                           epoch_number + 1)

        writer.flush()
        scheduler.step()
        # Check if this is the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch_number
            best_model_state = deepcopy(model.state_dict())
            torch.save(best_model_state, f'saved_models/best_vae_model_{time_str}.pth')
        elif epoch_number - best_epoch >= 12:  # 15
            print('Early stop for 12 epochs. Stopping training.')
            # torch.save(model.state_dict(), f'saved_models/cvae_model_early_stop_at_{epoch_number}_{time_str}.pth')
            break
        print(f'Best epoch: {best_epoch + 1}, Best loss: {best_loss:.4f}')
        epoch_number += 1
    print('Finished Training CVAE')
    writer.close()


# load test data and test
def test_single(config):
    now = datetime.datetime.now()
    # time_str = now.strftime("%d%m_%H%M")
    time_str = config['time_str']
    fname = f'test CVAE_mdd_ac_64_DC-{time_str}'
    writer = SummaryWriter(f'runs/test1_cvae_32d_mdd_ac_mega{time_str}')

    device = torch.device("cuda:1")
    device_ids = [1, 2]
    epochs = 10
    batch_size = 84  # from 32 # NOTE: Using the batch size must be divisible by (the number of GPUs * 2) # see

    # Instantiate the model
    input_dim = (1, 64, 64, 64)  # 784 = (1, 160, 192, 160)
    intermediate_dim = 128  # 256
    latent_dim = 32
    beta = config['beta']  # 1  # 1 WITH KL loss
    disentangle = True
    gamma = config['gamma']  # 1.  # 5 WITH TC loss
    recon_alpha = config['recon_alpha']
    model = ContrastiveVAE(input_dim, intermediate_dim, latent_dim, beta, disentangle, gamma)
    print(
        f'Model configuration: latent_dim={latent_dim}, beta={beta}, gamma={gamma},recon_alpha={recon_alpha}，time_str={time_str}')
    model = nn.DataParallel(model, device_ids=device_ids)
    # model = torch.compile(model)
    model.to(device)
    model.load_state_dict(torch.load(config['model_path']))
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

    healthy_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False, max_min=False)
    healthy_dataset.load_data_info(root_dir, csv_file, filter_func=filter_healthy)
    mdd_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False, max_min=False)
    mdd_dataset.load_data_info(root_dir, csv_file, filter_func=filter_depressed)
    generator = torch.Generator().manual_seed(0)  # 42 for fixing the split for uncontaminated min/max
    hc_train_dataset, hc_test_dataset, hc_true_test_dataset = torch.utils.data.random_split(healthy_dataset,
                                                                                            [0.7, 0.15, 0.15],
                                                                                            # [mdd_train_size,
                                                                                            #  mdd_val_size,
                                                                                            #  len(healthy_dataset) - mdd_train_size - mdd_val_size],
                                                                                            generator=generator)
    mdd_train_dataset, mdd_test_dataset, mdd_true_test_dataset = torch.utils.data.random_split(mdd_dataset,
                                                                                               [0.7, 0.15, 0.15],
                                                                                               # [mdd_train_size,
                                                                                               #  mdd_val_size,
                                                                                               #  len(mdd_dataset) - mdd_train_size - mdd_val_size],
                                                                                               generator=generator)

    tg_test_data = mdd_true_test_dataset
    bg_test_data = hc_true_test_dataset

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

    # Loop over the batches of data
    assert len(tg_test_loader) == len(bg_test_loader)  # TODO: check if this is true
    model.eval()
    # model.disentangle = True
    disentangle = True
    print('EPOCH {}:'.format(epoch_number + 1))

    running_vloss = 0.0
    running_vrecon, running_vbeta_kl, running_vdis, running_vgamma_tc = 0.0, 0.0, 0.0, 0.0
    running_vq_score, running_vq_bar_score = 0.0, 0.0
    running_vq, running_vq_bar = 0.0, 0.0
    num_val_batches = 0
    epoch = 0
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
                tc_loss, discriminator_loss, _, _, q_score, q_bar_score, q, q_bar = output
            # q = q.detach().cpu().numpy()
            # q_bar = q_bar.detach().cpu().numpy()
            # check if q and q_bar are the same distribution

            q_pred = (q_score > 0.5).float()
            correct_q = (q_pred == 1).sum().item()
            q_bar_pred = (q_bar_score > 0.5).float()
            correct_q_bar = (q_bar_pred == 0).sum().item()
            recon, beta_kl, dis, gamma_tc = cvae_loss(input_dim, tg_inputs, bg_inputs, tg_outputs, bg_outputs,
                                                      tg_z_mean, tg_z_log_var,
                                                      tg_s_mean, tg_s_log_var, bg_z_mean, bg_z_log_var, beta,
                                                      disentangle, gamma,
                                                      tc_loss, discriminator_loss, recon_alpha)
            loss = recon + beta_kl
            running_vloss += loss.item()
            running_vrecon += recon.item()
            running_vbeta_kl += beta_kl.item()
            running_vdis += dis.item()
            running_vgamma_tc += gamma_tc.item()
            running_vq_score += correct_q
            running_vq_bar_score += correct_q_bar
            running_vq += q_score.sum().item()
            running_vq_bar += q_bar_score.sum().item()

            num_val_batches += 1
            tg_z_mean_total.append(tg_z_mean.cpu().reshape(-1, 32))
            bg_z_mean_total.append(bg_z_mean.cpu().reshape(-1, 32))
            tg_labels_total.append(tg_labels.cpu().reshape(-1, 1))
            bg_labels_total.append(bg_labels.cpu().reshape(-1, 1))
    tg_z_mean_total = np.concatenate(tg_z_mean_total, axis=0)
    bg_z_mean_total = np.concatenate(bg_z_mean_total, axis=0)
    tg_labels_total = np.concatenate(tg_labels_total, axis=0)
    bg_labels_total = np.concatenate(bg_labels_total, axis=0)

    ss = plot_32d_latent_space(tg_z_mean_total, bg_z_mean_total, tg_label=tg_labels_total, bg_label=bg_labels_total,
                               name=fname, epoch=0)

    avg_val_loss = running_vloss / num_val_batches  # average validation loss
    avg_val_recon = running_vrecon / num_val_batches
    avg_val_beta_kl = running_vbeta_kl / num_val_batches
    avg_val_dis = running_vdis / num_val_batches
    avg_val_gamma_tc = running_vgamma_tc / num_val_batches
    avg_val_q_score = running_vq_score / num_val_batches / len(q_pred)
    avg_val_q_bar_score = running_vq_bar_score / num_val_batches / len(q_bar_pred)
    avg_val_q = running_vq / num_val_batches / len(q_score)
    avg_val_q_bar = running_vq_bar / num_val_batches / len(q_bar_score)
    print(
        f'Epoch: {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f},Recon Loss: {avg_val_recon:.4f}, KL Loss: {avg_val_beta_kl:.4f}, Discrimination Loss: {avg_val_dis:.4f}, Gamma TC Loss: {avg_val_gamma_tc:.4f},')
    writer.add_scalars('total loss', {'test': avg_val_loss, }, epoch)
    writer.add_scalars('Recon_loss', {'test': avg_val_recon, }, epoch)
    writer.add_scalars('Beta_KL_loss', {'test': avg_val_beta_kl, }, epoch)
    writer.add_scalars('Discrinination', {'test': avg_val_dis, }, epoch)
    writer.add_scalars('Gamma_TC_loss', {'test': avg_val_gamma_tc, }, epoch)
    writer.add_scalars('silhouette score', {'test': ss, }, epoch)
    writer.add_scalars('Q_and_q_bar', {'test_q': avg_val_q,
                                       'test_q_bar': avg_val_q_bar}, epoch)
    writer.add_scalars('Q_and_q_bar_accuracy', {'test_q_acc': avg_val_q_score,
                                                'test_q_bar_acc': avg_val_q_bar_score}, epoch)
    writer.add_scalars('q_score acc', {'test': avg_val_q_score, }, epoch)
    writer.add_scalars('q_bar_score acc', {'test': avg_val_q_bar_score, }, epoch)
    writer.add_scalars('D avg acc', {'test': (avg_val_q_score + avg_val_q_bar_score) / 2, }, epoch)

    # writer.add_scalar('Recon_loss/val', avg_val_recon, epoch)
    # writer.add_scalar('Beta_KL_loss/val', avg_val_beta_kl, epoch)
    # writer.add_scalar('Discrinination/val', avg_val_dis, epoch)
    # writer.add_scalar('Gamma_TC_loss/val', avg_val_gamma_tc, epoch)
    # writer.add_scalar('silhouette score/val', ss, epoch)
    # writer.add_scalar('q_score acc/val', avg_val_q_score, epoch)
    # writer.add_scalar('q_bar_score acc/val', avg_val_q_bar_score, epoch)
    # writer.add_scalar('D avg acc/val', (avg_val_q_score + avg_val_q_bar_score) / 2, epoch)

    writer.flush()
    print('Finished test VAE')
    writer.close()


# analysis = tune.run(
#     train_single,
#     config={"lr": tune.grid_search([0.001]),
#             "beta": tune.grid_search([1.0]),
#             "gamma": tune.grid_search([1.0]), }
# )

#
time_str = datetime.datetime.now().strftime("%d%m_%H%M")
config = {"lr": 0.0005,
          "beta": 1,
          "gamma": 1,
          "recon_alpha": 100 / (64 ** 3),  # 64 ^ 3 / 4
          "d_lr": 0.0005,
          'batch_size': 64,
          'time_str': time_str,
          'note': 'no iterative training, saved zz,ss discriminator = DenseLayers([2 * latent_dim, 1] THIS IS VAE， seed0'}
print('this is VAE training!')
# save the config to a new file'
dir_name = 'vae_config'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
with open(f'vae_config/config_{time_str}.json', 'w') as fp:
    json.dump(config, fp)
train_single(config)

exit()

# print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

config['time_str'] = '1108_1050'
time_str = config['time_str']
# load config
with open(f'vae_config/config_{time_str}.json', 'r') as fp:
    config = json.load(fp)
config['model_path'] = f'saved_models/best_cvae_model_{time_str}.pth'


# config['latent_dim'] = 32
# config['beta'] = 10
# config['gamma'] = 0.1
# config['recon_alpha'] = 4
# config['batch_size'] = 64


def check_file(file_path):
    if os.path.isfile(file_path):
        print(f"{file_path} exists.")
    else:
        print(f"{file_path} does not exist.")
    if os.access(file_path, os.R_OK):
        print('File is accessible to read')
    else:
        print('File is not accessible to read')


check_file(config['model_path'])
test_single(config)
