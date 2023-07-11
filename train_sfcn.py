import datetime
import os
import pickle
import shutil
import time
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from dataloader import CustomDataset, DataStoreDataset, filter_healthy, custom_collate_fn
from label_BAG import sfcn_loader
from dp_model import dp_loss as dpl
import nibabel as nib
from dp_model import dp_utils as dpu
from sfcn_helper import get_bin_range_step

# current time in ddmm_hhmm format
now = datetime.datetime.now()
time_str = now.strftime("%d%m_%H%M")
writer = SummaryWriter(f'runs/sfcn_train_{time_str}')


def transform_labels_to_distribution(labels_batch, sigma, device):
    y_batch = []

    for label in labels_batch:
        label = label.item()
        bin_range, bin_step = get_bin_range_step(label)
        y, _ = dpu.num2vect(label, bin_range, bin_step, sigma)
        y_batch.append(y)
    y_batch = np.array(y_batch)
    return torch.Tensor(y_batch).to(dtype=torch.float32, device=device)


def train_sfcn():
    # Use GPU for training if available.
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    gpu = True
    if str(device) == 'cpu':
        gpu = False
        print('Using CPU for training')
    else:
        print('Using GPU for training')

    sfcn = sfcn_loader(gpu=device, eval=False, weights='./brain_age/run_20190719_00_epoch_best_mae.p')
    # load the dataset
    HOME = os.environ['HOME']
    root_dir = f'{HOME}/GenScotDepression/data/ukb/imaging/raw/t1_structural_nifti_20252'
    csv_file = 'data/filtered_mdd_db_age.csv'

    init_learning_rate = 0.01
    epochs = 100
    weight_decay = 0.001
    batch_size = 8  # adjust as the paper

    # Instantiate the CustomDataset class
    healthy_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False)
    healthy_dataset.load_data_info(root_dir, csv_file, filter_func=filter_healthy)
    # split to train and test
    train_size = int(0.8 * len(healthy_dataset))
    val_size = len(healthy_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(healthy_dataset, [train_size, val_size])
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn,
                            num_workers=8)
    dataloader_val = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn,
                                num_workers=8)

    optimizer = optim.SGD(sfcn.parameters(), lr=init_learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.3)

    best_loss = np.inf  # Initialize the best loss to infinity
    best_epoch = 0  # Initialize the best epoch to zero
    epoch_number = 0  # Initialize the epoch number to zero
    # Training loop
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        running_loss = 0.0
        last_loss = 0.0
        num_batches = 0
        for i, batch in enumerate(dataloader):
            if (batch is None) or 'age_bin' not in batch.keys():
                print('Batch is None or age_bin is not in batch.keys()')
                continue
            inputs = torch.Tensor(batch['image_data']).to(dtype=torch.float32, device=device)
            labels = torch.Tensor(batch['age_bin']).to(dtype=torch.float32, device=device)

            optimizer.zero_grad()
            outputs = sfcn.module(inputs)
            output_tensor = outputs[0].reshape([batch['age_bin'].shape[0], -1])
            loss = dpl.my_KLDivLoss(output_tensor, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            if i % 20 == 19:  # print every 100 mini-batches
                last_loss = running_loss / num_batches
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_number * len(dataloader) + i + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                writer.flush()

        # Print statistics
        avg_epoch_loss = running_loss / num_batches
        print(f'Epoch: {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}')
        writer.add_scalar('training loss', avg_epoch_loss, epoch)
        writer.flush()
        running_vloss = 0.0
        num_val_batches = 0
        with torch.no_grad():  # Do not calculate gradients since we are not training
            for i, batch in enumerate(dataloader_val):
                if batch is None:
                    continue

                inputs = torch.Tensor(batch['image_data']).to(dtype=torch.float32, device=device)
                labels = torch.Tensor(batch['age_bin']).to(dtype=torch.float32, device=device)

                # Forward pass # output is a list
                outputs = sfcn.module(inputs)
                output_tensor = outputs[0].reshape([batch['age_bin'].shape[0], -1])

                # Compute loss
                loss = dpl.my_KLDivLoss(output_tensor, labels)
                running_vloss += loss.item()
                num_val_batches += 1

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
            torch.save(sfcn.state_dict(), 'best_model.pth')
        elif epoch_number - best_epoch >= 7:
            print('Early stop for 7 epochs. Stopping training.')
            torch.save(sfcn.state_dict(), f'early_stop_at_{epoch_number}.pth')
            break

        sfcn.train()  # Set the model back to training mode for the next epoch
        print(f' best validation loss: {best_loss}')
        epoch_number += 1
        scheduler.step()
    print('Finished Training')
    writer.close()


if __name__ == '__main__':
    # calculat the training time
    print(f"start training")
    t_s = time.time()
    train_sfcn()
    t_e = time.time()
    print(f"end training{t_e}")
    print(f"total time in seconds: {t_e - t_s}")
