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

writer = SummaryWriter('runs/experiment_name')


def train_sfcn():
    # Use GPU for training if available.
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    gpu = True
    if str(device) == 'cpu':
        gpu = False
        print('Using CPU for training')
    else:
        print('Using GPU for training')

    # sfcn = sfcn_loader(gpu=False, eval=False, weights='./brain_age/run_20190719_00_epoch_best_mae.p')

    sfcn = sfcn_loader(gpu=gpu)
    # load the dataset
    HOME = os.environ['HOME']
    root_dir = f'{HOME}/GenScotDepression/data/ukb/imaging/raw/t1_structural_nifti_20252'
    csv_file = 'data/filtered_mdd_db_age.csv'

    init_learning_rate = 0.01
    epochs = 1
    weight_decay = 0.001
    batch_size = 8  # adjust as the paper

    # Instantiate the CustomDataset class
    healthy_dataset = DataStoreDataset(root_dir, csv_file, )
    healthy_dataset.load_data_info(root_dir, csv_file, filter_func=filter_healthy)
    # split to train and test
    train_size = int(0.8 * len(healthy_dataset))
    test_size = len(healthy_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(healthy_dataset, [train_size, test_size])
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn,
                            num_workers=4)
    dataloader_val = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn,
                                num_workers=4)

    optimizer = optim.SGD(sfcn.parameters(), lr=init_learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.3)

    best_loss = np.inf  # Initialize the best loss to infinity
    best_epoch = 0  # Initialize the best epoch to zero

    # To store validation MAE for each epoch
    val_mae = []

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue
                # Get the inputs
            inputs = []
            labels = batch['age']
            inputs = batch['image_data']
            labels = labels.to(dtype=torch.float32, device=device)
            # Convert the paths into actual image data
            inputs = np.array(inputs)
            inputs = torch.Tensor(inputs)
            inputs = inputs.to(dtype=torch.float32, device=device)
            labels = labels.to(dtype=torch.float32, device=device)

            tmp_dirs = batch['tmp_dir']
            # check the shape of inputs and labels
            # inputs = inputs.to(dtype=torch.float32, device=device)
            # assert labels.shape == (batch_size), labels.shape
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = sfcn.module(inputs)
            # output is a list
            output_tensor = outputs[0]
            # output_tensor = output_tensor.reshape([batch_size, -1])
            # use it in the loss computation
            loss = dpl.my_KLDivLoss(output_tensor, labels)
            writer.add_scalar('Training Loss', loss, global_step=i + epoch * len(dataloader))
            # outputs_shaped = outputs.reshape([batch_size, -1])
            # x = outputs.reshape([batch_size, -1])
            # loss = dpl.my_KLDivLoss(outputs, labels).numpy()
            loss.backward()
            optimizer.step()

            # remove the temporary files
            for tmp_dir in tmp_dirs:
                if tmp_dir is not None:
                    shutil.rmtree(tmp_dir)
                    # print(f'Temporary directory removed: {tmp_dir}')
        scheduler.step()

        # Print statistics
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
        writer.add_scalar('training loss', loss, epoch)
        # After training for one epoch, we evaluate the model on the validation set
        val_loss = 0.0
        sfcn.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Do not calculate gradients since we are not training
            for i, batch in enumerate(dataloader_val, 0):
                if batch is None:
                    continue
                # Get the inputs
                labels = batch['age']
                inputs = batch['image_data']
                labels = labels.to(dtype=torch.float32, device=device)
                # Convert the paths into actual image data
                inputs = np.array(inputs)
                inputs = torch.Tensor(inputs)
                inputs = inputs.to(dtype=torch.float32, device=device)
                labels = labels.to(dtype=torch.float32, device=device)

                # Forward pass
                outputs = sfcn.module(inputs)
                # output is a list
                output_tensor = outputs[0]
                # output_tensor = output_tensor.reshape([batch_size, -1])
                # use it in the loss computation
                loss = dpl.my_KLDivLoss(output_tensor, labels)

                val_loss += loss.item()

        val_loss = val_loss / len(dataloader_val)  # Calculate average validation loss for this epoch

        # Check if this is the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(sfcn.state_dict(), 'best_model.pth')
            # calculat mae for the best model

        # Save a checkpoint every 5 epochs
        if epoch % 5 == 4:  # Check if epoch number is a multiple of 5
            torch.save({
                'epoch': epoch,
                'model_state_dict': sfcn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, 'checkpoint.pth')
        sfcn.train()  # Set the model back to training mode for the next epoch

        print('Minimum validation best_loss: ', min(best_loss))
    print('Finished Training')
    writer.close()


if __name__ == '__main__':
    t_s = time.time()
    print(f"start training{t_s}")
    train_sfcn()
    t_e = time.time()
    print(f"end training{t_e}")
    print(f"total time: {t_e - t_s}")
