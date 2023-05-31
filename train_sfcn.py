import pickle

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataloader import CustomDataset
from label_BAG import sfcn_loader
from dp_model import dp_loss as dpl


def train_sfcn():
    # Use GPU for training if available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sfcn = sfcn_loader(gpu=False, eval=False, weights='./brain_age/run_20190719_00_epoch_best_mae.p')

    # Set hyperparameters.
    init_learning_rate = 0.01
    epochs = 10
    weight_decay = 0.001
    batch_size = 8  # adjust as the paper

    optimizer = optim.SGD(sfcn.parameters(), lr=init_learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.3)
    # Instantiate the CustomDataset
    dataset_path = 'data/preprocessed'  # replace with the path to your data
    csv_file = 'data/clinical_data.csv'  # replace with the path to your CSV file
    dataset = CustomDataset(root_dir=dataset_path, csv_file=csv_file)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # # save the dataloader in pickle
    # with open('dataloader.pkl', 'wb') as f:
    #     pickle.dump(dataloader, f)

    # load the dataloader
    with open('dataloader.pkl', 'rb') as f:
        dataloader = pickle.load(f)

    # To store validation MAE for each epoch
    val_mae = []
    # Training loop
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            if data is None:
                continue
            # Get the inputs
            inputs = data['h5_data']
            labels = data['age'].to(dtype=torch.float32, device=device)
            # check the shape of inputs and labels
            inputs = inputs.to(dtype=torch.float32, device=device)
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
            # outputs_shaped = outputs.reshape([batch_size, -1])
            # x = outputs.reshape([batch_size, -1])
            # loss = dpl.my_KLDivLoss(outputs, labels).numpy()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Print statistics
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

    print('Minimum validation MAE: ', min(val_mae))
    print('Finished Training')


if __name__ == '__main__':
    train_sfcn()
