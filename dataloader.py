import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py


class CustomDataset(Dataset):
    def __init__(self, root_dir, csv_file):
        self.root_dir = root_dir
        self.data_info = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the age from the csv file
        age = self.data_info.iloc[idx]['age']

        # Get the participant_id from the csv file
        participant_id = self.data_info.iloc[idx]['participant_id']

        # Construct the path of the .h5 file
        # Assuming the structure is '{dataset_name}/ds{6digits}/sub-{participant_id}'
        study = self.data_info.iloc[idx]['study']  # e.g. 'AOMIC/ds002785'
        h5_path = os.path.join(self.root_dir, study, 'sub-' + str(participant_id))

        # Load the .h5 file
        try:
            with h5py.File(h5_path, 'r') as f:
                h5_data = f['preprocessed_volume'][:]
            sample = {'h5_data': h5_data, 'age': age}
            return sample
        except:  # TODO: specify the exception
            print('File not found, load useless f for further check: ' + h5_path)
            with h5py.File('data/preprocessed/AOMIC/ds002785/sub-0001/sub-0001_T1w.h5', 'r') as f:
                h5_data = f['preprocessed_volume'][:]
            sample = {'h5_data': h5_data, 'age': age}
            return sample


if __name__ == '__main__':
    dataset = CustomDataset(root_dir='data/preprocessed', csv_file='data/clinical_data.csv')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    dataloader_iter = iter(dataloader)
    sample = next(dataloader_iter)
    assert sample['h5_data'].shape == (4, 160, 192, 160)
