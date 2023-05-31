import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py


class CustomDataset(Dataset):
    def __init__(self, root_dir, csv_file):
        self.root_dir = root_dir
        data_info = pd.read_csv(csv_file)
        self.data_info = data_info[data_info.apply(lambda row: os.path.exists(self._get_h5_path(row)), axis=1)]

    def __len__(self):
        return len(self.data_info)

    def _get_h5_path(self, row):
        participant_id = row['participant_id']
        study = row['study']
        h5_path = os.path.join(self.root_dir, study, str(participant_id))
        h5_path = h5_path + f'/{participant_id}_T1w.h5'
        return h5_path

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        age = self.data_info.iloc[idx]['age']
        participant_id = self.data_info.iloc[idx]['participant_id']
        status = self.data_info.iloc[idx]['status']
        # Assuming the structure is '{dataset_name}/ds{6digits}/sub-{participant_id}'

        study = self.data_info.iloc[idx]['study']  # e.g. 'AOMIC/ds002785'
        h5_path = os.path.join(self.root_dir, study,
                               str(participant_id))  # e.g. 'data/preprocessed/AOMIC/ds002785/sub-0001'
        h5_path = h5_path + f'/{participant_id}_T1w.h5'  # e.g. 'data/preprocessed/AOMIC/ds002785/sub-0001/sub-0001_T1w.h5'

        # Load the .h5 file
        sample = {'age': age, 'root_dir': self.root_dir, 'study': study,
                  'participant_id': participant_id, 'status': status}
        try:

            with h5py.File(h5_path, 'r') as f:
                data = f['preprocessed_volume'][:]
                data = np.expand_dims(data, axis=0)
                sample['h5_data'] = torch.from_numpy(data)
            return sample
        except FileNotFoundError:
            print('File not found, load useless f for further check.' + h5_path)
            return None
            # sample['age'] = -1  # set flag for further check in label_BAG.py
            # sample['h5_data'] = torch.full((160, 192, 160), -1)  # dummy tensor
            # return sample
            # with h5py.File('data/preprocessed/AOMIC/ds002785/sub-0001/sub-0001_T1w.h5', 'r') as f:
            #     h5_data = f['preprocessed_volume'][:]
            # sample = {'h5_data': h5_data, 'age': age, 'root_dir': self.root_dir, 'study': study,
            #           'participant_id': participant_id}


if __name__ == '__main__':
    dataset = CustomDataset(root_dir='data/preprocessed', csv_file='data/clinical_data.csv')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    dataloader_iter = iter(dataloader)
    sample = next(dataloader_iter)
    assert sample['h5_data'].shape == (4, 1, 160, 192, 160), f"{sample['h5_data'].size()}"
    print(sample['age'])
    print(sample['participant_id'])
