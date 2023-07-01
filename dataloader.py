import csv
import os
import zipfile
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py


class CustomDataset(Dataset):
    def __init__(self, root_dir, csv_file):
        self.root_dir = root_dir
        self.data_info = self.load_data_info(root_dir, csv_file)
        self.missing_file_log = 'missing_files.csv'

    def load_data_info(self, root_dir, csv_file):
        data_info = pd.read_csv(csv_file)
        return data_info[data_info.apply(lambda row: os.path.exists(self._get_h5_path(row)), axis=1)]

    def __len__(self):
        return len(self.data_info)

    def _get_h5_path(self, row):
        participant_id = row['participant_id']
        study = row['study']
        h5_path = os.path.join(self.root_dir, study, str(participant_id))
        h5_path = h5_path + f'/{participant_id}_T1w.h5'
        return h5_path

    def _get_compressed_path(self, row):
        participant_id = row['participant_id']
        study = row['study']
        compressed_path = os.path.join(self.root_dir, study, str(participant_id))
        compressed_path = compressed_path + f'/{participant_id}_T1w.zip'
        return compressed_path

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


def filter_depressed(row):
    return row['depression'] == 1


def filter_healthy(row):
    return row['depression'] == 0


class DataStoreDataset(CustomDataset):
    def load_data_info(self, root_dir, csv_file, filter_func=None):
        data_info = pd.read_csv(csv_file)
        if filter_func is not None:
            self.data_info = data_info[data_info.apply(filter_func, axis=1)]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_info.iloc[idx]
        try:
            extracted_path, tmp_dir = self._extract_required_file(row)
            age = row['f.21003.2.0']
            item = (extracted_path, age)
        except Exception as e:
            # print(e)
            return None

        return item, tmp_dir

    def _extract_required_file(self, row):
        zip_filename = row['filename']
        full_compressed_path = os.path.join(self.root_dir, str(zip_filename))

        if not os.path.exists(full_compressed_path):
            with open(self.missing_file_log, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([zip_filename])
            # raise Exception(f"Zip file not found: {full_compressed_path}")
            return None

        with zipfile.ZipFile(full_compressed_path, 'r') as zip_ref:
            if 'T1/T1_brain_to_MNI.nii.gz' not in zip_ref.namelist():
                with open(self.missing_file_log, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([zip_filename])
                # raise Exception(f"Required file not found in zip archive: {full_compressed_path}")
                return None

            # Create a temporary directory
            tmp_dir = tempfile.mkdtemp()

            # Extract the required file into the temporary directory
            target_file_path = os.path.join(tmp_dir, 'T1/T1_brain_to_MNI.nii.gz')
            zip_ref.extract('T1/T1_brain_to_MNI.nii.gz', target_file_path)

        return target_file_path, tmp_dir


def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:  # fix for empty batch
        return []
    data, tmp_dirs = zip(*batch)  # unzip the data and the tmp_dirs
    return torch.utils.data.dataloader.default_collate(data), tmp_dirs


if __name__ == '__main__':
    # dataset = CustomDataset(root_dir='data/preprocessed', csv_file='data/clinical_data.csv')
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    # dataloader_iter = iter(dataloader)
    # sample = next(dataloader_iter)
    # assert sample['h5_data'].shape == (4, 1, 160, 192, 160), f"{sample['h5_data'].size()}"
    # print(sample['age'])
    # print(sample['participant_id'])
    HOME = os.environ['HOME']
    root_dir = f'{HOME}/GenScotDepression/data/ukb/imaging/raw/t1_structural_nifti_20252'
    csv_file = 'data/filtered_mdd_db_age.csv'
    depressed_dataset = DataStoreDataset(root_dir, csv_file)
    depressed_dataset.load_data_info(root_dir, csv_file, filter_func=filter_depressed)

    healthy_dataset = DataStoreDataset(root_dir, csv_file)
    healthy_dataset.load_data_info(root_dir, csv_file, filter_func=filter_healthy)

    # Create DataLoader objects
    depressed_loader = DataLoader(depressed_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    healthy_loader = DataLoader(healthy_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    for i, (data_n_tmp_dirs) in enumerate(healthy_loader):
        if (data_n_tmp_dirs is None) or (len(data_n_tmp_dirs) == 0):
            continue
        data, tmp_dirs = data_n_tmp_dirs
        # for path, age in zip(data[0], data[1]):
        #     print(f"Data batch {i}, Path: {path}, Age: {age}")

        # Clean up the batch of temporary files
        for tmp_dir in tmp_dirs:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir)

    # for i, (data, tmp_dirs) in enumerate(depressed_loader):
    #     if len(data) == 0:
    #         continue
    #     # Model training logic here
    #     # ...
    #     # Clean up the batch of temporary files
    #     for tmp_dir in tmp_dirs:
    #         if tmp_dir is not None:
    #             shutil.rmtree(tmp_dir)
