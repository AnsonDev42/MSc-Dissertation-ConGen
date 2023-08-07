import csv
import hashlib
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
import nibabel as nib
from dp_model import dp_utils as dpu
from sfcn_helper import get_bin_range_step
import skimage.transform as skt


class CustomDataset(Dataset):
    def __init__(self, root_dir, csv_file, on_the_fly=True, max_min=False):
        self.root_dir = root_dir
        self.data_info = self.load_data_info(root_dir, csv_file)
        self.missing_file_log = 'missing_files.csv'
        self.extract_dir = "/disk/scratch/s2341683/extracted_files"
        self.on_the_fly = on_the_fly
        self.max_min = max_min
        if max_min:
            self.min = torch.load('./../min_values.pt').unsqueeze(0)
            self.max = torch.load('./../max_values.pt').unsqueeze(0)
            self.diff = torch.load('./../diff.pt').unsqueeze(0)

        if not os.path.exists(self.extract_dir):
            os.makedirs(self.extract_dir)

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

    def _extract_required_file(self, row):
        zip_filename = row['filename']
        full_compressed_path = os.path.join(self.root_dir, str(zip_filename))

        if not os.path.exists(full_compressed_path):
            # ... your error handling ...
            return None, None

        # Define where to store or find the extracted file
        extracted_file_path = os.path.join(self.extract_dir, 'T1/T1_brain_to_MNI.nii.gz')

        if not os.path.exists(extracted_file_path):
            # If the file is not already extracted, extract it
            with zipfile.ZipFile(full_compressed_path, 'r') as zip_ref:
                # ... your error handling ...
                zip_ref.extract('T1/T1_brain_to_MNI.nii.gz', self.extract_dir)

        return extracted_file_path, None  # We're not using a temporary directory anymore

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
    return row['mdd_ac_status'] != 0. and row['BAG_BC_gt_1'] == 1


def filter_healthy(row):
    return row['mdd_ac_status'] == 0. and row['BAG_BC_gt_1'] == 1


class DataStoreDataset(CustomDataset):
    def load_data_info(self, root_dir, csv_file, filter_func=None):
        data_info = pd.read_csv(csv_file)
        if filter_func is not None:
            self.data_info = data_info[data_info.apply(filter_func, axis=1)]
        else:
            self.data_info = data_info

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_info.iloc[idx]

        try:
            extracted_path, tmp_dir = self._extract_required_file(row)
            if extracted_path is None:  # check for None values
                return None
            age = row['f.21003.2.0']
            # Load the image data and pre-process
            data, labels, bc = self.preprocessing(extracted_path, age)

        except Exception as e:
            print(f'get item exception in {extracted_path}:{e},try to remove and unzip again')
            self._cleanup_temp_dir(tmp_dir)
            extracted_path, tmp_dir = self._extract_required_file(row)
            if extracted_path is None:  # check for None values
                return None
            age = row['f.21003.2.0']
            data, labels, bc = self.preprocessing(extracted_path, age)

        sample = {'image_data': data, 'age': age, 'root_dir': self.root_dir, 'study': 'ukb',
                  'filename': row['filename'], 'mdd_status': row['depression'],
                  'extracted_path': extracted_path, 'age_bin': labels, 'bc': bc}
        if 'db' in row:  # add for db
            sample['db'] = row['db']
        if 'mdd_ac_status' in row:
            sample['mdd_ac_status'] = row['mdd_ac_status']
        # remove the temporary files
        if self.on_the_fly:
            self._cleanup_temp_dir(tmp_dir)

        return sample

    def preprocessing(self, extracted_path, age):
        data = nib.load(extracted_path).get_fdata()
        data = data.astype(np.float32)
        # for sfcn starts #
        """
        data = data / data.mean()  # disable here
        data = dpu.crop_center(data, (160, 192, 160))
        # normalise data by min max in all dimensions
        data = data.reshape([1, 160, 192, 160])
        """
        # for sfcn ends #

        # for cvae starts #
        min = data.min()
        max = data.max()
        diff = max - min
        data = (data - min) / diff
        # # resample to 64 * 64* 64
        data = skt.resize(data, (64, 64, 64), order=1, preserve_range=True, anti_aliasing=True)
        data = data.reshape([1, 64, 64, 64])
        # for cvae ends #

        # if self.max_min:
        #     data = torch.tensor(data)
        #     data = (data - self.min) / self.diff
        label = np.array([age, ])
        bin_range, bin_step = get_bin_range_step(age=label)
        labels, bc = dpu.num2vect(label, bin_range, bin_step, sigma=1)
        # labels, bc = -1, -1
        return data, labels, bc

    def _cleanup_temp_dir(self, tmp_dir):
        try:
            shutil.rmtree(tmp_dir)
        except Exception as e:
            print(f'Error while cleaning up temporary directory: {e}')

    def _extract_required_file(self, row):
        zip_filename = row['filename']
        full_compressed_path = os.path.join(self.root_dir, str(zip_filename))

        if not os.path.exists(full_compressed_path):
            with open(self.missing_file_log, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([zip_filename])
            # raise Exception(f"Zip file not found: {full_compressed_path}")
            return None, None
        # filename is xxxx.zip, so the extracted folder name is xxxx
        extracted_folder_name = zip_filename.split('.')[0]
        extracted_file_path = os.path.join(self.extract_dir, f'{extracted_folder_name}/T1/T1_brain_to_MNI.nii.gz')

        if os.path.exists(extracted_file_path):
            return extracted_file_path, os.path.join(self.extract_dir, f'{extracted_folder_name}')

        # if not exists, check availability and extract it
        with zipfile.ZipFile(full_compressed_path, 'r') as zip_ref:
            if 'T1/T1_brain_to_MNI.nii.gz' not in zip_ref.namelist():
                with open(self.missing_file_log, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([zip_filename])
                # raise Exception(f"Required file not found in zip archive: {full_compressed_path}")
                return None, None

            # Create a temporary directory
            # tmp_dir = tempfile.mkdtemp(dir="/disk/scratch/s2341683")
            tmp_dir = os.path.join(self.extract_dir, f'{extracted_folder_name}')
            # Extract the required file into the temporary directory
            target_file_path = os.path.join(tmp_dir, 'T1/T1_brain_to_MNI.nii.gz')
            zip_ref.extract('T1/T1_brain_to_MNI.nii.gz', tmp_dir)
            print(f'Extracted {zip_filename} to {target_file_path}')

        return target_file_path, tmp_dir


def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))  # remove None
    if len(batch) == 0:  # if batch is empty
        return {}  # return an empty dict
    return torch.utils.data.dataloader.default_collate(batch)  # use default collate on the filtered batch


def get_min_max():
    HOME = os.environ['HOME']
    root_dir = f'{HOME}/GenScotDepression/data/ukb/imaging/raw/t1_structural_nifti_20252'
    # csv_file = 'data/filtered_mdd_db_age.csv'
    # depressed_dataset = DataStoreDataset(root_dir, csv_file)
    # depressed_dataset.load_data_info(root_dir, csv_file, filter_func=filter_depressed)
    # healthy_dataset = DataStoreDataset(root_dir, csv_file)
    # healthy_dataset.load_data_info(root_dir, csv_file, filter_func=filter_healthy)
    # depressed_loader = DataLoader(depressed_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    # healthy_loader = DataLoader(healthy_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    csv_file = 'brain_age_info_retrained_sfcn_bc_filtered.csv'
    healthy_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False, max_min=False)
    healthy_dataset.load_data_info(root_dir, csv_file, filter_func=filter_healthy)
    mdd_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False, max_min=False)
    mdd_dataset.load_data_info(root_dir, csv_file, filter_func=filter_depressed)

    hc_train_size = int(0.8 * len(healthy_dataset))
    mdd_train_size = int(0.8 * len(mdd_dataset))
    hc_val_size = len(healthy_dataset) - hc_train_size
    mdd_val_size = len(mdd_dataset) - mdd_train_size
    generator = torch.Generator().manual_seed(42)  # for fixing the split for uncontaminated min/max
    hc_train_dataset, hc_test_dataset = torch.utils.data.random_split(healthy_dataset, [hc_train_size, hc_val_size],
                                                                      generator=generator)

    batch_size = 64
    bg_train_data = hc_train_dataset

    bg_train_loader = torch.utils.data.DataLoader(bg_train_data, batch_size=batch_size, shuffle=True,
                                                  collate_fn=custom_collate_fn,
                                                  num_workers=16, generator=generator)

    # Go through the rest of the dataset
    sample_shape = (1, 160, 192, 160)

    min_values = np.full(sample_shape, np.inf)
    max_values = np.full(sample_shape, -np.inf)

    for data in bg_train_loader:
        image_data = data['image_data']
        for img in image_data:
            min_values = np.minimum(min_values, img)
            max_values = np.maximum(max_values, img)
    # save the min/max values
    torch.save(min_values, 'min_values.pt')
    torch.save(max_values, 'max_values.pt')
    print(f'current min/max: {min_values}, {max_values}')  # current min/max: -189.0, 4647.0
    # Split the data into batches
    mdd_train_dataset, mdd_test_dataset = torch.utils.data.random_split(mdd_dataset, [mdd_train_size, mdd_val_size],
                                                                        generator=generator)

    tg_train_data = mdd_train_dataset
    tg_train_loader = torch.utils.data.DataLoader(tg_train_data, batch_size=batch_size, shuffle=True,
                                                  collate_fn=custom_collate_fn,
                                                  num_workers=16)

    for data in tg_train_loader:
        data = data['image_data']
        for img in image_data:
            min_values = np.minimum(min_values, img)
            max_values = np.maximum(max_values, img)
            print('2 in progress...')

    torch.save(min_values, 'min_val.pt')
    torch.save(max_values, 'max_val.pt')
    print(min_values, max_values)


if __name__ == '__main__':
    # get_min_max()
    HOME = os.environ['HOME']
    root_dir = f'{HOME}/GenScotDepression/data/ukb/imaging/raw/t1_structural_nifti_20252'
    csv_file = 'brain_age_info_retrained_sfcn_4label_mdd_ac_bc_masked_filtered.csv'
    healthy_dataset = DataStoreDataset(root_dir, csv_file, on_the_fly=False, max_min=False, )
    healthy_dataset.load_data_info(root_dir, csv_file, filter_func=filter_healthy)
    print(f'len of healthy:{len(healthy_dataset)}')
    dep = DataStoreDataset(root_dir, csv_file, on_the_fly=False, max_min=False, )
    dep.load_data_info(root_dir, csv_file, filter_func=filter_depressed)
    print(f'len of non-healthy:{len(dep)}')
    # iteraate first sample to see its shape
    for i in range(len(healthy_dataset)):
        sample = healthy_dataset[i]
        print(sample['image_data'].shape)
        print(sample['image_data'].min(), sample['image_data'].max())
        break
    # min = torch.load('min_values.pt')
    # max = torch.load('max_values.pt')
    # diff = max - min
    # torch.save(diff, 'diff.pt')
    # print(min.min(), max.max())
