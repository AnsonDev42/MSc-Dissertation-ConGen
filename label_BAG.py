import os
import shutil
import zipfile

import nibabel as nib

import h5py, csv
from dataloader import CustomDataset, DataStoreDataset, custom_collate_fn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from dp_model.model_files.sfcn import SFCN
from dp_model import dp_loss as dpl
from dp_model import dp_utils as dpu
import torch
import torch.nn.functional as F
from sfcn_helper import get_bin_range_step

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def label_writer(filename='brain_age_info.csv', gpu=False):
    """
    write the data path, participants ID,  age, brain age
    :return:
    """
    try:
        # create a new file if not exist, else ask for overwrite
        with open(filename, 'x') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['study', 'filename', 'age', 'brain_age', 'MDD_status'])
    except FileExistsError:
        # ask for overwrite
        if input("press 'y' or 'yes' to overwrite the brain age info file") in ['y', 'yes', 'Y', 'Yes', 'YES']:
            with open(filename, 'w') as csvfile:
                writer = csv.writer(csvfile)
            writer.writerow(['study', 'filename', 'age', 'brain_age', 'MDD_status'])
        else:
            print("brain age info file not overwritten")
            exit(0)
    # load the mdoel
    model = sfcn_loader(gpu=gpu)
    # load the dataset
    dataset = CustomDataset(root_dir='data/preprocessed', csv_file='data/clinical_data.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    # iterate through the dataset
    dataloader_iter = iter(dataloader)
    with open(filename, 'a') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(dataloader)):
            sample = next(dataloader_iter)
            study = sample['study'][0]
            participant_id = sample['participant_id'][0]
            age = sample['age'][0]
            status = sample['status'][0]

            if age != -1:
                # get the brain age by infer in sfcn
                brain_age = infer_sample_h5(sample['h5_data'][0], age, model)  # set [0] since batch is 1
                age_value = age.item()
                # write the info to the file
                writer.writerow([study, participant_id, age_value, brain_age, status])
                print(f"study: {study}, participant_id: {participant_id}, age: {age_value}, brain_age: {brain_age}, "
                      f"status: {status}")
            else:
                # writer.writerow([participant_id, -1, -1])
                ...
    print("brain age info file written")


def sfcn_loader(gpu=False, eval=True, weights='./brain_age/run_20190719_00_epoch_best_mae.p'):
    """
    load the sfcn model from han's code/weight and return the model
    :param gpu: use torch gpu or not
    :return: the sfcn model with pretrained weight
    """
    model = SFCN()
    model = torch.nn.DataParallel(model)
    if not gpu:
        if eval:
            model.eval()
            model.load_state_dict(torch.load(weights, map_location='cpu'))
    else:
        if eval:
            model.eval()
        fp_ = './brain_age/run_20190719_00_epoch_best_mae.p'
        model.load_state_dict(torch.load(fp_))
        model.to(device)
    return model


def infer_sample_h5(h5_data, age, model, gpu=False):
    # Example

    # Example data: some random brain in the MNI152 1mm std space
    data = np.array(h5_data)
    label = np.array([age, ])  # Assuming the random subject is 71.3-year-old.
    # Transforming the age to soft label (probability distribution)
    print(f'Label: {label[0]}')
    bin_range, bin_step = get_bin_range_step(label[0])
    sigma = 1
    y, bc = dpu.num2vect(label, bin_range, bin_step, sigma)
    y = torch.tensor(y, dtype=torch.float32)
    # print(f'Label shape: {y.shape}') # torch.Size([1, 40])

    # Preprocessing
    # data = data / data.mean()
    # data = dpu.crop_center(data, (160, 192, 160))
    print(f'Input data shape: {data.shape}')
    # Move the data from numpy to torch tensor on GPU
    sp = (1, 1) + data.shape
    data = data.reshape(sp)
    print(f'Final Input data shape: {data.shape}')
    if gpu:
        input_data = torch.tensor(data, dtype=torch.float32).to(device)
    else:
        input_data = torch.tensor(data, dtype=torch.float32)
    # print(f'Input data shape: {input_data.shape}')
    # print(f'dtype: {input_data.dtype}')

    # Evaluation
    with torch.no_grad():
        output = model.module(input_data)

    # Output, loss, visualisation
    x = output[0].reshape([1, -1])
    # Prediction
    x = x.numpy().reshape(-1)
    prob = np.exp(x)
    pred = prob @ bc
    return pred


def visualize_output(x, y, bc):
    # Prediction, Visualisation and Summary
    x = x.numpy().reshape(-1)
    y = y.numpy().reshape(-1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(bc, y)
    plt.title('Soft label')

    prob = np.exp(x)
    pred = prob @ bc
    plt.subplot(1, 2, 2)
    plt.bar(bc, prob)
    plt.title(f'Prediction: age={pred:.2f}\n')
    plt.show()


def infer_sample_ukb(h5_data, age, model, gpu=False):
    # Example

    # Example data: some random brain in the MNI152 1mm std space
    data = np.array(h5_data)
    label = np.array([age, ])  # Assuming the random subject is 71.3-year-old.
    # Transforming the age to soft label (probability distribution)
    print(f'Label: {label[0]}')
    bin_range, bin_step = get_bin_range_step(label[0])
    sigma = 1
    y, bc = dpu.num2vect(label, bin_range, bin_step, sigma)
    y = torch.tensor(y, dtype=torch.float32)
    # print(f'Label shape: {y.shape}') # torch.Size([1, 40])

    # Preprocessing
    data = data.astype(np.float32)
    data = data / data.mean()
    data = dpu.crop_center(data, (160, 192, 160))

    print(f'Input data shape: {data.shape}')
    # Move the data from numpy to torch tensor on GPU
    sp = (1, 1) + data.shape
    data = data.reshape(sp)
    # save data into npy file
    # np.save(f'/Users/yaowenshen/Downloads/yaowen_preprocessed.npy', data)

    print(f'Final Input data shape: {data.shape}')
    if gpu:
        input_data = torch.tensor(data, dtype=torch.float32).to(device)
    else:
        input_data = torch.tensor(data, dtype=torch.float32)
    # print(f'Input data shape: {input_data.shape}')
    # print(f'dtype: {input_data.dtype}')
    print(f'data is {data}')
    #
    # with open(f'/Users/yaowenshen/Downloads/{3303915}.npy', 'rb') as f:
    #     samples_arr = np.load(f)
    #     print(f'yf data: {samples_arr}')
    #     assert samples_arr.shape == data.shape, 'shape not match'
    #     print(f'sample 1d:{str(samples_arr[0][0].flatten().numpy())}')
    #     print(f'data 1d:{str(data[0][0].flatten().numpy())}')
    #     assert (samples_arr == data).all(), 'data not match'

    # exit()

    # compare samples_arr with data
    # print(f'samples_arr.shape: {samples_arr.shape}')

    # Evaluation
    with torch.no_grad():
        output = model.module(input_data)

    # Output, loss, visualisation
    x = output[0].reshape([1, -1])
    if gpu:
        x = x.cpu()
    # Prediction
    x = x.numpy().reshape(-1)
    prob = np.exp(x)
    pred = prob @ bc
    return pred


def label_writer_batch(filename='brain_age_info.csv', gpu=True):
    """
    write the data path, participants ID,  age, brain age
    :return:
    """
    try:
        # create a new file if not exist, else ask for overwrite
        with open(filename, 'x') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['study', 'filename', 'age', 'brain_age', 'MDD_status'])
    except FileExistsError:
        # ask for overwrite
        # if input("press 'y' or 'yes' to overwrite the brain age info file") in ['y', 'yes', 'Y', 'Yes', 'YES']:
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['study', 'filename', 'age', 'brain_age', 'MDD_status'])
        # else:
        #     print("brain age info file not overwritten")
        #     exit(0)
    # load the mdoel
    model = sfcn_loader(gpu=gpu)
    # load the dataset
    HOME = os.environ['HOME']
    root_dir = f'{HOME}/GenScotDepression/data/ukb/imaging/raw/t1_structural_nifti_20252'
    csv_file = 'data/filtered_mdd_db_age.csv'
    depressed_dataset = DataStoreDataset(root_dir, csv_file, )
    depressed_dataset.load_data_info(root_dir, csv_file, filter_func=None)
    dataloader = DataLoader(depressed_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # load the dataset in cuda

    # dataset = CustomDataset(root_dir='data/preprocessed', csv_file='data/clinical_data.csv')
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    dataloader_iter = iter(dataloader)

    with open(filename, 'a') as csvfile:
        writer = csv.writer(csvfile)
        for i, batch in enumerate(dataloader):
            if batch:  # check if batch is not an empty dictionary

                study = batch['study']
                filename = batch['filename']
                mdd_status = int(batch['mdd_status'].item()) if not torch.isnan(batch['mdd_status']) else 'nan'
                tmp_dirs = batch['tmp_dir']
                data = nib.load(batch['extracted_path'][0]).get_fdata()
                brain_age = infer_sample_ukb(data, age[0], model, gpu=gpu)  # set [0] since batch is 1
                age = int(batch['age'].item())
                writer.writerow([study[0], filename[0], age[0].item(), brain_age, mdd_status])
                print(
                    f"study: {study}, filename: {filename}, age: {age}, brain age: {brain_age}, MDD_status: {mdd_status}")

                # print(f"Age: {age}, Root Directory: {root_dir}, Study: {study}, Filename: {filename}")
                # print(f"MDD Status: {mdd_status}, Temp Directory: {tmp_dirs}")

                # Clean up the batch of temporary files
                for tmp_dir in tmp_dirs:
                    if tmp_dir is not None:
                        shutil.rmtree(tmp_dir)
                        print(f'Temporary directory removed: {tmp_dir}')

    print("brain age info file written")


if __name__ == '__main__':
    # random seed
    torch.manual_seed(0)
    np.random.seed(0)
    # with open(f'/Users/yaowenshen/Downloads/{3303915}.npy', 'rb') as f:
    #     samples_arr = np.load(f)
    # data = samples_arr
    import nibabel as nib
    import tempfile

    full_compressed_path = '/afs/inf.ed.ac.uk/user/s23/s2341683/pycharm_remote_tmp/ConGeLe/data/3303915_20252_2_0.zip'

    with zipfile.ZipFile(full_compressed_path, 'r') as zip_ref:
        # Create a temporary directory
        tmp_dir = tempfile.mkdtemp(dir="/tmp/s2341683")

        # Extract the required file into the temporary directory
        target_file_path = os.path.join(tmp_dir, 'T1/T1_brain_to_MNI.nii.gz')
        zip_ref.extract('T1/T1_brain_to_MNI.nii.gz', tmp_dir)

    print(f'target_file_path: {target_file_path}')  # target for unzip
    print(f'tmp_dir: {tmp_dir}')

    data = nib.load(target_file_path).get_fdata()
    # save data
    # np.save(f'/Users/yaowenshen/Downloads/yaowen_unprocessed.npy', data)
    # print(data.shape)
    x = infer_sample_ukb(data, 72, sfcn_loader(gpu=True), gpu=True)
    print(x)
    exit(0)

    # load from T1/T1_brain_to_MNI.nii.gz to numpy array
    # using bash to ls the file under folders '/tmp/tmptzlmab22/T1/    #
    # data = nib.load('/tmp/tmptzlmab22/T1/T1_brain_to_MNI.nii.gz')
    # data = data / data.mean()
    # data = dpu.crop_center(data, (160, 192, 160))
    #
    # model = sfcn_loader()
    # # data = np.random.rand(160, 192, 160)
    # print(infer_sample_ukb(data, 71, model))
    # label_writer()
    # label_writer_batch()
    import csv

    # Load the CSV file
    with open('brain_age_info_backup.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        rows = list(reader)

    # Rewrite the CSV file
    with open('brain_age_info_clean.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['study', 'filename', 'age', 'brain_age', 'MDD_status'])  # Write the header
        for row in rows:
            study = row[0].replace("[", "").replace("]", "").replace("'", "")
            filename = row[1].replace("[", "").replace("]", "").replace("'", "")
            age = int(row[2].replace("tensor(", "").replace(")", "").replace("[", "").replace("]", ""))
            brain_age = float(row[3])
            MDD_status = row[4].replace("tensor(", "").replace(")", "").replace("[", "").replace("]", "").replace(".",
                                                                                                                  "").replace(
                " ", "").replace(",dtype=torchfloat64", "")
            if MDD_status == 'nan':
                MDD_status = 'nan'
            else:
                MDD_status = int(MDD_status)

            # Write the modified row back to the CSV
            writer.writerow([study, filename, age, brain_age, MDD_status])
