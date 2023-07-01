import os
import shutil
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
    if not gpu:
        model = SFCN()
        model = torch.nn.DataParallel(model)
        if eval:
            model.eval()
            model.load_state_dict(torch.load(weights, map_location='cpu'))
    else:
        model = SFCN()
        model = torch.nn.DataParallel(model)
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
    data = h5_data
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
    dataloader = DataLoader(depressed_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    # load the dataset in cuda

    # dataset = CustomDataset(root_dir='data/preprocessed', csv_file='data/clinical_data.csv')
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    dataloader_iter = iter(dataloader)

    with open(filename, 'a') as csvfile:
        writer = csv.writer(csvfile)
        for i, batch in enumerate(dataloader):
            if batch:  # check if batch is not an empty dictionary
                age = batch['age']
                study = batch['study']
                filename = batch['filename']
                mdd_status = batch['mdd_status']
                tmp_dirs = batch['tmp_dir']
                data = nib.load(batch['extracted_path'][0]).get_fdata()
                brain_age = infer_sample_ukb(data, age[0], model, gpu=gpu)  # set [0] since batch is 1
                writer.writerow([study, filename, age, brain_age, mdd_status])
                print(
                    f"study: {study}, filename: {filename}, age: {age}, brain age: {brain_age}, MDD_status: {mdd_status}")

                # print(f"Age: {age}, Root Directory: {root_dir}, Study: {study}, Filename: {filename}")
                # print(f"MDD Status: {mdd_status}, Temp Directory: {tmp_dirs}")

                # Clean up the batch of temporary files
                for tmp_dir in tmp_dirs:
                    if tmp_dir is not None:
                        shutil.rmtree(tmp_dir)

    print("brain age info file written")


if __name__ == '__main__':
    # random seed
    torch.manual_seed(0)
    np.random.seed(0)
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
    label_writer_batch()
