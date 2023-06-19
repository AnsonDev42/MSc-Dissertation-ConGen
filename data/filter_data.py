import pandas as pd
import pyreadr
import os


def create_cached_t1_filenames(create_csv=False):
    """
    Load the cached filenames from the txt file

    :return: a list of filenames
    """

    # subjectID_fieldID_definedInstance_arrayIndex
    # each line is a filename, e.g. 6022573_20252_2_0.zip is a filename that contains the data of subject 6022573,
    # field 20252, instance 2, and array index 0
    if create_csv:
        fs = []
        with open('t1_structural_nifti_20252_filenames.txt', 'rb') as f:
            for line in f:
                fs.append(line.strip())
        df = pd.read_csv('t1_structural_nifti_20252_filenames.txt', sep=' ', header=None)
        df.columns = ['filename']
        df['subjectID'] = df['filename'].apply(lambda x: x.split('_')[0])
        df['fieldID'] = df['filename'].apply(lambda x: x.split('_')[1])
        df['definedInstance'] = df['filename'].apply(lambda x: x.split('_')[2])
        df['arrayIndex'] = df['filename'].apply(lambda x: x.split('_')[-1].split('.')[0])

        df.to_csv('t1_structural_nifti_20252_filenames.csv', index=False)

    # read csv file
    df = pd.read_csv('t1_structural_nifti_20252_filenames.csv', sep=',')
    return df


def load_cached_filenames(csv_path=False):
    if not csv_path:
        print("no csv path given...")
        return None
    df = pd.read_csv(csv_path, sep=',')
    return df


def create_depression_csv():
    # check if major_depression.csv exists, if not create
    if os.path.exists('major_depression.csv'):
        print("already exists major_depression.csv...")
    else:
        md_data = pyreadr.read_r('2021-04-phenotypes-ukb44797/Touchscreen.rds')
        print(md_data.keys())
        md_df = md_data[None]
        md_df.to_csv('major_depression.csv', index=False)
    md_df = pd.read_csv('major_depression.csv', sep=',')
    return md_df


def filter_depression(curr_df=None):
    """
    # Data-Field 20125 Description:	Probable recurrent major depression (severe)
    # https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=20125
    """
    md_df = create_depression_csv()
    # data_field = '20125'  # Probable recurrent major depression (severe)
    # df = load_cached_filenames(create_csv=False)
    filtered = pd.merge(md_df, curr_df, left_on='f.eid', right_on='subjectID', how='inner')
    return filtered


if __name__ == '__main__':
    # load_cached_filenames(create_csv=True)
    curr_df = create_cached_t1_filenames(create_csv=False)
    filtered_df = filter_depression(curr_df=curr_df)
    print(filtered_df.head())
