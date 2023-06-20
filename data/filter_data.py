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
    if os.path.exists('Touchscreen.csv'):
        print("already exists touch_screen.csv...")
    else:
        # this package can not load the specific rds, have to convert to csv in Rstudio
        ts_data = pyreadr.read_r('2021-04-phenotypes-ukb44797/Touchscreen.rds')
        print(ts_data.keys())
        ts_data = ts_data[None]
        ts_data.to_csv('touch_screen.csv', index=False)
    ts_data = pd.read_csv('Touchscreen.csv', sep=',')
    md_df = ts_data[ts_data['f.20125.0.0'] == 1]
    return md_df


def filter_depression(curr_df=None):
    """
    # Data-Field 20125 Description:	Probable recurrent major depression (severe)
    # https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=20125
    """
    # md_df = create_depression_csv()
    ts_data = pd.read_csv('Touchscreen_20126md.csv', sep=',')
    filtered = pd.merge(ts_data, curr_df, left_on='f.eid', right_on='subjectID', how='inner')

    values = ['Probable Recurrent major depression (moderate)',
              'Probable Recurrent major depression (severe)']

    filtered = filtered[filtered['f.20126.0.0'].isin(values)]

    return filtered


if __name__ == '__main__':
    # print(len(create_depression_csv()))
    # load_cached_filenames(create_csv=True)
    curr_df = create_cached_t1_filenames(create_csv=False)
    filtered_df = filter_depression(curr_df=curr_df)
    print(len(filtered_df))
    print(filtered_df.head())
