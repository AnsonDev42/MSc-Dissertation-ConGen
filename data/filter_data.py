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


def map_20126_str_to_int():
    """
    convert the string to int
        0 No Bipolar or Depression
        1 Bipolar I Disorder
        2 Bipolar II Disorder
        3 Probable Recurrent major depression (severe)
        4 Probable Recurrent major depression (moderate)
        5 Single Probable major depression episode

    :return: mapped csv
    """

    ts_data = pd.read_csv('Touchscreen_20126md.csv', sep=',')
    print(ts_data.head())
    print(ts_data['f.20126.0.0'].unique())
    # convert
    ts_data['f.20126.0.0'] = ts_data['f.20126.0.0'].apply(lambda x: 0 if x == 'No Bipolar or Depression' else x)
    ts_data['f.20126.0.0'] = ts_data['f.20126.0.0'].apply(lambda x: 1 if x == 'Bipolar I Disorder' else x)
    ts_data['f.20126.0.0'] = ts_data['f.20126.0.0'].apply(lambda x: 2 if x == 'Bipolar II Disorder' else x)
    ts_data['f.20126.0.0'] = ts_data['f.20126.0.0'].apply(
        lambda x: 3 if x == 'Probable Recurrent major depression (severe)' else x)
    ts_data['f.20126.0.0'] = ts_data['f.20126.0.0'].apply(
        lambda x: 4 if x == 'Probable Recurrent major depression (moderate)' else x)
    ts_data['f.20126.0.0'] = ts_data['f.20126.0.0'].apply(
        lambda x: 5 if x == 'Single Probable major depression episode' else x)

    ts_data.to_csv('Touchscreen_20126md_int.csv', index=False)
    return ts_data


def filter_depression(curr_df=None):
    """
    # Data-Field 20125 Description:	Probable recurrent major depression (severe)
    # https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=20125
    """
    # md_df = create_depression_csv()
    ts_data = pd.read_csv('Touchscreen_20126md_int.csv', sep=',')
    # all = pd.merge(ts_data, curr_df, left_on='f.eid', right_on='subjectID', how='inner')
    print('before merge, all the keys in Touchscreen_20126md_int')
    print(f'length of ts_data {len(ts_data)}')
    print(ts_data['f.20126.0.0'].value_counts().sort_index())
    if curr_df is None:
        return ts_data
    else:
        # filtered = ts_data
        # print(f'ts_data{ts_data.keys()}')
        # print(f'curr_df{curr_df.keys()}')
        filtered = pd.merge(ts_data, curr_df, left_on='f.eid', right_on='subjectID', how='inner')
        print('after inner merge...')
        print(filtered['f.20126.0.0'].value_counts().sort_index())

        # Define a dictionary to map index to description
        index_description = {
            0.0: 'No Bipolar or Depression',
            1.0: 'Bipolar I Disorder',
            2.0: 'Bipolar I Disorder',
            3.0: 'Probable Recurrent major depression (severe)',
            4.0: 'Probable Recurrent major depression (moderate)',
            5.0: 'Single Probable major depression episode'
        }

        # Get the value counts
        value_counts = filtered['f.20126.0.0'].value_counts().rename('count').reset_index().rename(
            columns={'index': 'index_value'})

        # Add the description column using the map function
        value_counts['description'] = value_counts['index_value'].map(index_description)

        # Print the value counts with descriptions
        print(value_counts)

        return filtered


if __name__ == '__main__':
    # map_20126_str_to_int()
    # depression = filter_depression()
    # exit()
    curr_df = create_cached_t1_filenames(create_csv=False)
    filtered_df = filter_depression(curr_df=curr_df)
    # print(len(filtered_df))
    # print(filtered_df.head())
