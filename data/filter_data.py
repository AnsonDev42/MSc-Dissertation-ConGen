import bz2

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
    try:
        df = pd.read_csv('t1_structural_nifti_20252_filenames.csv', sep=',')
    except FileNotFoundError:
        print("no processed csv file found... check filter_data.py for more.")
        return None
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


def add_age_info(curr_df=None, create_csv=False, name='filtered_with_age.csv'):
    """
    This func adds Age when attended assessment centre (21003)  into the dataframe
     (https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21003)

    :param curr_df:
    :param create_csv:
    :return:
    """
    if curr_df is None:
        # load md int file
        with open('Touchscreen_20126md_int.csv', 'rb') as f:
            curr_df = pd.read_csv(f, sep=',')
            # load age file using 21003
            raise NotImplementedError

    ts_data = pyreadr.read_r('Recruitment.rds')
    ts_data = ts_data[None][['f.eid', 'f.21003.0.0', 'f.21003.1.0', 'f.21003.2.0', 'f.21003.3.0']]

    ts_data = ts_data.rename(columns={'f.eid': 'subjectID'})
    # intersect merge ts_data with curr_df
    curr_df = pd.merge(curr_df, ts_data, on='subjectID', how='inner')
    if create_csv:
        curr_df.to_csv(name, index=False)
    return curr_df


def filter_depression(curr_df=None, create_csv=False):
    """
    # Data-Field 20125 Description:	Probable recurrent major depression (severe)
    # https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=20125
    """
    # md_df = create_depression_csv()
    ts_data = pd.read_csv('Touchscreen_20126md_int.csv', sep=',')
    print('before merge, all the keys in Touchscreen_20126md_int')
    print(f'length of ts_data {len(ts_data)}')
    ts_data_counts = ts_data
    print(ts_data_counts['f.20126.0.0'].value_counts().sort_index())
    if curr_df is None:
        raise NotImplementedError
    else:
        ts_data = ts_data.rename(columns={'f.eid': 'subjectID'})
        # filtered = pd.merge(ts_data, curr_df, left_on='f.eid', right_on='subjectID', how='inner')

        filtered = pd.merge(ts_data, curr_df, left_on='subjectID', right_on='subjectID', how='inner')

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
        value_counts = filtered
        # Get the value counts
        # value_counts = value_counts['f.20126.0.0'].value_counts().rename('count').reset_index().rename(columns={'index': 'index_value'})

        # Add the description column using the map function
        # value_counts['description'] = value_counts['index_value'].map(index_description)

        # Print the value counts with descriptions
        # print(value_counts)
        if create_csv:
            filtered.to_csv('filtered_depression.csv', index=False)
        return filtered


def filter_diabetes(curr_df=None, create_csv=False):
    # Data-Field 2443: 	Age diabetes diagnosed
    # https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=2976
    """
    >=1	age diabetes diagnosed
    0	nan
    -1	Do not know
    -3	Prefer not to answer
    :return:
    """
    with open('Touchscreen.csv', 'rb') as f:
        ts_data = pd.read_csv(f, usecols=['f.eid', 'f.2976.0.0', 'f.2976.1.0', 'f.2976.2.0',
                                          'f.2976.3.0'], sep=',', low_memory=False)
    ts_data = ts_data.rename(columns={'f.eid': 'subjectID'})
    ts_data = pd.merge(curr_df, ts_data[['subjectID', 'f.2976.0.0', 'f.2976.1.0', 'f.2976.2.0',
                                         'f.2976.3.0']], on='subjectID',
                       how='left')
    # filter 2976._.0 that >=1 but some are nan so we need to filter them out
    ts_data = ts_data.fillna(0)
    filtered_data = ts_data[
        ts_data['f.2976.0.0'] >= 1 | ts_data['f.2976.1.0'] >= 1 | ts_data['f.2976.2.0'] >= 1 | ts_data[
            'f.2976.3.0'] >= 1]
    print(f'length of filtered_data from diabetes {len(filtered_data)}')
    filtered_data = filtered_data[['subjectID', 'f.2976.0.0', 'f.2976.1.0', 'f.2976.2.0', 'f.2976.3.0']]

    if curr_df is None:
        return filtered_data
    else:
        # only add the one column to the curr_df to state the diabetes status for existing subjects from curr_df
        print(f'length of filtered_data from diabetes {len(filtered_data)}')
        # Merge the two dataframes on subjectID
        result_df = pd.merge(curr_df, filtered_data[['subjectID', 'f.2443']], on='subjectID',
                             how='left')
        if create_csv:
            result_df.to_csv('filtered.csv', index=False)
        return result_df


def create_unfiltered_mdd_db_csv():
    with open('Touchscreen.csv', 'rb') as f:
        ts_data = pd.read_csv(f, usecols=['f.eid', 'f.2976.0.0', 'f.2976.1.0', 'f.2976.2.0',
                                          'f.2976.3.0', 'f.20126.0.0', ], sep=',', low_memory=False)
    # export to csv
    ts_data.to_csv('unfiltered_mdd_db.csv', index=False)
    return 'unfiltered_mdd_db.csv created'


def filter_na_data(unfiltered_file='unfiltered_mdd_db_age.csv'):
    # remove f.20126.0.0 if it is nan  20126: Depression
    with open(unfiltered_file, 'rb') as f:
        data = pd.read_csv(f, sep=',', low_memory=False)
    data = data[data['f.20126.0.0'].notna()]
    print(f'length of filtered_data from non-nan MDD {len(data)}')
    #  filter out f.21003._.0 (age) if it is nan
    data = data[data['f.21003.0.0'].notna() | data['f.21003.1.0'].notna() | data['f.21003.2.0'].notna() | data[
        'f.21003.3.0'].notna()]  # 21003: Age when attended assessment centre
    print(f'length of filtered_data from non-nan age {len(data)}')
    # filter out diabetes data if all 4 columns are nan (f.2976._.0)    2976: Age diabetes diagnosed
    data = data[data['f.2976.0.0'].isna() & data['f.2976.1.0'].isna() & data['f.2976.2.0'].isna() & data[
        'f.2976.3.0'].isna()]
    # data = data[data['f.2976.0.0'].notna() | data['f.2976.1.0'].notna() | data['f.2976.2.0'].notna() | data[
    #     'f.2976.3.0'].notna()]
    print(f'length of filtered_data from non-nan DB  {len(data)}')
    # data = data[
    #     (data['f.2976.0.0'] >= 1) | (data['f.2976.1.0'] >= 1) | (data['f.2976.2.0'] >= 1) | (data['f.2976.3.0'] >= 1)]

    data.to_csv('filtered_mdd_db_age.csv', index=False)
    print('filtered_mdd_db_age.csv created')
    return data


if __name__ == '__main__':
    curr_df = create_cached_t1_filenames(create_csv=False)
    # filtered_df = filter_depression(curr_df=curr_df, create_csv=True)
    # filtered_df = filter_diabetes(curr_df=filtered_df, create_csv=True)
    #
    # create_unfiltered_mdd_db_csv()
    # filtered_df = pd.read_csv('unfiltered_mdd_db.csv', sep=',')
    # filtered_df = filtered_df.rename(columns={'f.eid': 'subjectID'})
    # print('start merge with curr_df for mdd + db')
    # filtered_df = pd.merge(curr_df, filtered_df, on='subjectID', how='inner')
    # print('start add age info for mdd + db')
    # filtered_df = add_age_info(filtered_df, create_csv=True, name='unfiltered_mdd_db_age.csv')
    data = filter_na_data('unfiltered_mdd_db_age.csv')
    depression = data[
        (data['f.20126.0.0'] == 'Probable Recurrent major depression (severe)') |
        (data['f.20126.0.0'] == 'Probable Recurrent major depression (moderate)') |
        (data['f.20126.0.0'] == 'Single Probable major depression episode')
        ]

    print(f'length of all nan-db data {len(data)}')
    print(f'length of depression {len(depression)}')
    print(f'length of healthy ppl with nan db {len(data) - len(depression)}')
