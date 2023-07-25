# this scripts provides functions to filter BAG data after training+bias correction and write into a new csv file
import pandas as pd


def filter_BAG_data(f='brain_age_info_bc.csv', fw='brain_age_info_bc.csv'):
    """
    :param f: read from csv that contains brain age and age
    :param fw: write to csv that contains brain age and age after bias correction
    :return:
    """
    # load  brain_age_info_clean.csv in pandas
    df = pd.read_csv(f)
    len_all = len(df)
    df_non_hc = df[df['mdd_ac_status'] != 0.0]  # filter out MDD patients
    # filter using new filter
    print(df_non_hc.keys())
    df_non_hc = df_non_hc[df_non_hc['BAG_BC_gt_1'] == 1]
    df_mdd = df_non_hc

    # Get the minimum and maximum age of the selected MDD patients
    min_age = df_mdd['age'].min()
    max_age = df_mdd['age'].max()

    print(f"Minimum Age: {min_age}")
    print(f"Maximum Age: {max_age}")
    age_bins = pd.interval_range(start=min_age, freq=1, end=max_age)

    df_hc = df[df['mdd_ac_status'] == 0.0]  # filter out healthy control samples
    df_hc = df_hc[df_hc['BAG_BC_gt_1'] == 1]

    df_mdd.loc[:, 'age_bin'] = pd.cut(df_mdd['age'], bins=age_bins)
    df_hc.loc[:, 'age_bin'] = pd.cut(df_hc['age'], bins=age_bins)

    # print size of df_mdd and df_hc
    print(f"Number of MDD samples: {len(df_mdd)}")
    print(f"Number of HC samples: {len(df_hc)}")

    # Create an empty DataFrame to store the selected HC samples
    df_hc_selected = pd.DataFrame()
    df_mdd_selected = pd.DataFrame()
    # Randomly select the same number of samples from each age bin for HC as in MDD
    for bin in df_mdd['age_bin'].unique():
        hc_in_bin = df_hc[df_hc['age_bin'] == bin]
        mdd_in_bin = df_mdd[df_mdd['age_bin'] == bin]
        n_samples = min(len(hc_in_bin), len(mdd_in_bin))
        hc_samples = hc_in_bin.sample(n=n_samples, replace=False, random_state=1)
        df_hc_selected = pd.concat([df_hc_selected, hc_samples])
        # only add n_samples of MDD samples to df_mdd_selected
        df_mdd_selected = pd.concat([df_mdd_selected, mdd_in_bin.sample(n=n_samples, replace=False, random_state=1)])
    # Save the new DataFrame to a CSV file
    df_combined = pd.concat([df_mdd_selected, df_hc_selected])
    df_combined = df_combined.drop(columns='age_bin')
    df_combined['depression'] = df_combined['MDD_status']
    df_combined['f.21003.2.0'] = df_combined['age']

    df_combined.to_csv(fw)
    df = df_combined
    mask1 = (df['mdd_ac_status'] == 0.0) & (df['BAG_BC_gt_1'] == 1)
    print(f"Number of total samples: {len(mask1)}")
    print(f"Number of HC samples: {mask1.sum()}")
    print(f"Number of non-hc samples: {len(mask1) - mask1.sum()}")

    return df_combined


if __name__ == '__main__':
    filter_BAG_data('brain_age_info_retrained_sfcn_4label_mdd_ac_bc_masked.csv',
                    'brain_age_info_retrained_sfcn_4label_mdd_ac_bc_masked_filtered.csv')

    print('sanaty check')
    df = pd.read_csv('brain_age_info_retrained_sfcn_4label_mdd_ac_bc_masked_filtered.csv')
    mask1 = (df['mdd_ac_status'] == 0.0) & (df['BAG_BC_gt_1'] == 1)
    print(f"Number of total samples: {len(mask1)}")
    print(f"Number of HC samples: {mask1.sum()}")
    print(f"Number of non-hc samples: {len(mask1) - mask1.sum()}")
