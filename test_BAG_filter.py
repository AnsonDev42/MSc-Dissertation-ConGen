import pandas as pd
import unittest

from BAG_filter import filter_BAG_data

from scipy.stats import ks_2samp


class TestBrainAgeData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = filter_BAG_data('brain_age_info_retrained_sfcn_bc.csv', 'brain_age_info_retrained_sfcn_bc_test.csv')

    def test_age_distribution(self):
        df_mdd = self.df[self.df['MDD_status'] == 1.0]
        df_hc = self.df[self.df['MDD_status'] == 0.0]
        mdd_ages = df_mdd['age'].value_counts().sort_index()
        hc_ages = df_hc['age'].value_counts().sort_index()
        hc_ages = hc_ages.reindex(mdd_ages.index, fill_value=0)  # reindex the hc_ages series
        self.assertTrue(all(mdd_ages == hc_ages))

    def test_total_count(self):
        # Test that we have equal amount of MDD and HC samples
        self.assertEqual(self.df[self.df['MDD_status'] == 1.0].shape[0], self.df[self.df['MDD_status'] == 0.0].shape[0])

    def test_same_number_of_samples(self):
        df_mdd = self.df[self.df['MDD_status'] == 1.0]
        df_hc = self.df[self.df['MDD_status'] == 0.0]
        self.assertTrue(len(df_hc) <= len(df_mdd))
        from scipy.stats import ks_2samp

    def test_age_distribution(self):
        df_mdd = self.df[self.df['MDD_status'] == 1.0]
        df_hc = self.df[self.df['MDD_status'] == 0.0]

        # Run the Kolmogorov-Smirnov test
        ks_statistic, p_value = ks_2samp(df_mdd['age'], df_hc['age'])

        # Check if p_value is greater than the significance level (e.g., 0.05)
        self.assertTrue(p_value > 0.05), f"p_value: {p_value} is less than 0.05, reject null hypothesis"


if __name__ == '__main__':
    unittest.main()
