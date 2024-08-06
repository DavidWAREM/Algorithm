import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import os
import logging
from data.rawdata_preprocess import DataProcessor  # Adjust the import according to your file structure


class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case environment.
        """
        self.mock_dataframe = pd.DataFrame({
            'REM FLDNAM KNO': ['KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'LEI',
                               'LEI', 'LEI', 'LEI', 'LEI', 'LEI', 'LEI'],
            'XRECHTS': [491245.2363, 491168.0245, 491857.7938, 491817.6781, 491896.3958, 491787.058, 491207.4857,
                        491185.1055, 491154.3696, 491154.5747, 491644.0148, 'K0001', 'K0317', 'K0005', 'K0008', 'K0010',
                        'K0011', 'K0150'],
            'YHOCH': [5465338.373, 5465388.848, 5465818.563, 5465871.392, 5466172.364, 5466114.927, 5465564.878,
                      5465611.87, 5465367.341, 5465367.213, 5466102.006, 'K0002', 'K0003', 'K0006', 'K0007', 'K0009',
                      'K0012', 'K0013'],
            'KNAM': ['K0001', 'K0002', 'K0003', 'K0004', 'K0005', 'K0006', 'K0007', 'K0008', 'K0009', 'K0010', 'K0011',
                     1, 302, 5, 8, 10, 11, 146],
            'ZUFLUSS': [0, -0.038, -0.042, -0.006, -0.032, -0.12, -0.031, -0.076, -0.008, 0, -0.045, 2, 3, 6, 7, 9, 12,
                        13],
            'FSTATUS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'PMESS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'LEI00000162FD357ECA6F', 'LEI00000262FD357ED282',
                      'LEI00000362FD357ED32A', 'LEI00000462FD357ED39E', 'LEI00000562FD357ED3D3',
                      'LEI00000662FD357ED41C', 'LEI00000762FD357ED46F']
        })
        self.original_file_path = 'test.csv'
        self.data_processor = DataProcessor(self.mock_dataframe, self.original_file_path)
        logging.disable(logging.CRITICAL)  # Disable logging for testing

    def tearDown(self):
        """
        Clean up after each test case.
        """
        logging.disable(logging.NOTSET)  # Enable logging after testing

    def test_split_data(self):
        """
        Test splitting the DataFrame into 'KNO' and 'LEI' DataFrames.
        """
        mock_dataframe = pd.DataFrame({
            'REM FLDNAM KNO': ['KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'LEI',
                               'LEI', 'LEI', 'LEI', 'LEI', 'LEI', 'LEI'],
            'XRECHTS': [491245.2363, 491168.0245, 491857.7938, 491817.6781, 491896.3958, 491787.058, 491207.4857,
                        491185.1055, 491154.3696, 491154.5747, 491644.0148, 'K0001', 'K0317', 'K0005', 'K0008', 'K0010',
                        'K0011', 'K0150'],
            'YHOCH': [5465338.373, 5465388.848, 5465818.563, 5465871.392, 5466172.364, 5466114.927, 5465564.878,
                      5465611.87, 5465367.341, 5465367.213, 5466102.006, 'K0002', 'K0003', 'K0006', 'K0007', 'K0009',
                      'K0012', 'K0013'],
            'KNAM': ['K0001', 'K0002', 'K0003', 'K0004', 'K0005', 'K0006', 'K0007', 'K0008', 'K0009', 'K0010', 'K0011',
                     1, 302, 5, 8, 10, 11, 146],
            'ZUFLUSS': [0, -0.038, -0.042, -0.006, -0.032, -0.12, -0.031, -0.076, -0.008, 0, -0.045, 2, 3, 6, 7, 9, 12,
                        13],
            'FSTATUS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'PMESS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'LEI00000162FD357ECA6F', 'LEI00000262FD357ED282',
                      'LEI00000362FD357ED32A', 'LEI00000462FD357ED39E', 'LEI00000562FD357ED3D3',
                      'LEI00000662FD357ED41C', 'LEI00000762FD357ED46F']
        })
        data_processor = DataProcessor(mock_dataframe, self.original_file_path)

        kno_df, lei_df = data_processor.split_data()

        expected_kno_df = pd.DataFrame({
            'REM FLDNAM KNO': ['KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO'],
            'XRECHTS': [491245.2363, 491168.0245, 491857.7938, 491817.6781, 491896.3958, 491787.058, 491207.4857,
                        491185.1055, 491154.3696, 491154.5747, 491644.0148],
            'YHOCH': [5465338.373, 5465388.848, 5465818.563, 5465871.392, 5466172.364, 5466114.927, 5465564.878,
                      5465611.87, 5465367.341, 5465367.213, 5466102.006],
            'KNAM': ['K0001', 'K0002', 'K0003', 'K0004', 'K0005', 'K0006', 'K0007', 'K0008', 'K0009', 'K0010', 'K0011'],
            'ZUFLUSS': [0, -0.038, -0.042, -0.006, -0.032, -0.12, -0.031, -0.076, -0.008, 0, -0.045],
            'FSTATUS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'PMESS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        })
        expected_lei_df = pd.DataFrame({
            'REM FLDNAM KNO': ['LEI', 'LEI', 'LEI', 'LEI', 'LEI', 'LEI', 'LEI'],
            'XRECHTS': ['K0001', 'K0317', 'K0005', 'K0008', 'K0010', 'K0011', 'K0150'],
            'YHOCH': ['K0002', 'K0003', 'K0006', 'K0007', 'K0009', 'K0012', 'K0013'],
            'KNAM': [1, 302, 5, 8, 10, 11, 146],
            'ZUFLUSS': [2, 3, 6, 7, 9, 12, 13],
            'FSTATUS': [0, 0, 0, 0, 0, 0, 0],
            'PMESS': ['LEI00000162FD357ECA6F', 'LEI00000262FD357ED282', 'LEI00000362FD357ED32A',
                      'LEI00000462FD357ED39E', 'LEI00000562FD357ED3D3', 'LEI00000662FD357ED41C',
                      'LEI00000762FD357ED46F']
        })

        pd.testing.assert_frame_equal(kno_df.reset_index(drop=True), expected_kno_df)
        pd.testing.assert_frame_equal(lei_df.reset_index(drop=True), expected_lei_df)

    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_save_dataframes(self, mock_to_csv, mock_makedirs):
        """
        Test saving the split DataFrames to CSV files.
        """
        mock_dataframe = pd.DataFrame({
            'REM FLDNAM KNO': ['KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'KNO', 'LEI',
                               'LEI', 'LEI', 'LEI', 'LEI', 'LEI', 'LEI'],
            'XRECHTS': [491245.2363, 491168.0245, 491857.7938, 491817.6781, 491896.3958, 491787.058, 491207.4857,
                        491185.1055, 491154.3696, 491154.5747, 491644.0148, 'K0001', 'K0317', 'K0005', 'K0008', 'K0010',
                        'K0011', 'K0150'],
            'YHOCH': [5465338.373, 5465388.848, 5465818.563, 5465871.392, 5466172.364, 5466114.927, 5465564.878,
                      5465611.87, 5465367.341, 5465367.213, 5466102.006, 'K0002', 'K0003', 'K0006', 'K0007', 'K0009',
                      'K0012', 'K0013'],
            'KNAM': ['K0001', 'K0002', 'K0003', 'K0004', 'K0005', 'K0006', 'K0007', 'K0008', 'K0009', 'K0010', 'K0011',
                     1, 302, 5, 8, 10, 11, 146],
            'ZUFLUSS': [0, -0.038, -0.042, -0.006, -0.032, -0.12, -0.031, -0.076, -0.008, 0, -0.045, 2, 3, 6, 7, 9, 12,
                        13],
            'FSTATUS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'PMESS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'LEI00000162FD357ECA6F', 'LEI00000262FD357ED282',
                      'LEI00000362FD357ED32A', 'LEI00000462FD357ED39E', 'LEI00000562FD357ED3D3',
                      'LEI00000662FD357ED41C', 'LEI00000762FD357ED46F']
        })
        data_processor = DataProcessor(mock_dataframe, self.original_file_path)

        data_processor.save_dataframes()

        # Check if the to_csv method was called twice
        self.assertEqual(mock_to_csv.call_count, 2)

    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_combine_connected_pipes(self, mock_to_csv, mock_makedirs):
        """
        Test combining connected pipes based on specified criteria.
        """
        kno_df = pd.DataFrame({
            'REM': ['REM1', 'REM2'],
            'FLDNAM': ['FLD1', 'FLD2'],
            'KNO': ['KNO1', 'KNO2'],
            'KNAM': ['KNAM1', 'KNAM2'],
            'ZUFLUSS': ['ZUF1', 'ZUF2'],
            'FSTATUS': ['FSTAT1', 'FSTAT2'],
            'PMESS': ['PM1', 'PM2'],
            'DSTATUS': ['DSTAT1', 'DSTAT2'],
            'GEOH': ['GEO1', 'GEO2'],
            'PRECH': ['P1', 'P2'],
            'DP': ['DP1', 'DP2'],
            'XRECHTS': ['X1', 'X2'],
            'YHOCH': ['Y1', 'Y2'],
            'HP': ['HP1', 'HP2'],
            'SYMBOL': ['S1', 'S2'],
            'ABGAENGE': ['2', '0'],
            'NETZNR': ['NET1', 'NET2']
        })

        lei_df = pd.DataFrame({
            'REM': ['REM1', 'REM2'],
            'FLDNAM': ['FLD1', 'FLD2'],
            'LEI': ['LEI1', 'LEI2'],
            'ANFNAM': ['KNAM1', 'KNAM2'],
            'ENDNAM': ['KNAM2', 'KNAM1'],
            'ANFNR': ['ANF1', 'ANF2'],
            'ENDNR': ['END1', 'END2'],
            'RORL': ['ROR1', 'ROR2'],
            'DM': ['DM1', 'DM2'],
            'RAU': ['RAU1', 'RAU2'],
            'FLUSS': ['FLU1', 'FLU2'],
            'VM': ['VM1', 'VM2'],
            'DP': ['DP1', 'DP2'],
            'DPREL': ['DPR1', 'DPR2'],
            'ROHRTYP': ['RT1', 'RT2'],
            'RAISE': ['R1', 'R2']
        })

        data_processor = DataProcessor(kno_df, self.original_file_path)
        updated_lei_df = data_processor.combine_connected_pipes(kno_df, lei_df)

        # Check if GroupID column was added
        self.assertIn('GroupID', updated_lei_df.columns)

        # Check if GroupID values are assigned
        self.assertTrue(updated_lei_df['GroupID'].notnull().all())


if __name__ == '__main__':
    unittest.main()
