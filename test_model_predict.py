import unittest
import numpy as np
import submission as sub
import utils as ut
import h5py
from keras.applications.densenet import preprocess_input, DenseNet121
import pandas as pd

class TestModelPredict(unittest.TestCase):

    def test_model_predict(self):
        # Retrieve values from file
        file = h5py.File('./fixtures/data_rgb_384.h5', 'r')
        x_test = file['x_test']

        x_test = np.asarray(x_test).astype('float16')
        # Preprocess x_test data
        x_test = preprocess_input(x_test)

        # Returns boolean array of True and False [ True, False, False, ... ]
        y_pred = sub.model_predict(x_test, './fixtures/checkpoints_flagship_version1.h5', 8)

        # This is the standard encoding where we sum each row
        y_pred_sum = ut.get_sum_preds(y_pred)

        benchmark = pd.read_csv('./fixtures/kaggle_submission.csv')

        self.assertEqual(y_pred_sum.tolist(), benchmark['Expected'].tolist())

if __name__ == '__main__':
    unittest.main()