import unittest
import numpy as np
import submission as sub
import utils as ut

def create_test_arr(arr):
    return np.array([arr]).astype(bool)

class TestSubmission(unittest.TestCase):

    def test_get_sum_preds(self):
        test = create_test_arr([1, 0, 0, 0, 0])
        result = sub.get_sum_preds(test)
        self.assertEqual(result, [0])

        test = create_test_arr([0, 1, 0, 1, 1])
        result = sub.get_sum_preds(test)
        self.assertEqual(result, [2])

        test = create_test_arr([1, 1, 1, 1, 1])
        result = sub.get_sum_preds(test)
        self.assertEqual(result, [4])

    def test_get_best_preds(self):
        test = create_test_arr([1, 0, 0, 0, 0])
        result = ut.get_best_preds(test)
        self.assertEqual(result, [0])

        test = create_test_arr([1, 1, 0, 0, 1])
        result = ut.get_best_preds(test)
        self.assertEqual(result, [1])

        test = create_test_arr([1, 0, 0, 0, 0])
        result = ut.get_best_preds(test)
        self.assertEqual(result, [0])

        test = create_test_arr([0, 0, 0, 0, 1])
        result = ut.get_best_preds(test)
        self.assertEqual(result, [0])

    def test_get_pessimist_preds(self):
        test = create_test_arr([0, 1, 0, 0, 0])
        result = sub.get_pessimist_preds(test)
        self.assertEqual(result, [1])

        test = create_test_arr([0, 0, 0, 0, 1])
        result = sub.get_pessimist_preds(test)
        self.assertEqual(result, [4])

        test = create_test_arr([1, 0, 0, 0, 0])
        result = sub.get_pessimist_preds(test)
        self.assertEqual(result, [0])

if __name__ == '__main__':
    unittest.main()