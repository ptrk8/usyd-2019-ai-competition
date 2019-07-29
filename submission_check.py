import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.applications.densenet import preprocess_input, DenseNet121
import numpy as np
import h5py
import pandas as pd
from keras.models import load_model
from utils import get_cur_milliseconds, get_best_preds, get_pessimist_preds


TEST_DATA_PATH_NAME = './data/data_rgb_512_processed.h5'
MODEL_PATH_NAME = './models/1564294047155_kappa_0.8008_val_acc_0.9356_acc_0.9299.h5'
SUBMISSION_PATH_NAME = './submissions'
BATCH_SIZE = 1


def create_submission(predictions, milliseconds, encoding_name):
    submission = pd.read_csv('./data/SampleSubmission.csv')
    submission['Expected'] = predictions
    submission.to_csv('{}/{}_{}_submission.csv'.format(SUBMISSION_PATH_NAME, milliseconds, encoding_name), index=False)


def get_sum_preds(predictions):
    return predictions.astype(int).sum(axis=1) - 1


def main():
    milliseconds = get_cur_milliseconds()
    # Retrieve values from file
    file = h5py.File(TEST_DATA_PATH_NAME, 'r')
    x_test = file['x_test']
    x_test = np.asarray(x_test)
    y_test = file['y_test']

    model = load_model(MODEL_PATH_NAME)
    model.evaluate(


if __name__ == '__main__':
    main()
