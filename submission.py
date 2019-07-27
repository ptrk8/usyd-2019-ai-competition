from keras.applications.densenet import preprocess_input, DenseNet121
import numpy as np
import h5py
import pandas as pd
from keras.models import load_model
from utils import get_cur_milliseconds, get_best_preds


TEST_DATA_PATH_NAME = './data_rgb_384.h5'
MODEL_PATH_NAME = './'


def create_submission(predictions, milliseconds, encoding_name):
    submission = pd.read_csv('./data/SampleSubmission.csv')
    submission['Expected'] = predictions
    submission.to_csv('./submissions/{}_{}_submission.csv'.format(unique_id, encoding_name), index=False)


def get_score(prediction):
    for idx, val in reversed(list(enumerate(prediction))):
        if val:
            return idx
    return 0


def get_sum_preds(predictions):
    return predictions.astype(int).sum(axis=1) - 1


def get_pessimist_preds(predictions):
    return [get_score(prediction) for prediction in predictions]


def main():
    milliseconds = get_cur_milliseconds()
    # Retrieve values from file
    file = h5py.File(TEST_DATA_PATH_NAME, 'r')
    x_test = file['x_test']
    x_test = np.asarray(x_test)
    x_test = x_test.astype('float16')
    # Preprocess x_test data
    x_test = preprocess_input(x_test)

    model = load_model(MODEL_PATH_NAME)
    predictions = model.predict(x_test, batch_size=32, verbose=1)

    # Returns boolean array of True and False [ True, False, False, ... ]
    y_pred = predictions > 0.5
    # This is the standard encoding where we sum each row
    y_pred_sum = get_sum_preds(y_pred)
    y_pred_best = get_best_preds(y_pred)
    y_pred_pessimist = get_pessimist_preds(y_pred)

    create_submission(y_pred_sum, milliseconds, 'sum_encoding')
    create_submission(y_pred_best, milliseconds, 'best_encoding')
    create_submission(y_pred_pessimist, milliseconds, 'pessimistic_encoding')

if __name__ == '__main__':
    main()