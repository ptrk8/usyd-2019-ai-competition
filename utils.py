from keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score
import numpy as np
import sys
import json
import pytz
from datetime import datetime
import time
import os


def multi(arr):
    arr_new = np.copy(arr)

    for idx, val in enumerate(arr_new):
        if val == 1:
            break
        arr_new[idx] = 1

    return arr_new.astype('int16').tolist()


def to_multi_label(arr):
    return np.asarray([multi(output) for output in arr])


def get_multi_label_outputs(arr):
    return np.copy(arr).astype(int).sum(axis=1) - 1


def get_multi_class_outputs(arr):
    return np.argmax(arr, axis=1)


def get_curr_datetime():
    tz = pytz.timezone('Australia/Melbourne')
    now = datetime.now(tz)
    return now.strftime("%m/%d/%Y, %H:%M:%S")


def get_cur_milliseconds():
    return int(round(time.time() * 1000))


def get_best_preds(arr):
    arr = np.asarray(arr).astype(int)
    mask = arr == 0
    return np.clip(np.where(mask.any(1), mask.argmax(1), 5) - 1, 0, 4)


def get_score(prediction):
    for idx, val in reversed(list(enumerate(prediction))):
        if val:
            return idx
    return 0


def get_pessimist_preds(predictions):
    return [get_score(prediction) for prediction in predictions]


def get_sum_preds(predictions):
    return predictions.astype(int).sum(axis=1) - 1


class Metrics(Callback):

    def on_epoch_end(self, epoch, logs={}):
        # Get Current time
        curr_datetime = get_curr_datetime()
        curr_millisecond = get_cur_milliseconds()
        # Add current time to logs
        logs['datetime'] = curr_datetime
        logs['millisecond'] = curr_millisecond
        # Get x_validation and y_validation
        x_val, y_val = self.validation_data[:2]
        if self.out_type == 'multi_label':
            y_val = y_val.sum(axis=1) - 1
            y_pred = self.model.predict(x_val) > 0.5
            y_pred_sum = get_sum_preds(y_pred)
            # Get quadratic weighted kappa
            logs['kappa'] = cohen_kappa_score(y_val, y_pred_sum, weights='quadratic')
            # Get kappa for bestfittings method of encoding
            y_pred_bestfitting = get_best_preds(y_pred)
            logs['kappa_bestfitting'] = cohen_kappa_score(y_val, y_pred_bestfitting, weights='quadratic')
            print("val_kappa_bestfitting: {}".format(logs['kappa_bestfitting']))
            # Get kappa for bestfittings method of encoding
            y_pred_pessimistic = get_pessimist_preds(y_pred)
            logs['kappa_pessimistic'] = cohen_kappa_score(y_val, y_pred_pessimistic, weights='quadratic')
            print("val_kappa_pessimistic: {}".format(logs['kappa_pessimistic']))
        elif self.out_type == 'multi_class':
            y_val = get_multi_class_outputs(y_val)
            y_pred = self.model.predict(x_val)
            y_pred = get_multi_class_outputs(y_pred)
            print(y_val.shape)
            print(y_pred.shape)
            # Get quadratic weighted kappa
            logs['kappa'] = cohen_kappa_score(y_val, y_pred, weights='quadratic')

        # Add kappa to logs
        # logs['kappa'] = val_kappa

        # self.val_kappas.append(val_kappa)
        # Appends data to log file
        with open('{}/log.txt'.format(self.output_folder_path), 'a+') as log_file:
            log_file.write('{}\n'.format(json.dumps(logs)))

        decimal_places = 4
        self.model.save('{}/{}_kappa_{}_val_acc_{}_acc_{}.h5'.format(
            self.output_folder_path,
            curr_millisecond,
            round(logs['kappa'], decimal_places),
            round(logs['val_acc'], decimal_places),
            round(logs['acc'], decimal_places)
        ))

        # if _val_kappa == max(self.val_kappas):
        # print("Validation Kappa has improved. Saving model.")
        # self.model.save('model.h5')
        # Print Kappa
        print("val_kappa: {}".format(logs['kappa']))
        return


def get_custom_callback(out_type='multi_label', output_folder_path=None):
    if not output_folder_path:
        print('No output folder path was provided')
        sys.exit(1)
    callback = Metrics()
    # callback.val_kappas = []
    callback.out_type = out_type
    callback.output_folder_path = output_folder_path
    return callback

