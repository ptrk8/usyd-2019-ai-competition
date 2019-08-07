import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras.backend as K
from keras.models import load_model
from sklearn.metrics import cohen_kappa_score
import h5py
from keras.applications.densenet import preprocess_input
import numpy as np
from utils import f1_loss, multi_label_acc, f1_m, get_cur_milliseconds
from os import listdir
from os.path import isfile, join, getsize
from pprint import pprint
import pandas as pd

DIRECTORY = 'D:/diabetes/models/flagship'
model_names = listdir(DIRECTORY)
model_path_names = ['{}/{}'.format(DIRECTORY, name) for name in model_names if isfile(join(DIRECTORY, name))]


file = h5py.File('D:/diabetes/kaggle/data_check_384.h5', 'r')

x_test = file['x_check']
y_test = file['y_check']

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

x_test = preprocess_input(x_test)

kappas = []
for path in model_path_names:
    model = load_model(path, custom_objects={'f1_loss': f1_loss,
                                             'multi_label_acc': multi_label_acc,
                                             'f1_m': f1_m})
    predictions = model.predict(x_test, batch_size=5, verbose=1)
    predictions_bool = predictions > 0.5
    predictions_int = predictions_bool.astype(int).sum(axis=1) - 1
    val = {
        'path': path,
        'kappa': cohen_kappa_score(predictions_int, y_test, weights='quadratic')
    }
    kappas.append(val)
    pprint(val)


print(kappas)
kappas = pd.DataFrame(kappas)
milliseconds = get_cur_milliseconds()
kappas.to_csv('{}_brute.csv'.format(milliseconds), index=False)