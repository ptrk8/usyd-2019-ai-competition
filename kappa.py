from sklearn.metrics import cohen_kappa_score
import numpy as np
import h5py

y_val = np.array([1, 0, 3])
y_pred = np.array([1, 2, 3])

kappa = cohen_kappa_score(y_val, y_pred, weights='quadratic')

print('hello')
print(kappa)


def get_values():
    file = h5py.File('./data/data_rgb_384_processed.h5', 'r')
    return file['x_train'], file['y_train'], file['x_test'], file['y_test']


x_train, y_train, x_test, y_test = get_values()

print(y_test)

with open('y_test.txt', 'w') as f:
    f.write(str(list(y_test)))
