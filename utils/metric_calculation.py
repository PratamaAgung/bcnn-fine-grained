import h5py
from sklearn.metrics import classification_report
import numpy as np
import pickle

test_data = h5py.File('new_test.h5', 'r')
X_test, Y_test = test_data['X'], test_data['Y']

with open("class_pred_residual.pickle", "rb") as handle:
    pred_all = pickle.load(handle)

ground_truth = []
i = 0
for y,p in zip(Y_test[:], pred_all):
    # print(y)
    ground_truth.append(np.argmax(y))
    if (np.argmax(y) != p):
        print(str(i) + " " + str(np.argmax(y)))
    i += 1

# pred_all[77] = 3
print(classification_report(y_true= ground_truth, y_pred= pred_all, digits=4))
