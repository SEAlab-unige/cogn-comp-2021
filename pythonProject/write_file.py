"""
This file contains methods to write on a file.

Author: Tommaso Apicella
Email: tommaso.apicella@edu.unige.it
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def clear_file(file_path):
    open(file_path, 'w').close()
    return None


def write_acc_in_file(file_path, ytrue, ypred):
    acc = accuracy_score(ytrue, ypred)
    with open(file_path, 'a') as f:
        f.write("Accuracy: {}".format(acc))
    f.close()
    return None


def write_confusion_matrix_in_file(file_path, ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)
    with open(file_path, 'a') as f:
        f.write(np.array2string(cm, separator=', '))
    f.close()
    return None


def write_text_in_file(file_path, text):
    with open(file_path, 'a') as f:
        f.write(text)
    f.close()
    return None
