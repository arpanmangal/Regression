"""
Module for reading data from 'linearX.csv' and 'linearY.csv'
"""

import numpy as np

def loadData (x_file="ass1_data/linearX.csv", y_file="ass1_data/linearY.csv"):
    """
    Loads the X, Y matrices.
    Splits into training, validation and test sets
    """

    X = np.genfromtxt(x_file)
    Y = np.genfromtxt(y_file)
    Z = [X, Y]
    Z = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
    np.random.shuffle(Z)

    # Partition the data into three sets
    size = len(Z)
    training_size = int(0.8 * size)
    validation_size = int(0.1 * size)
    test_size = int(0.1 * size)

    training_Z = Z[0:training_size]
    validation_Z = Z[training_size:training_size+validation_size]
    test_Z = Z[training_size+validation_size:]

    return (Z[:,0], Z[:,1])
