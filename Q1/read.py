"""
Module for reading data from 'linearX.csv' and 'linearY.csv'
"""

import numpy as np

def loadData ():
    """
    Loads the X, Y matrices.
    Splits into training, validation and test sets
    """

    X = np.genfromtxt('../ass1_data/linearX.csv')
    Y = np.genfromtxt('../ass1_data/linearY.csv')
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
    # print (training_Z, validation_Z, test_Z)
    # print (Z)

    return ([training_Z[:,0], training_Z[:,1]], 
            [validation_Z[:,0], validation_Z[:,1]],
            [test_Z[:,0], test_Z[:,1]])
