"""
Module for reading data from 'weightedX.csv' and 'weightedY.csv'
"""

import numpy as np

def loadData ():
    """
    Loads the X, Y matrices.
    X as sorted
    Splits into training, validation and test sets
    """

    X = np.genfromtxt('../ass1_data/weightedX.csv')
    Y = np.genfromtxt('../ass1_data/weightedY.csv')
    Z = [X, Y]
    Z = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
    
    # Sorting the data
    tup = list(map(tuple, Z))
    tup.sort()
    Z = np.array(tup)

    # Partition the data into three sets
    size = len(Z)
    training_size = int(0.8 * size)
    validation_size = int(0.1 * size)
    test_size = int(0.1 * size)

    training_Z = Z[0:training_size]
    validation_Z = Z[training_size:training_size+validation_size]
    test_Z = Z[training_size+validation_size:]

    return (Z[:,0], Z[:,1])
