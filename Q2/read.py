"""
Module for reading data from 'weightedX.csv' and 'weightedY.csv'
"""

import numpy as np

def loadData (x_file="../ass1_data/weightedX.csv", y_file="../ass1_data/weightedY.csv"):
    """
    Loads the X, Y matrices.
    X as sorted
    """

    X = np.genfromtxt(x_file)
    Y = np.genfromtxt(y_file)
    Z = [X, Y]
    Z = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
    
    # Sorting the data
    tup = list(map(tuple, Z))
    tup.sort()
    Z = np.array(tup)

    return (Z[:,0], Z[:,1])
