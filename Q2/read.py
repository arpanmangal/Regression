"""
Module for reading data from 'weightedX.csv' and 'weightedY.csv'
"""

import numpy as np

def loadData ():
    """
    Loads the X, Y matrices.
    X as sorted
    """

    X = np.genfromtxt('../ass1_data/weightedX.csv')
    Y = np.genfromtxt('../ass1_data/weightedY.csv')
    Z = [X, Y]
    Z = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
    
    # Sorting the data
    tup = list(map(tuple, Z))
    tup.sort()
    Z = np.array(tup)

    return (Z[:,0], Z[:,1])
