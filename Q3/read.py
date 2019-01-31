"""
Module for reading data from 'logisticX.csv' and 'logisticY.csv'
"""

import numpy as np

def loadData ():
    """
    Loads the X, Y matrices.
    """

    X = np.genfromtxt('../ass1_data/logisticX.csv', delimiter=',')
    Y = np.genfromtxt('../ass1_data/logisticY.csv', dtype=int)

    return (X, Y)
