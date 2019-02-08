"""
Module for reading data from 'logisticX.csv' and 'logisticY.csv'
"""

import numpy as np

def loadData (x_file="../ass1_data/logisticX.csv", y_file="../logisticY.csv"):
    """
    Loads the X, Y matrices.
    """

    X = np.genfromtxt(x_file, delimiter=',')
    Y = np.genfromtxt(y_file, dtype=int)

    return (X, Y)
