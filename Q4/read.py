"""
Module for reading data from 'q4x.csv' and 'q4y.csv'
"""

import numpy as np

def loadData (x_file="../ass1_data/q4x.dat", y_file="../ass1_data/q4y.dat"):
    """
    Loads the X, Y matrices.
    """

    X = np.genfromtxt(x_file, delimiter='  ', dtype=int)
    labels = np.genfromtxt(y_file, dtype=str)
    Y = []
    for label in labels:
        if (label == "Alaska"):
            Y.append(1)
        else:
            Y.append(0)

    return (X, Y)
