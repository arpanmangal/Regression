"""
Module for reading data from 'q4x.csv' and 'q4y.csv'
"""

import numpy as np

def loadData ():
    """
    Loads the X, Y matrices.
    """

    X = np.genfromtxt('../ass1_data/q4x.dat', delimiter='  ', dtype=int)
    labels = np.genfromtxt('../ass1_data/q4y.dat', dtype=str)
    Y = []
    for label in labels:
        if (label == "Alaska"):
            Y.append(1)
        else:
            Y.append(0)

    return (X, Y)
