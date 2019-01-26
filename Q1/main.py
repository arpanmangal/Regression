"""
Main controller
"""

import plot as plot
import read as ld
import model as model

if __name__ == '__main__':

    # Load data
    data = ld.loadData()

    # visualise data
    # plot.scatterPlot(data[0][0], data[0][1], marker='rx')
    # plot.scatterPlot(data[1][0], data[1][1], marker='rx')
    # plot.scatterPlot(data[2][0], data[2][1], marker='rx')

    # Train data
    model.train(data[0][0], data[0][1],
                data[1][0], data[1][1],
                data[2][0], data[2][1])