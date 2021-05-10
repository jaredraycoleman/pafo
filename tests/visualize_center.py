from pafo.triangle import Triangle, PlanarTriangle
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pathlib
thisdir = pathlib.Path(__file__).resolve().parent
from functools import partial
import numpy as np 
from typing import Tuple

try:
    from tests.visualize_solution import plot
except:
    from visualize_solution import plot

def main():
    fig: plt.Figure
    ax: plt.Axes

    speed = 0.05

    R = PlanarTriangle([
        [0, 0], [1, 0], [0.6, 0.2]
    ])
    P = Triangle.equilateral() # Triangle([30, 60, 90], deg=True)
    Q = R.min_max_traversal_triangle(P)
    pad = 0.02
    center = np.sum(Q.points, axis=0) / 3

    x_min = np.min([center[0], *R.points[:,0], *Q.points[:,0]])-pad
    x_max = np.max([center[0], *R.points[:,0], *Q.points[:,0]])+pad
    y_min = np.min([center[1], *R.points[:,1], *Q.points[:,1]])-pad
    y_max = np.max([center[1], *R.points[:,1], *Q.points[:,1]])+pad

    savepath = thisdir.joinpath("center")
    for i in range(10):
        with plot(savepath, x_min, x_max, y_min, y_max) as (fig, ax):
            ax.scatter(*center, c="blue")
            
            _R = PlanarTriangle(R.points + (Q.points - R.points) * i / 9)
            Q.plot(fig, ax, "green")

            ax.scatter(*_R.points.T, c="black")
            ax.scatter(*Q.points.T, c="green")



if __name__ == "__main__":
    main()