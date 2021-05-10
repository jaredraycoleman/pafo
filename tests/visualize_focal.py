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

def perp(a: np.ndarray) -> np.ndarray:
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def main():
    fig: plt.Figure
    ax: plt.Axes

    speed = 0.05

    R = PlanarTriangle([
        [0, 0], [1, 0], [0.6, 0.2]
    ])
    P = Triangle([30, 60, 90], deg=True)
    Q = R.min_max_traversal_triangle(P)
    pad = 0.02
    focal = seg_intersect(R.points[0], Q.points[0], R.points[1], Q.points[1])

    x_min = np.min([focal[0], *R.points[:,0], *Q.points[:,0]])-pad
    x_max = np.max([focal[0], *R.points[:,0], *Q.points[:,0]])+pad
    y_min = np.min([focal[1], *R.points[:,1], *Q.points[:,1]])-pad
    y_max = np.max([focal[1], *R.points[:,1], *Q.points[:,1]])+pad

    savepath = thisdir.joinpath("focal")
    for i in range(10):
        with plot(savepath, x_min, x_max, y_min, y_max) as (fig, ax):
            ax.scatter(*focal, c="blue")
            
            _R = PlanarTriangle(R.points + (Q.points - R.points) * i / 9)
            Q.plot(fig, ax, "green")

            ax.add_collection(LineCollection(
                [[Q.points[2], focal]],
                linestyles="--", colors="black"
            ))
            ax.add_collection(LineCollection(
                [[R.points[0], focal], [R.points[1], focal]],
                linestyles="--", colors="black"
            ))

            ax.scatter(*_R.points.T, c="black")
            ax.scatter(*Q.points.T, c="green")



if __name__ == "__main__":
    main()