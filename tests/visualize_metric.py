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

def get_tris(A: Triangle, B: Triangle) -> Tuple[PlanarTriangle, PlanarTriangle]:
    _A = A.trivial_replication([0, 0.2], [1, 0])
    _B = B.trivial_replication([1.2, .1], [2.2, .5])
    return _A, _B


def plot_tris(A: Triangle, B: Triangle, savepath: pathlib.Path, pad: float, annotate: bool = True):    
    _A, _B = get_tris(A, B)
    with plot(savepath, 0-pad, 2.2+pad, 0-pad, 
              max(_A.points[2][1], _B.points[2][1])+pad) as (fig, ax):
        _A.plot(fig, ax, color="black")
        _A_offsets = np.array([
            [0.05, 0],
            [-0.05, 0.03],
            [-0.05, -0.06],
        ])
        _B.plot(fig, ax, color="blue")
        _B_offsets = np.array([
            [0.065, 0.06],
            [-0.07, 0.0],
            [0, -0.05],
        ])

        if annotate:
            for i in range(3):
                ax.annotate(f"$\\alpha_{i}$", _A.points[i] + _A_offsets[i])
                ax.annotate(f"$\\beta_{i}$", _B.points[i] + _B_offsets[i], color="blue")

def get_triv(A: Triangle, B: Triangle) -> Tuple[PlanarTriangle, PlanarTriangle]:
    _A = A.trivial_replication([0, 0], [1, 0])
    _B = B.trivial_replication([0, 0], [1, 0])
    return _A, _B

def plot_triv(fig: plt.Figure, ax: plt.Axes,
              A: Triangle, B: Triangle, dist: bool = True):
    _A, _B = get_triv(A, B)
    _A.plot(fig, ax, "black")
    _B.plot(fig, ax, "blue")
    ax.scatter([0, 1], [0, 0], c="black")
    ax.annotate("$(0, 0)$", [-0.01, -0.02])
    ax.annotate("$(1, 0)$", [1, -0.02])
    
    _A.plot(fig, ax, color="black")
    _A_offsets = np.array([
        [0.07, 0.02],
        [-0.05, 0.03],
        [-0.02, -0.04],
    ])
    _B.plot(fig, ax, color="blue")
    _B_offsets = np.array([
        [0.05, 0.0125],
        [-0.03, 0.01],
        [-0.02, -0.04],
    ])
    ax.annotate("$t$", _A.points[2] + np.array([-0.03, -0.005]))
    ax.annotate("$t^\prime$", _B.points[2] + np.array([-0.025, 0.01]))
    for i in range(3):
        ax.annotate(f"$\\alpha_{i}$", _A.points[i] + _A_offsets[i])
        ax.annotate(f"$\\beta_{i}$", _B.points[i] + _B_offsets[i], color="blue")


    if dist:
        ax.add_collection(
            LineCollection(
                [[_A.points[2], _B.points[2]]], 
                color="black", linestyles="--"
            )
        )
        ax.scatter(*_A.points[2], c="black")
        ax.scatter(*_B.points[2], c="black")

def main():
    fig: plt.Figure
    ax: plt.Axes
    alpha, beta = 45, 65
    A = Triangle(np.sort([alpha, beta, 180 - alpha - beta]), deg=True)
    B = Triangle(np.sort([30, 60, 90]), deg=True)
    pad = 0.02
    

    savepath = thisdir.joinpath("metric")
    plot_tris(A, B, savepath, pad, annotate=False)
    plot_tris(A, B, savepath, pad, annotate=True)

    _A, _B = get_tris(A, B)
    _plot = partial(plot, savepath, 0-pad, 1+pad, 0-pad, 
                    max(_A.points[2][1], _B.points[2][1])+pad)

    # Fix to (0, 0) (1, 0)
    with _plot() as (fig, ax):
        plot_triv(fig, ax, A, B, dist=False)

    # Fix to (0, 0) (1, 0)
    with _plot() as (fig, ax):
        plot_triv(fig, ax, A, B)
    

if __name__ == "__main__":
    main()