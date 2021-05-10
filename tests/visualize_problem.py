from pafo.triangle import Triangle, PlanarTriangle
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.collections import  PatchCollection
import numpy as np
import pathlib

from typing import Tuple

thisdir = pathlib.Path(__file__).resolve().parent

def plot(src: PlanarTriangle, dst: PlanarTriangle, color="black") -> Tuple[plt.Figure, plt.Axes]:
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    ax.set_xlim([0.1, .9])
    ax.set_ylim([0.1, .8])
    ax.axis("square")

    ax.scatter(*(src.points.T), c="black")
    dst.plot(fig, ax, color=color)

    patches = PatchCollection([
        FancyArrow(*s, *(d - s), length_includes_head=True, width=0.005, head_width=0.025, color="k")
        for s, d in zip(src.points, dst.points)
    ], edgecolors="k", facecolors="k")
    ax.add_collection(patches)

    plt.axis("off")
    return fig, ax

def visualize_problem():
    tri = PlanarTriangle([
        [0.2, 0.5],
        [0.4, 0.1],
        [0.6, 0.7],
    ])
    eq = Triangle.equilateral()

    ps = np.array([
        np.random.random(5) * 0.3 + 0.1,
        np.random.random(5) * 0.3 + 0.1
    ], dtype=np.float64).T
    qs = np.array([
        np.random.random(5) * 0.3 + 0.5,
        np.random.random(5) * 0.3 + 0.1
    ], dtype=np.float64).T

    savedir = thisdir.joinpath("visualize_problem")
    savedir.mkdir(exist_ok=True, parents=True)
    for i, (p, q) in enumerate(zip(ps, qs)):
        rep = eq.trivial_replication(p, q)
        fig, ax = plot(tri, rep)
        fig.savefig(str(savedir.joinpath(f"frame_{i}.png")), transparent=True)

    sol = tri.min_max_traversal_triangle(eq)
    fig, ax = plot(tri, sol, "green")
    fig.savefig(str(savedir.joinpath(f"sol.png")), transparent=True)


if __name__ == "__main__":
    visualize_problem()