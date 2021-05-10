from pafo.triangle import Triangle, PlanarTriangle
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Circle
import numpy as np
from typing import Tuple, List
from contextlib import contextmanager
from functools import partial
import pathlib

thisdir = pathlib.Path(__file__).resolve().parent


@contextmanager
def plot(savepath, xmin, xmax, ymin, ymax, frame=[1]) -> Tuple[plt.Figure, plt.Axes]:
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(11, 9))
    yield fig, ax
    ax.axis("square")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.axis("off")
    plt.draw()
    plt.tight_layout()

    if savepath:
        savepath = pathlib.Path(savepath).resolve()
        savepath.mkdir(exist_ok=True, parents=True)
        plt.savefig(str(savepath.joinpath(f"frame-{frame[0]}.png")), transparent=True, 
                    bbox_inches="tight", pad_inches=0, dpi=240)
        frame[0] += 1
    else:
        plt.show()
    plt.close(fig)

def plot_base(fig: plt.Figure, ax: plt.Axes, 
              R: PlanarTriangle, P_ref: PlanarTriangle,
              annotate_points: bool = True, annotate_sides: bool = True, 
              scale: bool = True) -> None:
    

    R.plot(fig, ax)
    ax.scatter(*(R.points.T), c="black")
    P_ref.plot(fig, ax, color="blue")

    if annotate_points:
        ax.annotate(f"$r_0$", R.points[0]-0.035)
        ax.annotate(f"$r_1$", R.points[1]+0.015)
        ax.annotate(f"$r_2$", R.points[2]+0.015)
        
        labels = np.array([f"$p_{i}$" for i in range(3)])[np.argsort(P_ref.angles)]
        ax.scatter(*(P_ref.points.T), c="blue")
        ax.annotate(labels[0], P_ref.points[0]-0.035)
        ax.annotate(labels[1], P_ref.points[1]+0.015)
        ax.annotate(labels[2], P_ref.points[2]+0.015)

    if annotate_sides:
        midpoints = (P_ref.points + np.roll(P_ref.points, -1, axis=0)) / 2
        labels = ["\sqrt{3}", "1", "2"]
        if scale:
            offsets = np.array([
                [-0.05, -0.05],
                [0.01, 0.0225],
                [-0.075, 0.025],
            ])
            labels = ["$\\frac{" + label + "}{3 + \sqrt{3}}$" for label in labels]
        else:
            offsets = np.array([
                [-0.05, -0.05],
                [0.01, 0.0225],
                [-0.01, 0.025],
            ])
            labels = [f"${label}$" for label in labels]

        for i in range(3):
            ax.annotate(labels[i], midpoints[i] + offsets[i])


def get_triv(R: PlanarTriangle, P: Triangle, i: int) -> PlanarTriangle:
    j, k = (i+1)%3, (i+2)%3
    return P.roll(-(i+1)).trivial_replication(R.points[j], R.points[k])

def plot_triv(fig: plt.Figure, ax: plt.Axes, 
              R: PlanarTriangle, P: Triangle, i: int,
              annotate: bool = False) -> None:
    triv = get_triv(R, P, i)
    ax.scatter(*triv.points[2], color="blue")
    triv.plot(fig, ax, "blue")

    offsets = np.array([
        [-0.05, 0],
        [0.015, 0],
        [0.015, 0]
    ])
    ax.annotate(f"$t_{i}$", triv.points[2] + offsets[i])

def get_span_circle(R: PlanarTriangle, P: Triangle, i: int) -> Tuple[np.ndarray, float]:
    j, k = (i+1)%3, (i+2)%3
    r = R.min_max_traversal(P)
    return P.roll(-(i+1)).replication_spanner_circle(R.points[j], R.points[k], r)

def plot_span_circle(fig: plt.Figure, ax: plt.Axes, 
                     R: PlanarTriangle, P: Triangle, i: int) -> None:
    c, r = get_span_circle(R, P, i)
    circle = Circle(c, r, linestyle="--", fill=None)
    ax.add_patch(circle)

def plot_radius_circle(fig: plt.Figure, ax: plt.Axes, 
                       R: PlanarTriangle, P: Triangle, i: int):
    sol = R.min_max_traversal_triangle(P)
    circle = Circle(R.points[i], np.linalg.norm(sol.points[i] - R.points[i]), linestyle="--", fill=None)
    ax.add_patch(circle)

def plot_arrow(fig: plt.Figure, ax: plt.Axes, 
               R: PlanarTriangle, P: Triangle, i: int, scale: bool):
    triv = get_triv(R, P, i)
    length = triv.points[2] - R.points[i]
    if scale:
        length *= P.sides[(i+1)%3]
    arrow = FancyArrow(*R.points[i], *length, length_includes_head=True,
                        head_width=0.035, color="black")
    ax.add_patch(arrow)

def plot_sol(fig: plt.Figure, ax: plt.Axes, 
             R: PlanarTriangle, P: Triangle) -> None:
    R.min_max_traversal_triangle(P).plot(fig, ax, color="green")  

def plot_sol_points(fig: plt.Figure, ax: plt.Axes,
                    R: PlanarTriangle, P: Triangle, points: List[int] = [0, 1, 2]) -> None:
    sol = R.min_max_traversal_triangle(P)
    label_offsets = np.array([
        [-0.05, -0.01],
        [-0.02, 0.02],
        [0.015, -0.0075]
    ])
    for point in points:
        ax.scatter(*sol.points[point], c="green")
        ax.annotate(f"$q_{point}$", sol.points[point] + label_offsets[point])

def main():
    fig: plt.Figure
    ax: plt.Axes

    R = PlanarTriangle([
        [0, 0], [1, 0], [0.6, 0.6]
    ])
    P = Triangle([30, 60, 90], deg=True)
    Q = R.min_max_traversal_triangle(P)
    p_u, p_v = [1.2, 0.5], [1.6, 0.5]
    pad = 0.02

    r = R.min_max_traversal(P)
    xmins, ymins = [p_u[0] - pad, p_v[0] + pad], [p_u[1] - pad, p_v[1] + pad]
    xmaxs, ymaxs = [p_u[0] - pad, p_v[0] + pad], [p_u[1] - pad, p_v[1] + pad]
    for i in range(3):
        c, r = get_span_circle(R, Triangle(np.sort(P.angles)), i)
        xmins.append(c[0] - r - pad)
        xmaxs.append(c[0] + r + pad)
        ymins.append(c[1] - r - pad)
        ymaxs.append(c[1] + r + pad)

        xmins.append(R.points[i][0] - r - pad)
        xmaxs.append(R.points[i][0] + r + pad)
        ymins.append(R.points[i][1] - r - pad)
        ymaxs.append(R.points[i][1] + r + pad)

    xmin = min(xmins)
    xmax = max(xmaxs)
    ymin = min(ymins)
    ymax = max(ymaxs)

    P_ref = P.trivial_replication(p_u, p_v)
    savepath = thisdir.joinpath("visualize_solution")
    _plot = partial(plot, savepath, xmin, xmax, ymin, ymax)
    
    # Show robots & Pattern
    with _plot() as (fig, ax):
        plot_base(fig, ax, R, P_ref, annotate_points=False, annotate_sides=True, scale=False)

    _plot_base = partial(plot_base, R=R, P_ref=P_ref, 
                         annotate_points=True, annotate_sides=True, scale=True)

    # robots are sorted by angles
    assert((np.diff(R.angles) >= 0).all())

    # Show robots & Pattern w/ annotations
    with _plot() as (fig, ax):
        _plot_base(fig, ax)

    # get correct assignment
    P = Triangle(np.sort(P.angles)) 

    # Show robots & Pattern with annotations
    for i in range(3):
        with _plot() as (fig, ax):
            _plot_base(fig, ax)
            plot_triv(fig, ax, R, P, i)
            plot_arrow(fig, ax, R, P, i, scale=False)
            plot_sol_points(fig, ax, R, P, list(range(i)))
        
        with _plot() as (fig, ax):
            _plot_base(fig, ax)
            plot_triv(fig, ax, R, P, i)
            plot_arrow(fig, ax, R, P, i, scale=True)
            plot_sol_points(fig, ax, R, P, list(range(i+1)))
        
    with _plot() as (fig, ax):
        _plot_base(fig, ax)
        plot_sol_points(fig, ax, R, P, list(range(i+1)))
        for i in range(3):
            plot_arrow(fig, ax, R, P, i, scale=True)
        plot_sol(fig, ax, R, P)

    # Show radius circles
    with _plot() as (fig, ax):
        _plot_base(fig, ax)
        plot_sol_points(fig, ax, R, P, list(range(3)))
        plot_sol(fig, ax, R, P)
        for i in range(3):
            plot_radius_circle(fig, ax, R, P, i)
            
    # Show spanner circles
    for i in range(3):
        with _plot() as (fig, ax):
            _plot_base(fig, ax)
            plot_sol_points(fig, ax, R, P, list(range(3)))
            plot_sol(fig, ax, R, P)

            plot_triv(fig, ax, R, P, i, annotate=True)
            plot_span_circle(fig, ax, R, P, i)
            for j in range(3):
                plot_radius_circle(fig, ax, R, P, j)


if __name__ == "__main__":
    main()