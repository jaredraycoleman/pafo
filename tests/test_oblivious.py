import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection

from pafo.triangle import Triangle, PlanarTriangle


def test_oblivious(speed: float = 0.01):
    eq: Triangle = Triangle.equilateral()
    tri: PlanarTriangle = PlanarTriangle.random()
    dst: PlanarTriangle = tri.min_max_traversal_triangle(eq)
    unit = dst.points - tri.points
    unit /= np.linalg.norm(unit)

    def next_dst() -> PlanarTriangle:
        nonlocal tri, unit, dst
        _dst: PlanarTriangle = tri.min_max_traversal_triangle(eq)
        if not np.isclose(dst.points, _dst.points).all():
            dst = _dst
            unit = dst.points - tri.points
            unit /= np.linalg.norm(unit)
        tri = PlanarTriangle(tri.points + unit * speed)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    ax.axis("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Initialize
    next_dst()
    tri_scatter = ax.scatter(*dst.points.T)
    dst_scatter = ax.scatter(*tri.points.T, c='green')

    def update_frame(frame):
        next_dst()
        tri_scatter.set_offsets(tri.points)
        dst_scatter.set_offsets(dst.points)

        return tri_scatter, dst_scatter

    ani = FuncAnimation(fig, update_frame, frames=None,
                        blit=True, repeat=False, interval=1,
                        cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    test_oblivious()
