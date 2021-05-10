from pafo.convex import convex_solution
from pafo.triangle import Triangle, PlanarTriangle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

def main():
    R = np.array([
        [0, 0], [1, 0],
        [0, 2], [1.5, 2]
    ])

    P = np.array([
        [0, 0], [1, 0],
        [0, 1], [1, 1]
    ])

    Q, r, _ = convex_solution(R, P)

    ax: plt.Axes
    fig, ax = plt.subplots()

    R_scat = ax.scatter(*R.T, c="black")
    Q_scat = ax.scatter(*Q.T, c="green")

    vecs = Q - R

    arrows = [FancyArrow(0, 0, *vec, color="black", head_width=0.02) for vec in vecs]
    arrows.append(
        FancyArrow(0, 0, *(vecs[0] + vecs[1]), color="green", head_width=0.02)
    )
    for arrow in arrows:
        ax.add_patch(arrow)

    for p in P[2:]:
        tri = Triangle(PlanarTriangle([*P[:2], p]).angles)
        c = tri.trivial_replication_point(*P[:2])
        ax.scatter(*c, color="blue")



    ax.axis("square")
    plt.show()

if __name__ == "__main__":
    main()