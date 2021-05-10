import subprocess

from pafo.triangle import Triangle, PlanarTriangle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import LineCollection
import shutil
import pathlib

import argparse
import numpy as np

thisdir = pathlib.Path(__file__).resolve().parent


def visualize_replications(mode: str, savepath: str) -> None:
    alpha, beta = 37, 52
    tri = Triangle([alpha, beta, 180 - alpha - beta], deg=True)
    anchors, radius = ([0, 0], [1, 0]), 0.1
    trivial = tri.trivial_replication(*anchors)
    if mode == "machine":
        rep_center, rep_radius = tri.replication_machine_circle(*anchors, radius)
        reps = tri.replication_machine(*anchors, radius, num=15)
    else:
        rep_center, rep_radius = tri.replication_spanner_circle(*anchors, radius)
        reps = tri.replication_spanner(*anchors, radius, num=10)

    savepath = pathlib.Path(savepath).resolve().with_suffix("")
    savepath.mkdir(parents=True)
    for i, rep in enumerate(reps[:-1]):
        # set up figure
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots()
        ax.axis("square")
        pad = 0.01
        ax.set_xlim(0-radius-pad, 1+radius+pad)
        ax.set_ylim(0-radius-pad, trivial.points[2][1]+rep_radius+pad)

        # ax.scatter(*(rep.points.T), c="blue")
        rep.plot(fig, ax, color="black")    # plot replication
        trivial.plot(fig, ax)
        
        ax.scatter(*(trivial.points.T), c="black")
        if mode == "spanner":
            ax.add_patch(Circle([0, 0], radius, color="black", fill=None))
        ax.add_patch(Circle([1, 0], radius, color="black", fill=None))
        ax.add_patch(Circle(rep_center, rep_radius, color="blue", fill=None, linestyle="--"))

        ax.annotate("$u$", trivial.points[0] - 0.03)
        ax.annotate("$v$", trivial.points[1] + 0.01)
        ax.annotate("$c$", trivial.points[2] + 0.015)

        lines = [
            [trivial.points[1], trivial.points[1] - np.array([0, radius])]
        ]
        if mode == "spanner":
            lines.append([
                trivial.points[0], trivial.points[0] - np.array([0, radius])
            ])

        ax.add_collection(LineCollection(lines, color="black", linestyles="--"))
        loc = np.array([lines[0][0][0]+0.01, (lines[0][0][1] + lines[0][1][1]) / 2])
        ax.annotate("r", loc)

        plt.axis("off")
        plt.tight_layout()
        fig.savefig(str(savepath.joinpath(f"frame-{i}.png")), transparent=True, 
                    bbox_inches="tight", dpi=240)
        plt.close(fig)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", required=True, metavar="PATH",
                        help="Save the animation. If suffix not .gif, saves as a directory of PNGs")
    parser.add_argument("--mode", choices=["machine", "spanner"], help="Plot replication machine or spanner")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    visualize_replications(args.mode, args.save)
