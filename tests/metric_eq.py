from pafo.triangle import Triangle, PlanarTriangle
import matplotlib.pyplot as plt
from typing import List, Tuple, Iterable
import string
import pathlib

thisdir = pathlib.Path(__file__).resolve().parent

def plot_grid(savepath: pathlib.Path, tris: List[PlanarTriangle], 
              dim: Tuple[int, int], labels: Iterable[str], frame=[0]) -> None:
    fig, axs = plt.subplots(*dim, figsize=(11, 9))
    for i, (ax, tri) in enumerate(zip(axs.flatten(), tris)):
        tri.plot(fig, ax)
        ax.axis("square")
        ax.axis("off")
        if labels:
            ax.set_title(labels[i])

    plt.tight_layout()
    plt.draw()

    if savepath:
        savepath = pathlib.Path(savepath).resolve()
        savepath.mkdir(exist_ok=True, parents=True)
        plt.savefig(str(savepath.joinpath(f"frame-{frame[0]}.png")), transparent=True, 
                    bbox_inches="tight", pad_inches=0, dpi=240)
        frame[0] += 1
    else:
        plt.show()

def main():
    savedir = thisdir.joinpath("metric_eq")
    eq = Triangle.equilateral()
    dim = (4, 6)
    num = dim[0] * dim[1]
    labels = list(string.ascii_uppercase[:num])
    tris = [Triangle.random() for i in range(num)]
    plot_grid(savedir, tris, dim, labels)

    labels, sorted_tris = zip(*sorted(zip(labels, tris), 
                                      key=lambda x: eq.distance(x[1]), 
                                      reverse=True))
    labels=[f"$\\tau({label}, EQ) = {eq.distance(tri):.2f}$" 
            for label, tri in zip(labels, sorted_tris)]
    plot_grid(savedir, sorted_tris, dim, labels)


if __name__ == "__main__":
    main()