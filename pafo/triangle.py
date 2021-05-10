import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Iterable, List, Optional, Tuple


class Triangle:
    """Represents a triangle object

    A Triangle is defined by its interior angles, which should sum to pi
    """

    def __init__(self, angles: Iterable[float], deg: bool = False,
                 reflected: bool = False) -> None:
        """Constructor for a triangle
        
        Args:
            angles: Interior angles of triangle.
        
        Keyword Args:
            deg: If true, angles are in degrees, otherwise they are assumed to be in radians (default: False)
            reflected: If True, the "negative" reflection is given for replications into the plane
        
        Raises:
            ValueError: If angles cannot be cast to a numpy array of three elements \
                or if the sum of the angles is not equal to pi (or 180 degrees)
        """
        try:
            self.angles = np.array(angles, dtype=np.float64)
            assert (self.angles.shape == (3,))
        except (AssertionError, ValueError):
            raise ValueError("angles must be castable to a 3x1 numpy array")

        if deg:
            self.angles *= (math.pi / 180.0)

        if not np.isclose(math.pi, np.sum(self.angles)):
            raise ValueError(f"Angles must add up to pi, not {np.sum(self.angles) / math.pi}*pi.")

        self.reflected = reflected

    @classmethod
    def equilateral(cls) -> "Triangle":
        return Triangle([math.pi / 3] * 3)

    @classmethod
    def isosceles(cls, angle: float, deg: bool = False) -> "Triangle":
        angle = angle * math.pi / 180 if deg else angle
        return Triangle([angle, angle, math.pi - 2 * angle])

    @classmethod
    def isosceles_right(cls) -> "Triangle":
        """Generates an isosceles right triangle

        Returns:
            A Triangle object with [90, 45, 45]
        """
        return Triangle([math.pi / 4, math.pi / 4, math.pi / 2])

    @classmethod
    def random(cls):
        angles = np.random.random(3)
        return Triangle(angles / np.sum(angles) * math.pi)

    @property
    def reflection(self) -> "Triangle":
        """Gets reflection of triangle

        Returns:
            Triangle: reflection of triangle with the same angles, but produces replications that are reflections \
                of each other
        """
        return Triangle(self.angles, deg=False, reflected=(not self.reflected))

    def __copy__(self) -> "Triangle":
        """Copy the triangle

        Returns:
            Triangle: triangle with same angles and handedness
        """
        return Triangle(self.angles.copy(), deg=False, reflected=self.reflected)

    def __str__(self):
        """Returns string representation of Triangle"""
        angle_str = [f"{angle / math.pi:.2f}pi" for angle in self.angles]
        return f"Triangle: [{', '.join(angle_str)}]"

    def trivial_replication(self, u: np.ndarray, v: np.ndarray) -> "PlanarTriangle":
        """Gets the trivial replication of a triangle on (u, v)

        Args:
            u: first point to replicate on
            v: second point to replicate on

        Raises:
            ValueError: if u or v cannot be cast to 2-element numpy arrays

        Returns:
            PlanarTriangle: Triangle embedded into the plane
        """
        try:
            u = np.array(u, dtype=np.float64)
            v = np.array(v, dtype=np.float64)
            assert (u.shape == (2,) and v.shape == (2,))
        except (ValueError, AssertionError):
            raise ValueError("u and v must be castable to 2x1 numpy arrays")

        return PlanarTriangle([u, v, self.trivial_replication_point(u, v)])

    def trivial_replication_point(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Gets the trivial replication point of the trivial replication \
            of this triangle on (u, v)
        
        Args:
            u: first point to replicate on
            v: second point to replicate on

        Raises:
            ValueError: if u or v cannot be cast to 2-element numpy arrays
        
        Returns:
            np.ndarray: Trivial Replication Point of this triangle on (u, v)
        """
        try:
            u = np.array(u, dtype=np.float64)
            v = np.array(v, dtype=np.float64)
            assert (u.shape == (2,) and v.shape == (2,))
        except (ValueError, AssertionError):
            raise ValueError("u and v must be castable to 2x1 numpy arrays")

        vec = v - u # uv vector
        k = (np.linalg.norm(vec) / self.sides[0]) * self.sides[2] # scale of triangle

        vec /= np.linalg.norm(vec) # uv-norm
        point = np.array([ # rotate uv-norm
            vec[0] * math.cos(self.angles[0]) - vec[1] * math.sin(self.angles[0]),
            vec[0] * math.sin(self.angles[0]) + vec[1] * math.cos(self.angles[0])
        ])
        point = (point * k) + u

        return point

    def replication_machine(self, u: np.ndarray, v: np.ndarray,
                            r: float, num: int = 10) -> List["PlanarTriangle"]:
        """Gets the replication machine for a triangle on (u, C(v, r))

        Args:
            u: first point to replicate on
            v: center of circle to replicate on
            r: radius of circle to replicate on
            num: number of trivial replications to generate
        
        Raises:
            ValueError: If u or v cannot be cast to 2-element numpy arrays
        
        Returns:
            List[PlanarTriangle]: List of PlanarTriangle objects in the Replication \
                Machine of this triangle on (u, C(v, r))
        """
        try:
            u = np.array(u, dtype=np.float64)
            v = np.array(v, dtype=np.float64)
            assert (u.shape == (2,) and v.shape == (2,))
        except (ValueError, AssertionError):
            raise ValueError("u and v must be castable to 2x1 numpy arrays")

        return [
            self.trivial_replication(
                u, [math.cos(_v)*r + v[0], math.sin(_v)*r + v[1]]
            )
            for _v in np.linspace(0, 2 * math.pi, num=num)
        ]

    def replication_machine_circle(self, u: np.ndarray, v: np.ndarray, r: float) -> Tuple[np.ndarray, float]:
        """Gets the replication machine circle of this triangle on (u, C(v, r))
        
        Args:
            u: first point to replicate on
            v: center of circle to replicate on
            r: radius of circle to replicate on
        
        Returns:
            Tuple[np.ndarray, float]: center point and radius of replication machine circle
        """
        try:
            u = np.array(u, dtype=np.float64)
            v = np.array(v, dtype=np.float64)
            assert (u.shape == (2,) and v.shape == (2,))
        except (ValueError, AssertionError):
            raise ValueError("u and v must be castable to 2x1 numpy arrays")

        center = self.trivial_replication_point(u, v)
        radius = r * np.linalg.norm(u - center) / np.linalg.norm(u - v)
        return center, radius

    def replication_spanner(self, u: np.ndarray, v: np.ndarray,
                            r: float, num: int = 10) -> List["PlanarTriangle"]:
        """Gets the replication spanner for a triangle on (C(u, r), C(v, r))

        Args:
            u: first point to replicate on
            v: center of circle to replicate on
            r: radius of circle to replicate on
            num: square root of the number of trivial replciations to generate. \
                default is 10, which means 100 trivial replications will be generated.
                
        Returns:
            List[PlanarTriangle]: List of PlanarTriangle objects in the Replication \
                Spanner of this triangle on (C(u, r), C(v, r))
        """
        try:
            u = np.array(u, dtype=np.float64)
            v = np.array(v, dtype=np.float64)
            assert (u.shape == (2,) and v.shape == (2,))
        except (ValueError, AssertionError):
            raise ValueError("u and v must be castable to 2x1 numpy arrays")

        return [
            self.trivial_replication(
                [np.cos(_u)*r + u[0], np.sin(_u)*r + u[1]],
                [np.cos(_v)*r + v[0], np.sin(_v)*r + v[1]]
            )
            for _u in np.linspace(0, 2 * math.pi, num=num)
            for _v in np.linspace(0, 2 * math.pi, num=num)
        ]

    def replication_spanner_circle(self, u: np.ndarray, v: np.ndarray, r: float) -> Tuple[np.ndarray, float]:
        """Gets the replication spanner circle of this triangle on (u, C(v, r))
        
        Args:
            u: center of first circle to replicate on
            v: center of second circle to replicate on
            r: radius of circle to replicate on
        
        Returns:
            Tuple[np.ndarray, float]: center point and radius of replication spanner cirlce
        """
        try:
            u = np.array(u, dtype=np.float64)
            v = np.array(v, dtype=np.float64)
            assert (u.shape == (2,) and v.shape == (2,))
        except (ValueError, AssertionError):
            raise ValueError("u and v must be castable to 2x1 numpy arrays")

        center = self.trivial_replication_point(u, v)
        radius = r * (np.linalg.norm(u - center) + np.linalg.norm(v - center)) / np.linalg.norm(u - v)
        return center, radius

    def distance(self, triangle: "Triangle") -> float:
        """Distance from one triangle to another

        This distance is proportional to the min-max traversal pattern formation solution.

        Args:
            triangle: triangle to get distance from
        """
        self_angles = np.sort(self.angles)
        tri_angles = np.sort(triangle.angles)
        # if self_angles[0] > tri_angles[0]:
        #     self_angles, tri_angles = tri_angles, self_angles
        self_sin = np.sin(self_angles)
        tri_sin = np.sin(tri_angles)
        return math.sqrt(
            (self_sin[1] / self_sin[2]) ** 2 + (tri_sin[1] / tri_sin[2]) ** 2 -
            2 * (self_sin[1] * tri_sin[1]) / (self_sin[2] * tri_sin[2]) * math.cos(self_angles[0] - tri_angles[0])
        )

    def roll(self, shift: int = 1) -> "Triangle":
        """Shift angles

        Args:
            shift: amount to shift angles

        Returns:
            Triangle: new triangle object with angles shifted
        """
        return Triangle(np.roll(self.angles, shift), deg=False, reflected=self.reflected)

    @property
    def sides(self) -> np.ndarray:
        """Get proportional side lengths of triangle

        This is equivalent to the side lengths of the triangle with a perimeter of 1

        Returns:
            three-element numpy array representing the side-lengths of the triangle
        """
        sin = np.sin(self.angles)
        sides = np.array([
            1,
            sin[0] / sin[2],
            sin[1] / sin[2],
        ], dtype=np.float64)
        return sides / np.sum(sides)

    def plot(self, fig: Optional[plt.Figure] = None,
             ax: Optional[plt.Axes] = None,
             color: str = 'black') -> Tuple[plt.Figure, plt.Axes]:
        return self.trivial_replication([0, 0], [1, 0]).plot(fig, ax, color)


class PlanarTriangle:
    """Represents a Triangle embedded in the plane

    Raises:
        ValueError: If the numpy arrays passed as Args do not match the correct shape.
    """

    def __init__(self, points: np.ndarray) -> None:
        """Constructor for a triangle embedded into the plane
        
        Args:
            points: points of triangle
        
        Raises:
            ValueError: Raised if points is not castable to a 3x2 numpy array
        """
        try:
            self.points = np.array(points, dtype=np.float64)
            assert (self.points.shape == (3, 2))
        except (AssertionError, ValueError):
            raise ValueError("angles must be castable to a 3x2 numpy array.")

    @classmethod
    def random(cls,
               x_range: Tuple[float, float] = (0.0, 1.0),
               y_range: Tuple[float, float] = (0.0, 1.0)) -> "PlanarTriangle":
        try:
            assert (len(x_range) == 2 and len(y_range) == 2)
            x_range = (float(x_range[0]), float(x_range[1]))
            y_range = (float(y_range[0]), float(y_range[1]))
        except (AssertionError, ValueError):
            raise ValueError(f"x_range and y_range must be castable to 2-tuples of floats. i.e. (low, high)")

        points = np.array([
            np.random.random(3) * (x_range[1] - x_range[0]) + x_range[0],
            np.random.random(3) * (y_range[1] - y_range[0]) + y_range[0]
        ], dtype=np.float64).T

        return PlanarTriangle(points)

    @property
    def angles(self) -> np.ndarray:
        """Gets the interior angles of the triangle
        
        Returns:
            np.ndarray: interior angles of triangle
        """
        dists = np.linalg.norm(self.points - np.roll(self.points, -1, axis=0), axis=1)

        def alpha(i):
            return np.arccos((dists[i - 2] ** 2 - dists[i] ** 2 - dists[i - 1] ** 2) / (-2 * dists[i - 1] * dists[i]))
        return np.apply_along_axis(alpha, 0, np.arange(dists.size))

    @property
    def perimeter(self) -> float:
        """Gets perimeter of triangle
        
        Returns:
            float: perimeter of the triangle
        """
        return np.sum(np.linalg.norm(self.points - np.roll(self.points, 1, axis=0), axis=1))

    def _min_max_traversal(self, pattern: Triangle,
                           assignment: bool = True,
                           reflection: bool = True) -> Tuple[np.ndarray, np.ndarray, Triangle, float]:
        """Computes the min-max traversal distance from this triangle to the given pattern

        This is the private version of the method that returns the computed assignment/permutation, \
        permuted points, the correctly reflected pattern, and the min-max traversal distance.
        This method should not be used - use min_max_traversal instead.

        Args:
            pattern: pattern to compute min-max traversal distance to
            assignment: if True, compute correct assignment, otherwise compute the optimal solution for the \
                prescribed assignment (default: True)
            reflection: if True, compute correct reflection, otherwise compute the optimal solution for the \
                current reflection of pattern

        Returns:
            Tuple[np.ndarray, np.ndarray, Triangle, float]: tuple of (assignment/permutation, points, \
                reflected pattern, min-max traversal distance)
        """
        if assignment:
            perm = np.argsort(self.angles)
            points = self.points[perm]
            pattern = Triangle(np.sort(pattern.angles), reflected=pattern.reflected)

        # if reflection:
        #     _pattern = pattern.reflection
        #     t = pattern.trivial_replication_point(points[0], points[1])
        #     _t = _pattern.trivial_replication_point(points[0], points[1])
        #     if np.linalg.norm(_t - points[2]) < np.linalg.norm(t - points[2]):
        #         pattern = _pattern

        c = pattern.trivial_replication_point(points[0], points[1])
        dist = np.linalg.norm(points[2] - c) * pattern.sides[0]
        return perm, points, pattern, dist

    def min_max_traversal(self, pattern: Triangle,
                          assignment: bool = True,
                          reflection: bool = True) -> float:
        """Computes the min-max traversal distance from this triangle to the given pattern

        Args:
            pattern: pattern to compute min-max traversal distance to
            assignment: if True, compute correct assignment, otherwise compute the optimal solution for the \
                prescribed assignment (default: True)
            reflection: if True, compute correct reflection, otherwise compute the optimal solution for the \
                current reflection of pattern

        Returns:
            float: min-max traversal distance
        """
        *_, dist = self._min_max_traversal(pattern, assignment, reflection)
        return dist

    def min_max_traversal_triangle(self, pattern: Triangle,
                                   assignment: bool = True,
                                   reflection: bool = True) -> "PlanarTriangle":
        """Computes a min-max traversal formation from this triangle to the given pattern

        Note there may be more than one triangle that satisfies the min-max traversal distance.
        This method returns *a* valid solution.

        Args:
            pattern: pattern to compute min-max traversal to
            assignment: if True, compute correct assignment, otherwise compute the optimal solution for the \
                prescribed assignment (default: True)
            reflection: if True, compute correct reflection, otherwise compute the optimal solution for the \
                current reflection of pattern

        Returns:
            PlanarTriangle: min-max traversal triangle
        """
        perm, points, pattern, r = self._min_max_traversal(pattern, assignment, reflection)
        form = np.zeros_like(points)
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            aux = pattern.roll(-(i+1)).trivial_replication_point(points[j], points[k])
            form [i] = points[i] + (aux - points[i]) * pattern.sides[j]
        tri = PlanarTriangle(form)
        return tri

    def plot(self, fig: Optional[plt.Figure] = None,
             ax: Optional[plt.Axes] = None,
             color: str = 'black') -> Tuple[plt.Figure, plt.Axes]:
        """Plots the triangle onto the plane

        Uses matplotlib.pyplot to plot the triangle
        
        Keyword Args:
            fig: figure to plot on, if None and ax is None, a figure and axis will be generated (default: None)
            ax: axis to plot on, if None and ax is None, a figure and axis will be generated (default: None)
            color: color of triangle (default: black)
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: figure and axes that triangle was plotted on
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.add_patch(plt.Polygon(self.points, fill=None, color=color))
        return fig, ax


def test_replication():
    triangle = Triangle([90, 45, 45], deg=True)
    tri = triangle.trivial_replication((1, 3), (4, 9))
    fig, ax = tri.plot()
    ax.axis("square")
    plt.show()


def test_sides():
    triangle = Triangle([90, 45, 45], deg=True)
    i_right = np.array([math.sqrt(2), 1, 1])
    i_right /= np.sum(i_right)
    assert(np.isclose(np.sort(i_right), np.sort(triangle.sides)).all())


def test_solution():
    robots = PlanarTriangle.random()
    pattern = Triangle.equilateral()

    formation = robots.min_max_traversal_triangle(pattern)
    fig, (top, btm) = plt.subplots(2)

    robots.plot(fig, top, "blue")
    pattern.plot(fig, btm)
    formation.plot(fig, top, "red")

    top.axis("equal")
    btm.axis("equal")
    plt.show()
