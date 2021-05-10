from cvxopt import matrix, solvers
import numpy as np
from itertools import permutations, combinations, chain
import math

from typing import List

solvers.options['show_progress'] = False


def format_eqs(lefts: List[List[float]], right: List[float]):
    """Formats equations as a Matrix

    Args:
        lefts: left-hand sides of equations
        right: right-hand sides of equations

    Returns:
        Matrix form of equations
    """
    *xs, cs = zip(*([right] + lefts))
    mat = []
    for elems in xs:
        mat.append([float(-elem) for elem in elems])

    return matrix(mat), matrix(cs)


def convex_solution_no_perm(positions: np.array, formation: np.array) -> np.array:
    """Uses Convex Optimization to solve Min-Max Pattern Formation

    Finds solution for the trivial reflection/assignment
    See Paper: https://academic.microsoft.com/paper/2025764057

    Args:
        positions: Initial Robot Positions
        formation: Formation to form

    Returns:
        Destinations similar to formation that minimize the maximum \
            distance any robot with initial positions "positions" \
            must travel    
    """
    positions = np.copy(positions)
    formation = np.copy(formation) - formation[0]  # adjust formation to origin
    angle = -math.atan2(formation[1][1], formation[1][0])
    xs, ys = formation[:, 0].copy(), formation[:, 1].copy()
    formation[:, 0] = math.cos(angle) * xs - math.sin(angle) * ys
    formation[:, 1] = math.sin(angle) * xs + math.cos(angle) * ys

    assert (positions.shape == formation.shape)

    s2norm = np.linalg.norm(formation[1])
    n = len(formation)

    # | q_i.x - p_i.x |    
    # |               |     <= r
    # | q_i.y - p_i.y |_2 
    # 
    # such that 
    # Aq = 0    <---    this says the shape is similar

    Gs = []
    hs = []

    eqs = np.append(np.eye(2 * n), np.zeros((2 * n, 2)), axis=1)
    eqs[:, -1][::2] = -positions[:, 0]
    eqs[:, -1][1::2] = -positions[:, 1]

    eq_r = np.zeros(2 * n + 2)
    eq_r[-2] = 1.0
    for i in range(n):
        G, h = format_eqs(
            [eqs[2 * i].tolist(), eqs[2 * i + 1].tolist()],
            eq_r
        )
        Gs.append(G)
        hs.append(h)

    # A has 2 (n-2) equations with 2n + 1 variables 
    A = np.zeros((2 * (n - 2), 2 * n + 1), dtype=float)
    b = np.zeros(2 * (n - 2), dtype=float)
    for i in range(2, n):
        x_idx = 2 * i
        A[x_idx - 4][0] = formation[i][0] - s2norm  # q_1.x
        A[x_idx - 4][1] = -formation[i][1]  # q_1.y
        A[x_idx - 4][2] = -formation[i][0]  # q_2.x
        A[x_idx - 4][3] = formation[i][1]  # q_2.y

        A[x_idx - 4][x_idx] = s2norm  # q_i.x

        y_idx = x_idx + 1
        A[y_idx - 4][0] = formation[i][1]  # q_1.x
        A[y_idx - 4][1] = formation[i][0] - s2norm  # q_1.y
        A[y_idx - 4][2] = -formation[i][1]  # q_2.x
        A[y_idx - 4][3] = -formation[i][0]  # q_2.y

        A[y_idx - 4][y_idx] = s2norm  # q_i.y

    # c is what we are trying to minimize
    #  we are trying to minimize t (the last element of (q, t))
    c = np.zeros(2 * n + 1)
    c[-1] = 1

    # Shape/data-type conversions for optimization function
    c = matrix(c.tolist())
    A = matrix(A.T.tolist())
    b = matrix([b.tolist()])

    sol = solvers.socp(c, A=A, b=b, Gq=Gs, hq=hs)  # second-order cone programming
    return np.array(sol['x'][:-1]).reshape((n, 2)), sol['x'][-1]


def convex_solution(positions: np.array, formation: np.array) -> np.array:
    """Uses Convex Optimization to solve Min-Max Pattern Formation

    Tests all reflections/assignments
    See Paper: https://academic.microsoft.com/paper/2025764057

    Args:
        positions: Initial Robot Positions
        formation: Formation to form

    Returns:
        Destinations similar to formation that minimize the maximum \
            distance any robot with initial positions "positions" \
            must travel    
    """
    positions = np.copy(positions)
    formation = np.copy(formation) - formation[0]
    ran = np.arange(len(positions))
    perm, target, radius = ran.copy(), None, None
    for _perm in map(np.array, permutations(ran)):
        _target, _radius = convex_solution_no_perm(positions[_perm], formation)
        if radius is None or _radius < radius:
            perm[_perm], target, radius = ran.copy(), _target, _radius

    return target[perm], radius, perm
