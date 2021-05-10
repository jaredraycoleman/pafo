import numpy as np 
import math

import matplotlib.pyplot as plt

from functools import partial 
from itertools import permutations, combinations, product
from typing import Callable, Dict, Tuple, List
from helpers import to_angles
from shapes import Triangle

from plotters import *

def get_transform(v1: np.array, v2: np.array) -> float:
    norm_1 = np.linalg.norm(v1)
    norm_2 = np.linalg.norm(v2)
    ratio = min(1, max(-1, np.dot(v1, v2) / (norm_1 * norm_2)))
    angle = math.acos(ratio)
    scale = norm_2 / norm_1

    if np.linalg.det([v1.astype(np.float32), v2.astype(np.float32)]) > 0:
        angle = -angle

    return angle, scale

def replication(formation: np.array, 
                q_0: np.array, q_1: np.array, 
                i: int=0, j: int=1) -> np.array:
    """Returns the replicated shape.

    This returns the shape P similar to formation such that
    P's ith and jth points are fixed to q_0 and q_1, respectively.

    Args:
        formation (np.array): formation to replicate
        q_0: Point to fix ith point in target formation to
        q_1: Point to fix jth point in target formation to
        i: index of point in target formation to fix to q_0
        j: index of point in target formation to fix to q_1

    Returns:
        np.array: The replicated shape
    """
    formation = np.copy(formation)
    q_0 = np.copy(np.array(q_0, dtype=np.float32))
    q_1 = np.copy(np.array(q_1, dtype=np.float32))

    assert(len(formation.shape) == 2)   # two-dimensional
    assert(formation.shape[1] == 2)     # collection of two-dimensional points
    assert(q_0.shape == (2,))           # two-dimensional point
    assert(q_1.shape == (2,))           # two-dimensional point

    formation -= (formation[i] - q_0)   # shift to i to q_0

    offset = np.copy(formation[i])
    formation -= offset     # temp shift i to origin
    q_1 -= offset

    theta, scale = get_transform(formation[j], q_1)

    c = math.cos(-theta)
    s = math.sin(-theta)

    formation = np.array(
        [
            formation[:,0] * c - formation[:,1] * s,
            formation[:,0] * s + formation[:,1] * c
        ]
    ).T * scale + offset

    return formation

def replication_machine_circles(formation: np.array,  
                                q_0: np.array, q_1: np.array, r: float,
                                i: int=0, j: int=1):
    T = replication(formation, q_0, q_1, i, j)
    radiuses = r * np.linalg.norm(T - q_0, axis=1) / np.linalg.norm(q_1 - q_0)
    return T, radiuses

def replication_spanner_circles(formation: np.array,  
                                q_0: np.array, q_1: np.array, r: float,
                                i: int=0, j: int=1):
    T = replication(formation, q_0, q_1, i, j)
    numerator = np.linalg.norm(T - q_0, axis=1) + np.linalg.norm(T - q_1, axis=1)
    denominator = np.linalg.norm(q_1 - q_0)
    radiuses = r * numerator / denominator
    return T, radiuses

def replication_machine(formation: np.array,  
                        q_0: np.array, q_1: np.array, r: float,
                        num: int, i: int=0, j: int=1):
    angles = np.linspace(0, 2*math.pi, num=num)
    q_1s = np.array([np.cos(angles) * r + q_1[0], np.sin(angles) * r + q_1[1]]).T
    fun = partial(replication, formation, q_0, i=i, j=j)
    return np.apply_along_axis(fun, axis=1, arr=q_1s)

def replication_spanner(formation: np.array,  
                        q_0: np.array, q_1: np.array, r: float,
                        num_0: int, num_1: int,
                        i: int=0, j: int=1):
    shapes = []
    for angle_0 in np.linspace(0, 2*math.pi, num=num_0):
        _q_0 = q_0 + r * np.array([math.cos(angle_0), math.sin(angle_0)])
        for angle_1 in np.linspace(0, 2*math.pi, num=num_1):
            _q_1 = q_1 + r * np.array([math.cos(angle_1), math.sin(angle_1)])
            shapes.append(replication(formation, _q_0, _q_1, i, j))
    return shapes

normalize = lambda x: x / np.linalg.norm(x, axis=1)[:, np.newaxis]
dists = lambda x, y: np.linalg.norm(x - y, axis=1)
get_sides = lambda x: np.linalg.norm(x - np.roll(x, 1, axis=0), axis=1)

def triangle_solution_no_perm(positions: np.array, formation: np.array):
    perim = np.sum(np.linalg.norm(formation - np.roll(formation, -1, axis=0), axis=1))
    target = np.zeros_like(positions)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        repl = replication(formation, positions[j], positions[k], j, k)
        side = np.linalg.norm(formation[j] - formation[k]) 
        radius = np.linalg.norm(positions[i] - repl[i]) * side / perim

        direction = repl[i] - positions[i]
        direction = direction / np.linalg.norm(direction)
        target[i] = positions[i] + direction * radius
        
    return target, radius
    # target = np.zeros_like(positions)
    # repls = []
    # norm_form = formation / get_sides(formation).sum()
    # side_12 = get_sides(norm_form)[2] # Get opposite side
    # radius = None
    # for i in range(3):
    #     j, k = (i + 1) % 3, (i + 2) % 3
    #     repl = replication(formation, positions[j], positions[k], j, k)
    #     repls.append(repl)
    #     direction = repl[i] - positions[i]
    #     direction = direction / np.linalg.norm(direction)
    #     if i == 0: # compute radius only once
    #         radius = np.linalg.norm(positions[0] - repl[0]) * side_12
    #     target[i] = positions[i] + direction * radius

    # return target, radius# , repls

def triangle_solution(positions: np.array, formation: np.array, do_refl: bool = True, do_perm: bool = True):
    assert(positions.shape[0] == 3)
    assert(formation.shape[0] == 3)

    # radius, perm, repl = None, None, None
    # for refl in list(product([1, -1], [1, -1])):
    #     refl_form = formation * refl
    #     for _perm in map(list, permutations(range(formation.shape[0]))):
    #         _formation = refl_form[_perm]
    #         norm_form = _formation / get_sides(_formation).sum()
    #         side_12 = get_sides(norm_form)[2] # Get opposite side

    #         _repl = replication(_formation, q_0=positions[1], q_1=positions[2], i=1, j=2)
    #         _radius = np.linalg.norm(positions[0] - _repl[0]) * side_12
    #         if radius is None or _radius < radius:
    #             radius, perm, repl = _radius, _perm, _repl

    target, radius, refl, perm = None, None, None, None
    refls = list(product([1, -1], [1, -1])) if do_refl else [[1, 1]]
    perms = [list(p) for p in permutations(range(3))] if do_perm else [[0, 1, 2]]
    for _refl, _perm in product(refls, perms):
        _target, _radius = triangle_solution_no_perm(positions, (formation * _refl)[_perm])
        if radius is None or _radius < radius:
            target, radius, refl, perm = _target, _radius, _refl, _perm

    ret = target, radius
    if do_refl:
        ret = *ret, refl
    if do_perm:
        ret = *ret, perm
    return ret

sgn = np.vectorize(lambda x: 1 if x > 0 else -1)

def TriangleMMD(R: np.array, P: np.array) -> float:
    assert(R.shape == (3, 2) and P.shape == (3, 2))
    R = R[np.argsort(to_angles(R))]
    P = P[np.argsort(to_angles(P))]
    
    P *= 2 * (
        np.sign(replication(R, (0, 0), (1, 0))[2]) == 
        np.sign(replication(P, (0, 0), (1, 0))[2])
    ) - 1
    T = replication(P, R[0], R[1])
    d = np.linalg.norm(P[0] - P[1]) / np.sum(np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1))
    return np.linalg.norm(R[2] - T[2]) * d

def TriangleMetric(T1: np.array, T2: np.array) -> float:
    assert(T1.shape == (3, 2) and T2.shape == (3, 2))
    _T1 = np.abs(T1[np.argsort(to_angles(T1))]) # Get assignment & match reflections
    _T2 = np.abs(T2[np.argsort(to_angles(T2))]) # Get assignment & match reflections
    _T1 = replication(T1, np.array([0.,0.]), np.array([1.,0.]))
    _T2 = replication(T2, np.array([0.,0.]), np.array([1.,0.]))
    return np.linalg.norm(_T2[2] - _T1[2]) # / (_T1_perim + _T2_perim)
    eq = Triangle.EQUILATERAL
    return abs(TriangleMMD(eq, T1) - TriangleMMD(eq, T2))
    # _T1 = replication(T1, np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    # _T1[2] = np.abs(_T1[2])
    # _T2 = replication(T2, np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    # _T2[2] = np.abs(_T2[2])
    # return TriangleMMD(_T1, _T2)
  
def approximation(positions: np.array, formation: np.array): #, return_replications: bool = False):
    maxd, target, radius = None, None, None
    for _formation in permutations(formation):
        tri_target, _radius, _ = triangle_solution_no_perm(positions[:3], _formation[:3])
        _target = replication(_formation, tri_target[0], tri_target[1])
        distances = np.linalg.norm(_target - positions, axis=1)
        _maxd = distances.max()
        if maxd is None or _maxd < maxd:
            target = _target
            radius = _radius
            maxd = _maxd

    return target, radius
  

import matplotlib.pyplot as plt
from shapes import random_shape
from plotters import get_colors

def test_optimization():
    from optimization import convex_solution_no_perm
    p = np.array([
        [0.02394594, 0.84032009],
        [0.53647123, 0.58477441],
        [0.78407296, 0.02645303],
        [0.96061581, 0.75271376],
    ])
    s = np.array([ # formation
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    p_perm = [0,1,2]
    s_perm = [0,3,1]

    target, radius, _ = triangle_solution_no_perm(p[p_perm], s[s_perm])
    print(radius)

    _s = s[s_perm] - s[s_perm][0]
    opt_target, opt_radius = convex_solution_no_perm(p[p_perm], _s)
    print(opt_radius)


    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.25, 1.25)

    ax.scatter(p[:,0], p[:,1], marker='o')
    ax.scatter(target[:,0], target[:,1], marker='X')

    ax.scatter(opt_target[:,0], opt_target[:,1], c='b', marker='x')

    plt.show()

def test_assignment():
    import matplotlib.pyplot as plt
    from matplotlib import collections as mc
    for i in range(100):
        T1 = random_shape(3)
        T2 = random_shape(3)

        # Assignment
        T1 = T1[np.argsort(to_angles(T1))]
        T2 = T2[np.argsort(to_angles(T2))]

        target, r1, refl, perm = triangle_solution(T1, T2)

        _target, _r1, _refl = triangle_solution(T1, T2, do_perm=False)

        if not np.isclose(r1, _r1):
            print(f'r1: {r1}, _r1: {_r1}')
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = plt.subplots()
            ax.axis('equal')
            
            colors = np.array(list('rgb'))
            ax.scatter(T1[:,0], T1[:,1], c=colors, marker='o')
            ax.scatter(target[:,0], target[:,1], c=colors[np.argsort(to_angles(target))], marker='x')
            ax.scatter(_target[:,0], _target[:,1], c=colors, marker='.')

            lines = mc.LineCollection(
                [ 
                    *zip(T1.tolist(), np.roll(T1, 1, axis=0).tolist()),
                    *zip(target.tolist(), np.roll(target, 1, axis=0).tolist()),
                    *zip(_target.tolist(), np.roll(_target, 1, axis=0).tolist()),
                ],
                # linewidths=2,
                colors='black'
            )
            ax.add_collection(lines)

            plt.show()

def test_solution():
    for i in range(100):
        T1 = random_shape(3)
        T2 = random_shape(3)

        target, r1, refl, perm = triangle_solution(T1, T2)
        r2 = TriangleMMD(T1, T2)

        assert(np.isclose(r1, r2))

def test_metric():
    for i in range(1000):
        T1 = random_shape(3)
        T2 = random_shape(3)
        T3 = random_shape(3)

        TriangleMetric(T1, T2)
        
        _T1 = np.copy(T1)
        _T2 = np.copy(T2)
        _T3 = np.copy(T3)

        fail = False
        #non-negativity
        T1T2 = TriangleMetric(T1, T2)
        if T1T2 >= 0.0:
            print(f'non-negativity: PASS')
        else:
            fail = True
            print(f'non-negativity: FAIL - {T1T2}')
        
        # identity
        T1T1 = TriangleMetric(T1, T1)
        if np.isclose(T1T1, 0.0):
            print(f'identity: PASS')
        else:
            fail = True
            print(f'identity: FAIL - {T1T1}')
        
        # symmetry
        T2T1 = TriangleMetric(T2, T1)
        if np.isclose(T1T2, T2T1):
            print(f'symmetry: PASS')
        else:
            fail = True
            print(f'symmetry: FAIL - {T1T2} != {T2T1}')
        
        # triangle inequality
        T2T3 = TriangleMetric(T2, T3)
        T1T3 = TriangleMetric(T1, T3)
        if np.isclose(T1T3, T1T2 + T2T3) or T1T3 < T1T2 + T2T3:
            print(f'triangle inequality: PASS')
        else:
            fail = True
            print(f'triangle inequality: FAIL - {T1T3} !<= {T1T2} + {T2T3} ({T1T2 + T2T3})')

        if fail:
            print(f'T1: {T1}')
            print(f'T2: {T2}')
            print(f'T3: {T3}')
            break


if __name__ == '__main__':
    test_solution()