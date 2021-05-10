
from typing import List, Tuple
import numpy as np
import math

def to_angles(P: np.array) -> np.array:
    dists = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)
    alpha = lambda i: np.arccos((dists[i-2]**2 - dists[i]**2 - dists[i-1]**2) / (-2 * dists[i-1] * dists[i]))
    return np.apply_along_axis(alpha, 0, np.arange(dists.size))

def d(p: np.array, q: np.array) -> float:
    assert(p.shape == (2,) and p.shape == (2,))
    return np.linalg.norm(p - q)

def sol(R, P):
    perimeter = d(P[0], P[1]) + d(P[1], P[2]) + d(P[2], P[0])
    P0P1 = d(P[0], P[1]) / perimeter
    P0P2 = d(P[0], P[1]) / perimeter
    P2P0 = d(P[0], P[1]) / perimeter
    
    alpha = arccos((P2P0 - P0P1 - P0P2) / (-2 * P0P1 * P0P2))

    

# def tri_solution(R: List[Tuple[float, float]], P: List[Tuple[float, float]]):
#     R = R[np.argsort(to_angles(R))]
#     P_angles = to_angles(P)
#     P_idx = np.argsort(to_angles(P))
#     P_angles = P_angles[P_idx]
#     P_norm = P[P_idx] / np.sum(P_dists)
#     # P = P[np.argsort(to_angles(P))]

#     for i in range(3):
#         side = d(R[i], R[(i+1)%R.shape[0]])


#     np.arange(np.shape[0])
#     np.sort

if __name__ == '__main__':
    arr = np.array([[1, 0], [0, 1], [0, 0]])
    angles = to_angles(arr) * 180 / math.pi

    print(angles)