import numpy as np

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1))))    # h for homogeneous
    l1 = np.cross(h[0], h[1])              # get first line
    l2 = np.cross(h[2], h[3])              # get second line
    x, y, z = np.cross(l1, l2)             # point of intersection
    if z == 0:                             # lines are parallel
        return np.array([float('inf'), float('inf')])
    return np.array([x/z, y/z])

def to_angles(P: np.array) -> np.array:
    dists = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)
    alpha = lambda i: np.arccos((dists[i-2]**2 - dists[i]**2 - dists[i-1]**2) / (-2 * dists[i-1] * dists[i]))
    return np.apply_along_axis(alpha, 0, np.arange(dists.size))

def dist(p: np.array, q: np.array) -> float:
    assert(p.shape == (2,) and p.shape == (2,))
    return np.linalg.norm(p - q)