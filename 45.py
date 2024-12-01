import numpy as np

def cal_essential(F, K1, K2):
    E = K2.T @ F @ K1
    return E

def cal_P_options(E):
    # SVD of E
    U, _, V_t = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(V_t) < 0:
        V_t *= -1

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    t = U[:, 2]
    
    # P2 has four options
    P2s = [
        np.hstack((U @ W @ V_t, t.reshape(-1, 1))),
        np.hstack((U @ W @ V_t, -t.reshape(-1, 1))),
        np.hstack((U @ W.T @ V_t, t.reshape(-1, 1))),
        np.hstack((U @ W.T, V_t, t.reshape(-1, 1)))
    ]

    return P2s

def triangulation(pt1, pt2, P1, P2):
    A = np.array([
        pt1[0] * P1[2] - P1[0],
        pt1[1] * P1[2] - P1[1],
        pt2[0] * P2[2] - P2[0],
        pt2[1] * P2[2] - P2[1],
    ])

    _, _, V_t = np.linalg.svd(A)
    X = V_t[-1]

    return X / X[3]

def find_P2(pts1, pts2, P1, P2s):
    max_front_points = 0
    best_P2 = None

    for P2 in P2s:
        # calculate 3D points that are in front of the cameras
        count = 0
        for pt1, pt2 in zip(pts1, pts2):
            X = triangulation(pt1, pt2, P1, P2)

            # the depth in two camera system
            z1 = X[2]
            z2 = (P2 @ X)[2]
            if z1 > 0 and z2 > 0:
                count += 1

        # find the biggest count
        if count > max_front_points:
            max_front_points = count
            best_P2 = P2
    
    return best_P2


def step45(F, pts1, pts2, K1, K2):
    E = cal_essential(F, K1, K2)

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2s = cal_P_options(E)

    P2 = find_P2(pts1, pts2, P1, P2s)
    return P1, P2