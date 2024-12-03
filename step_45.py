import numpy as np
from step_123 import step_123

def cal_essential(F, K1, K2):
    E = K1.T @ F @ K2
    return E

def cal_P_options(E):
    # SVD of E
    U, _, V_t = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(V_t) < 0:
        V_t[-1, :] *= -1

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    t = U[:, 2]
    
    # P2 has four options
    P2s = [
        np.hstack((U @ W @ V_t, t.reshape(-1, 1))),
        np.hstack((U @ W @ V_t, -t.reshape(-1, 1))),
        np.hstack((U @ W.T @ V_t, t.reshape(-1, 1))),
        np.hstack((U @ W.T @ V_t, -t.reshape(-1, 1)))
    ]

    return P2s

def triangulation(pt1, pt2, P1, P2, K1, K2):
    P1 = K1 @ P1
    P2 = K2 @ P2
    A = np.vstack([
        pt1[0] * P1[2, :] - P1[0, :],
        pt1[1] * P1[2, :] - P1[1, :],
        pt2[0] * P2[2, :] - P2[0, :],
        pt2[1] * P2[2, :] - P2[1, :],
    ])

    _, _, V_t = np.linalg.svd(A)
    X = V_t[-1]

    return X / X[3]

def find_P2(pts1, pts2, P1, P2s, K1, K2):
    max_front_points = 0
    best_P2 = None
    best_3D_points = []
    best_pts1 = []
    best_pts2 = []
    for P2 in P2s:
        # calculate 3D points that are in front of the cameras
        count = 0
        cur_3D_points = []
        cur_pts1 = []
        cur_pts2 = []
        for pt1, pt2 in zip(pts1, pts2):
            X = triangulation(pt1, pt2, P1, P2, K1, K2)

            # the depth in two camera system
            z1 = X[2]
            z2 = (P2 @ X)[2]
            if z1 > 0 and z2 > 0:
                count += 1
                cur_3D_points.append(X[:3])
                cur_pts1.append(pt1)
                cur_pts2.append(pt2)

        # find the biggest count
        if count > max_front_points:
            max_front_points = count
            best_P2 = P2
            best_3D_points = cur_3D_points
            best_pts1 = cur_pts1
            best_pts2 = cur_pts2
    
    return best_P2, best_3D_points, best_pts1, best_pts2

def read_calib(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    k1_start = lines.index("K1:\n") + 1
    k2_start = lines.index("K2:\n") + 1

    K1 = np.array([list(map(float, line.split())) for line in lines[k1_start:k1_start + 3]])
    K2 = np.array([list(map(float, line.split())) for line in lines[k2_start:k2_start + 3]])

    return K1, K2


def step45(F, pts1, pts2, K1, K2):
    E = cal_essential(F, K1, K2)

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2s = cal_P_options(E)

    P2, points_3D, pts1, pts2 = find_P2(pts1, pts2, P1, P2s, K1, K2)
    return P1, P2, points_3D, pts1, pts2


if __name__ == '__main__':
    calib_filepath = 'data/Statue_calib.txt'
    img1_name = 'data/Statue1.bmp'
    img2_name = 'data/Statue2.bmp'

    F, pts1, pts2 = step_123(img1_name, img2_name, 0.0005, 10000)
    K1, K2 = read_calib(calib_filepath)
    
    P1, P2, points_3D, pts1, pts2 = step45(F, pts1, pts2, K1, K2)