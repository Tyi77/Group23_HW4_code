import numpy as np
from step_123 import step_123
from step_45 import step45, read_calib
from scipy.io import savemat

# def manual_triangulation(points1, points2, P1, P2):
#     """
#     手動實現三角化 (Triangulation)，將匹配點轉換為 3D 點雲。

#     參數:
#         points1 (np.ndarray): 第一張影像中的點對應 (Nx2)。
#         points2 (np.ndarray): 第二張影像中的點對應 (Nx2)。
#         P1 (np.ndarray): 第一台相機的投影矩陣 (3x4)。
#         P2 (np.ndarray): 第二台相機的投影矩陣 (3x4)。

#     返回:
#         points3D (np.ndarray): 重建的 3D 點雲 (Nx3)。
#     """
#     num_points = points1.shape[0]
#     points3D = []

#     for i in range(num_points):
#         # 提取匹配點
#         x1, y1 = points1[i]
#         x2, y2 = points2[i]

#         # 構建 A 矩陣
#         A = np.array([
#             x1 * P1[2, :] - P1[0, :],
#             y1 * P1[2, :] - P1[1, :],
#             x2 * P2[2, :] - P2[0, :],
#             y2 * P2[2, :] - P2[1, :]
#         ])

#         # SVD 分解解 Ax = 0
#         _, _, Vt = np.linalg.svd(A)
#         X_homogeneous = Vt[-1]  # SVD 的最後一行是解

#         # 從齊次坐標轉換為非齊次坐標
#         X = X_homogeneous[:3] / X_homogeneous[3]
#         points3D.append(X)

#     return np.array(points3D)

if __name__ == '__main__':
    calib_filepath = 'data/Statue_calib.txt'
    img1_name = 'data/Statue1.bmp'
    img2_name = 'data/Statue2.bmp'
    # calib_filepath = 'data/Mesona_calib.txt'
    # img1_name = 'data/Mesona1.JPG'
    # img2_name = 'data/Mesona2.JPG'

    F, pts1, pts2 = step_123(img1_name, img2_name, 0.0005, 10000)
    K1, K2 = read_calib(calib_filepath)
    
    P1, P2, points_3D, pts1, pts2 = step45(F, pts1, pts2, K1, K2)

    # plot points in 3D space
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points_3D = np.array(points_3D)
    ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], c='b')

    # show the plot
    plt.show()

    savemat('output/Statue.mat', {'points_3D': points_3D, 'P1': P1, 'P2': P2, 'pts1': pts1, 'pts2': pts2})
    # savemat('output/Mesona.mat', {'points_3D': points_3D, 'P1': P1, 'P2': P2, 'pts1': pts1, 'pts2': pts2})