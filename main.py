import numpy as np
import matplotlib.pyplot as plt
from step_123 import step_123
from step_456 import step456, read_calib
from scipy.io import savemat

if __name__ == '__main__':
    calib_filepath = 'data/Statue_calib.txt'
    img1_name = 'data/Statue1.bmp'
    img2_name = 'data/Statue2.bmp'
    # calib_filepath = 'data/Mesona_calib.txt'
    # img1_name = 'data/Mesona1.JPG'
    # img2_name = 'data/Mesona2.JPG'

    F, pts1, pts2 = step_123(img1_name, img2_name, 0.01, 100000)
    K1, K2 = read_calib(calib_filepath)
    
    P1, P2, points_3D, pts1, pts2 = step456(F, pts1, pts2, K1, K2)

    # plot points in 3D space
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points_3D = np.array(points_3D)
    ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], c='b')

    # show the plot
    plt.show()

    savemat('output/Statue.mat', {'points_3D': points_3D, 'P1': P1, 'P2': P2, 'pts1': pts1, 'pts2': pts2})
    # savemat('output/Mesona.mat', {'points_3D': points_3D, 'P1': P1, 'P2': P2, 'pts1': pts1, 'pts2': pts2})