import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os 
from utlis import *
import camera_calibration_show_extrinsics as show
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D


def calibrateCamera(world_corners, all_images_corners):

    # Get Homography
    H_all = getAllH(all_images_corners, world_corners)
    B = getB(H_all)
    print("====== Estimated B matrix is ========: \n", B)
    # intrinsic matrix
    A_init = getA(B)
    print("\n===== Initialized A as:======= \n", A_init)
    # extrinsic matrix
    rvecs, tvecs = getRotAndTrans(A_init, H_all)


    """ Draw """

    Vr = np.array(rvecs)
    Tr = np.array(tvecs)
    extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)

    # show the camera extrinsics
    print('Show the camera extrinsics')
    # plot setting
    # You can modify it for better visualization
    fig = plt.figure(figsize=(10, 10))
    # ax = fig.gca(projection='3d')
    ax = fig.add_axes(Axes3D(fig))   
    # camera setting
    camera_matrix = A_init
    cam_width = 0.064/0.1
    cam_height = 0.032/0.1
    scale_focal = 1600
    # chess board setting
    board_width = 8
    board_height = 6
    square_size = 1
    # display
    # True -> fix board, moving cameras
    # False -> fix camera, moving boards
    min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                    scale_focal, extrinsics, board_width,
                                                    board_height, square_size, True)

    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, 0)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    ax.set_title('Extrinsic Parameters Visualization')
    plt.show()

def calibrate():
    
    """ Initialize """
    # Read image path
    image_dir = "./data"
    image_path_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
    # Number of corner point where black and white meet in the pattern.
    n_corner_x = 7
    n_corner_y = 7
    # the criteria of refined corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 2D points for each checkerboard image
    all_images_corners = []

    # Define the world coordinate for 3D points.
    world_corners = np.zeros((n_corner_x * n_corner_y, 2), np.float32)
    # Prepare real world(object) points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    world_corners[:, :2] = np.mgrid[0 : n_corner_x, 0 : n_corner_y].T.reshape(-1, 2)
    
    print('Start finding chessboard corners...')
    for fname in image_path_list:
        
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Get the total corner points in the chessboard pattern.
        print('find the chessboard corners of',fname)
        # corners: all coordination of corner points.
        ret, corners = cv2.findChessboardCorners(gray, (n_corner_x, n_corner_y), None)

        if ret == True:
    
            # Refined pixel coordinate for 2d points.
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            all_images_corners.append(corners_refined.reshape(-1, 2))
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (n_corner_x, n_corner_y), corners_refined, ret)
        
    """ camera calibration """

    print('Camera calibration...')
    world_corners = np.array(world_corners)
    all_images_corners = np.array(all_images_corners)
    print(world_corners.shape)
    print(all_images_corners.shape)
    calibrateCamera(world_corners, all_images_corners)

if __name__ == "__main__":

    calibrate()