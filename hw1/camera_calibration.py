import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os 
import camera_calibration_show_extrinsics as show
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

class CameraCalibration:

    def __init__(self, data_dir, n_corner_x, n_corner_y, num_test_image):
        
        # Number of corner point where black and white meet in the checkerboard image.
        self.n_corner_x = n_corner_x
        self.n_corner_y = n_corner_y
        # Read image path
        self.img_path_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        print("Total number of images are ", len(self.img_path_list))

        # test 
        self.num_test_image = num_test_image 

    def get_world_points_and_image_points(self):
        # the criteria of refined corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Creating list to store 2D image points in image coordinate system.
        self.list_images_points_2d = []
        # Define the 3D world points in world coordinate system, there is a trick that only create 2d array because of z = 0.
        self.world_points_3d = np.zeros((self.n_corner_x * self.n_corner_y, 2), np.float32)
        # Prepare world points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0). 
        self.world_points_3d[:, :2] = np.mgrid[0 : self.n_corner_x, 0 : self.n_corner_y].T.reshape(-1, 2)
        
        for idx, fname in enumerate(self.img_path_list):
            
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Get the all of the image points in the checkerboard image.
            print('find the chessboard corners of',fname)
            # corners: all coordination of image points.
            ret, corners = cv2.findChessboardCorners(gray, (self.n_corner_x, self.n_corner_y), None)
            if ret == True:
                # Refined pixel coordinate for 2d points.
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.list_images_points_2d.append(corners_refined.reshape(-1, 2))
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (self.n_corner_x, self.n_corner_y), corners_refined, ret)
            
            if idx + 1 == self.num_test_image:
                break

        self.world_points_3d = np.array(self.world_points_3d)
        self.list_images_points_2d = np.array(self.list_images_points_2d)
        print("3D world points shape: ", self.world_points_3d.shape)
        print("2D image points shape: ", self.list_images_points_2d[0].shape)

    def __get_homography(self, img_points):
        """ Gets Homography matrix each image"""

        # Create matrix L
        L = []
        for idx in range(img_points.shape[0]):
            u, v = img_points[idx]
            X, Y = self.world_points_3d[idx]
            L.append(np.array([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u]))
            L.append(np.array([0, 0, 0, -X, -Y, -1, v * X, v * Y, v]))

        # Apply SVD
        L = np.array(L)
        _, _, V_T = np.linalg.svd(L, True)
        V = V_T.T
        # last column of V corresponds to smallest eigen value
        H = V[:, -1]
        # scale ambiguity
        H = H / H[8]
        # H shape is 3 by 3
        H = np.reshape(H, (3, 3))

        return H

    def get_all_homography(self):
        """ Gets Homography of all images present in calibration images dataset """
        
        list_H = []

        for img_points in self.list_images_points_2d:
            H = self.__get_homography(img_points)
            list_H.append(H)

        return np.array(list_H)

    def __get_Vij(self, hi, hj):

        Vij = np.array([
            hi[0] * hj[0],
            hi[0] * hj[1] + hi[1] * hj[0],
            hi[1] * hj[1],
            hi[2] * hj[0] + hi[0] * hj[2],
            hi[2] * hj[1] + hi[1] * hj[2],
            hi[2] * hj[2]
        ])

        return Vij.T

    def __get_V(self, list_H):
        
        V = []
        # solve to V of all image, reference formula (8)
        for H in list_H:
            # the first column of H
            h1 = H[:, 0]
            # the second column of H
            h2 = H[:, 1]

            v12 = self.__get_Vij(h1, h2)
            v11 = self.__get_Vij(h1, h1)
            v22 = self.__get_Vij(h2, h2)
            V.append(v12.T)
            V.append((v11 - v22).T)

        # shape (2 * num_images, 6)
        return np.array(V) 

    def __get_B(self, list_H):
        """ B is symmetric matrix """
        V = self.__get_V(list_H)
        _, _, C_T = np.linalg.svd(V)
        b = C_T.T[:, -1] 

        # get B with b = [B11, B12, B22, B13, B23, B33]
        B = np.array([
            [b[0], b[1], b[3]],
            [b[1], b[2], b[4]],
            [b[3], b[4], b[5]]
        ])

        return B

    def get_intrinsic_matrix(self, list_H):
        """" Get intrinsic matrix """
        # Get B
        B = self.__get_B(list_H)
        print("----Estimated B Matrix is ----\n", B)
        # Get intrinsic matrix, which compute all parameters of intrinsic matrix from matrix B
        v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2)
        lambd = (B[2, 2] - (B[0, 2] ** 2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0])
        alpha = np.sqrt(lambd / B[0, 0])
        beta = np.sqrt((lambd * B[0, 0]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2))
        gamma = -1 * B[0, 1] * (alpha ** 2) * (beta) / lambd
        u0 = (gamma * v0 / beta) - (B[0, 2] * (alpha ** 2) / lambd)

        # Create Intrinsic matrix A
        A = np.array([
            [alpha, gamma, u0],
            [0, beta, v0],
            [0, 0, 1]
        ])

        return A

    def get_extrinsic_matrix(self, A, list_H):
        """ Get extrinsic matrix which is the rotation and translation of the each image """
        rvecs = []
        tvecs = []
        r_matrix = []
        t_matrix = []
        for H in list_H:
            # Get the i-th column of H
            h1 = H[:, 0]
            h2 = H[:, 1]
            h3 = H[:, 2]
            # Get extrinsic matrix, which compute all parameters of extrinsic matrix from matrix H and A
            lambd = 1 / np.linalg.norm(np.matmul(np.linalg.pinv(A), h1), 2)
            # transpose(ri) is i-th column of R 
            r1 = np.matmul(lambd * np.linalg.pinv(A), h1)
            r2 = np.matmul(lambd * np.linalg.pinv(A), h2)
            r3 = np.cross(r1, r2)
            t = np.matmul(lambd * np.linalg.pinv(A), h3)
            # cv2.Rodrigus can transform matrix to vector
            rvec, _ = cv2.Rodrigues(np.vstack((r1, r2, r3)).T)
            rvecs.append(rvec)
            tvecs.append(np.vstack((t)))
            r_matrix.append(np.vstack((r1, r2, r3)).T)
            t_matrix.append(t.T)

        return np.array(rvecs), np.array(tvecs), np.array(r_matrix), np.array(t_matrix)

    def reprojection_error(self, A, rvecs, tvecs, dist_coeffs= np.zeros((5, ))):
        
        # Define the 3D world points in world coordinate system, this part need 3D array.
        world_points = np.zeros((self.n_corner_x * self.n_corner_y, 3), np.float32)
        # Prepare world points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0). 
        world_points[:, :2] = np.mgrid[0 : self.n_corner_x, 0 : self.n_corner_y].T.reshape(-1, 2)
        list_world_points_3d = []
        for _ in range(self.list_images_points_2d.shape[0]):
            list_world_points_3d.append(world_points)
        list_world_points_3d = np.array(list_world_points_3d) 
        mean_error = 0
        for i in range(len(list_world_points_3d)):
            imgpoints2, _ = cv2.projectPoints(list_world_points_3d[i], rvecs[i], tvecs[i], A, dist_coeffs)
            # from (num_points, 1, 2) to (num_points, 2)
            imgpoints2 = np.squeeze(imgpoints2)
            error = cv2.norm(self.list_images_points_2d[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        
        print( "total error: {}".format(mean_error/len(list_world_points_3d)) )

        return mean_error/len(list_world_points_3d)

    def project_points(self, world_points, R, T, A, dist_coeffs):
        
        camera_points = np.dot(R, world_points.T).T + T
        # x', y'
        x = camera_points[:,0] / camera_points[:,2]
        y = camera_points[:,1] / camera_points[:,2]
        # k1, k2: radial distoration coefficients, p1: p2: tangential distoration coefficient
        k1, k2, k3, p1, p2 = dist_coeffs
        # r ^ 2
        r2 = x ** 2 + y ** 2
        # r ^ 4
        r4 = r2 ** 2

        x_radial = x * (1 + k1 * r2 + k2 * r4)
        y_radial = y * (1 + k1 * r2 + k2 * r4)
        # x'', y''
        x_tangential = x_radial + (2 * p1 * x * y + p2 * (r2 + 2 * x ** 2))
        y_tangential = y_radial + (p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y)
        # fx: A[0, 0], fy: A[1, 1], cx: A[0, 2], cy: A[1, 2]
        u = A[0, 0] * x_tangential + A[0, 2]
        v = A[1, 1] * y_tangential + A[1, 2]
        return np.array([u, v], dtype= np.float32).T
    
    def reprojection_error_with_distoration(self, params, list_world_points_3d, list_images_points_2d, r_matrix, t_matrix, A):
        
        # get distoration coeffsions, update with least_squares
        dist_coeffs = params

        total_error = []
        # calcluate projection error
        for i in range(len(list_world_points_3d)):
            R = r_matrix[i]
            T = t_matrix[i]
            image_points = self.project_points(list_world_points_3d[i], R, T, A, dist_coeffs)
            # calculate loss 
            error = cv2.norm(self.list_images_points_2d[i], image_points, cv2.NORM_L2) / len(image_points)
            total_error.append(error)

        return np.array(total_error)
    
    def optimize_camera_calibration(self, r_matrix, t_matrix, A):
        from scipy.optimize import least_squares
        
        # initialize the distoration coeffsions [k1, k2, k3, p1, p2].
        dist_coeffs = np.zeros(5)
        # wolrd points list
        world_points = np.zeros((self.n_corner_x * self.n_corner_y, 3), np.float32)
        world_points[:, :2] = np.mgrid[0 : self.n_corner_x, 0 : self.n_corner_y].T.reshape(-1, 2)
        list_world_points_3d = []
        for _ in range(self.list_images_points_2d.shape[0]):
            list_world_points_3d.append(world_points)
        list_world_points_3d = np.array(list_world_points_3d) 

        result = least_squares(self.reprojection_error_with_distoration, dist_coeffs, method="lm", args= (list_world_points_3d, self.list_images_points_2d, r_matrix, t_matrix, A))

        optimized_dist_coeffs = result.x
        print("Optimized Distortion Coefficients: ", optimized_dist_coeffs)
        return optimized_dist_coeffs

def visualize(A, rvecs, tvecs):
    """ Visualize picture """

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
    camera_matrix = A
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

def run_camera_calibration(args):
    
    path = 'output.txt'
    f = open(path, 'w')

    for idx in range(args.start_test_num, args.end_test_num + 1):

        camera_calibration = CameraCalibration(args.data, args.corner_x, args.corner_y, idx)
        camera_calibration.get_world_points_and_image_points() 
        list_H = camera_calibration.get_all_homography()
        A = camera_calibration.get_intrinsic_matrix(list_H)
        print("----Estimated intrinsic matrix is ----\n", A)
        rvecs, tvecs, r_matrix, t_matrix = camera_calibration.get_extrinsic_matrix(A, list_H)
        # print(rvecs.shape, tvecs.shape, r_matrix.shape, t_matrix.shape)
        # print("----Estimated  extrinsic matrix is ----\n", np.concatenate((rvecs, tvecs), axis=1).reshape(-1,6))
      
        total_error = camera_calibration.reprojection_error(A, rvecs, tvecs)
        f.write(str(total_error) + ",")
    f.close()

    # optimized_dist_coeffs = camera_calibration.optimize_camera_calibration(r_matrix, t_matrix, A)
    # camera_calibration.reprojection_error(A, rvecs, tvecs, optimized_dist_coeffs)
    visualize(A, rvecs, tvecs)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-d', '--data', default= "./data/", type= str, help= 'path of data dir')
    parse.add_argument('-x', '--corner_x', default= 10, type= int, help= 'Number of corner point where black and white meet in the pattern.')
    parse.add_argument('-y', '--corner_y', default= 7, type= int, help= 'Number of corner point where black and white meet in the pattern.')
    # sn ~ en
    parse.add_argument('-sn', '--start_test_num', default= 30, type= int, help= 'what number you start test')
    parse.add_argument('-en', '--end_test_num', default= 30, type= int, help= 'what number you end testing')
    args = parse.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()
    run_camera_calibration(args)