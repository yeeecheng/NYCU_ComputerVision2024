import numpy as np
import cv2

def getH(img_corners, world_corners):
    """ Gets Homography matrix """

    h = []

    num_points = img_corners.shape[0]
    for i in range(num_points):
        u, v = img_corners[i]
        X, Y = world_corners[i]

        h.append(np.array([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u]))
        h.append(np.array([0, 0, 0, -X, -Y, -1, v * X, v * Y, v]))

    # apply SVD

    h = np.array(h)
    U, S, V_T = np.linalg.svd(h, True)
    V = V_T.T
    # last column of V corresponds to smallest eigen value
    H = V[:, -1]
    # scale ambiguity
    H = H / H[8]
    # H shape is 3 by 3
    H = np.reshape(H, (3, 3))

    return H

def getAllH(images_corners, world_corners):
    """ Gets Homography of all images present in calibration images dataset """

    H_all = []

    for img_corners in images_corners:
        H = getH(img_corners, world_corners)
        H_all.append(H)

    return np.array(H_all)


def getVij(hi, hj):

    Vij = np.array([
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2]
    ])

    return Vij.T


def getV(H_all):

    v = []
    for H in H_all:
        # the first column of H
        h1 = H[:, 0]
        # the second column of H
        h2 = H[:, 1]

        v12 = getVij(h1, h2)
        v11 = getVij(h1, h1)
        v22 = getVij(h2, h2)
        v.append(v12.T)
        v.append((v11 - v22).T)

    # shape (2 * images, 6)
    return np.array(v) 

def arrangeB(b):
    
    # B is symmetric
    # 0, 1, 3
    # 1, 2, 4
    # 3, 4, 5

    B = np.zeros((3, 3))
    B[0, 0] = b[0]
    B[0, 1] = b[1]
    B[0, 2] = b[3]
    B[1, 0] = b[1]
    B[1, 1] = b[2]
    B[1, 2] = b[4]
    B[2, 0] = b[3]
    B[2, 1] = b[4]
    B[2, 2] = b[5]

    return B

def getB(H_all):

    v = getV(H_all)
    U, S, V_T = np.linalg.svd(v)
    b = V_T.T[:, -1] 
    B = arrangeB(b)

    return B


def getA(B):
    
    v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2)
    lambd = (
        B[2, 2] - (B[0, 2] ** 2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
        )

    alpha = np.sqrt(lambd / B[0, 0])

    beta = np.sqrt((lambd * B[0, 0]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2))
    gamma = -1 * B[0, 1] * (alpha ** 2) * (beta) / lambd
    u0 = (gamma * v0 / beta) - (B[0, 2] * (alpha ** 2) / lambd)

    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])


    return A

def getRotAndTrans(A, H_all):
    """Gets the rotation and translation of the each image"""

    rvecs = []
    tvecs = []
    # RT_all = []

    for H in H_all:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        lambd = 1 / np.linalg.norm(np.matmul(np.linalg.pinv(A), h1), 2)
        r1 = np.matmul(lambd * np.linalg.pinv(A), h1)
        r2 = np.matmul(lambd * np.linalg.pinv(A), h2)
        r3 = np.cross(r1, r2)
        t = np.matmul(lambd * np.linalg.pinv(A), h3)
    
        rvec, _ = cv2.Rodrigues(np.vstack((r1, r2, r3)).T)
        rvecs.append(rvec)
        tvecs.append(np.vstack((t)))
        # RT = np.vstack((r1, r2, r3, t)).T
        # RT_all.append(RT)
    
    return rvecs, tvecs


def extractParamFromA(A, K_distortion_init):
    """ Extract the individual intrinsics parameters from A matrix """

    alpha = A[0, 0]
    gamma = A[0, 1]
    u0 = A[0, 2]
    beta = A[1, 1]
    v0 = A[1, 2]
    k1 = K_distortion_init[0]
    k2 = K_distortion_init[1]

    return np.array([alpha, gamma, beta, u0, v0, k1, k2])

def reprojectionRMSError(A, K_distortion, RT_all, images_corners, world_corners):
    """ Calculates the reprojection error of all images """

    alpha, gamma, beta, u0, v0, k1, k2 = extractParamFromA(A, K_distortion)

    error_all_images = []
    reprojected_corners_all = []

    # images_corners: shape [images, no_corners, 2]
    num_img = images_corners.shape[0]
    for i in range(num_img):
        img_corners = images_corners[i]
        RT = RT_all[i]
        P = np.dot(A, RT)
        error_per_img = 0
        reprojected_img_corners = []

        num_points = img_corners.shape[0]
        for j in range(num_points):
            world_point_corners_nonHomo_2d = world_corners[j]
            world_point_3d_Homo = np.array([
                [world_point_corners_nonHomo_2d[0]],
                [world_point_corners_nonHomo_2d[1]],
                [0],
                [1],
            ], dtype= float)

            img_corner_nonHomo = img_corners[j]
            img_corner_Homo = np.array([
                [img_corner_nonHomo[0]],
                [img_corner_nonHomo[1]],
                [1]
            ], dtype= float)

            # pixel coordinates (u, v) using Projection Matrix P
            pixel_coords = np.matmul(P, world_point_3d_Homo)
            u = pixel_coords[0] / pixel_coords[2]
            v = pixel_coords[1] / pixel_coords[2]

            # image coordinates (or coordinates in camera plane) using only RT matrix
            image_coords = np.matmul(RT, world_point_3d_Homo)
            x_norm = image_coords[0] / image_coords[2]
            y_norm = image_coords[1] / image_coords[2]

            r = np.sqrt(x_norm ** 2 + y_norm ** 2)

            u_hat = u + (u - u0) * (k1 * r ** 2 + k2 * (r ** 4))
            v_hat = v + (v - v0) * (k1 * r ** 2 + k2 * (r ** 4))

            img_corner_Homo_hat = np.array(
                [u_hat, v_hat, [1]], dtype= float
            )

            reprojected_img_corners.append((img_corner_Homo_hat[0], img_corner_Homo_hat[1]))

            error_per_corner = np.linalg.norm(
                (img_corner_Homo - img_corner_Homo_hat), 2
            )
            error_per_img = error_per_img + error_per_corner

        reprojected_corners_all.append(reprojected_img_corners)
        error_all_images.append(error_per_img / 54)

    return np.array(error_all_images), np.array(reprojected_corners_all)