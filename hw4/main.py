import argparse
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import trimesh
import matplotlib.pyplot as plt

class SFM:

    def __init__(self, img1, img2):
        
        self.img1 = img1
        self.img2 = img2
    
    """ step1. find out correspondence across images """
    def extract_keypoints_and_features(self):

        sift_detector = cv2.SIFT_create()
        gray_imag1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        gray_imag2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        # find keypoints of images
        keypts1, des1 = sift_detector.detectAndCompute(gray_imag1, None)
        keypts2, des2 = sift_detector.detectAndCompute(gray_imag2, None)
        # create matcher with Euclidean ditance calculating
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        good_matches = sorted(matches, key=lambda x: x.distance)
        
        # find the first-nearest and second nearest point
        # matches = bf.knnMatch(des1, des2, k=2)
        # good_matches = []
        # ratio_threshold = 0.7
        # for m, n in matches:
        #     if (m.distance / n.distance) < ratio_threshold :
        #         good_matches.append(m)
        # get the point pair from the good matches list, then create the points1 and points2.  
        pts1 = np.array([keypts1[m.queryIdx].pt for m in good_matches])
        pts2 = np.array([keypts2[m.trainIdx].pt for m in good_matches])
        
        return pts1, pts2
    
    # step2. estimate the fundamental matrix across images (normalized 8 points)
    def __normalize_pts(self, pts):
        """Normalize points to have mean = 0 and average distance = sqrt(2)."""
        centroid = np.mean(pts, axis=0)
        shifted_points = pts - centroid
        mean_dist = np.mean(np.sqrt(np.sum(shifted_points**2, axis=1)))
        scale = np.sqrt(2) / mean_dist
        T = np.array([[scale, 0, -scale * centroid[0]],
                    [0, scale, -scale * centroid[1]],
                    [0, 0, 1]])
        normalized_points = (T @ np.column_stack((pts, np.ones(len(pts)))).T).T
        return normalized_points[:, :2], T

    def __construct_matrix_A(self, pts1, pts2):
        """Construct matrix A for the 8-point algorithm."""
        A = []
        for (x1, y1), (x2, y2) in zip(pts1, pts2):
            A.append([x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1])
        return np.array(A)

    def estimate_fundamental_matrix(self, pts1, pts2):
        # Normalize points
        pts1_norm, T1 = self.__normalize_pts(pts1)
        pts2_norm, T2 = self.__normalize_pts(pts2)

        # Construct A matrix
        A = self.__construct_matrix_A(pts1_norm, pts2_norm)

        # Solve for F using SVD
        _, _, Vt = np.linalg.svd(A)
        F = Vt[-1].reshape(3, 3)

        # Enforce rank-2 constraint
        U, S, Vt = np.linalg.svd(F)
        S[-1] = 0
        F = U @ np.diag(S) @ Vt

        # Denormalize
        F = T2.T @ F @ T1

        return F / F[2, 2]
    
    # step3. draw the interest points on you found in step.1 in one image and the corresponding epipolar lines in another
    def draw_epipolar_lines(self, F, pts1, pts2):
        
        image1 = self.img1
        image2 = self.img2
        
        lines = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines = lines.reshape(-1, 3)
       
        h, w = image1.shape[:2]
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [w, -(r[2] + r[0] * w) / r[1]])
            image1 = cv2.line(image1, (x0, y0), (x1, y1), color, 1)
            image1 = cv2.circle(image1, tuple(map(int, pt1)), 5, color, -1)
            image2 = cv2.circle(image2, tuple(map(int, pt2)), 5, color, -1)
        
        # plt.figure(figsize=(10, 5))
        # plt.subplot(121), plt.imshow(cv2.cvtColor(imag1, cv2.COLOR_BGR2RGB))
        # plt.title('Epipolar Lines on Image 1')
        # plt.subplot(122), plt.imshow(cv2.cvtColor(imag2, cv2.COLOR_BGR2RGB))
        # plt.title('Epipolar Lines on Image 2')
        cv2.imwrite(f'result/epilines_image1.jpg', image1)
        cv2.imwrite(f'result/epilines_image2.jpg', image2)

    # step4. get 4 possible solutions of essential matrix from fundamental matrix
    def get_essential_matrix(self, K1, K2, F):
        # computte essential matrix 
        E = K2.T @ F @ K1
        # decompose E using SVD
        U, S, Vt = np.linalg.svd(E)
        # Correct signular values to enforces rank-2 constraint
        new_S = np.diag([1, 1, 0])
        E = U @ new_S @ Vt
        return E
    
    def __decompose_essential_matrix(self, E):
        """ Decompose Essential Matrix into possible rotations and translation. """
        U, _, Vt = np.linalg.svd(E)
        
        # Ensure proper orientation
        if np.linalg.det(U) < 0:
            U *= -1
        if np.linalg.det(Vt) < 0:
            Vt *= -1
        
        # Define W matrix
        W = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])

        # Compute possible rotations
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        
        # Ensure rotations are valid (det(R) = 1)
        if np.linalg.det(R1) < 0:
            R1 *= -1
        if np.linalg.det(R2) < 0:
            R2 *= -1
    
        # Compute translation vector
        t = U[:, 2]  # Third column of U
        return R1, R2, t
    
     # step5. find out the most appropriate solution of essential matrix
    def find_appropriate_essential_matrix(self, E, pts1, pts2, K1, K2):
        """ Determine the correct (R, t) solution from Essential Matrix. """
        R1, R2, t = self.__decompose_essential_matrix(E)
        solutions = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
        
        # Projection matrices for the first camera (identity rotation, no translation)
        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        
        # Test all four combinations
        best_solution = None
        max_positive_depth = 0
        
        for R, t in solutions:
            # Projection matrix for the second camera
            P2 = K2 @ np.hstack((R, t.reshape(-1, 1)))
            pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T  # 2D point to homo
            pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3).T  # 2D point to homo

            # Triangulate points
            points_4d_h = cv2.triangulatePoints(P1, P2, pts1_h[:2], pts2_h[:2])    
            # Convert to inhomogeneous coordinates
            points_3d = points_4d_h[:3, :] / points_4d_h[3, :]
            # Count points with positive depth
            positive_depth = np.sum(points_3d[2, :] > 0)
            
            if positive_depth > max_positive_depth:
                max_positive_depth = positive_depth
                best_solution = (R, t)
                
        return best_solution
    
    # setp6. apply triangulation to get 3D points
    def get_3d_points(self, R, t, K1, K2, pts1, pts2):
        # Projection matrices for the first camera (identity rotation, no translation)
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        # Intrinsic matrix for both cameras
        P1 = K1 @ P1
        # Projection matrix for the second camera
        P2 = K2 @ np.hstack((R, t.reshape(-1, 1)))
        pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T  # 2D point to homo
        pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3).T  # 2D point to homo
        # Triangulate points
        points_4d_h = cv2.triangulatePoints(P1, P2, pts1_h[:2], pts2_h[:2])
        # Convert to inhomogeneous coordinates
        points_3d = points_4d_h[:3, :] / points_4d_h[3, :]
        P = points_3d.T
        # 2D projections
        projected_points_h = P2 @ np.column_stack((P, np.ones(P.shape[0]))).T
        p_img2 = (projected_points_h[:2, :] / projected_points_h[2, :]).T  
        return P, p_img2, P2

    # step6.
    def plot_3d_points(self, points_3d):
        """
        Visualize 3D points using matplotlib.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("Triangulated 3D Points")
        plt.savefig("./result/point3d.png")

    # step7 use texture mapping to get a 3D model
    def generate_mesh(self, points_3d):
        """
        Generate a triangular mesh from 3D points using Delaunay triangulation.
        """
        tri = Delaunay(points_3d[:, :2])  # Use X and Y for triangulation
        return tri
    
    # step8. use 3d software like Blender to visualize the resulting 3D model(import .obj and .mtl files)
    def create_3d_model_with_texture_trimesh(self, points_3d, tri, image, image_size):
        """
        Create and save a textured 3D model using Trimesh.
        """
        # Create a mesh object from the points and triangles
        mesh = trimesh.Trimesh(vertices=points_3d, faces=tri.simplices)
        
        # Create a UV map for texture coordinates
        uv_map = np.zeros((len(points_3d), 2))
        for i, point in enumerate(points_3d):
            uv_map[i, 0] = point[0] / image_size[0]
            uv_map[i, 1] = point[1] / image_size[1]
        
        # Add texture
        mesh.visual.face_colors = [255, 255, 255, 255]  # White faces
        mesh.visual.texture = image
        mesh.visual.uv = uv_map
        
        # Export to OBJ file
        mesh.export("textured_model.obj")

        mesh = trimesh.load_mesh("textured_model.obj")
        # mesh.show()

def run(args):
    
    img_name = args.img_name
    fundamental_matrix_algorithm = args.fundamental_matrix_algorithm

    img1 = cv2.imread(args.data1)
    img2 = cv2.imread(args.data2)
    sfm = SFM(img1, img2)
    pts1, pts2 = sfm.extract_keypoints_and_features()
    
    if fundamental_matrix_algorithm == "nor_point8":
        # normalized 8 point
        F = sfm.estimate_fundamental_matrix(pts1, pts2)
    
    elif fundamental_matrix_algorithm == "RANSAC":
        # RANSAC
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=0.9, confidence=0.99)
   
    sfm.draw_epipolar_lines(F, pts1, pts2)

    if img_name == "Mesona":
        K1 = np.array([
            [1.4219, 0.0005, 0.5092],
            [0, 1.4219, 0],
            [0, 0, 0.0010]
        ])

        K2 = np.array([
            [1.4219, 0.0005, 0.5092],
            [0, 1.4219, 0],
            [0, 0, 0.0010]
        ])
    elif img_name == "Statue":
        K1 = np.array([
            [5426.566895, 0.678017, 330.096680],
            [0.000000, 5423.133301, 648.950012],
            [0.000000, 0.000000, 1.000000]
        ])

        K2 = np.array([
            [5426.566895, 0.678017, 387.430023],
            [0.000000, 5423.133301, 620.616699],
            [0.000000, 0.000000, 1.000000]
        ])
    elif img_name == "trash":
        K1 = np.array([
            [2945.377, 0, 1284.862],
            [0, 2945.377, 954.52],
            [0, 0, 1]
        ])

        K2 = np.array([
            [2945.377, 0, 1455.543],
            [0, 2945.377, 954.52],
            [0, 0, 1]
        ])

    E = sfm.get_essential_matrix(K1, K2, F)
    if fundamental_matrix_algorithm == "RANSAC":
        pts1 = pts1[mask.ravel() == 1].reshape(-1, 2)
        pts2 = pts2[mask.ravel() == 1].reshape(-1, 2)
    R, t = sfm.find_appropriate_essential_matrix(E, pts1, pts2, K1, K2)
    points_3d, p_img2, M = sfm.get_3d_points(R, t, K1, K2, pts1, pts2)
    
    import scipy.io as sio
    data = {
        'P': points_3d,
        'p_img2': p_img2,
        'M': M,
        'tex_name': "test",
        'im_index': 1,
        'output_dir': './output'
    }
    sio.savemat('data.mat', data)

    sfm.plot_3d_points(points_3d)
    tri = sfm.generate_mesh(points_3d)
    sfm.create_3d_model_with_texture_trimesh(points_3d, tri, img1, img1.shape)
   
def parse_args():

    parse = argparse.ArgumentParser()
    # ./data/Mesona1.JPG, 
    # ./data/Statue1.bmp
    # ./my_data/trash1.png
    parse.add_argument('-d1', '--data1', default= "./my_data/trash1.png", type= str, help= 'path of image1')
    parse.add_argument('-d2', '--data2', default= "./my_data/trash2.png", type= str, help= 'path of image2')
    parse.add_argument('-N', '--img_name', default= "Mesona", type= str, 
                        choices= ["Mesona", "Statue", "trash"], help= 'calibration parameter')
    parse.add_argument('-F', '--fundamental_matrix_algorithm', default= "nor_point8", 
                       type= str, choices=["RANSAC", "nor_point8"], help= 'fundamental_matrix_algorithm')
    args = parse.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    run(args)
