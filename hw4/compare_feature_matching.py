import cv2
import numpy as np

def extract_keypoints_and_features(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    points = np.array([kp.pt for kp in keypoints])
    return points, descriptors


def compute_fundamental_matrix(pts1, pts2):
    # F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=0.5, confidence=0.99)
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_8POINT, ransacReprojThreshold=0.5, confidence=0.99)
    inliers = np.hstack((pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]))
    return F, inliers

def analyze_geometric_stability(F, pts1, pts2):
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    distances = np.abs(np.sum(lines1 * np.hstack((pts1, np.ones((pts1.shape[0], 1)))), axis=1))
    mean_distance = np.mean(distances)
    inlier_ratio = np.mean(distances < 1.0)
    return inlier_ratio, mean_distance



def feature_match(desc1, desc2, method="BFMatcher", ratio=0.7):
    if method == "BFMatcher":
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(desc1, desc2, k=2)
    elif method == "FLANN":
        index_params = dict(algorithm=1, trees=5) 
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
    elif method == "KAZE":
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(desc1, desc2, k=2)
    elif method == "AKAZE":
        norm_type = cv2.NORM_HAMMING if desc1.dtype == np.uint8 else cv2.NORM_L2
        bf = cv2.BFMatcher(norm_type)
        matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]

    return good_matches



img_pair = ['Mesona1.JPG', 'Mesona2.JPG']

img1 = cv2.imread(f'./data/{img_pair[0]}', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(f'./data/{img_pair[1]}', cv2.IMREAD_GRAYSCALE)

points1, desc1 = extract_keypoints_and_features(img1)
points2, desc2 = extract_keypoints_and_features(img2)

methods = ["BFMatcher", "FLANN", "KAZE", "AKAZE"]
results = []

for method in methods:
    print(f"Using {method} for feature matching...")
    try:
        matches = feature_match(desc1, desc2, method=method)
        matched_points1 = np.float32([points1[m.queryIdx] for m in matches])
        matched_points2 = np.float32([points2[m.trainIdx] for m in matches])

        F, inliers = compute_fundamental_matrix(matched_points1, matched_points2)
        inlier_points1 = inliers[:, :2]
        inlier_points2 = inliers[:, 2:]

        inlier_ratio, mean_distance = analyze_geometric_stability(F, inlier_points1, inlier_points2)
        results.append((method, len(matches), inlier_ratio, mean_distance))

    except NotImplementedError as e:
        print(f"{method} is not implemented: {e}")
        results.append((method, 0, 0, 0))

print("\nFeature Matching Method Comparison:")
for method, match_count, inlier_ratio, mean_distance in results:
    print(f"Method: {method}, Matches: {match_count}, Inlier Ratio: {inlier_ratio:.2f}, Mean Distance: {mean_distance:.2f}")