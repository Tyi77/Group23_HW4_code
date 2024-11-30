import cv2
import numpy as np
import matplotlib.pyplot as plt


# =Interest points detection & feature description by SIFT=
def detect_and_describe(img, img_name="Image"):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    # img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # cv2.imshow(f"{img_name} - Keypoints", img_with_keypoints)
    # cv2.imwrite(f"./output/{img_name} - Keypoints.png", img_with_keypoints)
    # cv2.waitKey(0)
    
    return keypoints, descriptors

# =Feature matching by SIFT features=
def match_features(img1, img2, descriptors1, descriptors2, keypoints1, keypoints2, output_name="matching", ratio=0.75):
    matches = []
    
    for i, desc1 in enumerate(descriptors1):
        distances = np.linalg.norm(descriptors2 - desc1, axis=1)
        sorted_indices = np.argsort(distances)
        
        if distances[sorted_indices[0]] < ratio * distances[sorted_indices[1]]:
            matches.append((i, sorted_indices[0], distances[sorted_indices[0]]))

    matches = sorted(matches, key=lambda x: x[2])
    matches = [(i, j) for i, j, _ in matches]
    
    # img_matches = np.hstack((img1, img2))
    # h1, w1 = img1.shape[:2]
    
    # for idx1, idx2 in matches[:]:
    #     pt1 = tuple(np.round(keypoints1[idx1].pt).astype(int))
    #     pt2 = tuple(np.round(keypoints2[idx2].pt).astype(int) + np.array([w1, 0]))
    #     color = tuple(np.random.randint(0, 255, 3).tolist())
    #     cv2.line(img_matches, pt1, pt2, color, 1)
    #     cv2.circle(img_matches, pt1, 4, color, 1)
    #     cv2.circle(img_matches, pt2, 4, color, 1)
    
    # cv2.imshow(f"{output_name} - Feature Matching", img_matches)
    # cv2.imwrite(f"./output/{output_name} - Feature Matching.png", img_matches)
    # cv2.waitKey(0)
    
    return matches
# ==

# =Compute Fundamental Matrix=
def normalize_points(points):
    # Compute the centroid and Shift the points
    centroid = np.mean(points, axis=0)
    shifted_points = points - centroid

    # Compute the average distance
    avg_dist = np.mean(np.linalg.norm(shifted_points, axis=1))

    # Scale the points
    scale = np.sqrt(2) / avg_dist
    normalized_points = shifted_points * scale

    # Construct the normalization matrix
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

    return normalized_points, T


def compute_fundamental(pts1, pts2):
    """Compute the fundamental matrix using the normalized 8-point algorithm."""
    # # Step 1: Normalize points
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)
    
    # =Construct A matrix=
    A = []
    for (x1, y1), (x2, y2) in zip(pts1_norm, pts2_norm):
        A.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])
    
    # =Solve Af=0 =
    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)  # Last column of V gives the solution
    
    # Step 4: Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F_norm)
    S[-1] = 0  # Set the smallest singular value to 0
    F_norm = U @ np.diag(S) @ Vt
    
    # Step 5: Denormalize the fundamental matrix
    F = T2.T @ F_norm @ T1
    return F
# ==

# =Use Ransac to optimize the Fundamental Matrix=
def find_fundamental_matrix(pts1, pts2, threshold=0.001, max_iters=10000):
    """Robustly estimate the fundamental matrix using RANSAC."""
    best_F = None
    best_inliers = None
    max_inliers = 0

    for _ in range(max_iters):
        # Step 1: Randomly sample 8 points
        sample_indices = np.random.choice(len(pts1), 8, replace=False)
        sampled_pts1 = pts1[sample_indices]
        sampled_pts2 = pts2[sample_indices]

        # Step 2: Compute the fundamental matrix from the sampled points
        F = compute_fundamental(sampled_pts1, sampled_pts2)

        # Step 3: Compute epipolar errors for all points
        n_points = pts1.shape[0]
        pts1_h = np.concat((pts1, np.ones((n_points, 1))), axis=1)
        pts2_h = np.concat((pts2, np.ones((n_points, 1))), axis=1)
        errors = np.abs(np.sum(pts2_h @ F * pts1_h, axis=1))
        
        # Filter out the inliers
        inliers = errors < threshold
        num_inliers = np.sum(inliers)
        
        # Update the best model if the number of inliers is greater
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_F = F
            best_inliers = inliers
    
    if best_inliers is not None:
        best_F = compute_fundamental(pts1[best_inliers], pts2[best_inliers])

    return best_F, best_inliers
# ==

# =Draw epipolar lines=
def my_find_epilines(pts, from_img_idx, F):
    homo_pts = np.concat((pts, np.ones((pts.shape[0], 1))), axis=1)
    
    if from_img_idx == 1:
        lines = (F @ homo_pts.T).T
    elif from_img_idx == 2:
        lines = (F.T @ homo_pts.T).T
    else:
        raise ValueError("from_img_idx must be 1 or 2")

    # Normalize each line so that sqrt(a^2 + b^2) = 1
    lines /= np.sqrt(lines[:, 0]**2 + lines[:, 1]**2).reshape(-1, 1)

    return lines

def draw_epilines(img1, img2, pts1, pts2, F):
    """Draw epilines on both images."""
    # # Epilines for points in the first image (draw on the second image)
    # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    # img1_with_lines = img1.copy()
    # for r, pt1 in zip(lines1, pts1):
    #     color = tuple(np.random.randint(0, 255, 3).tolist())
    #     x0, y0 = map(int, [0, -r[2] / r[1]])
    #     x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
    #     img1_with_lines = cv2.line(img1_with_lines, (x0, y0), (x1, y1), color, 1)
    #     img1_with_lines = cv2.circle(img1_with_lines, tuple(map(int, pt1)), 5, color, -1)

    # # Epilines for points in the second image (draw on the first image)
    # lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    # img2_with_lines = img2.copy()
    # for r, pt2 in zip(lines2, pts2):
    #     color = tuple(np.random.randint(0, 255, 3).tolist())
    #     x0, y0 = map(int, [0, -r[2] / r[1]])
    #     x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]])
    #     # img2_with_lines = cv2.line(img2_with_lines, (x0, y0), (x1, y1), color, 1)
    #     img2_with_lines = cv2.circle(img2_with_lines, tuple(map(int, pt2)), 5, color, -1)
    
    # =2=
    # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    # lines1 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines1 = my_find_epilines(pts2, 2, F)
    img2_with_lines = img2.copy()
    img1_with_lines = img1.copy()
    w = img1.shape[1]
    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist()) # RGB color
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [w, -(r[2] + r[0] * w) / r[1]])
        img1_with_lines = cv2.line(img1_with_lines, (x0, y0), (x1, y1), color, 1)
        # img1_with_lines = cv2.circle(img1_with_lines, tuple(map(int, pt1)), 5, color, -1)
        img2_with_lines = cv2.circle(img2_with_lines, tuple(map(int, pt2)), 5, color, -1)

    return img1_with_lines, img2_with_lines

def step_123(img1_name, img2_name):
    # Load the images
    img1 = cv2.imread('./data/Statue1.bmp')  # Replace with your actual image paths
    img2 = cv2.imread('./data/Statue2.bmp')

    # K1 = np.array([
    #     [5426.566895, 0.678017, 330.096680],
    #     [0.000000, 5423.133301, 648.950012],
    #     [0.000000, 0.000000, 1.000000]
    # ])

    # K2 = np.array([
    #     [5426.566895, 0.678017, 387.430023],
    #     [0.000000, 5423.133301, 620.616699],
    #     [0.000000, 0.000000, 1.000000]
    # ])

    # =1=
    # # Step 1: Find correspondences across images
    # sift = cv2.SIFT_create()
    # keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    # keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # # Match features using BFMatcher
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # # Apply ratio test to keep good matches
    # good_matches = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good_matches.append(m)

    # =2=
    keypoints1, descriptors1 = detect_and_describe(img1)
    keypoints2, descriptors2 = detect_and_describe(img2)

    matches = match_features(img1, img2, descriptors1, descriptors2, keypoints1, keypoints2)

    # Extract point coordinates from good matches
    pts1 = np.float32([keypoints1[m[0]].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([keypoints2[m[1]].pt for m in matches]).reshape(-1, 2)

    # Step 2: Estimate the fundamental matrix
    # F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_8POINT)
    F, inliers = find_fundamental_matrix(pts1, pts2)

    # # Filter inliers based on the mask
    # inliers_pts1 = pts1[mask.ravel() == 1]
    # inliers_pts2 = pts2[mask.ravel() == 1]

    # Draw epipolar lines
    img1_lines, img2_lines = draw_epilines(img1, img2, pts1[inliers], pts2[inliers], F)

    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_lines)
    plt.title("Epilines on Image 1")
    plt.subplot(1, 2, 2)
    plt.imshow(img2_lines)
    plt.title("Epilines on Image 2")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    img1_name = './data/Statue1.bmp'
    img2_name = './data/Statue2.bmp'
    step_123(img1_name, img2_name)