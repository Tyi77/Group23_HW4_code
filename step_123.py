import cv2
import numpy as np
import matplotlib.pyplot as plt


# =Interest points detection & feature description by SIFT=
def detect_and_describe(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    return keypoints, descriptors

# =Feature matching by SIFT features=
def match_features(descriptors1, descriptors2, ratio=0.75):
    matches = []
    
    for i, desc1 in enumerate(descriptors1):
        distances = np.linalg.norm(descriptors2 - desc1, axis=1)
        sorted_indices = np.argsort(distances)
        
        if distances[sorted_indices[0]] < ratio * distances[sorted_indices[1]]:
            matches.append((i, sorted_indices[0], distances[sorted_indices[0]]))

    matches = sorted(matches, key=lambda x: x[2])
    matches = [(i, j) for i, j, _ in matches]
    
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
    # Step 1: Normalize points
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)
    
    # Construct A matrix
    A = []
    for (x1, y1), (x2, y2) in zip(pts1_norm, pts2_norm):
        A.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])
    
    # Solve Af=0
    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F_norm)
    S[-1] = 0  # Set the last singular value to 0
    F_norm = U @ np.diag(S) @ Vt
    
    # Denormalize the fundamental matrix
    F = T2.T @ F_norm @ T1
    return F
# ==

# =Use Ransac to optimize the Fundamental Matrix=
def find_fundamental_matrix(pts1, pts2, threshold=0.0005, max_iters=10000):
    """Robustly estimate the fundamental matrix using RANSAC."""
    best_F = None
    best_inliers = None
    max_inliers = 0

    for _ in range(max_iters):
        # Randomly sample 8 points
        sample_indices = np.random.choice(len(pts1), 8, replace=False)
        sampled_pts1 = pts1[sample_indices]
        sampled_pts2 = pts2[sample_indices]

        # Compute the fundamental matrix from the sampled points
        F = compute_fundamental(sampled_pts1, sampled_pts2)

        # Compute epipolar errors for all points
        n_points = pts1.shape[0]
        pts1_h = np.concatenate((pts1, np.ones((n_points, 1))), axis=1)
        pts2_h = np.concatenate((pts2, np.ones((n_points, 1))), axis=1)
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
    homo_pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    
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

def step_123(img1_name, img2_name, fundamental_threshold, fundamental_iter):
    # Load the images
    img1 = cv2.imread(img1_name)
    img2 = cv2.imread(img2_name)

    # find corresponding feature points
    keypoints1, descriptors1 = detect_and_describe(img1)
    keypoints2, descriptors2 = detect_and_describe(img2)

    matches = match_features(descriptors1, descriptors2)

    # Extract point coordinates from matches
    pts1 = np.float32([keypoints1[m[0]].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([keypoints2[m[1]].pt for m in matches]).reshape(-1, 2)

    # Estimate the fundamental matrix and inliers
    F, inliers = find_fundamental_matrix(pts1, pts2, threshold=fundamental_threshold, max_iters=fundamental_iter)

    # Draw epipolar lines
    img1_lines, img2_lines = draw_epilines(img1, img2, pts1[inliers], pts2[inliers], F)

    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_lines)
    plt.title("Epilines on Image 1")
    plt.subplot(1, 2, 2)
    plt.imshow(img2_lines)
    plt.title("Feature points on Image 2")
    plt.tight_layout()
    plt.show()

    return F, pts1[inliers], pts2[inliers]

if __name__ == '__main__':
    # img1_name = './data/Statue1.bmp'
    # img2_name = './data/Statue2.bmp'
    img1_name = './data/Mesona1.JPG'
    img2_name = './data/Mesona2.JPG'
    step_123(img1_name, img2_name)