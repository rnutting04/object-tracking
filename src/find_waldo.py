import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def find_waldo_in_image(scene_path, waldo_ref_path):
    # 1. Load Images
    print(f"Loading Scene: {scene_path}")
    print(f"Loading Reference: {waldo_ref_path}")
    
    scene_img = cv2.imread(scene_path)
    waldo_ref = cv2.imread(waldo_ref_path, cv2.IMREAD_GRAYSCALE)

    if scene_img is None:
        raise FileNotFoundError(f"Could not load scene image at {scene_path}")
    if waldo_ref is None:
        raise FileNotFoundError(f"Could not load reference image at {waldo_ref_path}")

    # Convert scene to grayscale for detection
    scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    h_ref, w_ref = waldo_ref.shape

    # 2. Initialize SIFT
    # nfeatures=0 means "find as many as possible"
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.03, edgeThreshold=10)

    # 3. Detect and Compute Features
    print("Detecting features...")
    kp_ref, des_ref = sift.detectAndCompute(waldo_ref, None)
    kp_scene, des_scene = sift.detectAndCompute(scene_gray, None)

    if des_ref is None or len(des_ref) < 5:
        print("Error: Not enough features in the Waldo reference image.")
        return
    
    print(f"Reference features: {len(des_ref)}")
    print(f"Scene features:     {len(des_scene)}")

    # 4. Match Features using FLANN
    index_params = dict(algorithm=1, trees=5) 
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_ref, des_scene, k=2)

    # 5. Filter Matches (Lowe's Ratio Test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Good matches found: {len(good_matches)}")

    MIN_MATCH_COUNT = 10
    
    if len(good_matches) > MIN_MATCH_COUNT:
        # 6. Locate Object (Homography)
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            # Get corners of the reference image
            pts = np.float32([[0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)
            
            # Project corners into the scene
            dst = cv2.perspectiveTransform(pts, M)

            # Draw bounding box
            result_img = scene_img.copy()
            result_img = cv2.polylines(result_img, [np.int32(dst)], True, (0, 0, 255), 5, cv2.LINE_AA)
            
            # Label it
            label_pos = (int(dst[0][0][0]), int(dst[0][0][1]) - 10)
            cv2.putText(result_img, "Found Waldo!", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Draw the matching keypoints for visualization
            matchesMask = mask.ravel().tolist()
            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
            match_vis = cv2.drawMatches(waldo_ref, kp_ref, scene_gray, kp_scene, good_matches, None, **draw_params)

            # Show results using Matplotlib (easier for large images)
            plt.figure(figsize=(15, 10))
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title("Waldo Found")
            plt.axis('off')
            plt.show()

            # Optional: Show the feature matches
            plt.figure(figsize=(15, 10))
            plt.imshow(match_vis)
            plt.title("Feature Matches")
            plt.axis('off')
            plt.show()
            
        else:
            print("Could not calculate Homography (transformation matrix).")
    else:
        print(f"Not enough matches found - {len(good_matches)}/{MIN_MATCH_COUNT}")
        print("Try a clearer reference image or adjust contrast threshold.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    SCENE_PATH = "data/test1.jpg"   # The big puzzle image
    REF_PATH   = "data/ref/waldo.png"       # The small crop of Waldo
    
    find_waldo_in_image(SCENE_PATH, REF_PATH)