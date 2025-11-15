import cv2
import numpy as np


def main():
    # -----------------------------
    # 1. Load multiple reference images
    # -----------------------------
    ref_paths = [
        "data/ref/ref1.jpg",  # close, centered
        "data/ref/ref2.jpg",  # slight left angle
        "data/ref/ref3.jpg",  # slight right angle
        "data/ref/ref4.jpg",  # farther / smaller view
    ]

    ref_imgs = []
    for p in ref_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not load reference image: {p}")
        ref_imgs.append(img)

    # Filter out any that failed to load
    valid_refs = []
    for i, img in enumerate(ref_imgs):
        if img is not None:
            valid_refs.append((i, img))

    if len(valid_refs) == 0:
        raise RuntimeError("No valid reference images loaded. Check paths in ref_paths.")

    # -----------------------------
    # 2. Open video source
    # -----------------------------
    # For prerecorded video:
    cap = cv2.VideoCapture("data/videos/test1.mp4")

    # For live webcam instead, comment the line above and uncomment this:
    # cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open video source.")

    # -----------------------------
    # 3. Create SIFT + FLANN
    # -----------------------------
    sift = cv2.SIFT_create(
        nfeatures=6000,
        contrastThreshold=0.02,
        edgeThreshold=5,
        sigma=1.2,
    )

    # Precompute keypoints/descriptors for all references
    ref_kp = []
    ref_des = []
    ref_shapes = []

    for idx, img in valid_refs:
        kp, des = sift.detectAndCompute(img, None)
        if des is None:
            print(f"[WARN] No descriptors found in reference #{idx} ({ref_paths[idx]}). Skipping.")
            continue
        ref_kp.append(kp)
        ref_des.append(des)
        ref_shapes.append(img.shape)

    if len(ref_des) == 0:
        raise RuntimeError("No descriptors available for any reference images.")

    # FLANN for SIFT (L2)
    index_params = dict(algorithm=1, trees=5)  # KDTree = 1
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # CLAHE object (reuse every frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Minimum inliers to accept a detection
    MIN_INLIERS = 20

    # -----------------------------
    # 4. Main processing loop
    # -----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Preprocessing ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Contrast limited adaptive histogram equalization
        gray_eq = clahe.apply(gray)

        # Mild sharpen (unsharp mask)
        blur = cv2.GaussianBlur(gray_eq, (0, 0), 1.0)
        gray_proc = cv2.addWeighted(gray_eq, 1.5, blur, -0.5, 0)

        # Detect SIFT features on processed frame
        kp, des = sift.detectAndCompute(gray_proc, None)
        if des is None or len(des) < 2:
            cv2.imshow("Multi-Ref Tracking", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        best_inliers = -1
        best_H = None
        best_projected = None
        best_ref_idx = -1

        # For each reference, try matching and keep the best one
        for i in range(len(ref_des)):
            des_ref = ref_des[i]
            kp_ref_i = ref_kp[i]
            h_ref, w_ref = ref_shapes[i]

            # KNN matches from ref → frame
            matches = flann.knnMatch(des_ref, des, k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) < 8:
                continue

            src_pts = np.float32(
                [kp_ref_i[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            # Adaptive RANSAC threshold based on how many good matches we have
            rth = 5.0 if len(good) < 40 else 3.0
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, rth, maxIters=2000)

            if H is None or mask is None:
                continue

            inliers = int(mask.sum())
            if inliers < MIN_INLIERS:
                continue

            # Compute projected box for this reference
            box = np.float32([
                [0, 0],
                [w_ref, 0],
                [w_ref, h_ref],
                [0, h_ref]
            ]).reshape(-1, 1, 2)

            projected = cv2.perspectiveTransform(box, H)

            # Optionally, reject extremely distorted boxes
            # (simple heuristic: area not too small / not NaN)
            area = cv2.contourArea(projected.astype(np.float32))
            if area < 10:
                continue

            # Keep best reference by inlier count
            if inliers > best_inliers:
                best_inliers = inliers
                best_H = H
                best_projected = projected
                best_ref_idx = i

        # Draw best detection (if any)
        if best_projected is not None and best_H is not None:
            frame = cv2.polylines(frame, [np.int32(best_projected)], True, (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"Ref #{best_ref_idx}  Inliers: {best_inliers}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Multi-Ref Tracking", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
