import cv2
import numpy as np


def main():
    # -----------------------------
    # 1. Load ONE good reference image
    # -----------------------------
    ref_path = "data/ref/ref5.jpg"   # <- CHANGE THIS IF NEEDED

    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise RuntimeError(f"Could not load reference image at {ref_path}")

    print("[INFO] Loaded reference:", ref_path, "shape:", ref.shape)

    # -----------------------------
    # 2. Open video source
    # -----------------------------
    # For prerecorded video:
    cap = cv2.VideoCapture("data/videos/test3.mp4")  # <- CHANGE IF NEEDED

    # For live webcam, comment line above and uncomment:
    # cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    # -----------------------------
    # 3. Create SIFT + FLANN
    # -----------------------------
    sift = cv2.SIFT_create(
        nfeatures=3000,          # moderate
        contrastThreshold=0.04,  # default-ish
        edgeThreshold=10,
        sigma=1.6,
    )

    kp_ref, des_ref = sift.detectAndCompute(ref, None)
    if des_ref is None or len(des_ref) < 5:
        raise RuntimeError("Not enough descriptors in reference image")

    print("[INFO] Reference keypoints:", len(kp_ref))

    # FLANN for SIFT
    index_params = dict(algorithm=1, trees=5)  # KDTree
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # -----------------------------
    # 4. Parameters
    # -----------------------------
    MIN_INLIERS = 8          # very lenient for now
    LOWE_RATIO = 0.75
    RANSAC_THRESH = 5.0

    frame_idx = 0

    # -----------------------------
    # 5. Main loop
    # -----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video or cannot read frame.")
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # SIFT on frame
        kp_frame, des_frame = sift.detectAndCompute(gray, None)
        if des_frame is None or len(des_frame) < 5:
            cv2.putText(frame, "No frame descriptors", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # KNN matches ref -> frame
        matches = flann.knnMatch(des_ref, des_frame, k=2)

        good = []
        for m, n in matches:
            if m.distance < LOWE_RATIO * n.distance:
                good.append(m)

        num_good = len(good)

        # Default overlay with debug info
        cv2.putText(frame, f"Frame {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Ref KP: {len(kp_ref)}  Frame KP: {len(kp_frame)}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Good matches: {num_good}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if num_good >= 4:
            src_pts = np.float32(
                [kp_ref[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp_frame[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts,
                                         cv2.RANSAC, RANSAC_THRESH,
                                         maxIters=2000)

            if H is not None and mask is not None:
                inliers = int(mask.sum())
                cv2.putText(frame, f"Inliers: {inliers}",
                            (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if inliers >= MIN_INLIERS:
                    h_ref, w_ref = ref.shape
                    box = np.float32([
                        [0, 0],
                        [w_ref, 0],
                        [w_ref, h_ref],
                        [0, h_ref]
                    ]).reshape(-1, 1, 2)

                    projected = cv2.perspectiveTransform(box, H)
                    frame = cv2.polylines(frame, [np.int32(projected)], True,
                                          (0, 255, 0), 3)
                else:
                    cv2.putText(frame, "Inliers below threshold",
                                (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Homography failed",
                            (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Too few good matches",
                        (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
