# single_cam_trajectory.py
import cv2
import numpy as np

VIDEO_PATH = "data/videos/test9.mp4"        # change if needed
REF_PATH   = "data/ref/ref10.jpg"           # good reference image
H_INV_PATH = "data/calibration/H_inv.npy"  # from calibration step


def load_h_inv(path):
    H_inv = np.load(path)
    if H_inv.shape != (3, 3):
        raise RuntimeError("H_inv.npy must be a 3x3 matrix")
    return H_inv


def image_to_world(cx, cy, H_inv):
    """Map image center (cx, cy) to world (X, Y) using inverse homography."""
    pt_img = np.array([cx, cy, 1.0], dtype=np.float32)
    pt_world_h = H_inv @ pt_img
    pt_world_h /= pt_world_h[2]
    X = float(pt_world_h[0])
    Y = float(pt_world_h[1])
    return X, Y


def world_to_map(X, Y, img_size=(600, 600), scale=200.0, margin=50):
    """
    Convert world coords (X, Y) to pixel coords on a top-down map image.
    Assumes world roughly in [0,1]x[0,1] for now; adjust scale/margin as needed.
    """
    h, w = img_size
    mx = int(X * scale + margin)
    my = int((1.0 - Y) * scale + margin)  # flip Y so up is up
    return mx, my


def main():
    # -----------------------------
    # Load calibration and reference
    # -----------------------------
    H_inv = load_h_inv(H_INV_PATH)

    ref = cv2.imread(REF_PATH, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise RuntimeError(f"Could not load reference image at {REF_PATH}")

    # SIFT
    sift = cv2.SIFT_create(
        nfeatures=3000,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )
    kp_ref, des_ref = sift.detectAndCompute(ref, None)
    if des_ref is None or len(des_ref) < 5:
        raise RuntimeError("Not enough descriptors in reference image")

    # FLANN
    index_params = dict(algorithm=1, trees=5)  # KDTree for SIFT
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    # Trajectory: list of (X, Y)
    trajectory = []

    # Map image (top-down)
    map_h, map_w = 600, 600
    map_img = np.zeros((map_h, map_w, 3), dtype=np.uint8)

    # Parameters
    LOWE_RATIO = 0.7
    MIN_INLIERS = 15
    RANSAC_THRESH = 4.0

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(gray, None)

        if des_frame is None or len(des_frame) < 5:
            cv2.putText(frame, "No frame descriptors", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # FLANN matches
            matches = flann.knnMatch(des_ref, des_frame, k=2)
            good = []
            for m, n in matches:
                if m.distance < LOWE_RATIO * n.distance:
                    good.append(m)

            if len(good) >= 4:
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
                    cv2.putText(frame, f"Inliers: {inliers}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if inliers >= MIN_INLIERS:
                        h_ref, w_ref = ref.shape
                        box = np.float32([
                            [0, 0],
                            [w_ref, 0],
                            [w_ref, h_ref],
                            [0, h_ref],
                        ]).reshape(-1, 1, 2)
                        projected = cv2.perspectiveTransform(box, H)
                        frame = cv2.polylines(frame, [np.int32(projected)],
                                              True, (0, 255, 0), 3)

                        # Compute center of box in image coords
                        pts = projected.reshape(-1, 2)
                        cx = float(pts[:, 0].mean())
                        cy = float(pts[:, 1].mean())

                        cv2.circle(frame, (int(cx), int(cy)), 4,
                                   (0, 0, 255), -1)

                        # Map to world coords
                        X, Y = image_to_world(cx, cy, H_inv)
                        trajectory.append((X, Y))

                        # Draw on top-down map
                        mx, my = world_to_map(X, Y,
                                              img_size=(map_h, map_w),
                                              scale=200, margin=50)
                        cv2.circle(map_img, (mx, my), 2,
                                   (0, 255, 0), -1)

                else:
                    cv2.putText(frame, "Homography failed", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Too few good matches", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Some debug overlay
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show both windows
        cv2.imshow("Video + Detection", frame)
        cv2.imshow("World Trajectory", map_img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
