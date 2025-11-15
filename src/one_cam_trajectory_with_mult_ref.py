import cv2
import numpy as np

VIDEO_PATH = "data/videos/test9.mp4"          # change if needed
REF_PATHS  = [
    "data/ref/ref10_near.jpg",
    "data/ref/ref10_mid.jpg",
    "data/ref/ref10_far.jpg",
]                                             # multiple reference images
H_INV_PATH = "data/calibration/H_inv.npy"     # from calibration step


def load_h_inv(path):
    H_inv = np.load(path)
    if H_inv.shape != (3, 3):
        raise RuntimeError("H_inv.npy must be a 3x3 matrix")
    return H_inv


def image_to_world(cx, cy, H_inv):
    """Map image point (cx, cy) to world (X, Y) using inverse homography."""
    pt_img = np.array([cx, cy, 1.0], dtype=np.float32)
    pt_world_h = H_inv @ pt_img
    pt_world_h /= pt_world_h[2]
    X = float(pt_world_h[0])
    Y = float(pt_world_h[1])
    return X, Y


def world_to_map(X, Y, img_size=(600, 600), scale=200.0, margin=50):
    """
    Convert world coords (X, Y) to pixel coords on a top-down map image.
    Assumes world roughly in [0,1]x[0,1]; adjust scale/margin as needed.
    """
    h, w = img_size
    mx = int(X * scale + margin)
    my = int((1.0 - Y) * scale + margin)  # flip Y so up is up
    return mx, my


def main():
    # -----------------------------
    # Load calibration and references
    # -----------------------------
    H_inv = load_h_inv(H_INV_PATH)

    # SIFT
    sift = cv2.SIFT_create(
        nfeatures=3000,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )

    # Load all refs and precompute keypoints/descriptors
    refs = []
    for path in REF_PATHS:
        ref_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            print(f"[WARN] Could not load reference image at {path}, skipping.")
            continue
        kp_ref, des_ref = sift.detectAndCompute(ref_img, None)
        if des_ref is None or len(des_ref) < 5:
            print(f"[WARN] Not enough descriptors in {path}, skipping.")
            continue
        refs.append({
            "path": path,
            "img": ref_img,
            "kp": kp_ref,
            "des": des_ref,
            "h": ref_img.shape[0],
            "w": ref_img.shape[1],
        })

    if not refs:
        raise RuntimeError("No valid reference images loaded.")

    print(f"Loaded {len(refs)} reference images.")

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
    LOWE_RATIO    = 0.7
    MIN_INLIERS   = 15
    RANSAC_THRESH = 4.0

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(gray, None)

        best_candidate = None  # will store dict with ref, H, mask, inliers

        if des_frame is None or len(des_frame) < 5:
            cv2.putText(frame, "No frame descriptors", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Try all references and pick the one with the most inliers
            for ref_idx, ref_data in enumerate(refs):
                des_ref = ref_data["des"]
                kp_ref  = ref_data["kp"]

                matches = flann.knnMatch(des_ref, des_frame, k=2)
                good = []
                for m, n in matches:
                    if m.distance < LOWE_RATIO * n.distance:
                        good.append(m)

                if len(good) < 4:
                    continue

                src_pts = np.float32(
                    [kp_ref[m.queryIdx].pt for m in good]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp_frame[m.trainIdx].pt for m in good]
                ).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH,
                    maxIters=2000
                )
                if H is None or mask is None:
                    continue

                inliers = int(mask.sum())

                # keep the ref with the highest inlier count
                if inliers >= MIN_INLIERS:
                    if (best_candidate is None) or (inliers > best_candidate["inliers"]):
                        best_candidate = {
                            "ref_idx": ref_idx,
                            "ref": ref_data,
                            "H": H,
                            "mask": mask,
                            "inliers": inliers,
                        }

            # If we found a good ref for this frame, use it
            if best_candidate is not None:
                ref_data = best_candidate["ref"]
                H = best_candidate["H"]
                inliers = best_candidate["inliers"]
                ref_idx = best_candidate["ref_idx"]

                cv2.putText(frame, f"Inliers: {inliers} (ref {ref_idx})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                h_ref, w_ref = ref_data["h"], ref_data["w"]
                box = np.float32([
                    [0, 0],
                    [w_ref, 0],
                    [w_ref, h_ref],
                    [0, h_ref],
                ]).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(box, H)
                frame = cv2.polylines(frame, [np.int32(projected)],
                                      True, (0, 255, 0), 3)

                # Use bottom midpoint as "contact point" on the floor
                pts = projected.reshape(-1, 2)  # [tl, tr, br, bl]
                bottom_mid = (pts[2] + pts[3]) / 2.0
                cx, cy = float(bottom_mid[0]), float(bottom_mid[1])

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
                cv2.putText(frame, "No ref passed inlier threshold", (10, 60),
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
