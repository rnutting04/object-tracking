import cv2
import numpy as np
import matplotlib.pyplot as plt 
import csv



def load_h_inv(path):
    H_inv = np.load(path)
    if H_inv.shape != (3, 3):
        raise RuntimeError("H_inv.npy must be a 3x3 matrix")
    return H_inv


def image_to_world(cx, cy, H_inv):
    pt_img = np.array([cx, cy, 1.0], dtype=np.float32)
    pt_world_h = H_inv @ pt_img
    pt_world_h /= pt_world_h[2]
    return float(pt_world_h[0]), float(pt_world_h[1])


def world_to_map(X, Y, img_size=(600, 600), margin=50):
    """Fixed mapping: assume world roughly in [0,1]x[0,1]."""
    h, w = img_size
    scale_x = w - 2 * margin
    scale_y = h - 2 * margin
    mx = int(X * scale_x + margin)
    my = int((1.0 - Y) * scale_y + margin)  # flip Y so up is up
    return mx, my


def process_camera(cam_id:str, video_path:str, ref_paths:list, h_inv_path:str, csv_out: str, plot_out: str):

    H_inv = load_h_inv(h_inv_path)

    # SIFT
    sift = cv2.SIFT_create(
        nfeatures=3000,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )

    # Load refs
    refs = []
    for path in ref_paths:
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
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    # Trajectories (raw + smoothed)
    trajectory_raw = []     # list of (X, Y)
    trajectory_smooth = []  # list of (Xs, Ys)

    map_h, map_w = 600, 600
    map_img = np.zeros((map_h, map_w, 3), dtype=np.uint8)

    LOWE_RATIO    = 0.7
    MIN_INLIERS   = 12      # slightly relaxed
    RANSAC_THRESH = 4.0

    # Smoothing state
    SMOOTH_ALPHA = 0.2      # 0=no smoothing, 1=heavy smoothing
    smooth_X = None
    smooth_Y = None

    frame_idx = 0
    csv_file = open("trajectory_run1.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "X", "Y", "Xs", "Ys", "inliers", "ref_idx"])
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(gray, None)

        best_candidate = None

        if des_frame is None or len(des_frame) < 5:
            cv2.putText(frame, "No frame descriptors", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Try all references, pick the one with most inliers
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

                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH, maxIters=2000
                )
                if H is None or mask is None:
                    continue

                inliers = int(mask.sum())
                if inliers >= MIN_INLIERS:
                    if best_candidate is None or inliers > best_candidate["inliers"]:
                        best_candidate = {
                            "ref_idx": ref_idx,
                            "ref": ref_data,
                            "H": H,
                            "mask": mask,
                            "inliers": inliers,
                        }

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

                # bottom midpoint (feet)
                pts = projected.reshape(-1, 2)  # tl, tr, br, bl
                bottom_mid = (pts[2] + pts[3]) / 2.0
                cx, cy = float(bottom_mid[0]), float(bottom_mid[1])

                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

                # World coords (raw)
                X, Y = image_to_world(cx, cy, H_inv)
                trajectory_raw.append((X, Y))

                # ---- smoothing ----
                if smooth_X is None:
                    smooth_X, smooth_Y = X, Y
                else:
                    smooth_X = (1.0 - SMOOTH_ALPHA) * smooth_X + SMOOTH_ALPHA * X
                    smooth_Y = (1.0 - SMOOTH_ALPHA) * smooth_Y + SMOOTH_ALPHA * Y

                csv_writer.writerow([frame_idx, X, Y, smooth_X, smooth_Y, inliers, ref_idx])

                trajectory_smooth.append((smooth_X, smooth_Y))

                # Draw smoothed path on map
                mx, my = world_to_map(smooth_X, smooth_Y,
                                      img_size=(map_h, map_w),
                                      margin=50)
                cv2.circle(map_img, (mx, my), 2, (0, 0, 255), -1)

            else:
                cv2.putText(frame, "No ref passed inlier threshold", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Video + Detection", frame)
        cv2.imshow("World Trajectory", map_img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    csv_file.close()
    # ------------- Final matplotlib plot -------------
    if trajectory_smooth:
        xs = [p[0] for p in trajectory_smooth]
        ys = [p[1] for p in trajectory_smooth]

        plt.figure(figsize=(5, 7))
        plt.scatter(xs, ys, s=5)
        plt.gca().invert_yaxis()   # match "up is up" like our map
        plt.axis('equal')
        plt.title("World trajectory (smoothed)")
        plt.xlabel("World X")
        plt.ylabel("World Y")
        plt.tight_layout()
        plt.savefig(plot_out, dpi=200)
        print("Saved plot to trajectory_plot.png")
    else:
        print("No trajectory points recorded – nothing to plot.")


if __name__ == "__main__":
    VIDEO_PATH = "data/videos/test9.mp4"
    REF_PATHS  = [
        "data/ref/ref10_near.jpg",
        "data/ref/ref10_mid.jpg",
        "data/ref/ref10_far.jpg",
    ]
    H_INV_PATH = "data/calibration/H_inv.npy"
    process_camera("cam1", VIDEO_PATH, REF_PATHS, H_INV_PATH, "trajectory_run1.csv", "trajectory_plot.png")
