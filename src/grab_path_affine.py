import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

VIDEO_PATH = "data/videos/test23_angle1.mov"
REF_PATHS  = [
    "data/ref/ref14_1.jpg",
    "data/ref/ref14_2.jpg",
    "data/ref/ref14_3.jpg",
    "data/ref/ref14_4.jpg",
]
H_INV_PATH = "data/calibration/H_test23_angle1.npy"
CSV_OUT    = "cam1_ref_affine_run.csv"
PLOT_OUT   = "cam1_ref_affine_run.png"


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
    h, w = img_size
    scale_x = w - 2 * margin
    scale_y = h - 2 * margin
    mx = int(X * scale_x + margin)
    my = int((1.0 - Y) * scale_y + margin)  # flip Y so up is up
    return mx, my


def main():
    H_inv = load_h_inv(H_INV_PATH)

    # ---- SIFT ----
    sift = cv2.SIFT_create(
        nfeatures=3000,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )

    # ---- Load reference images ----
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

    # ---- FLANN ----
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    map_h, map_w = 600, 600
    map_img = np.zeros((map_h, map_w, 3), dtype=np.uint8)

    LOWE_RATIO      = 0.7
    MIN_INLIERS_REF = 12     # threshold for ref-based homography
    RANSAC_THRESH   = 4.0

    MIN_INLIERS_AFFINE = 10  # threshold for frame-to-frame affine

    frame_idx = 0

    # CSV writer
    csv_file = open(CSV_OUT, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "X", "Y", "Xs", "Ys", "inliers", "ref_idx", "mode"])

    # ---- State for affine tracking ----
    prev_gray      = None
    prev_kp_frame  = None
    prev_des_frame = None
    prev_box       = None  # (4,2) corners of object box in previous frame

    # trajectory in world space (for plotting)
    trajectory_world = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(gray, None)

        # ---------------------------
        # 1) Frame-to-frame AFFINE prediction (image space)
        # ---------------------------
        affine_box = None
        affine_inliers = 0

        if (
            prev_des_frame is not None
            and des_frame is not None
            and len(des_frame) >= 5
            and prev_kp_frame is not None
        ):
            matches_pf = flann.knnMatch(prev_des_frame, des_frame, k=2)
            good_pf = []
            for m, n in matches_pf:
                if m.distance < LOWE_RATIO * n.distance:
                    good_pf.append(m)

            if len(good_pf) >= 4:
                src_pts_pf = np.float32(
                    [prev_kp_frame[m.queryIdx].pt for m in good_pf]
                ).reshape(-1, 1, 2)
                dst_pts_pf = np.float32(
                    [kp_frame[m.trainIdx].pt for m in good_pf]
                ).reshape(-1, 1, 2)

                # estimate affine transform prev_frame -> current_frame
                A, inliers_aff = cv2.estimateAffine2D(
                    src_pts_pf, dst_pts_pf,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=2000,
                    confidence=0.99,
                    refineIters=10
                )
                if A is not None and inliers_aff is not None:
                    affine_inliers = int(inliers_aff.sum())
                    if affine_inliers >= MIN_INLIERS_AFFINE and prev_box is not None:
                        prev_box_reshaped = prev_box.reshape(1, -1, 2)  # (1,4,2)
                        pred_box = cv2.transform(prev_box_reshaped, A)[0]  # (4,2)
                        affine_box = pred_box

        # ---------------------------
        # 2) Reference-based detection (multi-ref SIFT + homography)
        # ---------------------------
        best_candidate = None

        if des_frame is not None and len(des_frame) >= 5:
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

                H_obj, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH, maxIters=2000
                )
                if H_obj is None or mask is None:
                    continue

                inliers = int(mask.sum())
                if inliers >= MIN_INLIERS_REF:
                    if best_candidate is None or inliers > best_candidate["inliers"]:
                        best_candidate = {
                            "ref_idx": ref_idx,
                            "ref": ref_data,
                            "H": H_obj,
                            "mask": mask,
                            "inliers": inliers,
                        }

        # ---------------------------
        # 3) Choose which box to use: REF vs AFFINE
        # ---------------------------
        final_box = None
        final_mode = "none"
        inliers_used = 0
        ref_idx_used = -1

        if best_candidate is not None:
            # Prefer ref-based when strong enough
            ref_data = best_candidate["ref"]
            H_obj    = best_candidate["H"]
            inliers  = best_candidate["inliers"]
            ref_idx_used = best_candidate["ref_idx"]

            h_ref, w_ref = ref_data["h"], ref_data["w"]
            box = np.float32([
                [0, 0],
                [w_ref, 0],
                [w_ref, h_ref],
                [0, h_ref],
            ]).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(box, H_obj)  # (4,1,2)
            final_box = projected.reshape(-1, 2)
            final_mode = "ref"
            inliers_used = inliers

            # GREEN: reference-based homography
            cv2.polylines(frame, [np.int32(final_box)], True, (0, 255, 0), 3)
            cv2.putText(frame, f"Mode: REF  inliers={inliers}  ref={ref_idx_used}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif affine_box is not None:
            # Fallback: AFFINE prediction
            final_box = affine_box
            final_mode = "affine"
            inliers_used = affine_inliers

            # CYAN: affine-predicted box
            cv2.polylines(frame, [np.int32(final_box)], True, (255, 255, 0), 2)
            cv2.putText(frame, f"Mode: AFFINE  inliers={affine_inliers}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # No detection at all
            cv2.putText(frame, "Mode: NONE (no valid detection)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ---------------------------
        # 4) If we have a final box, map to world & log
        # ---------------------------
        if final_box is not None:
            # convention: tl, tr, br, bl — for ref-based; affine inherits that order
            pts = final_box
            bottom_mid = (pts[1] + pts[3]) / 2.0
            cx, cy = float(bottom_mid[0]), float(bottom_mid[1])

            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            X, Y = image_to_world(cx, cy, H_inv)
            trajectory_world.append((X, Y))

            # Different color on world map depending on mode
            if final_mode == "ref":
                color = (0, 255, 0)         # green
            elif final_mode == "affine":
                color = (255, 255, 0)       # cyan/yellow-ish
            else:
                color = (0, 0, 255)

            mx, my = world_to_map(X, Y, img_size=(map_h, map_w), margin=50)
            cv2.circle(map_img, (mx, my), 2, color, -1)

            # For now Xs, Ys == X, Y (you can later plug Kalman filtered coords)
            csv_writer.writerow([frame_idx, X, Y, X, Y, inliers_used, ref_idx_used, final_mode])

            # Update state for next frame's affine estimation
            prev_box       = final_box.copy()
            prev_gray      = gray.copy()
            prev_kp_frame  = kp_frame
            prev_des_frame = des_frame
        else:
            # No final box; we still keep prev_* so we can try affine next frame
            pass

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Video + Detection (REF + AFFINE)", frame)
        cv2.imshow("World Trajectory", map_img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

    # ---- Final plot in world space ----
    if trajectory_world:
        xs = [p[0] for p in trajectory_world]
        ys = [p[1] for p in trajectory_world]

        plt.figure(figsize=(5, 7))
        plt.plot(xs, ys, "g.", markersize=3)
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.title("World Trajectory (REF + AFFINE)")
        plt.xlabel("World X")
        plt.ylabel("World Y")
        plt.tight_layout()
        plt.savefig(PLOT_OUT, dpi=200)
        print(f"Saved plot to {PLOT_OUT}")
    else:
        print("No trajectory points recorded – nothing to plot.")


if __name__ == "__main__":
    main()
