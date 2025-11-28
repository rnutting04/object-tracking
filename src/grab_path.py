import cv2
import numpy as np
import matplotlib.pyplot as plt 
import csv
import os

import calibrate_homography
import fuse_trajectories



def load_h_inv(path):
    """
    Loads the homography matrix from a .npy file
    """

    H_inv = np.load(path)
    if H_inv.shape != (3, 3):
        raise RuntimeError("H_inv.npy must be a 3x3 matrix")
    return H_inv


def image_to_world(cx, cy, H_inv):
    """
    Converts a point in the image to the world coordinates
    """

    pt_img = np.array([cx, cy, 1.0], dtype=np.float32)
    pt_world_h = H_inv @ pt_img
    pt_world_h /= pt_world_h[2]
    return float(pt_world_h[0]), float(pt_world_h[1])


def world_to_map(X, Y, img_size=(600, 600), margin=50):
    """
    Converts a point in the world to the image coordinates
    """
    h, w = img_size
    scale_x = w - 2 * margin
    scale_y = h - 2 * margin
    mx = int(X * scale_x + margin)
    my = int((1.0 - Y) * scale_y + margin)  # flip Y so up is up
    return mx, my


def process_camera(cam_id:str, video_path:str, ref_paths:list, h_inv_path:str, csv_out: str, plot_out: str):
    """
    Processes a single camera video
    """

    if not os.path.exists(h_inv_path):
        print(f"Calibrating homography for {cam_id}...")
        calibrate_homography.calibrate_now(video_path, h_inv_path)
    else:
        print(f"Loading homography for {cam_id} from {h_inv_path}...")

    H_inv = load_h_inv(h_inv_path)

    # SIFT
    sift = cv2.SIFT_create(
        nfeatures=3000,
        contrastThreshold=0.06,
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
    RANSAC_THRESH = 3.0

    # Smoothing state
    SMOOTH_ALPHA = 0.2      # 0=no smoothing, 1=heavy smoothing
    smooth_X = None
    smooth_Y = None

    frame_idx = 0
    csv_file = open(csv_out, "w", newline="")
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
                            "matches": good,
                        }

            if best_candidate is not None:
                ref_data = best_candidate["ref"]
                H = best_candidate["H"]
                inliers = best_candidate["inliers"]
                ref_idx = best_candidate["ref_idx"]

                good_matches = best_candidate["matches"]
                mask = best_candidate["mask"]
                kp_ref = ref_data["kp"]

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
                bottom_mid = (pts[1] + pts[3]) / 2.0
                cx, cy = float(bottom_mid[0]), float(bottom_mid[1])

                COLOR_POINT = (0, 0, 255)
                cv2.circle(frame, (int(cx), int(cy)), 4, COLOR_POINT, -1)

                """
                TESTING CODE TO PLOT REFERENCE POINT PROJECTION
                """

                ref_vis = cv2.cvtColor(ref_data["img"], cv2.COLOR_GRAY2BGR)

                matches_mask = mask.ravel().tolist()    

                for i, (m, is_inlier) in enumerate(zip(good_matches, matches_mask)):
                    if is_inlier:
                        # 1. Draw on Reference Image (Existing logic)
                        ref_pt_idx = m.queryIdx
                        (rx_pt, ry_pt) = kp_ref[ref_pt_idx].pt
                        cv2.circle(ref_vis, (int(rx_pt), int(ry_pt)), 3, (0, 255, 0), -1)

                        # 2. Draw on Video Frame (NEW logic)
                        frame_pt_idx = m.trainIdx
                        (fx, fy) = kp_frame[frame_pt_idx].pt
                        # Drawing Blue dots (255, 0, 0) to distinguish from the Red foot point
                        cv2.circle(frame, (int(fx), int(fy)), 3, (255, 0, 0), -1)

                try:
                    H_inv_ref = np.linalg.inv(H)
                    
                    pt_frame_h = np.array([cx, cy, 1.0], dtype=np.float32)
                    pt_ref_h = H_inv_ref @ pt_frame_h
                    
                    # Check for division by zero in perspective division
                    if abs(pt_ref_h[2]) > 1e-6:
                        pt_ref_h /= pt_ref_h[2]
                        rx, ry = float(pt_ref_h[0]), float(pt_ref_h[1])
                        cv2.circle(ref_vis, (int(rx), int(ry)), 6, COLOR_POINT, -1)
                except np.linalg.LinAlgError:
                    print(f"[WARN] Frame {frame_idx}: Singular matrix encountered. Skipping reference projection.")

                ref_vis = cv2.resize(ref_vis, (600, 400))

                cv2.imshow("Current Reference Image", ref_vis)

                """
                END TESTING CODE
                """

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

        plt.figure(figsize=(6, 6))

        plt.scatter(xs, ys, s=5, c='blue', alpha=0.6)

        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

        ticks = np.arange(0.0, 1.1, 0.1)
        plt.xticks(ticks)
        plt.yticks(ticks)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
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
    """
    Script used to run the entire pipeline.
    """

    REF_PATHS  = [
        "data/ref/ref14_vid_1.png",
        'data/ref/ref14_vid_2.png',
        # 'data/ref/ref14_vid_3.png',
        # 'data/ref/ref14_vid_4.png',
        # "data/ref/ref14_vid_5.png",
        "data/ref/ref14_5_blurry.png",
        # "data/ref/ref14_vid_side1.png",
        # "data/ref/ref14_vid_side2.png",
        # "data/ref/ref14_vid_top1.png",
        # "data/ref/ref14_vid_top2.png",
    ]

    TEST_CONFIGS = [
        {
            "id": "test32_1",
            "ext": "mov",
        },
        {
            "id": "test32_2",
            "ext": "mov",
        }
    ]

    GROUND_TRUTH = [
        {
            "id": "test32_ground",
            "ext": "mov",
        }
    ]


    # Run the pipeline for eachh test file.
    for config in TEST_CONFIGS:
        cam_id = config["id"]
        extension = config["ext"]
        video_path = f'data/videos/{cam_id}.{extension}'
        h_inv_path = f'data/calibration/H_{cam_id}.npy'
        
        # Output files based on the configuration ID
        csv_out = f"output/trajectory_run_{cam_id}.csv"
        plot_out = f"output/trajectory_plot_{cam_id}.png"
        
        print(f"\n--- Running Camera: {cam_id} ---")
        
        process_camera(
            cam_id, 
            video_path, 
            REF_PATHS, 
            h_inv_path, 
            csv_out,
            plot_out
        )

    # Run the ground truth pipeline
    process_camera(GROUND_TRUTH[0]["id"], f'data/videos/{GROUND_TRUTH[0]["id"]}.{GROUND_TRUTH[0]["ext"]}', REF_PATHS, f'data/calibration/H_{GROUND_TRUTH[0]["id"]}.npy', f'output/ground_run_{GROUND_TRUTH[0]["id"]}.csv', f'output/ground_plot_{GROUND_TRUTH[0]["id"]}.png')

    print("\n--- All runs complete ---")

    # Fuse the trajectories if there are two test files
    if len(TEST_CONFIGS) >= 2:
        print("Fusing trajectories...")
        id1 = TEST_CONFIGS[0]["id"]
        id2 = TEST_CONFIGS[1]["id"]
        csv1 = f"trajectory_run_{id1}.csv"
        csv2 = f"trajectory_run_{id2}.csv"
        cam1_data = fuse_trajectories.load_csv(csv1)
        cam2_data = fuse_trajectories.load_csv(csv2)
        off_x , off_y = fuse_trajectories.calculate_offset(cam1_data, cam2_data)
        print(f"Offset X: {off_x:.4f}, Y: {off_y:.4f}")
        fuse_trajectories.fuse(
            cam1=cam1_data,
            cam2=cam2_data,
            out_csv=f"output/fused_trajectory_{id1}_{id2}.csv",
            plot_png=f"output/fused_plot_{id1}_{id2}.png",
            cam1_offset=(0, 0),
            cam2_offset=(off_x, off_y),
        )