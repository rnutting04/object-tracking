import cv2
import numpy as np
import csv
import os
import math
import matplotlib.pyplot as plt

# Import YOLO
from ultralytics import YOLO

import calibrate_homography
import fuse_yolo
import align_planes
from grab_path import load_h_inv, image_to_world

SMOOTH_ALPHA = 0.2
MAP_SIZE = (600, 600)

def map_to_window(x, y, r_width, r_height, img_w, img_h, margin=50):
    """
    Maps world coordinates (x, y) to image coordinates (u, v)
    preserving aspect ratio within a static window size.
    """ 

    # Determine safe drawing area
    draw_w = img_w - 2 * margin
    draw_h = img_h - 2 * margin

    # Calculate scale to fit the largest dimension
    scale_x = draw_w / r_width
    scale_y = draw_h / r_height
    scale = min(scale_x, scale_y)

    # Center the plot
    occupied_w = r_width * scale
    occupied_h = r_height * scale

    offset_x = margin + (draw_w - occupied_w) / 2
    offset_y = margin + (draw_h - occupied_h) / 2

    # Map coordinates
    u = int(x * scale + offset_x)
    v = int(img_h - (y * scale + offset_y)) 
    
    return u, v

def process_camera_yolo(cam_id, video_path, h_inv_path, csv_out, plot_out, r_width=1.0, r_height=1.0):
    """ 
    Detects a person using YOLOv8 and maps their feet to world coordinates.
    """
    
    if not os.path.exists(h_inv_path):
        print(f"Calibrating homography for {cam_id}...")
        calibrate_homography.calibrate_now(video_path, h_inv_path, real_width=r_width, real_height=r_height)
    else:
        if FORCE_CALIBRATE:
            print(f"Calibrating homography for {cam_id}...")
            calibrate_homography.calibrate_now(video_path, h_inv_path, real_width=r_width, real_height=r_height)
        print(f"Loading homography for {cam_id} from {h_inv_path}...")

    H_inv = load_h_inv(h_inv_path)

    # Load YOLO
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    trajectory_smooth = []
    smooth_state = {}
    frame_idx = 0

    map_img = np.zeros((MAP_SIZE[0], MAP_SIZE[1], 3), dtype=np.uint8)

    MAX_REAL_HEIGHT = r_height * 1.5
    MAX_REAL_WIDTH = r_width * 1.5

    with open(csv_out, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "id", "Xs", "Ys"]) 

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # YOLO TRACKING 
            results = model.track(frame, classes=[0], conf=0.3, persist=True, verbose=False)

            # Iterate through results
            for result in results:
                boxes = result.boxes

                if boxes is None:
                    continue

                for box in boxes:
                    if box.id is None:
                        continue

                    track_id = int(box.id.item())
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Get feet coordinates
                    feet_x = (x1 + x2) / 2.0
                    feet_y = y2

                    # Map to World
                    raw_X, raw_Y = image_to_world(feet_x, feet_y, H_inv)

                    # Smoothing
                    if track_id not in smooth_state:
                        sx, sy = raw_X, raw_Y
                    else:
                        prev_sx, prev_sy = smooth_state[track_id]
                        sx = (1.0 - SMOOTH_ALPHA) * prev_sx + SMOOTH_ALPHA * raw_X
                        sy = (1.0 - SMOOTH_ALPHA) * prev_sy + SMOOTH_ALPHA * raw_Y

                    smooth_state[track_id] = (sx, sy)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(feet_x), int(feet_y)), 5, (0, 0, 255), -1)

                    # Filter out outliers
                    if -2.0 <= sx <= MAX_REAL_WIDTH and -2.0 <= sy <= MAX_REAL_HEIGHT:
                        csv_writer.writerow([frame_idx, track_id, f"{sx:.4f}", f"{sy:.4f}"])
                        trajectory_smooth.append((sx, sy))

                        mx, my = map_to_window(sx, sy,
                                        r_width=r_width, r_height=r_height,
                                        img_w = MAP_SIZE[0], img_h = MAP_SIZE[1],  
                                        margin=50)
                        
                        # Draw on map checking the bounds
                        # if 0 <= mx < MAP_SIZE[0] and 0 <= my < MAP_SIZE[1]:
                        cv2.circle(map_img, (mx, my), 2, (0, 0, 255), -1)


            cv2.putText(frame, f"Frame: {frame_idx}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(f"YOLO Multi-Track: {cam_id}", frame)
            cv2.imshow("World Trajectory", map_img)

            key = cv2.waitKey(1)

            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    csv_file.close()

    if trajectory_smooth:
        print("Generating plot...")

        xs = [p[0] for p in trajectory_smooth]
        ys = [p[1] for p in trajectory_smooth]

        plt.figure(figsize=(6, 6))

        plt.scatter(xs, ys, s=5, c='blue', alpha=0.6)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"World Trajectories: {cam_id}")
        plt.xlabel("World X")
        plt.ylabel("World Y")

        plt.tight_layout()

        plt.savefig(plot_out, dpi=200)
        print(f"Saved plot to {plot_out}.")
        plt.close()
    else:
        print("No trajectory points recorded – nothing to plot.")

    print(f"Finished processing {cam_id}.")

if __name__ == "__main__":
    """
    Script used to run the entire pipeline.
    """

    FORCE_CALIBRATE = False

    TEST_CONFIGS = [
        {
            "id": "yolo_demo_1",
            "ext": "mov",
            "width": 6.25,
            "height": 1.0,
        },
        {
            "id": "yolo_demo_2",
            "ext": "mov",
            "width": 1.25,
            "height": 3.5,
        },
    ]

    # Run the pipeline for each test file.
    for config in TEST_CONFIGS:
        cam_id = config["id"]
        extension = config["ext"]

        rw = config.get("width", 1.0)
        rh = config.get("height", 1.0)

        video_path = f'data/videos/{cam_id}.{extension}'
        h_inv_path = f'data/calibration/H_{cam_id}.npy'
        
        # Output files based on the configuration ID

        csv_out = f"output/trajectory_run_{cam_id}.csv"
        plot_out = f"output/trajectory_plot_{cam_id}.png"
        
        print(f"\n--- Running Camera: {cam_id} ---")
        
        process_camera_yolo(
            cam_id, 
            video_path, 
            h_inv_path, 
            csv_out,
            plot_out,
            r_width=rw,
            r_height=rh,
        )

    print("\n--- All runs complete ---")

    # Fuse the trajectories if there are two test files

    if len(TEST_CONFIGS) >= 2:
        print("Fusing trajectories...")

        w1, h1 = TEST_CONFIGS[0]["width"], TEST_CONFIGS[0]["height"]
        w2, h2 = TEST_CONFIGS[1]["width"], TEST_CONFIGS[1]["height"]

        cam1_bounds = (0, 0, w1, h1)
        cam2_bounds = (0, 0, w2, h2)

        id1 = TEST_CONFIGS[0]["id"]
        id2 = TEST_CONFIGS[1]["id"]

        csv1 = f"output/trajectory_run_{id1}.csv"
        csv2 = f"output/trajectory_run_{id2}.csv"

        final_plot = f"output/fused_plot_{id1}_{id2}.png"

        off_x, off_y = align_planes.get_alignment_offset(w1, h1, w2, h2)

        fuse_yolo.create_global_plot(csv1, csv2, off_x, off_y, final_plot, cam1_bounds, cam2_bounds)
    