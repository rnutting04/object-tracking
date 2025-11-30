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
# Import your existing calibration helper
from grab_path import load_h_inv, image_to_world, world_to_map

def get_dynamic_map(trajectory, r_width=1.0, r_height=1.0, base_size=600, margin=50):
    """
    Generates a map image with a fixed aspect ratio based on real-world dimensions.
    """
    # 1. Calculate Canvas Dimensions based on Aspect Ratio
    aspect_ratio = r_width / r_height

    if aspect_ratio >= 1.0:
        w = base_size
        h = int(base_size / aspect_ratio)
    else:
        h = base_size
        w = int(base_size * aspect_ratio)

    # Use a black background
    canvas = np.zeros((h, w, 3), dtype=np.uint8) 

    if not trajectory:
        return canvas

    # 2. Calculate Scale Factors (taking margin into account)
    # This matches the logic in grab_path.py world_to_map
    scale_draw_x = w - 2 * margin
    scale_draw_y = h - 2 * margin

    # Convert to numpy for vectorized math
    pts = np.array(trajectory, dtype=np.float32)

    # 3. Transform World Coordinates to Pixel Coordinates
    # X_pixel = (X_world / r_width) * draw_area + margin
    pts[:, 0] = (pts[:, 0] / r_width) * scale_draw_x + margin
    
    # Y_pixel = (1.0 - (Y_world / r_height)) * draw_area + margin  <-- Flip Y
    pts[:, 1] = (1.0 - (pts[:, 1] / r_height)) * scale_draw_y + margin

    # 4. Draw the Trajectory
    # Convert to int32 for OpenCV
    pts_int = pts.astype(np.int32)
    
    # Reshape for polylines: (number_of_points, 1, 2)
    pts_reshaped = pts_int.reshape((-1, 1, 2))

    # Draw the path (Blue line, thickness 2, Anti-Aliased)
    cv2.polylines(canvas, [pts_reshaped], isClosed=False, color=(255, 100, 0), thickness=2, lineType=cv2.LINE_AA)

    # Draw the current position (Red dot)
    if len(pts_int) > 0:
        curr = pts_int[-1]
        # Ensure we don't crash if the point is way out of bounds (though it shouldn't be)
        if 0 <= curr[0] < w and 0 <= curr[1] < h:
            cv2.circle(canvas, (curr[0], curr[1]), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    return canvas

def process_camera_yolo(cam_id, video_path, h_inv_path, csv_out, plot_out, r_width=1.0, r_height=1.0):
    """
    Detects a person using YOLOv8 and maps their feet to world coordinates.
    """
    
    if not os.path.exists(h_inv_path):
        print(f"Calibrating homography for {cam_id}...")
        calibrate_homography.calibrate_now(video_path, h_inv_path, real_width=r_width, real_height=r_height)
    else:
        if RE_CALIBRATE:
            print(f"Calibrating homography for {cam_id}...")
            calibrate_homography.calibrate_now(video_path, h_inv_path, real_width=r_width, real_height=r_height)
        print(f"Loading homography for {cam_id} from {h_inv_path}...")

    H_inv = load_h_inv(h_inv_path)

    # Initialize YOLO (v8 Nano model is fastest)
    # It will download 'yolov8n.pt' automatically if not present
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    csv_file = open(csv_out, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "X", "Y", "Xs", "Ys"]) 

    map_h, map_w = 600, 600
    map_img = np.zeros((map_h, map_w, 3), dtype=np.uint8)

    # --- Smoothing & Tracking Variables ---
    SMOOTH_ALPHA = 0.2
    smooth_X, smooth_Y = None, None

    trajectory_smooth = []
    
    # Track the previous center of the bounding box (image coordinates)
    prev_box_center = None 
    MAX_PIXEL_JUMP = 150  # Increased slightly for YOLO as it might detect better at edges
    
    frame_idx = 0
    smooth_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Run YOLO inference
        # classes=[0] ensures we only look for 'person'
        # verbose=False keeps the console clean
        results = model.predict(frame, classes=[0], conf=0.4, verbose=False)

        valid_boxes = []

        # Extract boxes from YOLO results
        # YOLO returns x1, y1, x2, y2. We convert to x, y, w, h to match your logic
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                # Basic size filter to ignore tiny background people/noise
                if w * h < 1000: 
                    continue
                
                valid_boxes.append((x, y, w, h))

        selected_box = None

        # 2. Logic based on History (Kept from your original script)
        if len(valid_boxes) > 0:
            if prev_box_center is None:
                # No history? Pick the largest box by area (assumes subject is main focus)
                selected_box = max(valid_boxes, key=lambda b: b[2] * b[3])
            else:
                # We have history. Find the box closest to previous position.
                best_dist = float('inf')
                px, py = prev_box_center

                for (x, y, w, h) in valid_boxes:
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    
                    # Euclidean distance
                    dist = math.sqrt((cx - px)**2 + (cy - py)**2)

                    # Only consider if it's within jump limit
                    if dist < best_dist and dist < MAX_PIXEL_JUMP:
                        best_dist = dist
                        selected_box = (x, y, w, h)

        # 3. Process the Selection
        if selected_box is not None:
            (raw_x, raw_y, raw_w, raw_h) = selected_box

            # Box Smoothing Logic
            if smooth_box is None:
                smooth_box = [float(raw_x), float(raw_y), float(raw_w), float(raw_h)]
            else:
                alpha_box = 0.3 # increased slightly as YOLO is more stable than HOG
                smooth_box[0] = (1 - alpha_box) * smooth_box[0] + alpha_box * raw_x
                smooth_box[1] = (1 - alpha_box) * smooth_box[1] + alpha_box * raw_y
                smooth_box[2] = (1 - alpha_box) * smooth_box[2] + alpha_box * raw_w
                smooth_box[3] = (1 - alpha_box) * smooth_box[3] + alpha_box * raw_h

            sx, sy, sw, sh = smooth_box

            # YOLO boxes are usually tight to the body. 
            # We don't need the massive shrink factor from the HOG code.
            tight_x = int(sx)
            tight_y = int(sy)
            tight_w = int(sw)
            tight_h = int(sh)
            
            # Update history tracker
            curr_cx = sx + sw / 2.0
            curr_cy = sy + sh / 2.0
            prev_box_center = (curr_cx, curr_cy)

            # Draw "Good" Box
            cv2.rectangle(frame, (tight_x, tight_y), (tight_x + tight_w, tight_y + tight_h), (0, 255, 0), 2)

            # Calculate feet (bottom center)
            feet_x = float(tight_x + tight_w / 2.0)
            # We use 1.0 * height (bottom) because YOLO boxes are tight
            feet_y = float(tight_y + tight_h) 

            cv2.circle(frame, (int(feet_x), int(feet_y)), 5, (0, 0, 255), -1)

            # Map to World
            X, Y = image_to_world(feet_x, feet_y, H_inv)

            # Smooth World Coordinates
            if smooth_X is None:
                smooth_X, smooth_Y = X, Y
            else:
                smooth_X = (1.0 - SMOOTH_ALPHA) * smooth_X + SMOOTH_ALPHA * X
                smooth_Y = (1.0 - SMOOTH_ALPHA) * smooth_Y + SMOOTH_ALPHA * Y

            trajectory_smooth.append((smooth_X, smooth_Y))

            csv_writer.writerow([frame_idx, X, Y, smooth_X, smooth_Y])

        else:
            cv2.putText(frame, "Lost Tracking", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        map_img = get_dynamic_map(
            trajectory_smooth, 
            r_width=r_width, 
            r_height=r_height,
            base_size=600,
            )

        cv2.imshow(f"YOLO Detection: {cam_id}", frame)
        cv2.imshow("World Trajectory", map_img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

    if trajectory_smooth:
        xs = [p[0] for p in trajectory_smooth]
        ys = [p[1] for p in trajectory_smooth]

        plt.figure(figsize=(6, 6))

        plt.scatter(xs, ys, s=5, c='blue', alpha=0.6)

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

    print(f"Finished processing {cam_id}.")

if __name__ == "__main__":
    """
    Script used to run the entire pipeline.
    """

    RE_CALIBRATE = True

    TEST_CONFIGS = [
        {
            "id": "yolo_test_2",
            "ext": "mov",
            "width": 1.0,
            "height": 2.0,
        },
        {
            "id": "yolo_test_1",
            "ext": "mov",
            "width": 3.0,
            "height": 1.0,
        }
    ]

    # Run the pipeline for eachh test file.
    for config in TEST_CONFIGS:
        cam_id = config["id"]
        extension = config["ext"]

        rw = config.get("width", 1.0)
        rh = config.get("height", 1.0)

        video_path = f'data/videos/{cam_id}.{extension}'
        h_inv_path = f'data/calibration/H_{cam_id}.npy'
        
        # Output files based on the configuration ID

        csv_out = f"output/1trajectory_run_{cam_id}.csv"
        plot_out = f"output/1trajectory_plot_{cam_id}.png"
        
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

    # if len(TEST_CONFIGS) >= 2:
    #     print("Fusing trajectories...")
    #     id1 = TEST_CONFIGS[0]["id"]
    #     id2 = TEST_CONFIGS[1]["id"]
    #     csv1 = f"output/trajectory_run_{id1}.csv"
    #     csv2 = f"output/trajectory_run_{id2}.csv"
    #     cam1_data = fuse_yolo.load_csv(csv1)
    #     cam2_data = fuse_yolo.load_csv(csv2)
    #     off_x , off_y = fuse_yolo.calculate_offset(cam1_data, cam2_data)
    #     print(f"Offset X: {off_x:.4f}, Y: {off_y:.4f}")
    #     fuse_yolo.fuse(
    #         cam1=cam1_data,
    #         cam2=cam2_data,
    #         out_csv=f"output/fused_trajectory_{id1}_{id2}.csv",
    #         plot_png=f"output/fused_plot_{id1}_{id2}.png",
    #         cam1_offset=(0, 0),
    #         cam2_offset=(off_x, off_y),
    #     )