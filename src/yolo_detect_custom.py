import cv2
import numpy as np
import csv
import os
import math
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

# Import Custom Model
from custom_yolo import TinyYOLO, IMG_SIZE, GRID_SIZE, DEVICE

# Import Pipeline Helpers
import calibrate_homography
import fuse_yolo
import align_planes
from grab_path import load_h_inv, image_to_world, world_to_map

# --- CONFIGURATION ---
MODEL_WEIGHTS = "simple_yolo_weights.pth"
CONFIDENCE_THRESHOLD = 0.4

class RobustTracker:
    def __init__(self, max_dist=500, max_disappeared=10):
        self.next_id = 0
        self.objects = {}       # id: (x, y)
        self.disappeared = {}   # id: count of frames missing
        self.max_dist = max_dist
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        # rects: list of (center_x, center_y, w, h)
        
        # 1. If no new detections, increment disappeared for everyone
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = [(r[0], r[1]) for r in rects]

        # 2. If no existing objects, register all inputs
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
            return self.objects

        # 3. Match inputs to existing objects
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Compute distances
        D = np.zeros((len(object_ids), len(input_centroids)))
        for i in range(len(object_ids)):
            for j in range(len(input_centroids)):
                # BUG FIX: Use object_centroids instead of object_ids for coordinates
                dist = math.hypot(object_centroids[i][0] - input_centroids[j][0], 
                                  object_centroids[i][1] - input_centroids[j][1])
                D[i, j] = dist

        # Find smallest matches
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # If distance is too far, don't match (it's a new person)
            if D[row, col] > self.max_dist:
                continue

            # Match found: Update centroid and reset disappeared counter
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        # 4. Handle Disappeared (Existing objects that weren't matched)
        unused_rows = set(range(len(object_ids))).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # 5. Handle New Objects (Inputs that weren't matched)
        unused_cols = set(range(len(input_centroids))).difference(used_cols)
        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects

def get_custom_yolo_boxes(model, frame):
    H, W = frame.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(img_tensor).squeeze(0)

    pixel_boxes = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            conf = preds[i, j, 4]
            if conf > CONFIDENCE_THRESHOLD:
                x_cell, y_cell, w, h = preds[i, j, 0:4]
                norm_x = (j + x_cell) / GRID_SIZE
                norm_y = (i + y_cell) / GRID_SIZE
                px = norm_x * W
                py = norm_y * H
                pw = w * W
                ph = h * H
                pixel_boxes.append([float(px), float(py), float(pw), float(ph)])
    return pixel_boxes

def process_camera_yolo(cam_id, video_path, h_inv_path, csv_out, plot_out, r_width=1.0, r_height=1.0):
    if not os.path.exists(h_inv_path) or RE_CALIBRATE:
        print(f"Calibrating homography for {cam_id}...")
        calibrate_homography.calibrate_now(video_path, h_inv_path, real_width=r_width, real_height=r_height)
    
    H_inv = load_h_inv(h_inv_path)
    print(f"Loading Custom TinyYOLO from {MODEL_WEIGHTS}...")
    
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"ERROR: {MODEL_WEIGHTS} not found. Run custom_yolo.py to train first.")
        return

    model = TinyYOLO().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    
    # UPDATED: Tracker with Patience=10 frames and MaxDistance=500 pixels
    tracker = RobustTracker(max_dist=500, max_disappeared=10)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Could not open video: {video_path}")

    csv_file = open(csv_out, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "id", "Xs", "Ys"]) 

    trajectory_smooth = []
    smooth_state = {}
    SMOOTH_ALPHA = 0.2
    
    BASE_SIZE = 600
    aspect_ratio = r_width / r_height
    if aspect_ratio >= 1.0: map_w, map_h = BASE_SIZE, int(BASE_SIZE / aspect_ratio)
    else: map_h, map_w = BASE_SIZE, int(BASE_SIZE * aspect_ratio)

    map_img = np.zeros((map_h, map_w, 3), dtype=np.uint8)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        raw_boxes = get_custom_yolo_boxes(model, frame)
        tracked_objects = tracker.update(raw_boxes)
        
        for obj_id, centroid in tracked_objects.items():
            cx, cy = centroid
            
            # Find closest box dim
            best_wh = (50, 100)
            min_dist = 9999
            for rb in raw_boxes:
                rcx, rcy, rw, rh = rb
                dist = math.hypot(rcx - cx, rcy - cy)
                if dist < min_dist:
                    min_dist = dist
                    best_wh = (rw, rh)
            w, h = best_wh
            
            x1, y1 = int(cx - w/2), int(cy - h/2)
            x2, y2 = int(cx + w/2), int(cy + h/2)
            feet_x, feet_y = cx, y2
            
            raw_X, raw_Y = image_to_world(feet_x, feet_y, H_inv)

            if obj_id not in smooth_state:
                sx, sy = raw_X, raw_Y
                smooth_state[obj_id] = (sx, sy)
            else:
                prev_sx, prev_sy = smooth_state[obj_id]
                sx = (1.0 - SMOOTH_ALPHA) * prev_sx + SMOOTH_ALPHA * raw_X
                sy = (1.0 - SMOOTH_ALPHA) * prev_sy + SMOOTH_ALPHA * raw_Y
                smooth_state[obj_id] = (sx, sy)

            csv_writer.writerow([frame_idx, obj_id, f"{sx:.4f}", f"{sy:.4f}"])
            trajectory_smooth.append((sx, sy))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (int(feet_x), int(feet_y)), 5, (0, 0, 255), -1)

            mx, my = world_to_map(sx, sy, r_width=r_width, r_height=r_height, img_size=(map_h, map_w), margin=50)
            if 0 <= mx < map_w and 0 <= my < map_h:
                cv2.circle(map_img, (mx, my), 2, (0, 255, 0), -1)

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(f"Custom YOLO Track: {cam_id}", frame)
        cv2.imshow("World Trajectory", map_img)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    if trajectory_smooth:
        xs = [p[0] for p in trajectory_smooth]
        ys = [p[1] for p in trajectory_smooth]
        plt.figure(figsize=(6, 6))
        plt.scatter(xs, ys, s=5, c='blue', alpha=0.6)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"World Trajectories: {cam_id}")
        plt.savefig(plot_out, dpi=200)
        plt.close()
    print(f"Finished processing {cam_id}.")

if __name__ == "__main__":
    RE_CALIBRATE = True # Set to True to re-click floor points

    # Update this with your filenames
    TEST_CONFIGS = [
        {
            "id": "yolo_global_2",
            "ext": "mov",
            "width": 7.9,
            "height": 1.0,
        },
    ]

    for config in TEST_CONFIGS:
        cam_id = config["id"]
        extension = config["ext"]
        rw = config.get("width", 1.0)
        rh = config.get("height", 1.0)
        video_path = f'data/videos/{cam_id}.{extension}'
        h_inv_path = f'data/calibration/H_{cam_id}.npy'
        csv_out = f"output/trajectory_run_{cam_id}.csv"
        plot_out = f"output/trajectory_plot_{cam_id}.png"
        
        print(f"\n--- Running Camera: {cam_id} ---")
        os.makedirs("output", exist_ok=True)
        
        process_camera_yolo(cam_id, video_path, h_inv_path, csv_out, plot_out, r_width=rw, r_height=rh)

    print("\n--- All runs complete ---")