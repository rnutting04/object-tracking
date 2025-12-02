import matplotlib
matplotlib.use('Agg') # Fixes the Qt crash
import matplotlib.pyplot as plt

import cv2
import numpy as np
import csv
import os
import math
import torch
import torchvision.transforms as transforms
from PIL import Image

# --- IMPORTS FROM YOUR PIPELINE ---
import calibrate_homography
import fuse_yolo
import align_planes
from grab_path import load_h_inv, image_to_world, world_to_map

# --- IMPORT CUSTOM MODEL ---
from custom_yolo import TinyYOLO, IMG_SIZE, GRID_SIZE, DEVICE

# --- CONFIGURATION ---
MODEL_WEIGHTS = "simple_yolo_weights.pth"

# 1. LOWER CONFIDENCE: Catch people even if model is unsure
CONFIDENCE_THRESHOLD = 0.2 

# 2. NMS THRESHOLD: Controls overlap
# Lower (0.2) = Aggressively removes duplicates (Fixes "1 person is 2")
# Higher (0.5) = Allows overlaps (Fixes "2 people are 1")
NMS_THRESHOLD = 0.3 

# --- ORIENTATION CONTROLS ---
SWAP_XY  = False   
INVERT_X = False   
INVERT_Y = False   

# --- NEW: NMS FUNCTION (Fixes Double Detection) ---
def non_max_suppression(boxes, confidences, threshold):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    confidences = np.array(confidences)

    # Coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2] # x + w
    y2 = boxes[:, 1] + boxes[:, 3] # y + h

    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(confidences) # Sort by confidence (weakest to strongest)

    pick = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the intersection
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # Compute Overlap (IoU)
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes that overlap too much
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))

    return boxes[pick].tolist()

# --- IMPROVED TRACKER (More Patient) ---
class SimpleTracker:
    def __init__(self, max_dist=150, max_disappeared=15):
        self.next_id = 0
        self.objects = {} 
        self.max_dist = max_dist
        self.disappeared = {} 
        # Wait 15 frames (0.5 sec) before forgetting an ID
        self.max_disappeared_frames = max_disappeared 

    def update(self, rects):
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared_frames: 
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (cx, cy, w, h) in enumerate(rects):
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(rects[i])
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid][:2] for oid in object_ids]
        
        D = np.zeros((len(object_ids), len(rects)))
        for i in range(len(object_ids)):
            for j in range(len(rects)):
                dist = math.hypot(object_centroids[i][0] - input_centroids[j][0],
                                  object_centroids[i][1] - input_centroids[j][1])
                D[i, j] = dist

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols: continue
            if D[row, col] > self.max_dist: continue

            obj_id = object_ids[row]
            self.objects[obj_id] = rects[col] 
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_ids))).difference(used_rows)
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
            if self.disappeared[obj_id] > self.max_disappeared_frames:
                del self.objects[obj_id]
                del self.disappeared[obj_id]

        unused_cols = set(range(len(rects))).difference(used_cols)
        for col in unused_cols:
            self.register(rects[col])

        return self.objects

    def register(self, box):
        self.objects[self.next_id] = box
        self.disappeared[self.next_id] = 0
        self.next_id += 1

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

    raw_boxes = []
    raw_confs = []
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            conf1 = preds[i, j, 4].item()
            conf2 = preds[i, j, 9].item()
            
            if conf1 > CONFIDENCE_THRESHOLD or conf2 > CONFIDENCE_THRESHOLD:
                if conf1 > conf2:
                    x_cell, y_cell, w, h = preds[i, j, 0:4]
                    conf = conf1
                else:
                    x_cell, y_cell, w, h = preds[i, j, 5:9]
                    conf = conf2

                norm_x = (j + x_cell.item()) / GRID_SIZE
                norm_y = (i + y_cell.item()) / GRID_SIZE
                
                px = norm_x * W
                py = norm_y * H
                pw = w.item() * W
                ph = h.item() * H
                
                # Store as Top-Left format for NMS calculation
                tl_x = px - (pw/2)
                tl_y = py - (ph/2)
                
                raw_boxes.append([tl_x, tl_y, pw, ph])
                raw_confs.append(conf)

    # --- APPLY NMS ---
    # This filters out the overlapping double detections
    final_boxes_tl = non_max_suppression(raw_boxes, raw_confs, NMS_THRESHOLD)
    
    # Convert back to Center format for tracker
    final_boxes_center = []
    for (x, y, w, h) in final_boxes_tl:
        cx = int(x + w/2)
        cy = int(y + h/2)
        final_boxes_center.append((cx, cy, int(w), int(h)))
                
    return final_boxes_center

def process_camera_yolo(cam_id, video_path, h_inv_path, csv_out, plot_out, r_width=1.0, r_height=1.0):
    
    if not os.path.exists(h_inv_path):
        print(f"Calibrating homography for {cam_id}...")
        calibrate_homography.calibrate_now(video_path, h_inv_path, real_width=r_width, real_height=r_height)
    else:
        if RE_CALIBRATE:
            print(f"Calibrating homography for {cam_id}...")
            calibrate_homography.calibrate_now(video_path, h_inv_path, real_width=r_width, real_height=r_height)
        print(f"Loading homography for {cam_id} from {h_inv_path}...")

    H_inv = load_h_inv(h_inv_path)

    print(f"Loading Custom TinyYOLO from {MODEL_WEIGHTS}...")
    if not os.path.exists(MODEL_WEIGHTS):
        print("ERROR: Weights file not found.")
        return

    model = TinyYOLO().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    
    # Tracker with Patience
    tracker = SimpleTracker(max_dist=150, max_disappeared=15)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    csv_file = open(csv_out, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "id", "Xs", "Ys"]) 

    trajectory_smooth = []
    smooth_state = {}
    SMOOTH_ALPHA = 0.2

    BASE_SIZE = 600
    aspect_ratio = r_width / r_height
    if aspect_ratio >= 1.0:
        map_w, map_h = BASE_SIZE, int(BASE_SIZE / aspect_ratio)
    else:
        map_h, map_w = BASE_SIZE, int(BASE_SIZE * aspect_ratio)

    map_img = np.zeros((map_h, map_w, 3), dtype=np.uint8)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        raw_boxes = get_custom_yolo_boxes(model, frame)
        tracked_objects = tracker.update(raw_boxes)
        
        for obj_id, (cx, cy, w, h) in tracked_objects.items():
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            feet_x = cx
            feet_y = y2
            
            raw_X, raw_Y = image_to_world(feet_x, feet_y, H_inv)

            # --- APPLY ORIENTATION FIXES ---
            if SWAP_XY:
                raw_X, raw_Y = raw_Y, raw_X
            if INVERT_X:
                raw_X = -raw_X 
            if INVERT_Y:
                raw_Y = -raw_Y

            if obj_id not in smooth_state:
                sx, sy = raw_X, raw_Y
                smooth_state[obj_id] = (sx, sy)
            else:
                prev_sx, prev_sy = smooth_state[obj_id]
                # Jump check
                if math.hypot(raw_X - prev_sx, raw_Y - prev_sy) > 1.5:
                     sx, sy = raw_X, raw_Y
                else:
                    sx = (1.0 - SMOOTH_ALPHA) * prev_sx + SMOOTH_ALPHA * raw_X
                    sy = (1.0 - SMOOTH_ALPHA) * prev_sy + SMOOTH_ALPHA * raw_Y
                
                smooth_state[obj_id] = (sx, sy)

            csv_writer.writerow([frame_idx, obj_id, f"{sx:.4f}", f"{sy:.4f}"])
            trajectory_smooth.append((sx, sy))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (int(feet_x), int(feet_y)), 5, (0, 0, 255), -1)

            # Draw on Map
            draw_x = abs(sx)
            draw_y = abs(sy)
            mx, my = world_to_map(draw_x, draw_y,
                                  r_width=r_width, r_height=r_height,
                                  img_size=(map_h, map_w),
                                  margin=50)

            if 0 <= mx < map_w and 0 <= my < map_h:
                cv2.circle(map_img, (mx, my), 2, (0, 0, 255), -1) 

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(f"Custom YOLO: {cam_id}", frame)
        cv2.imshow("World Trajectory", map_img)

        key = cv2.waitKey(1)
        if key == ord('q'): break

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
    RE_CALIBRATE = True 

    TEST_CONFIGS = [
        {
            "id": "yolo_two_1", 
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