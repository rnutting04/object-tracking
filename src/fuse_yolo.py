import csv
import numpy as np
import matplotlib.pyplot as plt
import os

def load_trajectory(csv_path, bounds):
    xs, ys = [], []

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return np.array([]), np.array([])
    
    min_x, min_y, max_x, max_y = bounds

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Support 'Xs'/'Ys' from your YOLO script or 'x'/'y' generic
                x_str = row.get('Xs') or row.get('x') or row.get('X')
                y_str = row.get('Ys') or row.get('y') or row.get('Y')

                if x_str and y_str:
                    x_val = float(x_str)
                    y_val = float(y_str)

                    if min_x <= x_val <= max_x and min_y <= y_val <= max_y:
                        xs.append(x_val)
                        ys.append(y_val)

            except ValueError:
                continue

    return np.array(xs), np.array(ys)

def create_global_plot(csv1_path, csv2_path, off_x, off_y, output_png, bounds1, bounds2):
    """
    Loads two CSVs, applies (off_x, off_y) to the second one, and saves a plot.
    """
    print(f"[FUSE] Merging trajectories with Offset X={off_x:.4f}, Y={off_y:.4f}")

    # 1. Load Trajectories
    c1_x, c1_y = load_trajectory(csv1_path, bounds=bounds1)

    c2_x, c2_y = load_trajectory(csv2_path, bounds=bounds2)

    if len(c1_x) == 0 and len(c2_x) == 0:
        print("[ERROR] No data found in CSVs.")
        return

    # 2. Apply Offset to Camera 2 (Local -> Global)
    c2_x_global = c2_x + off_x
    c2_y_global = c2_y + off_y

    # 3. Plotting
    plt.figure(figsize=(10, 8))
    
    if len(c1_x) > 0:
        plt.scatter(c1_x, c1_y, c='blue', s=5, alpha=0.6, label='Cam 1 (Anchor)')

    if len(c2_x) > 0:
        plt.scatter(c2_x_global, c2_y_global, c='red', s=5, alpha=0.6, label='Cam 2 (Shifted)')

    plt.title(f"Global Trajectory View\n(Cam 2 Offset: x={off_x:.2f}, y={off_y:.2f})")
    plt.xlabel("Global X (meters)")
    plt.ylabel("Global Y (meters)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"[SUCCESS] Merged plot saved to {output_png}")
    plt.close()