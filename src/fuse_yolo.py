import csv
import numpy as np
import matplotlib.pyplot as plt
import os

def load_trajectory(csv_path):
    """
    Loads a CSV file containing X and Y coordinates.
    """
    xs, ys = [], []

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return np.array([]), np.array([])

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Get X and Y coordinates
                x_str = row.get('Xs') or row.get('x') or row.get('X')
                y_str = row.get('Ys') or row.get('y') or row.get('Y')

                if x_str and y_str:
                    x_val = float(x_str)
                    y_val = float(y_str)

                    xs.append(x_val)
                    ys.append(y_val)

            except ValueError:
                continue

    return np.array(xs), np.array(ys)

def create_global_plot(csv1_path, csv2_path, off_x, off_y, output_png, bounds1, bounds2):
    """
    Loads two CSVs representing two planes, applies offset to the second one, and saves a plot.
    """

    print(f"Merging trajectories with Offset X={off_x:.4f}, Y={off_y:.4f}")

    # Load Trajectories
    c1_x, c1_y = load_trajectory(csv1_path)

    c2_x, c2_y = load_trajectory(csv2_path)

    if len(c1_x) == 0 and len(c2_x) == 0:
        print("No data found in CSVs.")
        return

    # Apply Offset to Camera 2
    c2_x_global = c2_x + off_x
    c2_y_global = c2_y + off_y

    # Plotting
    plt.figure(figsize=(10, 8))
    
    if len(c1_x) > 0:
        plt.scatter(c1_x, c1_y, c='blue', s=5, alpha=0.6, label='Cam 1 (Anchor)')

    if len(c2_x) > 0:
        plt.scatter(c2_x_global, c2_y_global, c='blue', s=5, alpha=0.6, label='Cam 2 (Shifted)')

    plt.title(f"Global Trajectory View\n(Cam 2 Offset: x={off_x:.2f}, y={off_y:.2f})")
    plt.xlabel("Global X (meters)")
    plt.ylabel("Global Y (meters)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"Merged plot saved to {output_png}")
    plt.close()