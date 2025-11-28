import csv
import matplotlib.pyplot as plt
import numpy as np

def load_csv(path):
    """Return dict: frame -> {X, Y, Xs, Ys, inliers}"""
    data = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            data[frame] = {
                "X": float(row["X"]),
                "Y": float(row["Y"]),
                "Xs": float(row["Xs"]),
                "Ys": float(row["Ys"]),
                "inliers": int(row["inliers"]),
            }
    return data

def calculate_offset(cam1, cam2):
    """
    Calculates the (x, y) translation required to align cam2 to cam1
    based on the average difference of overlapping frames.
    """

    # Find frames present in both datasets
    common_frames = set(cam1.keys()) & set(cam2.keys())
    
    if not common_frames:
        print("[WARN] No overlapping frames found between cameras. Returning (0,0) offset.")
        return (0, 0)

    diff_x = []
    diff_y = []

    print(f"Found {len(common_frames)} overlapping frames for alignment.")

    for f in common_frames:
        # We compare the SMOOTHED coordinates (Xs, Ys)
        x1 = cam1[f]["Xs"]
        y1 = cam1[f]["Ys"]
        
        x2 = cam2[f]["Xs"]
        y2 = cam2[f]["Ys"]

        # Calculate how far Cam2 is from Cam1
        diff_x.append(x1 - x2)
        diff_y.append(y1 - y2)

    # The offset is the average difference
    offset_x = np.mean(diff_x)
    offset_y = np.mean(diff_y)

    return (offset_x, offset_y)

def fuse(cam1, cam2, out_csv, plot_png, cam1_offset = (0, 0), cam2_offset = (0, 0)):
    """
    Fuses two trajectories together, using the offset calculated by calculate_offset()
    """

    # Collect all frames across both cameras
    all_frames = sorted(set(cam1.keys()) | set(cam2.keys()))

    fused = []

    for frame in all_frames:
        row = {"frame": frame}

        c1 = cam1.get(frame)
        c2 = cam2.get(frame)

        if c1:
            c1_X_global = c1["Xs"] + cam1_offset[0]
            c1_Y_global = c1["Ys"] + cam1_offset[1]
        if c2:
            c2_X_global = c2["Xs"] + cam2_offset[0]
            c2_Y_global = c2["Ys"] + cam2_offset[1]

        if c1 and not c2:
            # Camera 1 only
            row["X"] = c1_X_global
            row["Y"] = c1_Y_global
            row["source"] = "cam1"
            row["inliers"] = c1["inliers"]

        elif c2 and not c1:
            # Camera 2 only
            row["X"] = c2_X_global
            row["Y"] = c2_Y_global
            row["source"] = "cam2"
            row["inliers"] = c2["inliers"]

        else:
            # BOTH cams have data → fuse using inlier weighting
            n1 = c1["inliers"]
            n2 = c2["inliers"]

            # prevent division by zero
            if n1 + n2 == 0:
                continue

            Xf = (c1_X_global * n1 + c2_X_global * n2) / (n1 + n2)
            Yf = (c1_Y_global * n1 + c2_Y_global * n2) / (n1 + n2)

            row["X"] = Xf
            row["Y"] = Yf
            row["source"] = "fused"
            row["inliers"] = n1 + n2

        fused.append(row)

    # Write fused CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "X", "Y", "source", "inliers"])
        for r in fused:
            writer.writerow([r["frame"], r["X"], r["Y"], r["source"], r["inliers"]])

    print(f"Saved fused CSV to {out_csv}")

    # Plot fused trajectory
    xs = [r["X"] for r in fused]
    ys = [r["Y"] for r in fused]

    plt.figure(figsize=(6, 8))
    plt.plot(xs, ys, "g.", markersize=3)
    plt.title("Fused Trajectory")
    plt.xlabel("World X")
    plt.ylabel("World Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(plot_png, dpi=200)
    print(f"Saved fused plot to {plot_png}")


if __name__ == "__main__":
    """
    Stand alone script to run the fuse function
    """
    
    cam1 = load_csv("trajectory_run_test_stich1.csv")
    cam2 = load_csv("trajectory_run_test_stich2.csv")

    fuse(cam1, cam2,
         out_csv="fused_run.csv",
         plot_png="fused_plot.png",
         cam1_offset=(0, 0),
         cam2_offset=(0.2, -0.38),
         )
