import csv
import matplotlib.pyplot as plt

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


def fuse(cam1, cam2, out_csv="fused_trajectory.csv", plot_png="fused_plot.png"):
    # Collect all frames across both cameras
    all_frames = sorted(set(cam1.keys()) | set(cam2.keys()))

    fused = []

    for frame in all_frames:
        row = {"frame": frame}

        c1 = cam1.get(frame)
        c2 = cam2.get(frame)

        if c1 and not c2:
            # Camera 1 only
            row["X"] = c1["Xs"]
            row["Y"] = c1["Ys"]
            row["source"] = "cam1"
            row["inliers"] = c1["inliers"]

        elif c2 and not c1:
            # Camera 2 only
            row["X"] = c2["Xs"]
            row["Y"] = c2["Ys"]
            row["source"] = "cam2"
            row["inliers"] = c2["inliers"]

        else:
            # BOTH cams have data → fuse using inlier weighting
            n1 = c1["inliers"]
            n2 = c2["inliers"]

            # prevent division by zero
            if n1 + n2 == 0:
                continue

            Xf = (c1["Xs"] * n1 + c2["Xs"] * n2) / (n1 + n2)
            Yf = (c1["Ys"] * n1 + c2["Ys"] * n2) / (n1 + n2)

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
    plt.gca().invert_yaxis()  # match your OpenCV world map orientation
    plt.title("Fused Trajectory")
    plt.xlabel("World X")
    plt.ylabel("World Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(plot_png, dpi=200)
    print(f"Saved fused plot to {plot_png}")


if __name__ == "__main__":
    cam1 = load_csv("output/cam1_run.csv")
    cam2 = load_csv("output/cam2_run.csv")

    fuse(cam1, cam2,
         out_csv="output/fused_run.csv",
         plot_png="output/fused_plot.png")
