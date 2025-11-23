import cv2
import numpy as np
import matplotlib.pyplot as plt 
import csv
import os

# --- TUNING PARAMETERS ---
# 1. Scale Tuning: If path is too far away (high Y), DECREASE this slightly.
#    If path is too close (low Y), INCREASE this.
REF_OBJECT_WIDTH_METERS = 0.21 

# 2. Floor Size: The real width of the square you clicked for calibration
FLOOR_SQUARE_WIDTH_METERS = 1.0  

NORMALIZED_OBJ_WIDTH = REF_OBJECT_WIDTH_METERS / FLOOR_SQUARE_WIDTH_METERS

def get_transform_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def process_camera_3d_corrected(video_path, ref_paths, calibration_path, csv_out, plot_out):
    
    if not os.path.exists(calibration_path):
        print("Run calibration first.")
        return

    calib_data = np.load(calibration_path)
    K = calib_data['K']
    D = calib_data['D']
    rvec_world = calib_data['rvec']
    tvec_world = calib_data['tvec']
    
    # World -> Camera
    T_world_to_cam = get_transform_matrix(rvec_world, tvec_world)
    # Camera -> World
    T_cam_to_world = np.linalg.inv(T_world_to_cam)

    sift = cv2.SIFT_create(nfeatures=2000)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    refs = []
    for path in ref_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        kp, des = sift.detectAndCompute(img, None)
        if des is None: continue
        
        h, w = img.shape
        scale = NORMALIZED_OBJ_WIDTH / w
        
        # Center the origin on the image so rotation doesn't swing wildly
        # Origin (0,0,0) is now the CENTER of the paper/face
        obj_pts_local = np.zeros((len(kp), 3), dtype=np.float32)
        for i, k in enumerate(kp):
            obj_pts_local[i] = [
                (k.pt[0] - w/2) * scale, 
                (k.pt[1] - h/2) * scale, 
                0
            ]

        refs.append({"des": des, "obj_pts": obj_pts_local, "kp": kp, "img": img})
    
    cap = cv2.VideoCapture(video_path)
    
    trajectory_head = [] 
    trajectory_feet = [] 

    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(gray, None)

        best_pos = None 

        if des_frame is not None and len(des_frame) > 5:
            for ref in refs:
                matches = flann.knnMatch(ref["des"], des_frame, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)

                if len(good) < 6: continue

                src_pts_3d = np.float32([ref["obj_pts"][m.queryIdx] for m in good])
                dst_pts_2d = np.float32([kp_frame[m.trainIdx].pt for m in good])

                success, rvec_obj, tvec_obj, inliers = cv2.solvePnPRansac(
                    src_pts_3d, dst_pts_2d, K, D, 
                    reprojectionError=8.0, flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    T_obj_to_cam = get_transform_matrix(rvec_obj, tvec_obj)
                    T_obj_to_world = T_cam_to_world @ T_obj_to_cam
                    
                    X, Y, Z = T_obj_to_world[:3, 3]

                    # --- FIX Z AXIS ---
                    # If calibration was inverted, Z might be negative. 
                    # We force absolute value to interpret "distance from floor"
                    Z = abs(Z)

                    # Simple outlier rejection
                    if -1.0 < X < 2.0 and -1.0 < Y < 3.0: 
                        best_pos = (X, Y, Z)
                        
                        # Project Axis
                        axis = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]])
                        imgpts, _ = cv2.projectPoints(axis, rvec_obj, tvec_obj, K, D)
                        frame = cv2.drawFrameAxes(frame, K, D, rvec_obj, tvec_obj, 0.1)
                        break 

        if best_pos:
            X, Y, Z = best_pos
            trajectory_head.append((X, Y, Z))
            
            # ESTIMATE FEET: Just drop Z to 0
            trajectory_feet.append((X, Y, 0))

            cv2.putText(frame, f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("3D Tracking Corrected", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    # --- PLOT 3D RESULT ---
    if trajectory_head:
        head = np.array(trajectory_head)
        feet = np.array(trajectory_feet)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Head (Real 3D Position)
        ax.plot(head[:, 0], head[:, 1], head[:, 2], c='blue', alpha=0.5, label='Head Path (Actual)')
        
        # Plot Feet (Shadow on Floor)
        ax.plot(feet[:, 0], feet[:, 1], feet[:, 2], c='red', linewidth=2, label='Feet Path (Projected)')
        
        # Connect Head to Feet every 10th frame to visualize height
        for i in range(0, len(head), 10):
            ax.plot([head[i,0], feet[i,0]], [head[i,1], feet[i,1]], [head[i,2], feet[i,2]], 'k-', alpha=0.1)

        # Draw Floor
        floor = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,0]])
        ax.plot(floor[:,0], floor[:,1], floor[:,2], 'k--', linewidth=2, label='Calibrated Square')

        # Set consistent view
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_zlim(0, 1.5) # Look from floor up to 1.5m
        ax.legend()
        
        plt.savefig(plot_out)
        print(f"Saved corrected plot to {plot_out}")
        plt.show()

# --- RUN IT ---
if __name__ == "__main__":
    REF_PATHS  = [
        "data/ref/ref14_vid_1.png",
        'data/ref/ref14_vid_2.png',
        'data/ref/ref14_vid_3.png',
        'data/ref/ref14_vid_4.png',
        "data/ref/ref14_vid_5.png",
        'data/ref/ref14_vid_6.png',
        "data/ref/ref14_5_blurry.png",
        "data/ref/ref14_vid_side1.png",
        "data/ref/ref14_vid_side2.png",
        "data/ref/ref14_vid_top1.png",
        "data/ref/ref14_vid_top2.png",
    ]

    process_camera_3d_corrected(
        video_path="data/videos/test3_3d.mov",
        ref_paths=REF_PATHS, # Add your refs here
        calibration_path="data/calibration/camera_pose.npz",
        csv_out="3d_track.csv",
        plot_out="3d_plot.png"
    )