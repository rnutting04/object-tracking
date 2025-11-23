import cv2
import numpy as np
import os

# --- CONFIGURATION ---
VIDEO_PATH = "data/videos/test3_3d.mov" 
OUTPUT_PATH = "data/calibration/camera_pose.npz"

# Define the REAL WORLD coordinates of the 4 points you will click.
# Example: A 1.0 x 1.0 meter square on the floor.
# Order: Bottom-Left, Bottom-Right, Top-Right, Top-Left (Counter-Clockwise)
WORLD_SQUARE_PTS = np.array([
    [0.0, 0.0, 0.0],  # Point 1 (BL)
    [1.0, 0.0, 0.0],  # Point 2 (BR)
    [1.0, 1.0, 0.0],  # Point 3 (TR)
    [0.0, 1.0, 0.0],  # Point 4 (TL)
], dtype=np.float32)

def get_camera_intrinsics(width, height):
    """Approximates camera matrix if you don't have calibration data."""
    focal_length = width  # Approximate focal length
    center_x = width / 2
    center_y = height / 2
    camera_matrix = np.array([[focal_length, 0, center_x],
                              [0, focal_length, center_y],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
    return camera_matrix, dist_coeffs

clicked_points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"Clicked point {len(clicked_points)}: {x}, {y}")
            cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Calibrate 3D", param)

def calibrate_now():
    global clicked_points
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    if not ret: raise ValueError("Could not read video")

    h, w = frame.shape[:2]
    K, D = get_camera_intrinsics(w, h)

    cv2.namedWindow("Calibrate 3D")
    cv2.setMouseCallback("Calibrate 3D", mouse_callback, param=frame)
    
    print(f"--- 3D CALIBRATION ---")
    print(f"Click the 4 floor points corresponding to: \n{WORLD_SQUARE_PTS}")
    print("Press 'q' when done or to quit.")

    while True:
        cv2.imshow("Calibrate 3D", frame)
        key = cv2.waitKey(1)
        if key == ord('q'): break
        if len(clicked_points) == 4:
            print("Computing Pose...")
            break

    cv2.destroyAllWindows()

    if len(clicked_points) != 4:
        print("Aborted.")
        return

    image_pts = np.array(clicked_points, dtype=np.float32)

    # SolvePnP calculates the Camera's Position relative to the Floor Points
    success, rvec, tvec = cv2.solvePnP(WORLD_SQUARE_PTS, image_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        print("Calibration failed!")
        return

    # Save everything needed for tracking
    np.savez(OUTPUT_PATH, rvec=rvec, tvec=tvec, K=K, D=D)
    print(f"Saved camera pose and intrinsics to {OUTPUT_PATH}")
    
    # Calculate Rotation Matrix for verification
    R, _ = cv2.Rodrigues(rvec)
    print(f"Camera Translation (tvec):\n{tvec}")
    print(f"Camera Rotation (R):\n{R}")

if __name__ == "__main__":
    calibrate_now()