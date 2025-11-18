# calibrate_homography.py
import cv2
import numpy as np

VIDEO_PATH = "data/videos/test22_angle1.mov"   # change if needed
H_INV_OUT = "data/calibration/H_inv_2.npy"

# world coordinates of the four floor markers (meters, for example)
# order: P0 -> P1 -> P2 -> P3 in a rectangle order
world_pts = np.array([
    [0.0, 0.0],   # bottom-left
    [1.0, 0.0],   # bottom-right
    [1.0, 1.0],   # top-right
    [0.0, 1.0],   # top-left
], dtype=np.float32)

clicked_pts = []  # to store image points

def mouse_callback(event, x, y, flags, param):
    global clicked_pts, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_pts) < 4:
            clicked_pts.append([x, y])
            print(f"Clicked: {x}, {y}")
            cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)

def calibrate_now(video_path, output_path):
    global clicked_pts, frame_copy

    clicked_pts = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: {video_path}")
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read frame from video")
    
    frame_copy = frame.copy()

    window_name = f"Calibrate: {video_path} (Click 4 points)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"--- Calibrating for {video_path} ---")
    print("Please click the 4 floor markers in order: BL, BR, TR, TL")
    print("Press 'q' to quit without saving.")

    while True:
        cv2.imshow(window_name, frame_copy)
        key = cv2.waitKey(1)

        if key == ord('q'):
            print("Quit calibration without saving.")
            cv2.destroyAllWindows()
            # Depending on your preference, you might want to exit the whole script 
            # or just return. Here we exit to stop processing.
            exit(0)

        if len(clicked_pts) == 4:
            print("Got 4 points, computing homography...")
            cv2.waitKey(500) # visual pause
            break

    cv2.destroyAllWindows()

    image_pts = np.array(clicked_pts, dtype=np.float32)

    # H: world -> image
    H, _ = cv2.findHomography(world_pts, image_pts, method=0)
    if H is None:
        raise RuntimeError("findHomography failed")

    H_inv = np.linalg.inv(H)
    np.save(output_path, H_inv)
    print(f"Saved H_inv to {output_path}")
    print("H_inv =\n", H_inv)

if __name__ == "__main__":
    calibrate_now(VIDEO_PATH, H_INV_OUT)
