# calibrate_homography.py
import cv2
import numpy as np

# Global Variables

clicked_pts = []
frame_copy = None

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback to record points clicked and draw them on the frame.
    """
    global clicked_pts, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_pts) < 4:
            clicked_pts.append([x, y])
            print(f"Clicked: {x}, {y}")

            if frame_copy is not None:
                cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)

def calibrate_now(video_path, output_path, real_width = 1.0, real_height = 1.0):
    """
    Calibrates the camera by creating a homography matrix that represents the virtual floor.
    """
    global clicked_pts, frame_copy

    clicked_pts = []

    # World Coordinates of the four floor markers
    # Order: Bottom Left -> Bottom Right -> Top Right -> Top Left
    world_pts = np.array([
        [0.0, 0.0],
        [real_width, 0.0],
        [real_width, real_height],
        [0.0, real_height],
    ], dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    
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

    # Compute homography
    image_pts = np.array(clicked_pts, dtype=np.float32)
    H, _ = cv2.findHomography(world_pts, image_pts, method=0)
    
    if H is None:
        raise RuntimeError("findHomography failed")

    H_inv = np.linalg.inv(H)
    np.save(output_path, H_inv)
    print(f"Saved H_inv to {output_path}")
    print("H_inv =\n", H_inv)

if __name__ == "__main__":
    """
    Stand alone script to run the calibrate function
    """

    VIDEO_PATH = "data/videos/yolo_demo_2.mov"
    H_INV_OUT = "data/calibration/H_yolo_demo_2.npy"

    calibrate_now(VIDEO_PATH, H_INV_OUT, real_width=1.25, real_height=3.5)
