# calibrate_homography.py
import cv2
import numpy as np

VIDEO_PATH = "data/videos/test9.mp4"   # change if needed
H_INV_OUT = "data/calibration/H_inv.npy"

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


if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    # Grab a single frame (you can change frame index if you like)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read frame from video")

    frame_copy = frame.copy()

    cv2.namedWindow("Click 4 floor points (in order)")
    cv2.setMouseCallback("Click 4 floor points (in order)", mouse_callback)

    while True:
        cv2.imshow("Click 4 floor points (in order)", frame_copy)
        key = cv2.waitKey(1)

        if key == ord('q'):
            print("Quit without saving homography.")
            cv2.destroyAllWindows()
            exit(0)

        if len(clicked_pts) == 4:
            print("Got 4 points, computing homography...")
            break

    cv2.destroyAllWindows()

    image_pts = np.array(clicked_pts, dtype=np.float32)

    # H: world -> image
    H, _ = cv2.findHomography(world_pts, image_pts, method=0)
    if H is None:
        raise RuntimeError("findHomography failed")

    H_inv = np.linalg.inv(H)
    np.save(H_INV_OUT, H_inv)
    print(f"Saved H_inv to {H_INV_OUT}")
    print("H_inv =\n", H_inv)
