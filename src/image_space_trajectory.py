# image_space_trajectory.py
import cv2
import numpy as np

VIDEO_PATH = "data/videos/test7.mp4"
REF_PATH   = "data/ref/ref8.jpg"


def main():
    ref = cv2.imread(REF_PATH, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise RuntimeError(f"Could not load reference image at {REF_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    sift = cv2.SIFT_create(
        nfeatures=3000,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )

    kp_ref, des_ref = sift.detectAndCompute(ref, None)
    if des_ref is None or len(des_ref) < 5:
        raise RuntimeError("Not enough descriptors in reference image")

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    LOWE_RATIO   = 0.7
    MIN_INLIERS  = 15
    RANSAC_THRESH = 4.0

    # image-space trajectory
    traj_img = np.zeros((600, 600, 3), dtype=np.uint8)
    traj_points = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        h_f, w_f = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(gray, None)

        if des_frame is not None and len(des_frame) >= 5:
            matches = flann.knnMatch(des_ref, des_frame, k=2)

            good = []
            for m, n in matches:
                if m.distance < LOWE_RATIO * n.distance:
                    good.append(m)

            if len(good) >= 4:
                src_pts = np.float32(
                    [kp_ref[m.queryIdx].pt for m in good]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp_frame[m.trainIdx].pt for m in good]
                ).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts,
                                             cv2.RANSAC, RANSAC_THRESH,
                                             maxIters=2000)
                if H is not None and mask is not None:
                    inliers = int(mask.sum())
                    if inliers >= MIN_INLIERS:
                        h_ref, w_ref = ref.shape
                        box = np.float32([
                            [0, 0],
                            [w_ref, 0],
                            [w_ref, h_ref],
                            [0, h_ref],
                        ]).reshape(-1, 1, 2)
                        projected = cv2.perspectiveTransform(box, H)
                        frame = cv2.polylines(frame, [np.int32(projected)],
                                              True, (0, 255, 0), 3)

                        pts = projected.reshape(-1, 2)
                        cx = float(pts[:, 0].mean())
                        cy = float(pts[:, 1].mean())
                        cv2.circle(frame, (int(cx), int(cy)), 4,
                                   (0, 0, 255), -1)

                        # normalize to [0,1]
                        nx = cx / w_f
                        ny = cy / h_f

                        # map to trajectory image
                        sx = int(nx * 500 + 50)
                        sy = int((1.0 - ny) * 500 + 50)
                        traj_points.append((sx, sy))
                        cv2.circle(traj_img, (sx, sy), 2,
                                   (0, 255, 0), -1)

                        cv2.putText(frame, f"Inliers: {inliers}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 0), 2)

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow("Video + Detection", frame)
        cv2.imshow("Image-Space Trajectory", traj_img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
