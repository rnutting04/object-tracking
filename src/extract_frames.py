import cv2
import os
import glob

# --- CONFIGURATION ---
VIDEO_DIR = "raw_videos"      # Where you put your .mov/.mp4 files
OUTPUT_DIR = "data/images"    # Where YOLO expects images
FRAMES_PER_SECOND = 1         # Extract 1 frame per second of video

def extract_frames():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
        print(f"Created '{VIDEO_DIR}'. Please put your video files in there and run this again.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all video files
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.*"))
    extensions = ['.mp4', '.mov', '.avi', '.mkv']
    video_files = [f for f in video_files if os.path.splitext(f)[1].lower() in extensions]

    print(f"Found {len(video_files)} videos.")

    total_images = 0

    for video_path in video_files:
        filename = os.path.basename(video_path).split('.')[0]
        print(f"Processing {filename}...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate how many frames to skip to get desired rate
        frame_interval = int(fps / FRAMES_PER_SECOND)
        if frame_interval < 1: frame_interval = 1

        count = 0
        saved = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            if count % frame_interval == 0:
                # Save Frame
                out_name = f"{filename}_{saved:04d}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                cv2.imwrite(out_path, frame)
                saved += 1
                total_images += 1
            
            count += 1
        
        cap.release()

    print(f"--- Done. Extracted {total_images} images to '{OUTPUT_DIR}' ---")

if __name__ == "__main__":
    extract_frames()