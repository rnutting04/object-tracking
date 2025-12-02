# Path Stitching

Create a virtual environment and install dependencies: 

python3 -m venv venv source venv/bin/activate

pip install opencv-python numpy matplotlib ultralytics

SIFT + RANSAC + FLANN Implementation: 

Run src/graph_path.py which starts the pipeline. Make sure to change the path to the correct video file you want to process and its corresponding reference images. 

YOLO Implementation: 

Run src/graph_path_yolo.py which starts the pipeline. Make sure to change the path to the correct video file you want to process.