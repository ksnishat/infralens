from roboflow import Roboflow

# 1. Authenticate
rf = Roboflow(api_key="6WGixmq7tLpLj2730zRn")

# 2. Connect to YOUR Project
print("Connecting to Roboflow...")
project = rf.workspace("nishat-workspace").project("rust-detection-6ogya-rl87b")

# 3. Download Version 1
print("Downloading YOLOv8 dataset...")
version = project.version(1)
dataset = version.download("yolov8")

print("SUCCESS! Dataset downloaded to /app/rust-detection-6ogya-rl87b-1")