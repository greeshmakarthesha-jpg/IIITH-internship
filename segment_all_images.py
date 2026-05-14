from ultralytics import YOLO
import os

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

# Input video folder
input_folder = "training_videos"

# Output folder
output_folder = "segmented_output"

os.makedirs(output_folder, exist_ok=True)

# Process all MP4 videos
for file in os.listdir(input_folder):

    if file.endswith(".MP4") or file.endswith(".mp4"):

        video_path = os.path.join(input_folder, file)

        print(f"Processing video: {file}")

        # Run segmentation
        results = model.predict(
            source=video_path,
            save=True,
            project=output_folder,
            name="results"
        )

print("Semantic segmentation completed!")
