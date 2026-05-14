from ultralytics import YOLO
import cv2

# Load YOLO segmentation model
model = YOLO("yolov8n-seg.pt")

# Input video
video_path = "input_video.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video
output_path = "segmented_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

print("Starting segmentation...")

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    # Perform segmentation
    results = model.predict(frame, conf=0.4)

    # Draw masks and boxes
    annotated_frame = results[0].plot()

    # Write frame
    out.write(annotated_frame)

    # Show live preview
    cv2.imshow("Segmentation", annotated_frame)

    # Press q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()

print("Segmentation completed successfully")
