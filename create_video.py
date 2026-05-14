import cv2
import os

image_folder = "segmented_output"
video_name = "semantic_segmentation_output.mp4"

# Read images
images = [img for img in os.listdir(image_folder)
          if img.endswith(".jpg") or img.endswith(".png")]

images.sort()

# First frame
frame = cv2.imread(os.path.join(image_folder, images[0]))

height, width, layers = frame.shape

# Video writer
video = cv2.VideoWriter(
    video_name,
    cv2.VideoWriter_fourcc(*'mp4v'),
    10,
    (width, height)
)

# Add images to video
for image in images:

    img_path = os.path.join(image_folder, image)

    frame = cv2.imread(img_path)

    video.write(frame)

video.release()

print("Video created successfully!")

