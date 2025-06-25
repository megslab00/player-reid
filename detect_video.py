from ultralytics import YOLO
import cv2
import os

# Load the YOLOv8 model
model = YOLO("weights/best.pt")

# Open the video
input_path = "inputs/15sec_input_720p.mp4"
cap = cv2.VideoCapture(input_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("outputs/detected_video.mp4", fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO detection on the frame
    results = model(frame)
    annotated_frame = results[0].plot()  # draw boxes

    out.write(annotated_frame)

    frame_count += 1
    print(f"Processed frame: {frame_count}")

# Release everything
cap.release()
out.release()
print("✅ Video saved to outputs")
