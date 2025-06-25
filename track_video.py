import cv2
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import cosine
from ultralytics import YOLO

# --- Configuration ---
# REID_THRESHOLD:
#   - If IDs are still changing too frequently for the SAME player: LOWER this value (e.g., 0.65-0.7).
#   - If DIFFERENT players are frequently getting the SAME ID: RAISE this value (e.g., 0.8-0.85).
#   - This is the most critical parameter to tune based on visual output.
REID_THRESHOLD = 0.70 # Slightly lowered from 0.75 to potentially reduce "ID changing" for same player

EMBEDDING_ALPHA = 0.2 

GALLERY_UPDATE_INTERVAL = 10 # Update gallery embedding only if player hasn't been updated for this many frames.
                              # This reduces computation and over-fitting to single noisy frames.

# --- Re-ID Model Setup ---
class ResNet18_ReID(nn.Module):
    """
    A simple Re-ID model using a pre-trained ResNet18 backbone.
    It removes the final classification layer to extract feature embeddings.
    """
    def __init__(self):
        super(ResNet18_ReID, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        # Remove the final fully connected layer (fc) to get feature vectors
        self.features = nn.Sequential(*list(resnet18.children())[:-1])

    def forward(self, x):
        """
        Forward pass through the feature extractor.
        Input: Tensor of shape (batch_size, 3, 224, 224)
        Output: Flattened feature vector of shape (batch_size, 512)
        """
        x = self.features(x)
        return torch.flatten(x, 1)

# Initialize the Re-ID model and set to evaluation mode
reid_model = ResNet18_ReID()
reid_model.eval() 

# Determine the device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reid_model.to(device)

# Image transformation pipeline for Re-ID model input
transform = transforms.Compose([
    transforms.ToPILImage(),                   # Convert OpenCV BGR NumPy array to PIL Image
    transforms.Resize((224, 224)),             # Resize image to ResNet's expected input size
    transforms.ToTensor(),                     # Convert PIL Image to PyTorch Tensor (scales to 0-1)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with ImageNet stats
])

# --- Player Gallery ---
# Dictionary to store known player embeddings and their metadata
# Format: {stable_id: {'embedding': np.array, 'last_seen_frame': int, 'first_seen_frame': int}}
player_gallery = {}
next_stable_id = 0 # Counter for assigning new unique stable IDs

def get_embedding(image_crop):
    """
    Generates a feature embedding for a given player image crop.
    Handles invalid or empty crops gracefully and returns None if processing fails.
    """
    # Defensive checks for invalid image crops
    if image_crop is None or image_crop.size == 0 or image_crop.shape[0] == 0 or image_crop.shape[1] == 0:
        # print("Debug: get_embedding received empty or invalid image crop.") # Uncomment for detailed debug
        return None
    
    # Convert OpenCV's BGR image to RGB format, as most PyTorch models expect RGB
    image_crop_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    
    try:
        # Apply transformations and add a batch dimension (unsqueeze(0)) for the model input
        img_tensor = transform(image_crop_rgb).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error transforming image crop for embedding: {e}")
        return None

    with torch.no_grad(): # Disable gradient calculation for inference
        embedding = reid_model(img_tensor)
    
    # Convert the PyTorch tensor embedding to a NumPy array on the CPU
    embedding_np = embedding.squeeze().cpu().numpy()
    
    # Check for NaN or Inf values, which can indicate issues in model output or input
    if np.isnan(embedding_np).any() or np.isinf(embedding_np).any():
        print("Warning: Generated embedding contains NaN or Inf values. Returning None.")
        return None

    # Normalize the embedding to unit length for robust cosine similarity calculation
    norm = np.linalg.norm(embedding_np)
    if norm > 0:
        embedding_np = embedding_np / norm
    else:
        # If the embedding is zero (e.g., all features are zero), it's not discriminative.
        # This is rare but can happen with bad inputs or model issues.
        print("Warning: Generated embedding is a zero vector. Returning None.")
        return None

    return embedding_np

def find_or_assign_stable_id(current_embedding, frame_num, track_id):
    """
    Compares the current player's embedding with the gallery of known players.
    Assigns an existing stable_id if a strong match is found,
    otherwise assigns a new stable_id. Implements running average for gallery embeddings.
    """
    global next_stable_id 

    if current_embedding is None:
        return None # Cannot assign ID if embedding is invalid

    best_match_id = -1
    highest_similarity = -1.0 

    # If the gallery is empty, this is the first player detected.
    # Assign the next available stable ID.
    if not player_gallery: # More Pythonic way to check if dictionary is empty
        new_id = next_stable_id
        player_gallery[new_id] = {
            'embedding': current_embedding,
            'last_seen_frame': frame_num,
            'first_seen_frame': frame_num,
            'track_ids_seen': {track_id} # Keep track of associated tracker IDs for debugging
        }
        next_stable_id += 1
        print(f"[Frame {frame_num}] Track {track_id}: Assigned NEW stable ID {new_id} (gallery empty).")
        return new_id

    # Iterate through existing players in the gallery to find the best match
    for stable_id, data in player_gallery.items():
        gallery_embedding = data['embedding']
        
        # Ensure gallery embedding is valid before comparison
        if gallery_embedding is None or np.isnan(gallery_embedding).any() or np.isinf(gallery_embedding).any():
            continue 

        try:
            # Calculate cosine similarity. Embeddings are normalized, so cosine similarity is dot product.
            similarity = np.dot(current_embedding, gallery_embedding) 
            # Alternatively: similarity = 1 - cosine(current_embedding, gallery_embedding)
            # using scipy's cosine distance. If embeddings are normalized, both are equivalent.
        except ValueError as e:
            print(f"Error calculating cosine similarity for Track {track_id} with ID {stable_id}: {e}")
            similarity = -1.0 

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_id = stable_id

    # Decision based on highest similarity found
    if highest_similarity >= REID_THRESHOLD:
        # Match found! Assign existing stable_id.
        # Update gallery embedding using an exponential moving average (EMA)
        old_embedding = player_gallery[best_match_id]['embedding']
        
        # Only update if enough frames have passed since last update
        if (frame_num - player_gallery[best_match_id]['last_seen_frame']) >= GALLERY_UPDATE_INTERVAL:
            new_averaged_embedding = (1 - EMBEDDING_ALPHA) * old_embedding + EMBEDDING_ALPHA * current_embedding
            
            # Re-normalize the averaged embedding to unit length
            norm = np.linalg.norm(new_averaged_embedding)
            if norm > 0:
                new_averaged_embedding /= norm
            else:
                # Fallback if normalization results in zero vector (should be rare)
                new_averaged_embedding = current_embedding 
                print(f"Warning: Averaged embedding for ID {best_match_id} became zero. Resetting to current.")

            player_gallery[best_match_id]['embedding'] = new_averaged_embedding
            player_gallery[best_match_id]['last_seen_frame'] = frame_num
            player_gallery[best_match_id]['track_ids_seen'].add(track_id) # Add current track_id to set
            print(f"[Frame {frame_num}] Track {track_id}: Matched existing ID {best_match_id} (Sim: {highest_similarity:.2f}). Gallery updated.")
        else:
            # If not updating gallery embedding, just update last seen frame and track_id
            player_gallery[best_match_id]['last_seen_frame'] = frame_num
            player_gallery[best_match_id]['track_ids_seen'].add(track_id)
            # print(f"[Frame {frame_num}] Track {track_id}: Matched existing ID {best_match_id} (Sim: {highest_similarity:.2f}). No gallery update due to interval.")

        return best_match_id
    else:
        # No strong match found, this is likely a new player
        new_id = next_stable_id
        player_gallery[new_id] = {
            'embedding': current_embedding,
            'last_seen_frame': frame_num,
            'first_seen_frame': frame_num,
            'track_ids_seen': {track_id}
        }
        next_stable_id += 1
        print(f"[Frame {frame_num}] Track {track_id}: Assigned NEW stable ID {new_id} (highest sim: {highest_similarity:.2f} < {REID_THRESHOLD}).")
        return new_id

# --- Main Video Processing Logic ---

# Load the YOLOv8 model
model = YOLO("weights/best.pt")

# Open the input video file
input_path = "inputs/15sec_input_720p.mp4"
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {input_path}")
    exit()

# Get video properties for output video creation
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Setup output video writer
output_path = "outputs/reidentified_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Codec for MP4 video
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    print(f"Error: Could not open video writer for {output_path}")
    exit()

frame_count = 0
print("\n--- Starting Video Processing ---")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print(f"End of video or failed to read frame at frame_count: {frame_count}")
        break

    # Run YOLO detection with tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False) 
    
    annotated_frame = frame.copy()
    
    # Process detections if available
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = box 
            
            stable_id = None # Initialize stable_id for each detection
            color = (255, 0, 0) # Default color (Blue for non-players or issues)
            label_text = ""

            if model.names[class_id] == 'player':
                # Ensure valid bounding box dimensions
                if x2 > x1 and y2 > y1:
                    player_crop = frame[y1:y2, x1:x2]
                    current_embedding = get_embedding(player_crop)

                    if current_embedding is not None:
                        # Pass the current YOLO track_id for better logging
                        stable_id = find_or_assign_stable_id(current_embedding, frame_count, track_id)
                    else:
                        label_text = f"Player (Track: {track_id}) - Embedding Failed"
                        color = (0, 0, 255) # Red for embedding failure
                else:
                    label_text = f"Player (Track: {track_id}) - Invalid Box"
                    color = (0, 0, 255) # Red for invalid box

                if stable_id is not None:
                    label_text = f"Player ID: {stable_id} (Track: {track_id})"
                    color = (0, 255, 0) # Green for successfully re-identified player
                elif stable_id is None and "Embedding Failed" not in label_text and "Invalid Box" not in label_text:
                    label_text = f"Player (Track: {track_id}) - Re-ID Failed (No Match/New)" # Should be rare with logging
                    color = (0, 165, 255) # Orange for no match found, indicates new player or too high threshold
            else:
                label_text = f"{model.names[class_id]}: Track: {track_id}"
                # Color remains default blue for non-players

            # Draw annotations
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    out.write(annotated_frame)
    frame_count += 1
    if frame_count % 50 == 0: # Log every 50 frames for less clutter
        print(f"Processed frame: {frame_count}")

# Release resources
cap.release()
out.release()
print("\n--- Video Processing Complete ---")
print(f"✅ Re-Identified video saved to {output_path}")

# Optional: Print final gallery state for review
print("\n--- Final Player Gallery Summary ---")
for s_id, data in player_gallery.items():
    print(f"ID {s_id}: First seen frame: {data['first_seen_frame']}, Last seen frame: {data['last_seen_frame']}, Associated Tracker IDs: {data['track_ids_seen']}")