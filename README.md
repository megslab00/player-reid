# player-reid
This project focuses on detecting, tracking, and consistently identifying players in sports videos using a combination of deep learning and computer vision techniques.
# Player Re-Identification in Sports Footage 🎯

This repository contains my solution to the Machine Learning Internship assignment from Liat AI. The objective was to detect, track, and re-identify players across frames in a sports video using deep learning and computer vision techniques.

---

## 📌 Problem Statement

Given a video feed of a sports game, the task involves:
1. Detecting players using an object detection model.
2. Tracking them consistently across frames using a multi-object tracking algorithm.
3. Re-identifying players when they exit and re-enter the frame, assigning consistent IDs using embedding-based similarity.

---

## 🚀 Solution Overview

The pipeline is divided into three core components:

### 1. Detection
- **Model Used**: YOLOv8 (`best.pt`) trained to detect players, ball, and referees.
- **Framework**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

### 2. Tracking
- **Algorithm**: [BYTETracker](https://github.com/ifzhang/ByteTrack)
- Handles short-term ID assignment based on object motion and spatial consistency.

### 3. Re-identification (Re-ID)
- **Model**: Feature extractor based on `ResNet18` pretrained on ImageNet.
- **Approach**:
  - Extracts appearance embeddings of players from cropped frames.
  - Computes cosine similarity against a dynamic gallery of stored embeddings.
  - Re-assigns a consistent `stable_id` to players using a similarity threshold.

---

## 🛠️ Technologies Used

| Component        | Details                                      |
|------------------|----------------------------------------------|
| Language         | Python 3                                     |
| Object Detection | Ultralytics YOLOv8                           |
| Tracking         | BYTETracker (via Ultralytics interface)      |
| Re-ID Model      | PyTorch ResNet18 (feature extractor)         |
| Video Processing | OpenCV                                       |
| Similarity       | Cosine similarity (via SciPy)                |

---

## 🧠 Challenges Faced

- **ID Switching**: Frequent switching due to occlusions or overlapping players.
- **Appearance Drift**: Changes in pose and partial visibility affected embeddings.
- **Threshold Tuning**: Balancing between false matches and missed matches.

**Solutions Implemented:**
- Used a gallery update interval to refine embeddings over time.
- Focused only on the `player` class from YOLO's predictions.
- Re-assigned IDs based on cosine similarity between embeddings and gallery.

---



