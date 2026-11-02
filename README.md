# 🚗 Ask-Pothole — Intelligent Pothole Detection & Driver Assistance System

**A Deep Learning-Based Road Hazard Detection and Lane Guidance System**

Ask-Pothole is a video-based Advanced Driver Assistance System (ADAS) designed to detect potholes, track road hazards, estimate their approximate distance, analyse lane-wise road conditions, and provide real-time directional guidance to drivers.

The system combines a trained **YOLO object detection model**, **ByteTrack multi-object tracking**, monocular camera geometry, and a custom lane-risk assessment algorithm to identify potentially safer driving paths.

---

## 🎯 Project Overview

Potholes are a major road safety concern, often causing vehicle damage, accidents, and unsafe driving conditions.

Ask-Pothole aims to address this problem through computer vision and deep learning. It processes road videos frame by frame, identifies potholes, evaluates their positions and proximity, and provides visual guidance to help drivers respond to road hazards.

The system also features a custom Heads-Up Display (HUD) that presents detection results, tracking information, lane statistics, and driving recommendations.

The project has two parts:

1. **Model training & evaluation** — a notebook that trains and evaluates the pothole detector (`best.pt`). See [Model Training & Evaluation](#-model-training--evaluation).
2. **Detection & guidance pipeline** — `pothole_detection.py`, which uses `best.pt` on road video. See [Getting Started](#-getting-started).

## ✨ Key Features

### 1. Deep Learning-Based Pothole Detection

- Uses a trained YOLO model (`best.pt`) to detect potholes in video frames.
- Generates bounding boxes and confidence scores for detected potholes.
- Configurable confidence threshold and inference image size.
- Highlights detected potholes using custom targeting-style bounding boxes.
- Model reaches **0.938 mAP50 / 0.676 mAP50-95** on the held-out test set (see [results](#-results)).

### 2. Multi-Object Tracking with ByteTrack

- Integrates ByteTrack through Ultralytics YOLO tracking.
- Assigns tracking IDs to detected potholes across consecutive frames.
- Helps maintain object identity as potholes move through the video.
- Uses persistent tracking to maintain IDs between frames.

### 3. Approximate Distance Estimation

- Estimates the distance between the camera and detected potholes using monocular camera geometry.
- Uses:
  - Camera mounting height
  - Vertical field of view (FOV)
  - Estimated horizon position
  - Bottom coordinate of the detected bounding box
- Displays the estimated distance alongside each detection.

**Note:** This is a rough geometric estimate, not a calibrated depth measurement. Its accuracy depends on camera calibration, road geometry, and the reliability of the detected ground-contact point.

### 4. Three-Lane Risk Analysis

The video frame is divided into three vertical regions:

- LEFT
- CENTER (Current Vehicle Path)
- RIGHT

The system analyses the number and relative risk of detected potholes in each region.

Each pothole contributes to a lane danger score based on its detection confidence and apparent proximity.

The scoring mechanism is:

`Danger Score = Confidence × (0.3 + 0.7 × Closeness)`

Where:

- **Confidence:** YOLO detection confidence score.
- **Closeness:** Normalized vertical position of the bounding box's bottom coordinate.

Potholes appearing closer to the bottom of the frame contribute more heavily to the lane's danger score.

### 5. Intelligent Direction Guidance

Based on the calculated lane danger scores, the system generates one of four driving recommendations:

| Condition                                     | Recommendation           |
| --------------------------------------------- | ------------------------ |
| Center lane is safest                         | GO STRAIGHT              |
| Left lane is uniquely safest                  | MOVE LEFT                |
| Right lane is uniquely safest                 | MOVE RIGHT               |
| Both side lanes are equally safer than center | SLOW DOWN – PATH UNCLEAR |

The system avoids unnecessary lane changes when the center lane is already the safest option.

### 6. Temporal Decision Smoothing

- Maintains a history of the previous 8 directional decisions.
- Uses majority voting to determine the final recommendation.
- Reduces rapid directional changes and arrow flickering caused by frame-to-frame detection variations.

### 7. Custom Heads-Up Display (HUD)

A real-time visual interface presents:

- Pothole detection count
- Detection confidence
- Approximate distance estimates
- Tracking IDs
- Inference time
- Processing FPS
- Lane-wise pothole counts
- Directional recommendations
- Video processing progress

The output is rendered on a consistent **1280 × 720 (16:9)** canvas while preserving the original video's aspect ratio.

---

## 🧠 Model Training & Evaluation

The detector (`best.pt`) is trained and evaluated end to end in a single Colab notebook:
[`notebooks/pothole_yolo_train_eval.ipynb`](notebooks/pothole_yolo_train_eval.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShreyaIlamurugu/Ask-pothole/blob/main/notebooks/pothole_yolo_train_eval.ipynb)

The notebook covers: setup and GPU check → dataset download → label sanity checks → dataset YAML → training → validation and test evaluation → sample predictions → saving metrics and weights.

### Dataset

[Pothole Detection Dataset: YOLOv11 Optimized](https://www.kaggle.com/datasets/muskanverma24/pothole-detection-dataset-yolov11-optimized) (Kaggle, from Roboflow project `vasanthakumar662005/pothole-pq3lu`, licence CC BY 4.0).

| Split      | Images |
| ---------- | -----: |
| Train      |  3,345 |
| Validation |    397 |
| Test       |    198 |

One class (`pothole`), YOLO-format labels, 10,111 annotated boxes in total, no malformed label lines. The notebook downloads the data itself — **no images are stored in this repo.**

### Training configuration

| Parameter       | Value        | Purpose                      |
| --------------- | ------------ | ---------------------------- |
| Initial weights | `yolo26n.pt` | Starting model (2.39 M params, 5.4 GFLOPs) |
| epochs          | 50           | Maximum training epochs      |
| patience        | 10           | Early-stopping patience      |
| imgsz           | 640          | Input image size (px)        |
| batch           | -1           | Automatic batch size         |
| workers         | 2            | Data-loading workers         |
| seed            | 42           | Reproducibility              |

Hardware: Google Colab, Tesla T4 GPU, Ultralytics 8.4.191.

### 📈 Results

Best weights, evaluated at 640 px:

| Split                  | Precision | Recall | mAP50 | mAP50-95 |
| ---------------------- | --------: | -----: | ----: | -------: |
| Validation (973 boxes) |     0.895 |  0.867 | 0.934 |    0.668 |
| Test (433 boxes)       |     0.913 |  0.872 | 0.938 |    0.676 |

Inference speed: about 5 ms per image on the T4. The notebook writes the exact figures of every run to `metrics.json`.

<!-- After a run, copy the plots from the notebook's outputs/ folder into results/ and uncomment: -->
<!-- ![Training curves](results/results.png) -->
<!-- ![Confusion matrix](results/confusion_matrix.png) -->

### How to reproduce

1. Open the notebook in Colab (badge above) and select `Runtime ▸ Change runtime type ▸ T4 GPU`.
2. Create a [Kaggle API token](https://www.kaggle.com/settings) and paste it when prompted (or save it as a Colab Secret named `KAGGLE_API_TOKEN`).
3. `Runtime ▸ Run all` (roughly 20–40 min on a T4).
4. Download `best.pt` and place it in the repository root.

Optional command-line equivalent (after writing the dataset YAML as the notebook does):

```bash
yolo detect train data=pothole.yaml model=yolo26n.pt epochs=50 imgsz=640 patience=10 seed=42
```

---

## 🛠️ Technology Stack

| Technology         | Purpose                                 |
| ------------------ | --------------------------------------- |
| Python             | Core implementation                     |
| YOLO (Ultralytics) | Pothole detection (training and inference) |
| Kaggle API         | Dataset download                        |
| Google Colab (T4)  | Model training                          |
| ByteTrack          | Multi-object tracking                   |
| OpenCV             | Video processing and HUD rendering      |
| NumPy              | Numerical operations                    |
| Math               | Camera geometry and distance estimation |
| Collections        | Temporal decision history and voting    |
| Time               | FPS and inference-time measurement      |

## ⚙️ System Architecture

The project follows this processing pipeline:

```text
             INPUT ROAD VIDEO
                    |
                    v
          VIDEO FRAME EXTRACTION
                    |
                    v
          FRAME RESIZING & FITTING
                    |
                    v
          YOLO POTHOLE DETECTION   <-- best.pt (trained in the notebook)
                    |
                    v
          BYTETRACK OBJECT TRACKING
                    |
          +---------+---------+
          |                   |
          v                   v
    DISTANCE             LANE-WISE
    ESTIMATION           RISK ANALYSIS
          |                   |
          |                   v
          |            DIRECTION DECISION
          |                   |
          |            TEMPORAL SMOOTHING
          |                   |
          +---------+---------+
                    |
                    v
             HUD GENERATION
                    |
                    v
          PROCESSED OUTPUT VIDEO
```

## 📁 Project Structure

```text
Ask-pothole/
│
├── notebooks/
│   └── pothole_yolo_train_eval.ipynb   # train + evaluate the detector
│
├── results/                            # metrics.json, training curves, confusion matrix, sample predictions
│
├── Videos/
│   ├── video_11.mp4
│   ├── video_12.mp4
│   └── video_13.mp4
│
├── best.pt                             # trained weights
├── pothole_detection.py                # detection + tracking + lane guidance + HUD
├── requirements.txt
├── .gitignore
├── README.md
└── pothole_hud_output.mp4              # generated when the application runs
```

_The output video is generated when the application runs. It does not need to be present beforehand._

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or later
- pip package manager
- A trained YOLO model (`best.pt`) — included, or train your own with the [notebook](#-model-training--evaluation)
- A road video for processing

### 1. Clone the Repository

```bash
git clone https://github.com/ShreyaIlamurugu/Ask-pothole.git
```

```bash
cd Ask-pothole
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

(or: `pip install ultralytics opencv-python numpy`)

### 3. Configure the Input Video

Open `pothole_detection.py` and configure the following parameters:

```python
MODEL_PATH = "best.pt"
VIDEO_SOURCE = "Videos/video_11.mp4"
OUTPUT_VIDEO = "pothole_hud_output.mp4"
```

Change the video filename to process a different sample video.

### 4. Configure Detection Parameters

```python
CONFIDENCE = 0.25
IMAGE_SIZE = 640
```

These parameters control the minimum detection confidence and inference image size.

### 5. Configure Camera Geometry

```python
CAMERA_HEIGHT_M = 1.2
VERTICAL_FOV_DEG = 45.0
HORIZON_RATIO = 0.40
```

These values represent approximate camera mounting and viewing geometry. They should be adjusted to match the actual camera setup for meaningful distance estimates.

### 6. Run the Application

```bash
python pothole_detection.py
```

The application will:

1. Load the trained YOLO model.
2. Open the selected road video.
3. Detect and track potholes.
4. Estimate approximate distances.
5. Analyse lane-wise danger scores.
6. Generate directional guidance.
7. Display the annotated video.
8. Save the processed video as `pothole_hud_output.mp4`.

> The live preview uses OpenCV windows, so run this locally rather than in Colab.

### Keyboard Controls

| Key     | Action          |
| ------- | --------------- |
| `SPACE` | Pause / Resume  |
| `Q`     | Exit processing |
| `ESC`   | Exit processing |

---

## 📊 Configurable Parameters

| Parameter        | Default Value | Description                                   |
| ---------------- | ------------: | --------------------------------------------- |
| CONFIDENCE       |          0.25 | Minimum detection confidence                  |
| IMAGE_SIZE       |           640 | YOLO inference image size                     |
| CAMERA_HEIGHT_M  |           1.2 | Camera mounting height                        |
| VERTICAL_FOV_DEG |           45° | Camera vertical field of view                 |
| HORIZON_RATIO    |          0.40 | Estimated horizon position                    |
| SAFER_MARGIN     |           0.6 | Reserved lane safety parameter                |
| SMOOTH_FRAMES    |             8 | Number of recent decisions used for smoothing |
| OUTPUT_WIDTH     |          1280 | Output video width                            |
| OUTPUT_HEIGHT    |           720 | Output video height                           |

## 🖥️ Output

The application generates an annotated road video with:

- Pothole bounding boxes and tracking identifiers.
- Detection confidence and approximate distance.
- Three-lane visual segmentation.
- Lane-wise pothole statistics.
- Colour-coded hazard indicators.
- Directional arrows and driver recommendations.
- FPS, inference time, and processing progress.

**Output file:** `pothole_hud_output.mp4`

---

## ⚠️ Limitations

- Distance estimation is approximate and depends on camera calibration and road conditions.
- Lane segmentation is based on fixed vertical image regions, not actual lane-marking detection.
- Lane recommendations consider detected pothole risk but do not account for surrounding traffic, pedestrians, road rules, or other driving hazards.
- Tracking performance depends on detection quality and object visibility.
- Performance depends on the available CPU/GPU hardware and video resolution.
- The detector is trained on a single public dataset (3,345 training images); accuracy may drop on unseen road types, lighting, or camera angles. Reported metrics come from one training run (seed 42).

**Safety Disclaimer:** This project is an experimental computer vision prototype developed for academic and demonstration purposes. Its directional recommendations should not be treated as autonomous driving instructions or as a substitute for driver judgement.

---

## 🔮 Future Enhancements

- Real-time pothole detection using a live camera feed.
- Camera calibration for improved distance estimation.
- Integration of vehicle speed for time-to-hazard estimation.
- Audio and visual driver alerts.
- Improved lane-risk assessment using road geometry and surrounding traffic.
- Deployment on embedded hardware for real-world ADAS applications.
- Larger and more varied training data (night, rain, different cameras) and a larger YOLO model variant.

---

## 📄 Licence & Credits

- Dataset: [Pothole Detection Dataset: YOLOv11 Optimized](https://www.kaggle.com/datasets/muskanverma24/pothole-detection-dataset-yolov11-optimized), CC BY 4.0 — credit the original Roboflow project `vasanthakumar662005/pothole-pq3lu`.
- Detection framework: [Ultralytics YOLO](https://docs.ultralytics.com).

## 👥 Project Information

**Project:** Ask-Pothole – Intelligent Pothole Detection & Driver Assistance System

**Domain:** Deep Learning | Computer Vision | Advanced Driver Assistance Systems (ADAS)

**Core Technologies:** YOLO | ByteTrack | OpenCV | Python

---

_Building safer roads through intelligent vision systems._ 🚗
