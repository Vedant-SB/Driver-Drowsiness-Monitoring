# Deep Learning-Based Real-Time Driver Drowsiness and Distraction Monitoring System

## Overview

Driver fatigue, drowsiness, and distraction are major contributors to road accidents.

This project implements a real-time driver monitoring system using Deep Learning and Computer Vision. It uses a webcam to monitor the driver's facial features, detect signs of drowsiness, identify yawning, and estimate head orientation to detect possible driver distraction.

The system combines two trained MobileNetV2 models with MediaPipe facial landmarks, geometric measurements, temporal analysis, and audio alerts.

The objective is to provide continuous monitoring of driver alertness and generate alerts when potentially unsafe driving behaviour is detected.

## Features

### 1. Real-Time Face Detection

- Uses MediaPipe Face Landmarker.
- Detects facial landmarks from webcam frames.
- Tracks facial features required for eye, mouth, and head-pose analysis.

### 2. Eye State Classification

- Uses a trained MobileNetV2 model.
- Classifies eye images into Open and Closed categories.
- Calculates Eye Aspect Ratio (EAR) using facial landmarks.
- Uses temporal monitoring to identify prolonged eye closure.

### 3. Drowsiness Detection

- Monitors continuous eye closure duration.
- Implements PERCLOS (Percentage of Eye Closure) monitoring.
- Combines eye-state classification and EAR-based analysis.
- Generates a drowsiness alert when configured conditions are met.

### 4. Yawning Detection

- Uses a trained MobileNetV2 classification model.
- Classifies facial images into Yawn and No Yawn categories.
- Uses Mouth Aspect Ratio (MAR) as an additional geometric indicator.
- Implements temporal confirmation and voting to reduce false detections.

### 5. Driver Distraction Detection

- Uses facial landmarks and OpenCV's solvePnP algorithm.
- Estimates head orientation using yaw and pitch angles.
- Detects when the driver appears to look away from the forward direction.
- Uses a time-based confirmation mechanism to avoid triggering alerts for brief head movements.

### 6. Audio Alert System

- Uses Pygame for audio alert generation and playback.
- Generates alerts for detected drowsiness, yawning, and distraction.
- Uses event tracking and cooldown mechanisms to avoid repeated unnecessary alerts.
- Implements alert prioritization when multiple conditions occur.

### 7. Fatigue Event Monitoring

- Tracks detected yawning and drowsiness episodes over a rolling time window.
- Counts detection events rather than continuously counting every frame.
- Identifies accumulated fatigue indicators based on configured conditions.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep Learning framework |
| Keras | Model loading and inference |
| MobileNetV2 | Transfer learning architecture |
| OpenCV | Image processing, webcam access, and head-pose estimation |
| MediaPipe | Facial landmark detection |
| NumPy | Numerical computations |
| Pygame | Audio alert generation |
| Matplotlib | Data visualization and model evaluation |

## Deep Learning Models

The system uses two independently trained MobileNetV2 models.

### 1. Eye State Classification Model

Classifies eye images into two categories:

- Open
- Closed

**Test Performance:**

| Metric | Result |
|---|---:|
| Test Accuracy | 99.08% |
| Correct Predictions | 216 / 218 |

### 2. Yawn Classification Model

Classifies facial images into two categories:

- Yawn
- No Yawn

**Test Performance:**

| Metric | Result |
|---|---:|
| Test Accuracy | 94.95% |
| F1-Score | Approximately 0.95 |

Both models use transfer learning with MobileNetV2.

The models are trained and evaluated separately for their respective classification tasks.

## Additional Computer Vision Techniques

Apart from deep learning classification, the system uses geometric and temporal techniques.

### Eye Aspect Ratio (EAR)

EAR measures eye openness using distances between facial landmarks around the eye.

It helps identify eye closure and supports the drowsiness detection mechanism.

### PERCLOS

PERCLOS estimates the proportion of monitored time during which the eyes remain closed.

It is used as a temporal indicator of possible driver drowsiness.

### Mouth Aspect Ratio (MAR)

MAR measures mouth opening using facial landmarks.

It provides an additional geometric indicator for yawning detection.

### Head-Pose Estimation

OpenCV's solvePnP algorithm estimates the driver's head orientation using facial landmarks and a 3D facial reference model.

Yaw and pitch angles are used to identify possible distraction caused by looking away from the road.

## Alert Management

The alert manager coordinates the different detection mechanisms.

It includes:

- Event-based tracking of yawning and drowsiness episodes.
- Rolling time-window monitoring.
- Fatigue condition evaluation.
- Alert prioritization.
- Cooldown mechanisms to prevent repeated alerts.
- Audio alerts using Pygame.

## Project Structure

```text
Driver-Drowsiness-Monitoring/
│
├── app/
│   ├── __init__.py
│   ├── face_detector.py
│   ├── eye_monitor.py
│   ├── yawn_detector.py
│   ├── distraction_detector.py
│   ├── alert_manager.py
│   └── webcam.py
│
├── models/
│   ├── eye_mobilenetv2_final.keras
│   ├── yawn_mobilenetv2_final.keras
│   └── face_landmarker.task
│
├── sounds/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── test_distraction.py
└── test_yawn_crop.py
```

## Installation and Setup

### Prerequisites

- Python 3.13
- Webcam
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Vedant-SB/Driver-Drowsiness-Monitoring.git
cd Driver-Drowsiness-Monitoring
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python main.py
```

Make sure the required model files are present inside the `models/` directory before running the application.

A working webcam is also required.

## Dataset

The models were trained using the Drowsiness Dataset available on Kaggle:

https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset

The dataset contains four categories:

- Open
- Closed
- Yawn
- No Yawn

Separate classification models were trained for eye-state classification and yawning detection.

## Model Evaluation

| Model | Architecture | Test Accuracy |
|---|---|---:|
| Eye Classification | MobileNetV2 | 99.08% |
| Yawn Classification | MobileNetV2 | 94.95% |

These results represent performance on the held-out test dataset.

Actual webcam performance may vary depending on lighting conditions, camera placement, facial orientation, glasses, and individual differences.

## Future Improvements

- Improve inference speed and overall real-time performance.
- Improve robustness under different lighting conditions.
- Evaluate performance across a wider range of users and camera positions.
- Improve detection reliability under challenging webcam conditions.
- Explore further optimization of facial landmark processing and model inference.

## Disclaimer

This project is developed for educational and research purposes.

It is not a certified vehicle safety system and must not be relied upon as the sole means of preventing driver fatigue-related accidents.