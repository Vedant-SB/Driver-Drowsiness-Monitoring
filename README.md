# Deep Learning-Based Real-Time Driver Drowsiness and Distraction Monitoring System

## Overview

Driver fatigue and drowsiness are major contributors to road accidents. This project aims to develop a real-time driver monitoring system using Deep Learning and Computer Vision to identify signs of driver drowsiness and distraction.

The system uses a webcam to monitor the driver's facial features and detect indicators such as eye closure and yawning. It combines trained deep learning models with facial landmark-based analysis to monitor driver alertness.

## Features

### Implemented Features

- Real-time face detection using MediaPipe Face Landmarker.
- Eye state classification (Open / Closed) using a trained MobileNetV2 model.
- Yawning detection using a trained MobileNetV2 model.
- Eye Aspect Ratio (EAR) calculation.
- PERCLOS-based eye closure monitoring.
- Continuous eye closure duration monitoring.
- Real-time webcam-based monitoring.

### Planned Features

- Head-pose estimation for driver distraction detection using OpenCV solvePnP.
- Audio alerts for detected drowsiness and distraction.
- Further performance optimization.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep Learning framework |
| Keras | Model development and inference |
| MobileNetV2 | Transfer learning architecture |
| OpenCV | Image processing and webcam access |
| MediaPipe | Facial landmark detection |
| NumPy | Numerical computations |
| Pygame | Audio alert integration (planned) |

## Deep Learning Models

The project uses two independently trained MobileNetV2 models.

### 1. Eye Classification Model

Classifies eye images into two categories:

- Open
- Closed

Test Performance:
- Accuracy: 99.08%
- Correct predictions: 216 / 218

### 2. Yawn Classification Model

Classifies facial images into two categories:

- Yawn
- No Yawn

Test Performance:
- Accuracy: 94.95%
- F1-score: Approximately 0.95

Both models use transfer learning with MobileNetV2.

## Additional Monitoring Techniques

Apart from deep learning classification, the system uses facial geometry and temporal analysis.

Eye Aspect Ratio (EAR): Measures the degree of eye openness using facial landmarks.

PERCLOS: Estimates the proportion of time the eyes remain closed over a monitoring window.

Continuous Eye Closure: Tracks how long the driver's eyes remain closed continuously.

These techniques help identify prolonged eye closure and potential drowsiness.

## Project Structure

    Driver-Drowsiness-Monitoring/
    |
    |-- app/
    |   |-- __init__.py
    |   |-- face_detector.py
    |   |-- webcam.py
    |   |-- eye_monitor.py
    |   |-- yawn_detector.py
    |
    |-- models/
    |   |-- eye_mobilenetv2_final.keras
    |   |-- yawn_mobilenetv2_final.keras
    |   |-- face_landmarker.task
    |
    |-- sounds/
    |
    |-- main.py
    |-- requirements.txt
    |-- README.md
    |-- .gitignore

## Installation and Setup

### Prerequisites

- Python 3.13
- Webcam
- Git

### 1. Clone the Repository

    git clone https://github.com/YOUR_USERNAME/Driver-Drowsiness-Monitoring.git
    cd Driver-Drowsiness-Monitoring

### 2. Create a Virtual Environment

    python -m venv .venv

Activate the environment on Windows:

    .venv\Scripts\activate

### 3. Install Dependencies

    pip install -r requirements.txt

### 4. Run the Application

    python main.py

Make sure all required model files are present inside the models/ directory before running the application.

## Dataset

The models were trained using the Drowsiness Dataset available on Kaggle:

https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset

The dataset contains images belonging to four categories:

- Open
- Closed
- Yawn
- No Yawn

Separate classification models were trained for eye state and yawning detection.

## Model Evaluation

| Model | Architecture | Test Accuracy |
|---|---|---:|
| Eye Classification | MobileNetV2 | 99.08% |
| Yawn Classification | MobileNetV2 | 94.95% |

These results represent performance on the held-out test dataset and do not guarantee equivalent real-world webcam performance.

## Future Improvements

- Implement real-time head-pose-based distraction detection.
- Integrate audio alerts.
- Improve robustness under different lighting conditions.
- Evaluate performance across different users and camera positions.
- Optimize real-time inference performance.

## Disclaimer

This project is developed for educational and research purposes. It is not a certified vehicle safety system and should not be relied upon as the sole means of preventing driver fatigue-related accidents.