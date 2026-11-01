import cv2
import numpy as np
import time


# MediaPipe Face Landmarker landmark indices
# Only stable facial landmarks are used.
POSE_LANDMARK_IDX = {
    "forehead": 10,
    "nose_tip": 1,
    "right_eye_outer": 33,
    "right_eye_inner": 133,
    "left_eye_inner": 362,
    "left_eye_outer": 263,
}


# Corresponding approximate 3D facial reference points
MODEL_POINTS_3D = np.array([
    (0.0, 330.0, -65.0),       # Forehead
    (0.0, 0.0, 0.0),           # Nose tip
    (225.0, 170.0, -135.0),    # Right eye outer
    (75.0, 170.0, -135.0),     # Right eye inner
    (-75.0, 170.0, -135.0),    # Left eye inner
    (-225.0, 170.0, -135.0),   # Left eye outer
], dtype=np.float64)


# Distraction thresholds (degrees)
YAW_THRESHOLD = 30.0
PITCH_THRESHOLD = 25.0

# Minimum continuous duration before triggering alert
DISTRACTION_DURATION = 2.5


class DistractionDetector:

    def __init__(self):

        self.distraction_start_time = None

    def estimate_head_pose(self, frame, landmarks):

        height, width = frame.shape[:2]

        # Ensure all required landmarks are available
        if len(landmarks) <= max(POSE_LANDMARK_IDX.values()):
            return None

        # Convert normalized MediaPipe coordinates to pixel coordinates
        image_points = []

        for index in POSE_LANDMARK_IDX.values():

            landmark = landmarks[index]

            x = landmark.x * width
            y = landmark.y * height

            image_points.append((x, y))

        image_points = np.array(
            image_points,
            dtype=np.float64
        )

        # Approximate camera intrinsic matrix
        focal_length = width

        camera_matrix = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        distortion_coefficients = np.zeros(
            (4, 1),
            dtype=np.float64
        )

        # Estimate head orientation
        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Extract Euler angles
        sy = np.sqrt(
            rotation_matrix[0, 0] ** 2 +
            rotation_matrix[1, 0] ** 2
        )

        singular = sy < 1e-6

        if not singular:

            pitch = np.arctan2(
                rotation_matrix[2, 1],
                rotation_matrix[2, 2]
            )

            yaw = np.arctan2(
                -rotation_matrix[2, 0],
                sy
            )

            roll = np.arctan2(
                rotation_matrix[1, 0],
                rotation_matrix[0, 0]
            )

        else:

            pitch = np.arctan2(
                -rotation_matrix[1, 2],
                rotation_matrix[1, 1]
            )

            yaw = np.arctan2(
                -rotation_matrix[2, 0],
                sy
            )

            roll = 0

        # Convert radians to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)

        return {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll)
        }

    def process(self, frame, landmarks):

        pose = self.estimate_head_pose(frame, landmarks)

        if pose is None:
            self.reset()
            return None

        yaw = pose["yaw"]
        pitch = pose["pitch"]
        roll = pose["roll"]

        # Determine whether the driver is looking away
        looking_away = (
            abs(yaw) > YAW_THRESHOLD or
            abs(pitch) > PITCH_THRESHOLD
        )

        current_time = time.monotonic()

        if looking_away:

            if self.distraction_start_time is None:
                self.distraction_start_time = current_time

            elapsed_time = (
                current_time - self.distraction_start_time
            )

        else:

            self.distraction_start_time = None
            elapsed_time = 0.0

        # Trigger alert only after continuous distraction
        alert = (
            looking_away and
            elapsed_time >= DISTRACTION_DURATION
        )

        # Determine approximate direction
        if abs(yaw) > YAW_THRESHOLD:
            direction = "Looking Left/Right"

        elif abs(pitch) > PITCH_THRESHOLD:
            direction = "Looking Up/Down"

        else:
            direction = "Forward"

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "looking_away": looking_away,
            "alert": alert,
            "elapsed_time": elapsed_time,
            "direction": direction
        }

    def reset(self):

        self.distraction_start_time = None