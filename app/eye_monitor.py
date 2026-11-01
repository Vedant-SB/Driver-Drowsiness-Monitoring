import cv2
import numpy as np
import time
from collections import deque


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.23
CLOSED_EYE_DURATION = 2.0
PERCLOS_WINDOW = 60.0
PERCLOS_THRESHOLD = 0.75

EYE_OPEN_THRESHOLD = 0.40

INFERENCE_INTERVAL = 3


class EyeMonitor:

    def __init__(self, eye_model):

        self.eye_model = eye_model

        self.closed_start_time = None
        self.eye_history = deque()

        self.frame_count = 0

        self.last_left_prediction = None
        self.last_right_prediction = None

    def reset(self):

        self.closed_start_time = None
        self.eye_history.clear()

        self.frame_count = 0

        self.last_left_prediction = None
        self.last_right_prediction = None

    # -----------------------------------------
    # CALCULATE EYE ASPECT RATIO
    # -----------------------------------------

    def calculate_ear(self, landmarks, indices):

        points = np.array([
            [landmarks[i].x, landmarks[i].y]
            for i in indices
        ])

        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])

        horizontal = np.linalg.norm(points[0] - points[3])

        if horizontal == 0:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    # -----------------------------------------
    # ALIGNED EYE CROP EXTRACTION
    # -----------------------------------------

    def extract_eye_crop(self, frame, landmarks, indices):

        height, width = frame.shape[:2]

        points = np.array([
            [
                landmarks[i].x * width,
                landmarks[i].y * height
            ]
            for i in indices
        ], dtype=np.float32)

        # Eye corner coordinates
        first_corner = points[0]
        second_corner = points[3]

        # Calculate eye center
        center = (
            first_corner + second_corner
        ) / 2.0

        # Calculate eye width
        eye_width = np.linalg.norm(
            second_corner - first_corner
        )

        if eye_width < 5:
            return None

        # Calculate eye rotation angle
        delta_x = second_corner[0] - first_corner[0]
        delta_y = second_corner[1] - first_corner[1]

        angle = np.degrees(
    np.arctan2(delta_y, delta_x)
)

        # Normalize rotation to handle both eye orientations
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        # Rotate frame to align eye horizontally
        rotation_matrix = cv2.getRotationMatrix2D(
            tuple(center),
            angle,
            1.0
        )

        aligned_frame = cv2.warpAffine(
            frame,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # Transform eye landmarks after rotation
        transformed_points = cv2.transform(
            points.reshape(1, -1, 2),
            rotation_matrix
        )[0]

        x_min = transformed_points[:, 0].min()
        y_min = transformed_points[:, 1].min()

        x_max = transformed_points[:, 0].max()
        y_max = transformed_points[:, 1].max()

        eye_height = y_max - y_min

        # Preserve surrounding eye features
        padding_x = int(eye_width * 1.2)
        padding_y = int(eye_height * 2.0)

        x_min = int(max(0, x_min - padding_x))
        y_min = int(max(0, y_min - padding_y))

        x_max = int(min(width, x_max + padding_x))
        y_max = int(min(height, y_max + padding_y))

        crop = aligned_frame[
            y_min:y_max,
            x_min:x_max
        ]

        if crop.size == 0:
            return None

        return cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )

    # -----------------------------------------
    # EYE CLASSIFICATION
    # -----------------------------------------

    def predict_eye(self, eye_image):

        processed = cv2.resize(
            eye_image,
            (224, 224)
        )

        processed = processed.astype("float32") / 255.0

        prediction = float(
            self.eye_model.predict(
                np.expand_dims(processed, axis=0),
                verbose=0
            )[0][0]
        )

        return {
            "state": (
                "Open"
                if prediction >= EYE_OPEN_THRESHOLD
                else "Closed"
            ),

            "confidence": (
                prediction
                if prediction >= EYE_OPEN_THRESHOLD
                else 1 - prediction
            )
        }

    # -----------------------------------------
    # TEMPORAL DROWSINESS MONITORING
    # -----------------------------------------

    def update_temporal_monitor(self, eyes_closed, current_time):

        if eyes_closed:

            if self.closed_start_time is None:
                self.closed_start_time = current_time

            closed_duration = (
                current_time - self.closed_start_time
            )

        else:

            self.closed_start_time = None
            closed_duration = 0.0

        self.eye_history.append(
            (current_time, eyes_closed)
        )

        while (
            self.eye_history
            and current_time - self.eye_history[0][0] > PERCLOS_WINDOW
        ):

            self.eye_history.popleft()

        history = list(self.eye_history)

        closed_time = 0.0
        total_time = 0.0

        for i in range(1, len(history)):

            previous_time, previous_closed = history[i - 1]

            current_observation_time, _ = history[i]

            interval = (
                current_observation_time - previous_time
            )

            total_time += interval

            if previous_closed:
                closed_time += interval

        perclos = (
            closed_time / total_time
            if total_time > 0
            else 0.0
        )

        alert = (
            closed_duration >= CLOSED_EYE_DURATION
            or (
                total_time >= PERCLOS_WINDOW
                and perclos >= PERCLOS_THRESHOLD
            )
        )

        return {
            "closed_duration": closed_duration,
            "perclos": perclos,
            "drowsiness_alert": alert
        }

    # -----------------------------------------
    # MAIN PROCESSING PIPELINE
    # -----------------------------------------

    def process(self, frame, landmarks):

        current_time = time.monotonic()

        # Calculate EAR
        left_ear = self.calculate_ear(
            landmarks,
            LEFT_EYE
        )

        right_ear = self.calculate_ear(
            landmarks,
            RIGHT_EYE
        )

        average_ear = (
            left_ear + right_ear
        ) / 2.0

        self.frame_count += 1

        # Run CNN inference every 3 frames
        if self.frame_count % INFERENCE_INTERVAL == 1:

            # Account for mirrored webcam display
            left_crop = self.extract_eye_crop(
                frame,
                landmarks,
                RIGHT_EYE
            )

            right_crop = self.extract_eye_crop(
                frame,
                landmarks,
                LEFT_EYE
            )

            if left_crop is None or right_crop is None:

                self.reset()
                return None

            self.last_left_prediction = self.predict_eye(
                left_crop
            )

            self.last_right_prediction = self.predict_eye(
                right_crop
            )

        if (
            self.last_left_prediction is None
            or self.last_right_prediction is None
        ):

            return None

        # Preserve existing drowsiness decision logic
        eyes_closed = (
            self.last_left_prediction["state"] == "Closed"
            and
            self.last_right_prediction["state"] == "Closed"
            and
            average_ear < EAR_THRESHOLD
        )

        temporal_result = self.update_temporal_monitor(
            eyes_closed,
            current_time
        )

        return {
            "left_eye": self.last_left_prediction["state"],

            "right_eye": self.last_right_prediction["state"],

            "left_confidence": self.last_left_prediction["confidence"],

            "right_confidence": self.last_right_prediction["confidence"],

            "ear": average_ear,

            "eyes_closed": eyes_closed,

            **temporal_result
        }