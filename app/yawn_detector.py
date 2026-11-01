import numpy as np
import cv2
import math
import time
from collections import deque


class YawnDetector:

    def __init__(self, model):

        self.model = model

        # Temporal confirmation
        self.prediction_history = deque(maxlen=5)
        self.yawn_start_time = None
        self.yawn_active = False
        self.mouth_closed_start_time = None

        # CNN + MAR detection thresholds
        self.YAWN_THRESHOLD = 0.50
        self.LOW_CONFIDENCE_THRESHOLD = 0.25

        self.MAR_THRESHOLD = 1.15
        self.HIGH_MAR_THRESHOLD = 1.35
        self.MAR_RELEASE_THRESHOLD = 1.0

        # Temporal thresholds
        self.YAWN_CONFIRMATION_TIME = 0.6
        self.MOUTH_CLOSED_DURATION = 0.5

    # Reset temporal detection state
    def reset(self):

        self.prediction_history.clear()

        self.yawn_start_time = None
        self.yawn_active = False
        self.mouth_closed_start_time = None

    # Extract face region
    def extract_face(self, frame, landmarks):

        height, width = frame.shape[:2]

        points = np.array([
            [int(point.x * width), int(point.y * height)]
            for point in landmarks
        ])

        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        face_width = x_max - x_min
        face_height = y_max - y_min

        padding_x = int(face_width * 0.20)
        padding_y = int(face_height * 0.20)

        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)

        x_max = min(width, x_max + padding_x)
        y_max = min(height, y_max + padding_y)

        face_crop = frame[y_min:y_max, x_min:x_max]

        if face_crop.size == 0:
            return None

        return face_crop

    # Calculate Mouth Aspect Ratio
    def calculate_mar(self, landmarks):

        upper_lower_pairs = [
            (13, 14),
            (82, 87),
            (312, 317)
        ]

        mouth_corners = (61, 291)

        def distance(point1, point2):

            return math.sqrt(
                (point1.x - point2.x) ** 2 +
                (point1.y - point2.y) ** 2
            )

        vertical_distances = [
            distance(landmarks[a], landmarks[b])
            for a, b in upper_lower_pairs
        ]

        horizontal_distance = distance(
            landmarks[mouth_corners[0]],
            landmarks[mouth_corners[1]]
        )

        if horizontal_distance == 0:
            return 0.0

        mar = sum(vertical_distances) / (
            2 * horizontal_distance
        )

        return float(mar)

    # CNN prediction
    def predict(self, face_image):

        resized = cv2.resize(face_image, (224, 224))

        resized = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        processed = resized.astype("float32") / 255.0

        processed = np.expand_dims(processed, axis=0)

        prediction = float(
            self.model.predict(processed, verbose=0)[0][0]
        )

        return prediction

    # Hybrid CNN + MAR decision
    def detect_yawn(self, yawn_probability, mar):

        strong_detection = (
            yawn_probability >= self.YAWN_THRESHOLD
            and mar >= self.MAR_THRESHOLD
        )

        tolerant_detection = (
            yawn_probability >= self.LOW_CONFIDENCE_THRESHOLD
            and mar >= self.HIGH_MAR_THRESHOLD
        )

        return strong_detection or tolerant_detection

    # Main yawn detection pipeline
    def process(self, frame, landmarks):

        face_crop = self.extract_face(frame, landmarks)

        if face_crop is None:
            self.reset()
            return None

        # Calculate MAR
        mar = self.calculate_mar(landmarks)

        # CNN prediction
        yawn_probability = self.predict(face_crop)

        # Hybrid CNN + MAR detection
        is_yawn = self.detect_yawn(
            yawn_probability,
            mar
        )

        current_time = time.monotonic()

        # ---------------------------------
        # YAWN CONFIRMATION
        # ---------------------------------

        if not self.yawn_active:

            self.prediction_history.append(is_yawn)

            positive_predictions = sum(
                self.prediction_history
            )

            confirmed_by_voting = (
                len(self.prediction_history) == 5
                and positive_predictions >= 3
            )

            if confirmed_by_voting:

                if self.yawn_start_time is None:
                    self.yawn_start_time = current_time

                elapsed_time = (
                    current_time - self.yawn_start_time
                )

                if elapsed_time >= self.YAWN_CONFIRMATION_TIME:

                    self.yawn_active = True
                    self.mouth_closed_start_time = None

            else:

                self.yawn_start_time = None

        # ---------------------------------
        # YAWN RELEASE
        # ---------------------------------

        else:

            if mar < self.MAR_RELEASE_THRESHOLD:

                if self.mouth_closed_start_time is None:
                    self.mouth_closed_start_time = current_time

                elif (
                    current_time - self.mouth_closed_start_time
                    >= self.MOUTH_CLOSED_DURATION
                ):

                    self.yawn_active = False
                    self.yawn_start_time = None
                    self.mouth_closed_start_time = None

                    self.prediction_history.clear()

            else:

                self.mouth_closed_start_time = None

        # ---------------------------------
        # RESULT
        # ---------------------------------

        return {
            "state": "Yawn" if self.yawn_active else "No Yawn",

            "confidence": (
                yawn_probability
                if yawn_probability >= 0.5
                else 1 - yawn_probability
            ),

            "yawn_probability": yawn_probability,

            "mar": mar,

            "confirmed": self.yawn_active
        }