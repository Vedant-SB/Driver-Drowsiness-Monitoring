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
        self.YAWN_CONFIRMATION_TIME = 0.5

        self.YAWN_THRESHOLD = 0.65

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

    def process(self, frame, landmarks):

        face_crop = self.extract_face(frame, landmarks)

        if face_crop is None:

            self.prediction_history.clear()
            self.yawn_start_time = None

            return None

        yawn_probability = self.predict(face_crop)

        # Frame-based voting
        is_yawn = (
            yawn_probability >= self.YAWN_THRESHOLD
        )

        self.prediction_history.append(is_yawn)

        positive_predictions = sum(
            self.prediction_history
        )

        confirmed_by_voting = (
            len(self.prediction_history) == 5
            and positive_predictions >= 3
        )

        current_time = time.monotonic()

        # Sustained yawn confirmation
        if confirmed_by_voting:

            if self.yawn_start_time is None:
                self.yawn_start_time = current_time

            elapsed_time = (
                current_time - self.yawn_start_time
            )

            confirmed_yawn = (
                elapsed_time >= self.YAWN_CONFIRMATION_TIME
            )

        else:

            self.yawn_start_time = None
            confirmed_yawn = False

        mar = self.calculate_mar(landmarks)

        return {
            "state": "Yawn" if confirmed_yawn else "No Yawn",

            "confidence": (
                yawn_probability
                if yawn_probability >= self.YAWN_THRESHOLD
                else 1 - yawn_probability
            ),

            "yawn_probability": yawn_probability,

            "mar": mar,

            "confirmed": confirmed_yawn
        }