import cv2
import mediapipe as mp
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FACE_MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"


class FaceDetector:

    def __init__(self):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(FACE_MODEL_PATH)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.face_landmarks:
            return result.face_landmarks[0]

        return None

    def close(self):
        self.detector.close()