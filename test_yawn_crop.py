import cv2
import time
from collections import deque

from app.face_detector import FaceDetector
from app.yawn_detector import YawnDetector


face_detector = FaceDetector()
yawn_detector = YawnDetector()

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

start_time = time.monotonic()

prediction_history = deque(maxlen=5)

last_prediction = None
display_state = "Detecting..."
display_probability = 0.0
display_mar = 0.0

frame_count = 0
INFERENCE_INTERVAL = 3


try:
    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        landmarks = face_detector.detect(frame, timestamp_ms)

        if landmarks:

            # Calculate MAR on every frame
            display_mar = yawn_detector.calculate_mar(landmarks)

            face_crop = yawn_detector.extract_face(
                frame, landmarks
            )

            if face_crop is not None:

                frame_count += 1

                if frame_count % INFERENCE_INTERVAL == 1:

                    last_prediction = yawn_detector.predict(
                        face_crop
                    )

                    display_probability = last_prediction[
                        "yawn_probability"
                    ]

                    print(
                        f"MAR: {display_mar:.4f} | "
                        f"Yawn Probability: {display_probability:.4f} | "
                        f"Prediction: {last_prediction['state']}"
                    )

                    prediction_history.append(
                        last_prediction["state"]
                    )

                    # Majority voting over recent predictions
                    yawn_count = prediction_history.count("Yawn")

                    if yawn_count >= 3:
                        display_state = "YAWNING"
                    else:
                        display_state = "NO YAWN"

                # Display extracted face
                cv2.imshow("Extracted Face", face_crop)

                # Display prediction on webcam
                color = (
                    (0, 0, 255)
                    if display_state == "YAWNING"
                    else (0, 255, 0)
                )

                cv2.putText(
                    frame,
                    f"Status: {display_state}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"MAR: {display_mar:.4f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Yawn Probability: {display_probability:.2%}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            else:
                cv2.putText(
                    frame,
                    "Face crop unavailable",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        else:
            display_state = "FACE NOT DETECTED"

            cv2.putText(
                frame,
                display_state,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("Yawning Detection System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    face_detector.close()
    cv2.destroyAllWindows()