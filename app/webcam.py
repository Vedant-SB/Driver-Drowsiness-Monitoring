import cv2
import time

from app.face_detector import FaceDetector
from app.eye_monitor import EyeMonitor
from app.yawn_detector import YawnDetector
from app.distraction_detector import DistractionDetector


def start_webcam(eye_model, yawn_model):

    # Initialize detectors
    face_detector = FaceDetector()
    eye_monitor = EyeMonitor(eye_model)
    yawn_detector = YawnDetector(yawn_model)
    distraction_detector = DistractionDetector()

    # Initialize webcam
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(
    "Actual Camera Resolution:",
    camera.get(cv2.CAP_PROP_FRAME_WIDTH),
    "x",
    camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

    print(
        "Actual Camera Resolution:",
        camera.get(cv2.CAP_PROP_FRAME_WIDTH),
        "x",
        camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if not camera.isOpened():
        print("Error: Could not access webcam.")
        face_detector.close()
        camera.release()
        return

    print("Driver Monitoring System Started")
    print("Press Q to exit.")

    start_time = time.monotonic()

    # Yawn inference optimization
    yawn_frame_count = 0
    yawn_result = None
    YAWN_INFERENCE_INTERVAL = 5

    try:

        while True:

            success, frame = camera.read()

            if not success:
                break

            # Mirror webcam frame
            frame = cv2.flip(frame, 1)

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            # Face landmark detection
            landmarks = face_detector.detect(
                frame, timestamp_ms
            )

            if landmarks:

                # -----------------------------
                # Eye Monitoring
                # -----------------------------

                eye_result = eye_monitor.process(
                    frame, landmarks
                )

                if eye_result:

                    status = (
                        "DROWSINESS ALERT!"
                        if eye_result["drowsiness_alert"]
                        else "Normal"
                    )

                    color = (
                        (0, 0, 255)
                        if eye_result["drowsiness_alert"]
                        else (0, 255, 0)
                    )

                    cv2.putText(
                        frame,
                        f"Left Eye: {eye_result['left_eye']}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2
                    )

                    cv2.putText(
                        frame,
                        f"Right Eye: {eye_result['right_eye']}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2
                    )

                    cv2.putText(
                        frame,
                        f"EAR: {eye_result['ear']:.3f}",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"PERCLOS: {eye_result['perclos']:.2%}",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Closed: {eye_result['closed_duration']:.1f}s",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        status,
                        (20, 205),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2
                    )

                else:

                    cv2.putText(
                        frame,
                        "Eye detection unavailable",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2
                    )

                # -----------------------------
                # Yawn Detection
                # -----------------------------

                yawn_frame_count += 1

                # Run yawn model every 5 frames
                if (
                    yawn_frame_count
                    % YAWN_INFERENCE_INTERVAL == 0
                ):

                    yawn_result = yawn_detector.process(
                        frame, landmarks
                    )

                # Display latest yawn result
                if yawn_result:

                    yawn_status = yawn_result["state"]

                    yawn_color = (
                        (0, 0, 255)
                        if yawn_status == "Yawn"
                        else (0, 255, 0)
                    )

                    cv2.putText(
                        frame,
                        f"Yawn: {yawn_status}",
                        (20, 260),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, yawn_color, 2
                    )

                    cv2.putText(
                        frame,
                        f"Yawn Probability: "
                        f"{yawn_result['yawn_probability']:.2%}",
                        (20, 295),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"MAR: {yawn_result['mar']:.3f}",
                        (20, 330),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2
                    )

                # -----------------------------
                # Distraction Detection
                # -----------------------------

                distraction_result = distraction_detector.process(
                    frame, landmarks
                )

                if distraction_result:

                    yaw = distraction_result["yaw"]
                    pitch = distraction_result["pitch"]
                    roll = distraction_result["roll"]

                    distraction_alert = distraction_result["alert"]

                    distraction_status = (
                        "DISTRACTION ALERT!"
                        if distraction_alert
                        else distraction_result["direction"]
                    )

                    distraction_color = (
                        (0, 0, 255)
                        if distraction_alert
                        else (0, 255, 0)
                    )

                    cv2.putText(
                        frame,
                        f"Yaw: {yaw:.1f} deg",
                        (20, 385),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Pitch: {pitch:.1f} deg",
                        (20, 420),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        f"Roll: {roll:.1f} deg",
                        (20, 455),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        distraction_status,
                        (20, 500),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, distraction_color, 2
                    )

                    cv2.putText(
                        frame,
                        f"Distraction Duration: "
                        f"{distraction_result['elapsed_time']:.1f}s",
                        (20, 535),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2
                    )

            else:

                # Reset detection states when face disappears
                yawn_result = None
                yawn_frame_count = 0

                yawn_detector.prediction_history.clear()
                yawn_detector.yawn_start_time = None

                distraction_detector.reset()

                cv2.putText(
                    frame,
                    "No Face Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2
                )

            # Display webcam
            cv2.imshow(
                "Driver Monitoring System",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        camera.release()
        face_detector.close()
        cv2.destroyAllWindows()

        print("Driver Monitoring System Stopped")