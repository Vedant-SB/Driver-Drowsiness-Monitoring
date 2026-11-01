import cv2
import time

from app.face_detector import FaceDetector
from app.eye_monitor import EyeMonitor
from app.yawn_detector import YawnDetector
from app.distraction_detector import DistractionDetector
from app.alert_manager import AlertManager


def start_webcam(eye_model, yawn_model):

    # Initialize detectors
    face_detector = FaceDetector()
    eye_monitor = EyeMonitor(eye_model)
    yawn_detector = YawnDetector(yawn_model)
    distraction_detector = DistractionDetector()
    alert_manager = AlertManager()

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

    if not camera.isOpened():
        print("Error: Could not access webcam.")
        face_detector.close()
        alert_manager.close()
        camera.release()
        return

    print("Driver Monitoring System Started")
    print("Press R to reset all monitoring values.")
    print("Press Q to exit.")

    start_time = time.monotonic()

    # Yawn inference optimization
    yawn_frame_count = 0
    yawn_result = None
    YAWN_INFERENCE_INTERVAL = 5

    # Current detection states
    drowsiness_alert_state = False
    yawn_alert_state = False
    distraction_alert_state = False

    try:
        while True:

            success, frame = camera.read()

            if not success:
                print("Error: Could not read frame from webcam.")
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

                    drowsiness_alert_state = (
                        eye_result["drowsiness_alert"]
                    )

                    status = (
                        "DROWSINESS ALERT!"
                        if drowsiness_alert_state
                        else "Normal"
                    )

                    color = (
                        (0, 0, 255)
                        if drowsiness_alert_state
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

                    drowsiness_alert_state = False

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
                if yawn_frame_count % YAWN_INFERENCE_INTERVAL == 0:

                    yawn_result = yawn_detector.process(
                        frame, landmarks
                    )

                # Display latest yawn result
                if yawn_result:

                    yawn_status = yawn_result["state"]

                    yawn_alert_state = (
                        yawn_status == "Yawn"
                        and yawn_result["confirmed"]
                    )

                    yawn_color = (
                        (0, 0, 255)
                        if yawn_alert_state
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

                else:
                    yawn_alert_state = False

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

                    distraction_alert_state = (
                        distraction_result["alert"]
                    )

                    distraction_status = (
                        "DISTRACTION ALERT!"
                        if distraction_alert_state
                        else distraction_result["direction"]
                    )

                    distraction_color = (
                        (0, 0, 255)
                        if distraction_alert_state
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
                    distraction_alert_state = False

                # -----------------------------
                # CENTRAL ALERT MANAGER
                # -----------------------------

                alert_result = alert_manager.update(
                    drowsiness_alert=drowsiness_alert_state,
                    yawn_alert=yawn_alert_state,
                    distraction_alert=distraction_alert_state
                )

                # Display fatigue event counts
                cv2.putText(
                    frame,
                    f"Yawns (90s): {alert_result['yawn_count']}",
                    (20, 590),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

                cv2.putText(
                    frame,
                    f"Drowsiness Episodes (90s): "
                    f"{alert_result['drowsiness_count']}",
                    (20, 625),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

                # Combined fatigue warning
                if alert_result["fatigue_alert"]:

                    cv2.putText(
                        frame,
                        "FATIGUE WARNING!",
                        (20, 680),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 3
                    )

            else:

                # Reset detection states when face disappears
                yawn_result = None
                yawn_frame_count = 0

                drowsiness_alert_state = False
                yawn_alert_state = False
                distraction_alert_state = False

                eye_monitor.reset()
                yawn_detector.reset()
                distraction_detector.reset()

                # Reset active states, but preserve recent event history
                alert_manager.reset_active_states()

                # Update manager so expired events are removed
                alert_result = alert_manager.update()

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

            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):

                eye_monitor.reset()
                yawn_detector.reset()
                distraction_detector.reset()

                yawn_result = None
                yawn_frame_count = 0

                drowsiness_alert_state = False
                yawn_alert_state = False
                distraction_alert_state = False

                alert_manager.reset_all()

            elif key == ord('q'):
                break

    finally:

        camera.release()
        face_detector.close()
        alert_manager.close()
        cv2.destroyAllWindows()

        print("Driver Monitoring System Stopped")