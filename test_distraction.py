import cv2
import time

from app.face_detector import FaceDetector
from app.distraction_detector import DistractionDetector


def main():
    face_detector = FaceDetector()
    distraction_detector = DistractionDetector()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        face_detector.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Distraction detection started.")
    print("Press Q to exit.")

    start_time = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            landmarks = face_detector.detect(
                frame,
                timestamp_ms
            )

            if landmarks:
                result = distraction_detector.process(
                    frame,
                    landmarks
                )

                if result:
                    cv2.putText(
                        frame,
                        f"Yaw: {result['yaw']:.2f}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Pitch: {result['pitch']:.2f}",
                        (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Roll: {result['roll']:.2f}",
                        (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    status = (
                        "DISTRACTION ALERT!"
                        if result["alert"]
                        else result["direction"]
                    )

                    color = (
                        (0, 0, 255)
                        if result["alert"]
                        else (0, 255, 0)
                    )

                    cv2.putText(
                        frame,
                        status,
                        (30, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Duration: {result['elapsed_time']:.1f}s",
                        (30, 195),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )

            else:
                distraction_detector.reset()

                cv2.putText(
                    frame,
                    "No Face Detected",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Distraction Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        face_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()