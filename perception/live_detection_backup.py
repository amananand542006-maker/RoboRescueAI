from pathlib import Path
import time

import cv2
from ultralytics import YOLO

from rescue_intelligence import initialize_log, record_detection

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.40
LOG_INTERVAL_SECONDS = 2.0


def main():
    print("=" * 40)
    print("ROBORESCUE AI")
    print("Live Detection + Rescue Intelligence")
    print("=" * 40)

    initialize_log()

    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    print("Opening camera...")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open the camera.")
        return

    print("Camera started. Press Q to stop.")

    last_log_time = 0.0
    previous_people_count = None

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                classes=[0],
                verbose=False,
            )

            detections = results[0].boxes
            people_count = len(detections)

            confidence_scores = [
                float(box.conf[0].item())
                for box in detections
            ]

            annotated_frame = results[0].plot()

            cv2.putText(
                annotated_frame,
                f"People detected: {people_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            current_time = time.monotonic()

            count_changed = people_count != previous_people_count
            interval_elapsed = (
                current_time - last_log_time >= LOG_INTERVAL_SECONDS
            )

            if count_changed or interval_elapsed:
                record_detection(people_count, confidence_scores)

                last_log_time = current_time
                previous_people_count = people_count

                confidence_text = ", ".join(
                    f"{score:.2f}" for score in confidence_scores
                )

                print(
                    f"People detected: {people_count}"
                    + (
                        f" | Confidence: {confidence_text}"
                        if confidence_text
                        else ""
                    )
                )

            cv2.imshow("RoboRescue AI", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed. Detection stopped.")


if __name__ == "__main__":
    main()
