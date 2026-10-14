import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "yolo11n.pt"
OUTPUT_DIR = PROJECT_DIR / "output"
LOG_PATH = OUTPUT_DIR / "person_tracking_log.csv"
VIDEO_PATH = OUTPUT_DIR / "person_tracking_recording.mp4"

CONFIDENCE_THRESHOLD = 0.40
LOG_INTERVAL_SECONDS = 5.0
CAMERA_INDEX = 0


def initialize_log():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        with LOG_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "people_detected",
                "tracking_ids",
                "confidence_scores",
                "event",
            ])


def log_detection(people_count, track_ids, confidences, event):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp,
            people_count,
            ";".join(map(str, track_ids)),
            ";".join(f"{score:.3f}" for score in confidences),
            event,
        ])


def main():
    parser = argparse.ArgumentParser(
        description="RoboRescue AI person tracking"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save the camera session to an MP4 file",
    )
    args = parser.parse_args()

    print("=" * 48)
    print("ROBORESCUE AI - PERSON TRACKING")
    print("=" * 48)

    initialize_log()

    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    print("Opening camera...")
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Could not open the camera.")
        print("Check macOS camera permissions and close other camera apps.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    video_writer = None
    if args.record:
        fps = camera.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 20.0

        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_writer = cv2.VideoWriter(
            str(VIDEO_PATH),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not video_writer.isOpened():
            print("WARNING: Video recording could not be started.")
            video_writer.release()
            video_writer = None
        else:
            print(f"Recording enabled: {VIDEO_PATH}")

    print("Camera started.")
    print("Press Q in the camera window to stop.")

    last_log_time = 0.0
    previous_ids = set()
    previous_count = None
    frame_count = 0
    start_time = time.monotonic()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("\nCould not read a camera frame.")
                break

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONFIDENCE_THRESHOLD,
                classes=[0],
                verbose=False,
            )

            result = results[0]
            annotated_frame = result.plot()

            people_count = 0
            track_ids = []
            confidences = []

            if result.boxes is not None:
                people_count = len(result.boxes)

                if result.boxes.conf is not None:
                    confidences = [
                        float(value)
                        for value in result.boxes.conf.cpu().tolist()
                    ]

                if result.boxes.id is not None:
                    track_ids = (
                        result.boxes.id.int().cpu().tolist()
                    )

            current_ids = set(track_ids)
            new_ids = current_ids - previous_ids
            lost_ids = previous_ids - current_ids

            if new_ids:
                print(f"\nNew tracking IDs: {sorted(new_ids)}")

            if lost_ids:
                print(f"\nNo longer visible: {sorted(lost_ids)}")

            elapsed = max(time.monotonic() - start_time, 0.001)
            fps = frame_count / elapsed
            frame_count += 1

            cv2.putText(
                annotated_frame,
                f"People detected: {people_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}",
                (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            now = time.monotonic()
            count_changed = people_count != previous_count
            interval_elapsed = now - last_log_time >= LOG_INTERVAL_SECONDS

            if count_changed or interval_elapsed:
                if new_ids:
                    event = "NEW_TRACK"
                elif lost_ids:
                    event = "TRACK_LOST"
                elif people_count > 0:
                    event = "PERSON_VISIBLE"
                else:
                    event = "NO_PERSON_VISIBLE"

                log_detection(
                    people_count,
                    track_ids,
                    confidences,
                    event,
                )

                last_log_time = now
                previous_count = people_count

            if video_writer is not None:
                video_writer.write(annotated_frame)

            cv2.imshow(
                "RoboRescue AI - Person Tracking",
                annotated_frame,
            )

            previous_ids = current_ids

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()

        if video_writer is not None:
            video_writer.release()

        cv2.destroyAllWindows()

        print("\nTracking stopped.")
        print(f"CSV log: {LOG_PATH}")

        if args.record and VIDEO_PATH.exists():
            print(f"Video: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
