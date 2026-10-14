import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from perception.detection_bridge import publish_detection
except ModuleNotFoundError:
    from detection_bridge import publish_detection

from rescue_intelligence import (
    initialize_log as initialize_rescue_logs,
    record_detection,
    record_rescue_assessment,
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "yolo11n.pt"
OUTPUT_DIR = PROJECT_DIR / "output"

LOG_PATH = OUTPUT_DIR / "person_tracking_log.csv"
VIDEO_PATH = OUTPUT_DIR / "person_tracking_recording.mp4"

CONFIDENCE_THRESHOLD = 0.40
LOG_INTERVAL_SECONDS = 5.0
TRACK_LOST_GRACE_SECONDS = 3.0
CAMERA_INDEX = 0


# ============================================================
# PERSON TRACKING CSV LOGGING
# ============================================================

def initialize_log():
    """Create the person-tracking CSV log if it does not exist."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        with LOG_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "timestamp",
                "people_detected",
                "tracking_ids",
                "confidence_scores",
                "event",
            ])


def log_detection(
    people_count,
    track_ids,
    confidences,
    event,
):
    """Append a person-tracking event to the CSV log."""

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    with LOG_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            people_count,
            ";".join(map(str, track_ids)),
            ";".join(
                f"{score:.3f}"
                for score in confidences
            ),
            event,
        ])


# ============================================================
# RESCUE INTELLIGENCE INTEGRATION
# ============================================================

def process_rescue_assessment(
    people_count,
    confidences,
):
    """
    Record the latest detection and create a rescue assessment.

    The priority is a demonstration rule based on the number
    of people detected, not a real emergency severity score.
    """

    record_detection(
        people_count,
        confidences,
    )

    assessment = record_rescue_assessment(
        people_count,
        confidences,
    )

    print(
        f"\nRescue assessment | "
        f"People: {people_count} | "
        f"Priority: {assessment['priority']} | "
        f"Status: {assessment['status']}"
    )

    print(
        f"Recommended action: "
        f"{assessment['recommended_action']}"
    )

    return assessment


# ============================================================
# MAIN APPLICATION
# ============================================================

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

    print("=" * 50)
    print("ROBORESCUE AI - PERSON TRACKING")
    print("=" * 50)

    # Initialize both logging systems.
    initialize_log()
    initialize_rescue_logs()

    print("Person-tracking log initialized.")
    print("Rescue Intelligence logs initialized.")

    # --------------------------------------------------------
    # LOAD YOLO MODEL
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        print(
            f"ERROR: YOLO model not found: {MODEL_PATH}"
        )
        return

    print("\nLoading YOLO model...")

    model = YOLO(str(MODEL_PATH))

    print("YOLO model loaded successfully.")

    # --------------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------------

    print("\nOpening camera...")

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Could not open the camera.")
        print(
            "Check macOS camera permissions and close "
            "other camera applications."
        )

        camera.release()
        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    video_writer = None

    # --------------------------------------------------------
    # OPTIONAL VIDEO RECORDING
    # --------------------------------------------------------

    if args.record:

        fps_setting = camera.get(
            cv2.CAP_PROP_FPS
        )

        if not fps_setting or fps_setting <= 1:
            fps_setting = 20.0

        width = int(
            camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        video_writer = cv2.VideoWriter(
            str(VIDEO_PATH),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_setting,
            (width, height),
        )

        if not video_writer.isOpened():

            print(
                "WARNING: Video recording could not "
                "be started."
            )

            video_writer.release()
            video_writer = None

        else:

            print(
                f"Recording enabled: {VIDEO_PATH}"
            )

    # --------------------------------------------------------
    # TRACKING INITIALIZATION
    # --------------------------------------------------------

    print("\nCamera started successfully.")

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD}"
    )

    print(
        f"Tracking grace period: "
        f"{TRACK_LOST_GRACE_SECONDS:.1f} seconds"
    )

    print(
        f"Rescue assessment interval: "
        f"{LOG_INTERVAL_SECONDS:.1f} seconds"
    )

    print(
        "\nPress Q in the camera window to stop."
    )

    last_log_time = 0.0
    previous_count = None

    frame_count = 0
    start_time = time.monotonic()

    # Remember when each tracking ID was last observed.
    last_seen = {}

    announced_ids = set()
    reported_lost_ids = set()

    # --------------------------------------------------------
    # CAMERA PROCESSING LOOP
    # --------------------------------------------------------

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "\nCould not read a camera frame."
                )

                break

            # -----------------------------------------------
            # YOLO + BYTETRACK
            # -----------------------------------------------

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

            # -----------------------------------------------
            # EXTRACT PERSON DETECTIONS
            # -----------------------------------------------

            if result.boxes is not None:

                people_count = len(
                    result.boxes
                )

                if result.boxes.conf is not None:

                    confidences = [
                        float(value)
                        for value in (
                            result.boxes.conf
                            .cpu()
                            .tolist()
                        )
                    ]

                if result.boxes.id is not None:

                    track_ids = (
                        result.boxes.id
                        .int()
                        .cpu()
                        .tolist()
                    )

            now = time.monotonic()

            current_ids = set(track_ids)

            # Publish actual webcam detections for the dashboard/simulation.
            publish_detection(
                people_detected=people_count,
                tracking_ids=track_ids,
                confidence_scores=confidences,
            )


            # -----------------------------------------------
            # UPDATE TRACKING IDS
            # -----------------------------------------------

            for track_id in current_ids:

                last_seen[track_id] = now

                if track_id not in announced_ids:

                    print(
                        f"\nNew tracking ID: {track_id}"
                    )

                    announced_ids.add(track_id)

                # Report a previously lost ID if it returns.
                if track_id in reported_lost_ids:

                    print(
                        f"\nTracking ID {track_id} "
                        f"visible again."
                    )

                    reported_lost_ids.remove(
                        track_id
                    )

            # -----------------------------------------------
            # DETECT LOST TRACKING IDS
            # -----------------------------------------------

            for track_id, last_time in list(
                last_seen.items()
            ):

                missing_for = now - last_time

                if (
                    track_id not in current_ids
                    and missing_for
                    >= TRACK_LOST_GRACE_SECONDS
                    and track_id
                    not in reported_lost_ids
                ):

                    print(
                        f"\nTracking ID {track_id} "
                        f"not seen for "
                        f"{missing_for:.1f} seconds."
                    )

                    reported_lost_ids.add(
                        track_id
                    )

            # -----------------------------------------------
            # CALCULATE FPS
            # -----------------------------------------------

            elapsed = max(
                now - start_time,
                0.001,
            )

            fps = frame_count / elapsed

            frame_count += 1

            # -----------------------------------------------
            # DISPLAY PERSON COUNT
            # -----------------------------------------------

            cv2.putText(
                annotated_frame,
                f"People detected: {people_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            # -----------------------------------------------
            # DISPLAY TRACKING COUNT
            # -----------------------------------------------

            cv2.putText(
                annotated_frame,
                f"Tracking IDs: {len(current_ids)}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            # -----------------------------------------------
            # DISPLAY FPS
            # -----------------------------------------------

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # -----------------------------------------------
            # DISPLAY LOST-TRACK GRACE PERIOD
            # -----------------------------------------------

            cv2.putText(
                annotated_frame,
                (
                    f"Lost grace: "
                    f"{TRACK_LOST_GRACE_SECONDS:.0f}s"
                ),
                (20, 141),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # -----------------------------------------------
            # STAGE 5: RESCUE INTELLIGENCE
            # -----------------------------------------------

            count_changed = (
                people_count != previous_count
            )

            interval_elapsed = (
                now - last_log_time
                >= LOG_INTERVAL_SECONDS
            )

            if count_changed or interval_elapsed:

                if current_ids:

                    event = "PERSON_VISIBLE"

                else:

                    event = "NO_PERSON_VISIBLE"

                # Keep the original tracking log.
                log_detection(
                    people_count,
                    track_ids,
                    confidences,
                    event,
                )

                # Record and assess the latest detection.
                process_rescue_assessment(
                    people_count,
                    confidences,
                )

                last_log_time = now
                previous_count = people_count

            # -----------------------------------------------
            # SAVE VIDEO IF RECORDING IS ENABLED
            # -----------------------------------------------

            if video_writer is not None:

                video_writer.write(
                    annotated_frame
                )

            # -----------------------------------------------
            # DISPLAY CAMERA WINDOW
            # -----------------------------------------------

            cv2.imshow(
                "RoboRescue AI - Person Tracking",
                annotated_frame,
            )

            # Press Q to exit.
            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):

                break

    except KeyboardInterrupt:

        print(
            "\nTracking interrupted by user."
        )

    except Exception as error:

        print(
            f"\nERROR during tracking: {error}"
        )

        raise

    finally:

        # -----------------------------------------------
        # CLEANUP
        # -----------------------------------------------

        camera.release()

        if video_writer is not None:

            video_writer.release()

        cv2.destroyAllWindows()

        print("\nTracking stopped.")

        print(
            f"Person tracking log: {LOG_PATH}"
        )

        print(
            f"Rescue detection log: "
            f"{PROJECT_DIR / 'output' / 'rescue_detection_log.csv'}"
        )

        print(
            f"Rescue incident log: "
            f"{PROJECT_DIR / 'output' / 'rescue_incidents.csv'}"
        )

        if args.record:

            print(
                f"Recording location: {VIDEO_PATH}"
            )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()