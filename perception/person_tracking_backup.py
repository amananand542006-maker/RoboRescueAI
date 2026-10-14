import cv2
from ultralytics import YOLO

MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.40


def main():
    print("=" * 40)
    print("ROBORESCUE AI")
    print("PERSON TRACKING - BYTETRACK")
    print("=" * 40)

    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    print("Opening camera...")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open the camera.")
        return

    print("Camera started. Press Q to stop.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
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

            tracked_people = 0

            if result.boxes is not None:
                tracked_people = len(result.boxes)

                if result.boxes.id is not None:
                    track_ids = (
                        result.boxes.id.int().cpu().tolist()
                    )

                    for track_id in track_ids:
                        print(
                            f"\rTracking person ID: {track_id}",
                            end="",
                            flush=True,
                        )

            cv2.putText(
                annotated_frame,
                f"People detected: {tracked_people}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "RoboRescue AI - ByteTrack",
                annotated_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\nTracking stopped.")


if __name__ == "__main__":
    main()
