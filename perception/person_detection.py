from pathlib import Path

import cv2
from ultralytics import YOLO


# Project configuration
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.40

# COCO class ID 0 represents a person.
PERSON_CLASS_ID = 0


def detect_people(image_path: str) -> None:
    """Detect people in an image and save the annotated result."""

    image_file = Path(image_path).expanduser().resolve()

    if not image_file.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_file}\n"
            "Place an image in assets/images and provide its path."
        )

    image = cv2.imread(str(image_file))

    if image is None:
        raise ValueError(f"Could not read image: {image_file}")

    model = YOLO(MODEL_NAME)

    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )

    annotated_image = results[0].plot()

    output_path = OUTPUT_DIR / f"{image_file.stem}_detected.jpg"

    if not cv2.imwrite(str(output_path), annotated_image):
        raise OSError(f"Could not save output image: {output_path}")

    person_count = len(results[0].boxes)

    print("\n===== ROBORESCUE AI =====")
    print(f"Input image: {image_file.name}")
    print(f"People detected: {person_count}")
    print(f"Output saved: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RoboRescue AI person detection"
    )

    parser.add_argument(
        "image",
        help="Path to the image you want to analyze",
    )

    args = parser.parse_args()

    detect_people(args.image)