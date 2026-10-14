import csv
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "rescue_detection_log.csv"


def initialize_log():
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "event",
                "people_detected",
                "confidence",
            ])


def record_detection(people_detected, confidence_scores):
    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    confidence_text = ";".join(
        f"{score:.3f}" for score in confidence_scores
    )

    with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp,
            "PERSON_DETECTED" if people_detected > 0 else "NO_PERSON",
            people_detected,
            confidence_text,
        ])


def report_rescue_status(people_detected):
    if people_detected > 0:
        print(f"People visible: {people_detected} | Status: PERSON DETECTED")
    else:
        print("People visible: 0 | Status: NO PERSON DETECTED")


if __name__ == "__main__":
    initialize_log()
    record_detection(1, [0.95])

    print("ROBORESCUE AI - Rescue Intelligence")
    print(f"Log file: {LOG_FILE}")
    report_rescue_status(1)
    print("Test complete.")
