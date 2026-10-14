import csv
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "rescue_detection_log.csv"
INCIDENT_FILE = OUTPUT_DIR / "rescue_incidents.csv"

DETECTION_HEADERS = [
    "timestamp",
    "event",
    "people_detected",
    "confidence",
]

INCIDENT_HEADERS = [
    "timestamp",
    "people_detected",
    "average_confidence",
    "priority",
    "status",
    "recommended_action",
]


def initialize_log():
    """Create CSV files and headers if they do not already exist."""
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
        with LOG_FILE.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(DETECTION_HEADERS)

    if not INCIDENT_FILE.exists() or INCIDENT_FILE.stat().st_size == 0:
        with INCIDENT_FILE.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(INCIDENT_HEADERS)


def record_detection(people_detected, confidence_scores):
    """Append a detection event to the existing detection log."""
    initialize_log()

    people_detected = max(0, int(people_detected))
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    confidence_text = ";".join(
        f"{float(score):.3f}" for score in confidence_scores
    )

    with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([
            timestamp,
            "PERSON_DETECTED" if people_detected > 0 else "NO_PERSON",
            people_detected,
            confidence_text,
        ])

    return timestamp


def assess_rescue_priority(people_detected, confidence_scores):
    """
    Assign a simple demonstration priority.

    This uses person count and detector confidence only.
    It does not establish injury, danger, or a real emergency.
    """
    people_detected = max(0, int(people_detected))

    scores = [
        float(score)
        for score in confidence_scores
        if 0.0 <= float(score) <= 1.0
    ]

    average_confidence = (
        sum(scores) / len(scores) if scores else 0.0
    )

    if people_detected == 0:
        return {
            "average_confidence": average_confidence,
            "priority": "NONE",
            "status": "NO_PERSON_DETECTED",
            "recommended_action": "Continue monitoring",
        }

    if people_detected >= 3:
        priority = "HIGH"
        action = "Flag for operator review"
    elif people_detected >= 1:
        priority = "NORMAL"
        action = "Review detection and continue monitoring"
    else:
        priority = "NONE"
        action = "Continue monitoring"

    return {
        "average_confidence": average_confidence,
        "priority": priority,
        "status": "PERSON_DETECTED",
        "recommended_action": action,
    }


def record_rescue_assessment(people_detected, confidence_scores):
    """Record an operator-review recommendation in a separate CSV."""
    initialize_log()

    people_detected = max(0, int(people_detected))
    scores = [float(score) for score in confidence_scores]
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    assessment = assess_rescue_priority(
        people_detected,
        scores,
    )

    with INCIDENT_FILE.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([
            timestamp,
            people_detected,
            f"{assessment['average_confidence']:.3f}",
            assessment["priority"],
            assessment["status"],
            assessment["recommended_action"],
        ])

    return assessment


def report_rescue_status(people_detected):
    if people_detected > 0:
        print(
            f"People visible: {people_detected} | "
            "Status: PERSON DETECTED"
        )
    else:
        print("People visible: 0 | Status: NO PERSON DETECTED")


if __name__ == "__main__":
    initialize_log()

    test_people = 1
    test_scores = [0.95]

    record_detection(test_people, test_scores)
    assessment = record_rescue_assessment(test_people, test_scores)

    print("ROBORESCUE AI - Rescue Intelligence")
    print(f"Detection log: {LOG_FILE}")
    print(f"Incident log: {INCIDENT_FILE}")

    report_rescue_status(test_people)

    print(f"Priority: {assessment['priority']}")
    print(f"Recommended action: {assessment['recommended_action']}")
    print("Test complete.")
