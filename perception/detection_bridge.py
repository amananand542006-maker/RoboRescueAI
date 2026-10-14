"""
RoboRescueAI - Stage 7
Detection bridge between AI perception and simulation.

This module shares the latest detection snapshot through
a JSON file. It does not assign real-world coordinates
to people detected by the webcam.
"""

import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
SNAPSHOT_PATH = OUTPUT_DIR / "latest_detection.json"

_lock = threading.Lock()


def publish_detection(
    people_detected: int,
    tracking_ids=None,
    confidence_scores=None,
):
    """
    Publish the latest detection snapshot.

    Args:
        people_detected: Number of people detected.
        tracking_ids: Optional list of tracker IDs.
        confidence_scores: Optional list of confidence scores.
    """
    if people_detected < 0:
        raise ValueError("people_detected cannot be negative")

    tracking_ids = list(tracking_ids or [])
    confidence_scores = list(confidence_scores or [])

    snapshot = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "people_detected": int(people_detected),
        "tracking_ids": tracking_ids,
        "confidence_scores": confidence_scores,
        "source": "yolo_bytetrack",
        "coordinates_available": False,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with _lock:
        temporary_path = None

        try:
            # Write to a temporary file first so readers
            # don't see a partially written JSON document.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=OUTPUT_DIR,
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                json.dump(snapshot, temporary_file, indent=2)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)

            os.replace(temporary_path, SNAPSHOT_PATH)

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

    return snapshot


def read_latest_detection():
    """
    Read the latest detection snapshot.

    Returns None if the AI system has not published data yet.
    """
    with _lock:
        if not SNAPSHOT_PATH.exists():
            return None

        try:
            with SNAPSHOT_PATH.open(
                "r", encoding="utf-8"
            ) as file:
                snapshot = json.load(file)

            if not isinstance(snapshot, dict):
                return None

            return snapshot

        except (OSError, json.JSONDecodeError):
            return None