# RoboRescueAI

An AI-powered person detection and tracking system designed to support rescue monitoring and situational awareness.

## Features

- Person detection using YOLO.
- Multi-object tracking using ByteTrack.
- Live camera processing with OpenCV.
- Tracking IDs and detection confidence scores.
- Rescue intelligence and priority assessment.
- CSV logging for detection and incident records.
- JSON bridge for sharing the latest detection state.
- Streamlit dashboard.
- Experimental rescue simulation.

## Technology Stack

- Python
- YOLO
- ByteTrack
- OpenCV
- Streamlit
- NumPy
- CSV and JSON

## Project Structure

- `assets/` — Test images and other assets.
- `dashboard/` — Streamlit dashboard.
- `docs/` — Project documentation.
- `output/` — Detection snapshots and generated logs.
- `perception/` — Detection, tracking, and perception modules.
- `tests/` — Automated tests.
- `simulation.py` — Rescue simulation.
- `yolo11n.pt` — YOLO model weights.

## Requirements

- Python 3.12 recommended.
- A compatible webcam for live detection.
- Required Python packages.
- YOLO model weights.

## Installation

Clone the repository:

    git clone YOUR_GITHUB_REPOSITORY_URL
    cd RoboRescueAI

Create a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install the project dependencies before running the application.

## Run Person Detection

    python perception/person_detection.py

## Run Person Tracking

    python perception/person_tracking.py

Press Q in the camera window to stop tracking.

## Run the Dashboard

    python -m streamlit run dashboard/app.py

Open the local address displayed in the Terminal.

## Detection Bridge

The detection bridge publishes the latest detection snapshot to:

    output/latest_detection.json

The snapshot includes a timestamp, detected-person count, tracking IDs, confidence scores, and source information.

The current webcam pipeline does not provide real-world coordinates.

## Logs

The application can generate CSV records for person tracking, rescue detection assessments, and rescue incidents.

## Safety and Limitations

RoboRescueAI is an experimental prototype for person detection and rescue-monitoring research.

It does not replace trained rescue personnel, emergency services, or validated life-safety equipment. Detection results may be inaccurate or incomplete.

## Future Improvements

- Improved real-time dashboard integration.
- Spatial localization and mapping.
- Additional automated tests.
- Improved simulation and visualization.
- Deployment and performance optimization.
