# RoboRescueAI

RoboRescueAI is an experimental computer-vision prototype for detecting and
tracking people in camera footage. It combines YOLO, OpenCV, and ByteTrack
with basic rescue-priority assessments, CSV logs, a Streamlit dashboard, and a
simple simulation.

> **Important:** This project is for learning and demonstration only. It is
> not a validated emergency-response system and must not be used as a
> substitute for emergency services, trained responders, or safety equipment.

## Features

- Detect people in images and live webcam video.
- Track people across webcam frames and display tracking IDs.
- Record detection events and basic operator-review recommendations.
- Publish the latest tracking result as JSON for the dashboard.
- View detection data in a Streamlit dashboard.
- Run a basic rescue simulation.

## Requirements

- Python 3.10 or newer.
- A webcam for live detection and tracking.
- macOS, Windows, or Linux. Camera access and OpenCV GUI support depend on
  your system configuration.

## Getting started

Clone your GitHub repository and enter the project directory:

```bash
git clone https://github.com/amananand542006-maker/RoboRescueAI.git
cd RoboRescueAI
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python streamlit pandas streamlit-autorefresh pygame
```

The repository includes the `yolo11n.pt` model file used by the detection and
tracking scripts.

## Run the application

Run person detection on an image:

```bash
python perception/person_detection.py assets/images/rescue_test.jpg
```

The annotated image is saved under `output/`.

Run live webcam detection:

```bash
python perception/live_detection.py
```

Run webcam detection with person tracking:

```bash
python perception/person_tracking.py
```

Press **Q** in the camera window to stop.

Start the dashboard:

```bash
python -m streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit. The dashboard reads data from the
project's `output/` directory; run detection or tracking to populate it.

Run the simulation:

```bash
python simulation.py
```

## Project layout

```text
RoboRescueAI/
├── assets/images/       # Sample image
├── dashboard/           # Streamlit dashboard
├── output/              # Generated logs, snapshots, and tracking data
├── perception/          # Detection, tracking, and rescue-assessment code
├── simulation.py        # Rescue simulation
└── yolo11n.pt           # YOLO model weights
```

## Generated data

Detection and tracking runs may create CSV logs, a JSON snapshot, and annotated
images or video under `output/`. These files can change each time the scripts
run. Review generated data before publishing it; do not commit private footage,
personal information, or data you do not have permission to share.

## Limitations

Detection results can be inaccurate, incomplete, or affected by camera
placement, lighting, and environmental conditions. The rescue-priority logic
is a basic demonstration based on detections; it does not establish whether a
person is injured or in danger. Always have a qualified human review results.

## Contributing

Improvements and bug reports are welcome. Please include steps to reproduce
issues and avoid sharing sensitive footage or personal information.
