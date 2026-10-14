import csv
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
LOG_FILE = OUTPUT_DIR / "person_tracking_log.csv"
VIDEO_FILE = OUTPUT_DIR / "person_tracking_recording.mp4"

st.set_page_config(
    page_title="RoboRescue AI",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 RoboRescue AI")
st.subheader("AI-Powered Person Detection and Tracking")
st.caption("Monitoring dashboard | Local development")

if st.button("🔄 Refresh dashboard"):
    st.rerun()

st.divider()


def load_tracking_log():
    if not LOG_FILE.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "people_detected",
                "tracking_ids",
                "confidence_scores",
                "event",
            ]
        )

    try:
        return pd.read_csv(LOG_FILE)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame(
            columns=[
                "timestamp",
                "people_detected",
                "tracking_ids",
                "confidence_scores",
                "event",
            ]
        )


data = load_tracking_log()

if data.empty:
    st.warning(
        "No tracking history found. Run the person-tracking program first."
    )
    latest = None
else:
    latest = data.iloc[-1]

    people_count = int(latest["people_detected"])

    tracking_ids = (
        str(latest["tracking_ids"])
        if pd.notna(latest["tracking_ids"])
        else ""
    )

    confidence_values = []
    confidence_text = str(latest["confidence_scores"])

    if confidence_text and confidence_text.lower() != "nan":
        try:
            confidence_values = [
                float(value)
                for value in confidence_text.split(";")
                if value.strip()
            ]
        except ValueError:
            confidence_values = []

    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else None
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("People Currently Detected", people_count)

    col2.metric(
        "Tracking IDs",
        tracking_ids if tracking_ids else "None",
    )

    col3.metric(
        "Average Confidence",
        f"{average_confidence:.1%}"
        if average_confidence is not None
        else "N/A",
    )

    col4.metric("Logged Events", len(data))

    st.divider()

    st.subheader("📋 Latest Tracking Event")

    event_col1, event_col2 = st.columns(2)

    with event_col1:
        st.write("**Timestamp**")
        st.write(str(latest["timestamp"]))

        st.write("**Event type**")
        st.write(str(latest["event"]))

    with event_col2:
        st.write("**People detected**")
        st.write(people_count)

        st.write("**Confidence scores**")
        st.write(confidence_text if confidence_text else "N/A")

    st.divider()

    st.subheader("📈 Detection History")

    history = data.tail(100).copy()

    history["people_detected"] = pd.to_numeric(
        history["people_detected"],
        errors="coerce",
    )

    chart_data = history[
        ["timestamp", "people_detected"]
    ].copy()

    chart_data["timestamp"] = pd.to_datetime(
        chart_data["timestamp"],
        errors="coerce",
    )

    chart_data = chart_data.dropna()

    if not chart_data.empty:
        chart_data = chart_data.set_index("timestamp")
        st.line_chart(chart_data["people_detected"])

    st.subheader("🧾 Recent Events")

    st.dataframe(
        data.tail(20).iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.subheader("🎥 Detection Recording")

if VIDEO_FILE.exists():
    video_size_mb = VIDEO_FILE.stat().st_size / (1024 * 1024)

    st.success(
        f"Recording found: {VIDEO_FILE.name} "
        f"({video_size_mb:.1f} MB)"
    )

    with st.expander("Play recorded video"):
        st.video(str(VIDEO_FILE))
else:
    st.info(
        "No recording found yet. Start tracking with the --record option."
    )

st.divider()

st.caption(
    "RoboRescue AI | Local dashboard. "
    "Detection events are observations, not confirmation of an emergency."
)
