from pathlib import Path
from datetime import datetime
import time

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

TRACKING_LOG = OUTPUT_DIR / "person_tracking_log.csv"
RESCUE_LOG = OUTPUT_DIR / "rescue_detection_log.csv"
VIDEO_PATH = OUTPUT_DIR / "person_tracking_recording.mp4"

REFRESH_MS = 3000
STALE_AFTER_SECONDS = 15

st.set_page_config(
    page_title="RoboRescue AI | Command Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st_autorefresh(
    interval=REFRESH_MS,
    limit=None,
    key="roborescue_stage4_refresh",
)


# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    .hero {
        padding: 26px;
        border-radius: 18px;
        background: linear-gradient(120deg, #132b46, #173e46);
        border: 1px solid #28516a;
        margin-bottom: 20px;
    }

    .hero h1 {
        margin: 0;
        color: #f8fafc;
        font-size: 2.2rem;
    }

    .hero p {
        color: #b9d5e8;
        margin-bottom: 0;
    }

    div[data-testid="stMetric"] {
        background-color: #111c2e;
        border: 1px solid #24364c;
        padding: 16px;
        border-radius: 14px;
    }

    h2, h3 {
        color: #f1f5f9;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def parse_ids(value):
    if pd.isna(value) or not str(value).strip():
        return []

    ids = []

    for item in str(value).split(";"):
        item = item.strip()

        try:
            ids.append(int(float(item)))
        except ValueError:
            continue

    return ids


def parse_scores(value):
    if pd.isna(value) or not str(value).strip():
        return []

    scores = []

    for item in str(value).split(";"):
        try:
            scores.append(float(item.strip()))
        except ValueError:
            continue

    return scores


tracking_df = load_csv(TRACKING_LOG)
rescue_df = load_csv(RESCUE_LOG)

if not tracking_df.empty and "timestamp" in tracking_df.columns:
    tracking_df["timestamp"] = pd.to_datetime(
        tracking_df["timestamp"],
        errors="coerce",
        utc=True,
    )

    tracking_df = tracking_df.dropna(subset=["timestamp"])
    tracking_df = tracking_df.sort_values("timestamp")

    if "people_detected" in tracking_df.columns:
        tracking_df["people_detected"] = pd.to_numeric(
            tracking_df["people_detected"],
            errors="coerce",
        )

    if "confidence_scores" in tracking_df.columns:
        tracking_df["average_confidence"] = tracking_df[
            "confidence_scores"
        ].apply(
            lambda value: (
                sum(parse_scores(value)) / len(parse_scores(value))
                if parse_scores(value)
                else float("nan")
            )
        )

    if "tracking_ids" in tracking_df.columns:
        tracking_df["parsed_ids"] = tracking_df["tracking_ids"].apply(
            parse_ids
        )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🤖 RoboRescue AI")
st.sidebar.caption("AI Search & Rescue Monitoring")

page = st.sidebar.radio(
    "Navigation",
    [
        "Command Center",
        "Detection Analytics",
        "Event History",
        "Video Review",
        "System Information",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Local development prototype")
st.sidebar.caption("Dashboard refresh: 3 seconds")


# --------------------------------------------------
# HEADER AND DATA FRESHNESS
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>ROBORESCUE AI</h1>
        <p>Intelligent Person Detection & Search-and-Rescue Monitoring</p>
    </div>
    """,
    unsafe_allow_html=True,
)

latest = tracking_df.iloc[-1] if not tracking_df.empty else None

data_age = None

if latest is not None:
    latest_timestamp = latest["timestamp"]

    if latest_timestamp.tzinfo is None:
        latest_timestamp = latest_timestamp.tz_localize("UTC")

    data_age = max(
        0,
        (
            pd.Timestamp.now(tz="UTC") - latest_timestamp
        ).total_seconds(),
    )

if latest is None:
    st.error(
        "NO TRACKING DATA: Start the person-tracking program "
        "to generate CSV records."
    )
elif data_age > STALE_AFTER_SECONDS:
    st.warning(
        f"TRACKING DATA IS STALE: The newest CSV record is "
        f"{data_age:.0f} seconds old. Check whether the tracker is running."
    )
else:
    st.success(
        f"TRACKING DATA IS RECENT — Last CSV record {data_age:.1f} "
        "seconds ago. This confirms recent logging, not camera health."
    )


# --------------------------------------------------
# COMMAND CENTER
# --------------------------------------------------

if page == "Command Center":

    st.subheader("Mission Overview")

    if latest is None:
        people_count = 0
        current_ids = []
        average_confidence = None
        latest_event = "No data"
    else:
        count_value = latest.get("people_detected", 0)
        people_count = (
            int(count_value) if pd.notna(count_value) else 0
        )

        current_ids = parse_ids(latest.get("tracking_ids", ""))
        scores = parse_scores(latest.get("confidence_scores", ""))

        average_confidence = (
            sum(scores) / len(scores) * 100 if scores else None
        )

        latest_event = str(latest.get("event", "Unknown"))

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("People Detected", people_count)
    metric2.metric("Active Tracking IDs", len(current_ids))
    metric3.metric(
        "Average Confidence",
        f"{average_confidence:.1f}%"
        if average_confidence is not None
        else "N/A",
    )
    metric4.metric("Latest Logged Event", latest_event)

    st.caption(
        "Metrics reflect the latest CSV record, not necessarily the "
        "current camera frame."
    )

    st.divider()

    left, right = st.columns([1.6, 1])

    with left:
        st.subheader("Detection Timeline")

        if (
            not tracking_df.empty
            and "people_detected" in tracking_df.columns
        ):
            chart_data = tracking_df.dropna(
                subset=["people_detected"]
            ).tail(200)

            if not chart_data.empty:
                chart_data = chart_data.set_index("timestamp")
                st.line_chart(
                    chart_data[["people_detected"]],
                    height=300,
                )
            else:
                st.info("No valid detection counts are available.")
        else:
            st.info("Detection history will appear after tracking starts.")

    with right:
        st.subheader("Current Tracking IDs")

        if current_ids:
            for track_id in current_ids:
                st.write(f"🎯 Tracking ID **{track_id}**")
        else:
            st.info("No tracking IDs in the latest record.")

        st.subheader("Latest Event")

        st.write(latest_event)

        if latest is not None:
            st.caption(
                "Recorded at: "
                + latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
            )

    st.divider()
    st.subheader("Recent Tracking Records")

    if not tracking_df.empty:
        display_columns = [
            column
            for column in [
                "timestamp",
                "people_detected",
                "tracking_ids",
                "confidence_scores",
                "event",
            ]
            if column in tracking_df.columns
        ]

        recent = tracking_df[display_columns].tail(15).copy()

        if "timestamp" in recent.columns:
            recent["timestamp"] = recent["timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )

        st.dataframe(
            recent.iloc[::-1],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No tracking records available yet.")


# --------------------------------------------------
# DETECTION ANALYTICS
# --------------------------------------------------

elif page == "Detection Analytics":

    st.subheader("Detection Analytics")

    if tracking_df.empty:
        st.info("Run the tracker to build analytics.")
    else:
        if "people_detected" in tracking_df.columns:
            counts = tracking_df["people_detected"].dropna()

            if not counts.empty:
                a, b, c = st.columns(3)
                a.metric("Peak Recorded Count", int(counts.max()))
                b.metric("Average Recorded Count", f"{counts.mean():.2f}")
                c.metric("Records", len(counts))

                st.markdown("#### People Detected Over Time")
                chart = tracking_df.dropna(
                    subset=["people_detected"]
                ).set_index("timestamp")

                st.line_chart(chart[["people_detected"]])

        if "average_confidence" in tracking_df.columns:
            confidence_data = tracking_df.dropna(
                subset=["average_confidence"]
            )

            if not confidence_data.empty:
                st.markdown("#### Average Detection Confidence")
                confidence_chart = confidence_data.set_index("timestamp")
                st.line_chart(
                    confidence_chart[["average_confidence"]],
                )

                st.caption(
                    "Confidence describes model detection certainty, "
                    "not the likelihood that a person needs rescue."
                )


# --------------------------------------------------
# EVENT HISTORY
# --------------------------------------------------

elif page == "Event History":

    st.subheader("Event History")

    if tracking_df.empty:
        st.info("No tracking events are available.")
    else:
        if "event" in tracking_df.columns:
            event_options = sorted(
                tracking_df["event"].dropna().astype(str).unique()
            )

            selected_events = st.multiselect(
                "Filter events",
                event_options,
                default=event_options,
            )

            filtered = tracking_df[
                tracking_df["event"].astype(str).isin(selected_events)
            ].copy()
        else:
            filtered = tracking_df.copy()

        st.write(f"Matching records: {len(filtered)}")

        if "timestamp" in filtered.columns:
            filtered["timestamp"] = filtered["timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )

        st.dataframe(
            filtered.iloc[::-1],
            width="stretch",
            hide_index=True,
        )

        csv_bytes = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download filtered tracking history",
            data=csv_bytes,
            file_name="roborescue_tracking_history.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Rescue Detection Log")

    if rescue_df.empty:
        st.info("No rescue detection log records found.")
    else:
        st.dataframe(
            rescue_df.tail(100).iloc[::-1],
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download rescue detection log",
            data=rescue_df.to_csv(index=False).encode("utf-8"),
            file_name="roborescue_rescue_detection_log.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# VIDEO REVIEW
# --------------------------------------------------

elif page == "Video Review":

    st.subheader("Recorded Tracking Video")

    if VIDEO_PATH.exists() and VIDEO_PATH.stat().st_size > 0:
        st.video(str(VIDEO_PATH))

        st.caption(
            f"File size: {VIDEO_PATH.stat().st_size / (1024 * 1024):.1f} MB"
        )
    else:
        st.info(
            "No recorded video found. Run "
            "'python perception/person_tracking.py --record' "
            "to create a recording."
        )

    st.warning(
        "This is a saved recording, not a live camera feed."
    )


# --------------------------------------------------
# SYSTEM INFORMATION
# --------------------------------------------------

elif page == "System Information":

    st.subheader("System Information")

    st.write("Project directory:", str(PROJECT_DIR))
    st.write("Tracking CSV:", str(TRACKING_LOG))
    st.write("Rescue CSV:", str(RESCUE_LOG))
    st.write("Video recording:", str(VIDEO_PATH))

    for label, path in [
        ("Tracking log", TRACKING_LOG),
        ("Rescue log", RESCUE_LOG),
        ("Recording", VIDEO_PATH),
    ]:
        exists = path.exists()
        st.write(
            f"{'✅' if exists else '❌'} {label}: "
            f"{'Available' if exists else 'Not found'}"
        )

    if latest is not None:
        st.write(
            "Most recent tracking record age:",
            f"{data_age:.1f} seconds",
        )
    else:
        st.write("Most recent tracking record: unavailable")

    st.info(
        "This stage is a local monitoring dashboard. "
        "Robot navigation, ROS 2 integration, and autonomous rescue "
        "missions are future development stages."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "RoboRescue AI | Stage 4 | Local development prototype | "
    f"Last dashboard refresh: {datetime.now().astimezone().strftime('%H:%M:%S')}"
)
