"""
app.py

Streamlit dashboard for EchoAlert. Shows a live-updating feed of
confirmed detections pulled from the SQLite database, plus a count-by-
category chart of what's been detected most.

Run with:
    streamlit run app.py

This dashboard is read-only -- it never writes to the database.
mic_yamnet_live.py is the only thing that logs detections. Run that in
one terminal and this dashboard in another, both pointing at the same
detections.db file.
"""

import time
import pandas as pd
import streamlit as st

from database import get_recent_detections, init_db

st.set_page_config(page_title="EchoAlert Dashboard", page_icon="🔔", layout="centered")

# Safe to call even if mic_yamnet_live.py hasn't been run yet -- this
# just makes sure the table exists so the page doesn't error out on a
# fresh checkout with no detections.db yet.
init_db()

st.title("🔔 EchoAlert Live Dashboard")
st.caption("Live feed of confirmed sound detections, pulled from detections.db")

# --- Sidebar controls ---
refresh_seconds = st.sidebar.slider("Auto-refresh interval (seconds)", 2, 30, 5)
row_limit = st.sidebar.slider("Rows to show", 10, 200, 50)

# --- Pull latest data ---
rows = get_recent_detections(limit=row_limit)
columns = ["id", "timestamp", "predicted_label", "confidence", "actual_label"]
df = pd.DataFrame(rows, columns=columns)

if df.empty:
    st.info("No detections logged yet. Run mic_yamnet_live.py and make some noise!")
else:
    # --- Count-by-category summary ---
    st.subheader("Detections by category")
    counts = df["predicted_label"].value_counts()
    st.bar_chart(counts)

    # --- Recent detections table ---
    st.subheader(f"Most recent {len(df)} detections")
    display_df = df.copy()
    display_df["confidence"] = display_df["confidence"].round(3)
    st.dataframe(
        display_df[["timestamp", "predicted_label", "confidence", "actual_label"]],
        use_container_width=True,
        hide_index=True,
    )

st.caption(f"Auto-refreshing every {refresh_seconds}s. Last updated: {time.strftime('%H:%M:%S')}")

# --- Auto-refresh loop ---
# Streamlit has no built-in "poll every N seconds" primitive, so the
# simplest reliable approach is to sleep, then rerun the whole script.
# Fine for a dashboard like this where the data volume is small.
time.sleep(refresh_seconds)
st.rerun()