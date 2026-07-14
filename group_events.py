"""
group_events.py

Collapses consecutive same-category detection rows into single "events"
using a time-gap threshold. This is the Day 4 event-level evaluation
step (Option A): a sustained sound that produced 5 confirmed frame-pairs
should count as ONE event, not five, when we compute precision/recall.

Usage:
    python group_events.py
"""

import sqlite3
import pandas as pd

DB_PATH = "detections.db"
GAP_THRESHOLD_SECONDS = 4  # rows in the same category within this many
                            # seconds of each other are treated as one event
START_FROM = "2026-07-14T00:00:00"  # ignore older leftover test rows


def load_detections(db_path=DB_PATH, start_from=START_FROM):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT id, timestamp, predicted_label, confidence, actual_label "
        "FROM detections WHERE timestamp >= ? ORDER BY timestamp ASC",
        conn,
        params=(start_from,),
    )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def group_into_events(df, gap_threshold=GAP_THRESHOLD_SECONDS):
    """
    Walks through rows in timestamp order. Starts a new event whenever
    the category changes OR the gap since the last row in the current
    event exceeds gap_threshold seconds. Returns one row per event with
    start/end time, row count, and confidence stats.
    """
    if df.empty:
        return pd.DataFrame()

    events = []
    current_ids = [df.iloc[0]["id"]]
    current_label = df.iloc[0]["predicted_label"]
    current_start = df.iloc[0]["timestamp"]
    current_confidences = [df.iloc[0]["confidence"]]
    last_time = current_start

    for _, row in df.iloc[1:].iterrows():
        gap = (row["timestamp"] - last_time).total_seconds()
        same_category = row["predicted_label"] == current_label

        if same_category and gap <= gap_threshold:
            # still the same event
            current_ids.append(row["id"])
            current_confidences.append(row["confidence"])
        else:
            # close out the current event, start a new one
            events.append({
                "event_start": current_start,
                "event_end": last_time,
                "predicted_label": current_label,
                "row_count": len(current_ids),
                "row_ids": current_ids,
                "avg_confidence": sum(current_confidences) / len(current_confidences),
                "max_confidence": max(current_confidences),
                "actual_label": "",  # you'll fill this in by hand
                "notes": "",  # you'll fill this in by hand
            })
            current_ids = [row["id"]]
            current_label = row["predicted_label"]
            current_start = row["timestamp"]
            current_confidences = [row["confidence"]]

        last_time = row["timestamp"]

    # close out the final event
    events.append({
        "event_start": current_start,
        "event_end": last_time,
        "predicted_label": current_label,
        "row_count": len(current_ids),
        "row_ids": current_ids,
        "avg_confidence": sum(current_confidences) / len(current_confidences),
        "max_confidence": max(current_confidences),
        "actual_label": "",
    })

    return pd.DataFrame(events)


if __name__ == "__main__":
    df = load_detections()
    print(f"Loaded {len(df)} raw detection rows from {START_FROM} onward.\n")

    events = group_into_events(df)
    print(f"Collapsed into {len(events)} events (gap threshold = {GAP_THRESHOLD_SECONDS}s):\n")

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 140)
    print(events[["event_start", "predicted_label", "row_count", "avg_confidence", "max_confidence", "row_ids"]])

    # Save to CSV so you can label actual_label in a spreadsheet if you'd rather not edit in terminal
    events.to_csv("events_for_labeling.csv", index=False)
    print("\nSaved to events_for_labeling.csv — fill in the actual_label column there.")