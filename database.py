"""
database.py

Sets up and manages the SQLite database that logs every sound EchoAlert
detects. This is the persistence layer Week 3 builds alerts and the
Streamlit dashboard on top of.

Schema:
    detections
    ----------
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    timestamp       TEXT     -- ISO 8601, e.g. "2026-07-12T14:32:01"
    predicted_label TEXT     -- the mapped category (doorbell, alarm, etc.)
    confidence      REAL     -- YAMNet's confidence score for that frame
    actual_label    TEXT     -- left NULL for now; filled in manually during
                                Week 4's evaluation pass
"""

import sqlite3
from datetime import datetime

DB_PATH = "detections.db"


def init_db(db_path=DB_PATH):
    """
    Creates the detections table if it doesn't already exist. Safe to
    call every time the app starts up -- won't wipe existing data.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            actual_label TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_detection(predicted_label, confidence, db_path=DB_PATH):
    """
    Logs a single detection event. Call this from mic_yamnet_live.py
    whenever a frame's category + confidence crosses your threshold
    (and, once you decide on it, passes the frame-agreement rule).

    timestamp is generated here automatically -- no need to pass one in.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO detections (timestamp, predicted_label, confidence, actual_label)
        VALUES (?, ?, ?, NULL)
    """, (datetime.now().isoformat(timespec="seconds"), predicted_label, confidence))
    conn.commit()
    conn.close()


def get_recent_detections(limit=20, db_path=DB_PATH):
    """
    Returns the most recent N detections, newest first. This is what
    the Streamlit dashboard will call to show a live feed.
    Returns a list of tuples: (id, timestamp, predicted_label, confidence, actual_label)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, predicted_label, confidence, actual_label
        FROM detections
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    # Quick manual test: set up the table, insert a fake detection,
    # then read it back. Run this file directly to sanity check the
    # schema before wiring it into the live mic script.
    init_db()
    insert_detection("doorbell", 0.42)
    print("Inserted a test detection. Recent rows:")
    for row in get_recent_detections():
        print(row)