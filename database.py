"""
database.py

Handles the SQLite database that stores every sound EchoAlert detects.

Schema:
    detections
    ----------
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    timestamp       TEXT     -- ISO 8601, e.g. "2026-07-12T14:32:01"
    predicted_label TEXT     -- the mapped category (doorbell, alarm, etc.)
    confidence      REAL     -- YAMNet's confidence score for that frame
    actual_label    TEXT     -- the true label, filled in manually during evaluation
"""

import sqlite3
from datetime import datetime

DB_PATH = "detections.db"


def init_db(db_path=DB_PATH):
    """
    Creates the detections table if it doesn't already exist.
    Safe to call every time the app starts -- it never wipes existing data.
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
    Logs one confirmed detection event with the current timestamp.
    confidence is cast to a plain float, since SQLite stores numpy
    float types as raw bytes (BLOB) instead of numbers if left as-is.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO detections (timestamp, predicted_label, confidence, actual_label)
        VALUES (?, ?, ?, NULL)
    """, (datetime.now().isoformat(timespec="seconds"), predicted_label, float(confidence)))
    conn.commit()
    conn.close()


def get_recent_detections(limit=20, db_path=DB_PATH):
    """
    Returns the most recent N detections, newest first, as a list of
    tuples: (id, timestamp, predicted_label, confidence, actual_label).
    Used by the dashboard to show a live feed.
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
    # Quick manual check: create the table, insert one test row, print it back.
    init_db()
    insert_detection("doorbell", 0.42)
    print("Inserted a test detection. Recent rows:")
    for row in get_recent_detections():
        print(row)