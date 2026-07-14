"""
evaluate_events.py

Week 4 evaluation step. Reads the manually-labeled events_for_labeling.csv
(event-level ground truth, per the Option A grouping decision) and computes:
    - A confusion matrix: predicted_label vs actual_label
    - Precision and recall per category
    - A plain-text summary suitable for pasting into your Week 4 writeup

Usage:
    python evaluate_events.py
"""

import ast
import sqlite3

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

CSV_PATH = "events_for_labeling.csv"
DB_PATH = "detections.db"


def load_events(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    missing = df["actual_label"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} row(s) still have an empty actual_label. "
            "Fill those in before running evaluation."
        )
    return df


def parse_row_ids(row_ids_str):
    """
    The CSV stores row_ids as a string like "[75, 76, 77, 78, 79]" or
    "[np.int64(74)]" (depending on how it was exported). Handles both.
    """
    cleaned = row_ids_str.replace("np.int64(", "").replace(")", "")
    return ast.literal_eval(cleaned)


def compute_row_level_flicker(events_df, db_path=DB_PATH):
    """
    For each event, pulls back the raw per-frame rows from detections.db
    and checks how many of them disagree with the event's actual_label.
    This surfaces transient misclassification (e.g. a ringtone briefly
    flickering to "alarm") that gets hidden once rows are merged into a
    single event for the main confusion matrix.
    """
    conn = sqlite3.connect(db_path)

    flicker_rows = []
    for _, event in events_df.iterrows():
        row_ids = parse_row_ids(event["row_ids"])
        placeholders = ",".join("?" for _ in row_ids)
        raw = pd.read_sql_query(
            f"SELECT id, predicted_label, confidence FROM detections "
            f"WHERE id IN ({placeholders})",
            conn,
            params=row_ids,
        )

        total = len(raw)
        mismatched = (raw["predicted_label"] != event["actual_label"]).sum()
        flicker_rows.append({
            "event_start": event["event_start"],
            "actual_label": event["actual_label"],
            "total_frames": total,
            "mismatched_frames": mismatched,
            "flicker_rate": mismatched / total if total else 0.0,
            "mismatched_labels": sorted(set(
                raw.loc[raw["predicted_label"] != event["actual_label"], "predicted_label"]
            )),
        })

    conn.close()
    return pd.DataFrame(flicker_rows)


def evaluate(df):
    labels = sorted(set(df["predicted_label"]) | set(df["actual_label"]))

    cm = confusion_matrix(df["actual_label"], df["predicted_label"], labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"actual_{l}" for l in labels],
                          columns=[f"pred_{l}" for l in labels])

    precision, recall, f1, support = precision_recall_fscore_support(
        df["actual_label"], df["predicted_label"], labels=labels, zero_division=0
    )

    metrics_df = pd.DataFrame({
        "category": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support (num actual events)": support,
    })

    return cm_df, metrics_df


if __name__ == "__main__":
    df = load_events()
    print(f"Loaded {len(df)} labeled events.\n")

    cm_df, metrics_df = evaluate(df)

    print("=== Confusion Matrix (rows = actual, columns = predicted) ===")
    print(cm_df)
    print()

    print("=== Precision / Recall / F1 per category ===")
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(metrics_df.to_string(index=False))
    print()

    overall_correct = (df["predicted_label"] == df["actual_label"]).sum()
    print(f"Overall event-level accuracy: {overall_correct}/{len(df)} "
          f"({100 * overall_correct / len(df):.1f}%)")

    print()
    print("=== Row-level flicker report (per event) ===")
    print("This shows transient misclassification WITHIN each confirmed event --")
    print("frames that briefly disagreed with the event's true label. This is")
    print("hidden by event-level accuracy above, since a merged event only needs")
    print("agreement on the *overall* label, not every individual frame.\n")

    flicker_df = compute_row_level_flicker(df)
    pd.set_option("display.max_colwidth", None)
    print(flicker_df.to_string(index=False))

    total_frames = flicker_df["total_frames"].sum()
    total_mismatched = flicker_df["mismatched_frames"].sum()
    print(f"\nOverall row-level flicker rate: {total_mismatched}/{total_frames} "
          f"frames ({100 * total_mismatched / total_frames:.1f}%) disagreed with "
          f"their event's actual_label.")