"""
mic_yamnet_live.py

Feeds each live 1-second mic window into YAMNet, looks at frame-level
predictions (not averaged), and filters them down to EchoAlert's
target categories using label_mapping.py.

A detection is only "confirmed" (and written to the database) when two
consecutive frames -- possibly spanning chunk boundaries -- agree on
the same category. This filters out single-frame flicker without
diluting real sustained sounds the way full averaging would.

Press Ctrl+C to stop.
"""

import numpy as np
import pandas as pd
import sounddevice as sd
import tensorflow_hub as hub

from alerts import maybe_alert
from label_mapping import categorize_frame
from database import init_db, insert_detection

# --- Config ---
SAMPLE_RATE = 16000   # YAMNet requires 16kHz
BLOCK_SIZE = 16000    # 1 second of audio per chunk
CHANNELS = 1

# --- Load YAMNet once at startup ---
print("Loading YAMNet model...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = list(pd.read_csv(class_map_path)['display_name'])
print("Model loaded.\n")

# --- Set up the database once at startup ---
init_db()

# State tracked across audio callback calls, used for the
# 2-consecutive-frame confirmation rule. This has to live at module
# level (not as a local variable inside audio_callback) because
# sounddevice calls audio_callback fresh for every ~1s chunk -- a local
# variable would reset each time and could never "remember" the
# previous frame's category.
last_category = None
last_confidence = None


def predict_from_waveform(waveform, debug=False):
    """
    Runs YAMNet on a waveform and checks each frame individually
    (frame-level, not averaged -- averaging dilutes short sound bursts
    into near-zero confidence).

    Returns a list of (category, confidence) tuples, one per frame,
    where category is None if that frame didn't match a target sound.
    """
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()

    results = []
    for frame_scores in scores_np:
        category, confidence = categorize_frame(frame_scores, class_names, debug=debug)
        results.append((category, confidence))
    return results


def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for every new audio chunk; runs detection and logging."""
    global last_category, last_confidence

    if status:
        print(f"[status] {status}")

    waveform = indata[:, 0].astype(np.float32)
    frame_results = predict_from_waveform(waveform, debug=True)

    # A ~1 second chunk contains a couple of YAMNet frames (~0.48s each).
    # Each frame's result is printed separately so it's clear what's
    # happening inside a single chunk, rather than collapsing it into
    # one line.
    matched_any = False
    for i, (category, confidence) in enumerate(frame_results):
        if category is not None:
            print(f"  frame {i}: {category:10s} ({confidence:.3f})")
            matched_any = True

            # Confirmation rule: only log when this frame's category
            # matches the immediately preceding frame's category.
            # last_category persists across chunk boundaries.
            if category == last_category:
                insert_detection(category, confidence)
                maybe_alert(category, confidence)
                print(f"    -> CONFIRMED + logged ({category}, {confidence:.3f})")

            last_category = category
            last_confidence = confidence
        else:
            # An unmatched frame breaks the streak -- a stray silent or
            # background frame between two real detections shouldn't
            # let them count as "consecutive."
            last_category = None
            last_confidence = None

    if not matched_any:
        print("  (no target sound detected)")


def main():
    print(f"Listening at {SAMPLE_RATE} Hz, {BLOCK_SIZE / SAMPLE_RATE:.1f}s chunks. "
          f"Press Ctrl+C to stop.\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    ):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()