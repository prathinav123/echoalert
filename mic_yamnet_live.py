"""
mic_yamnet_live.py

Week 2, Step 2 (v2): Feed each live 1-second mic window into YAMNet,
look at frame-level predictions (not averaged), and filter them down
to EchoAlert's target categories using label_mapping.py.

Press Ctrl+C to stop.
"""

import numpy as np
import pandas as pd
import sounddevice as sd
import tensorflow_hub as hub

from label_mapping import categorize_frame

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



"""
Runs YAMNet on a waveform and returns a per-frame breakdown, checking
EACH frame individually (frame-level, not averaged -- averaging was
diluting short bursts, per your Week 1 notes and yesterday's test).

Returns a list of (category, confidence) tuples, one per frame,
where category is None if that frame didn't match a target sound.
"""
def predict_from_waveform(waveform, debug=False):
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()

    results = []
    for frame_scores in scores_np:
        category, confidence = categorize_frame(frame_scores, class_names, debug=debug)
        results.append((category, confidence))
    return results


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[status] {status}")

    waveform = indata[:, 0].astype(np.float32)
    frame_results = predict_from_waveform(waveform, debug=True)

    # A ~1 second chunk contains a couple of YAMNet frames (~0.48s each).
    # Print each frame's result so we can see what's actually happening
    # inside a single chunk, rather than collapsing it into one line.
    matched_any = False
    for i, (category, confidence) in enumerate(frame_results):
        if category is not None:
            print(f"  frame {i}: {category:10s} ({confidence:.3f})")
            matched_any = True

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