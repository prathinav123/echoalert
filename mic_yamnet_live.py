"""
mic_yamnet_live.py

Week 2, Step 2: Feed each live 1-second microphone window directly into
YAMNet and print the top prediction per chunk. Builds on:
  - mic_stream_test.py (confirmed mic capture works)
  - Week 1 classifier.py (confirmed YAMNet loading/inference works)

Press Ctrl+C to stop.
"""

import numpy as np
import pandas as pd
import sounddevice as sd
import tensorflow_hub as hub

# --- Config ---
SAMPLE_RATE = 16000   # YAMNet requires 16kHz
BLOCK_SIZE = 16000    # 1 second of audio per chunk
CHANNELS = 1

# --- Load YAMNet once at startup (same as Week 1) ---
print("Loading YAMNet model...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = list(pd.read_csv(class_map_path)['display_name'])
print("Model loaded.\n")


def predict_from_waveform(waveform):
    """
    Same idea as your Week 1 predict(), but takes a numpy waveform
    directly instead of loading it from a file. Returns the single
    top (label, confidence) across all frames in this chunk.
    """
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()  # shape: (num_frames, 521)

    # Average across frames for a single chunk-level prediction.
    # Week 1 notes flagged that averaging dilutes short bursts --
    # true for a full clip, but here each chunk is already just ~1 second,
    # so averaging its few frames is a reasonable starting point. We can
    # switch to frame-level / max-based logic later if it misses things.)
    mean_scores = scores_np.mean(axis=0)
    top_idx = np.argmax(mean_scores)
    return class_names[top_idx], mean_scores[top_idx]


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[status] {status}")

    waveform = indata[:, 0].astype(np.float32)
    label, confidence = predict_from_waveform(waveform)
    print(f"{label:20s} ({confidence:.3f})")


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