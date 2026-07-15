"""
mic_stream_test.py

Confirms that live microphone audio can be captured in rolling windows
at 16kHz, the sample rate YAMNet requires. No model is involved here --
this just verifies audio is flowing correctly before adding YAMNet.

Run it, make some noise (talk, clap, tap the mic), and watch the
printed stats update in the terminal. Press Ctrl+C to stop.
"""

import numpy as np
import sounddevice as sd

# --- Config ---
SAMPLE_RATE = 16000           # YAMNet requires 16kHz
BLOCK_SIZE = SAMPLE_RATE*1    # 1 second of audio per chunk (SAMPLE_RATE * seconds)
CHANNELS = 1                  # mono


def audio_callback(indata, frames, time_info, status):
    """
    Called automatically by sounddevice every time a new chunk of audio
    is ready. Runs on a background thread managed by sounddevice.
    """
    if status:
        # Things like "input overflow" show up here -- worth printing
        # while debugging, safe to ignore occasionally.
        print(f"[status] {status}")

    # indata is a numpy array of shape (frames, channels)
    audio_chunk = indata[:, 0]  # take the single mono channel

    volume = np.abs(audio_chunk).max()
    rms = np.sqrt(np.mean(audio_chunk ** 2))

    print(f"chunk shape={audio_chunk.shape} dtype={audio_chunk.dtype} "
          f"peak_amplitude={volume:.4f} rms={rms:.4f}")


def main():
    print(f"Starting mic stream at {SAMPLE_RATE} Hz, "
          f"block size {BLOCK_SIZE} samples (~{BLOCK_SIZE / SAMPLE_RATE:.1f}s per chunk)")
    print("Make some noise! Press Ctrl+C to stop.\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    ):
        try:
            while True:
                sd.sleep(1000)  # keep main thread alive; callback does the work
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()