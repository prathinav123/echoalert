"""
classifier.py

Loads YAMNet and runs it on a pre-recorded audio file, printing the
top predicted label for each frame of the file. This is the simplest
possible test of the model before wiring it into live microphone input.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import librosa


# Load the pretrained YAMNet model from TensorFlow Hub.
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load YAMNet's list of ~521 class names, in the same order as its output scores.
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = list(pd.read_csv(class_map_path)['display_name'])


def load_audio(file_path):
    """Loads an audio file as mono, 16kHz float32 -- the format YAMNet requires."""
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)
    return waveform.astype(np.float32)


def predict(file_path):
    """
    Runs YAMNet on a file and prints the top-scoring label for each
    frame. Frame-level output is used instead of averaging scores
    across the whole clip, since averaging tends to dilute short,
    sudden sounds down to near-zero confidence.
    """
    waveform = load_audio(file_path)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()  # shape: (num_frames, 521)

    print(f"\n--- {file_path} ---")
    print(f"Total frames: {scores_np.shape[0]} (~{scores_np.shape[0] * 0.48:.1f} sec of audio)")

    for frame_idx, frame_scores in enumerate(scores_np):
        top_idx = np.argmax(frame_scores)
        print(f"Frame {frame_idx}: {class_names[top_idx]} ({frame_scores[top_idx]:.3f})")


if __name__ == "__main__":
    predict("iphone_alarm.wav")
    predict("doorbell-chime.wav")