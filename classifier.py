import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import librosa


# Load YAMNet from TF Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load YAMNet's class names (the ~521 labels it can predict)
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = list(pd.read_csv(class_map_path)['display_name'])



def load_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)
    return waveform.astype(np.float32)

# def load_audio(file_path):
#     waveform, sr = sf.read(file_path, dtype='float32')
#     if sr != 16000:
#         raise ValueError(f"Expected 16kHz sample rate, got {sr}. Resample first.")
#     if waveform.ndim > 1:
#         waveform = waveform.mean(axis=1)  # convert stereo to mono
#     return waveform

def predict(file_path):
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

# Alarm-type sounds cluster across Telephone/Ringtone/Beep/Ding — needs multi-label grouping
# Doorbell decay/reverb reads as "Mallet percussion" — may need its own handling or just be ignored
# Frame-level analysis is essential — averaged predictions dilute short bursts into meaninglessness

# --Week 1:
# YAMNet loads and runs
# Correct audio loading/resampling to 16kHz
# Averaged top-5 predictions work
# Frame-by-frame predictions work and reveal real, useful behavior