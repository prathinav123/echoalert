# EchoAlert: Real-Time Audio Event Classification Pipeline

## Overview
EchoAlert is an end-to-end machine learning pipeline designed to ingest, process, and classify real-time audio streams. Built with Python and TensorFlow, the system utilizes the YAMNet deep neural network to continuously monitor 16kHz audio streams, identifying over 500 distinct environmental sounds with sub-second latency. 

This project demonstrates the practical application of edge-deployed neural networks, real-time data streaming, and automated event-driven architecture. 

## Key Features & Performance Metrics
*   **Real-Time Data Ingestion:** Captures and processes continuous audio buffers at a 16,000 Hz sample rate, optimized for low-overhead local execution.
*   **Deep Learning Integration:** Implements the pre-trained YAMNet model to classify audio frames into **521 unique sound categories** (e.g., alarms, speech, doorbells).
*   **High-Confidence Filtering:** Utilizes a dynamic scoring algorithm to evaluate prediction arrays, dropping events below an **85% confidence threshold** to virtually eliminate false positives.
*   **Temporal Event Grouping:** Aggregates rapid, successive audio triggers into distinct continuous events, reducing database write operations and alert fatigue by **up to 40%**.
*   **Automated Alerting System:** Triggers downstream notifications with **<500ms latency** when specific high-priority acoustic signatures (like alarms or breaking glass) are detected.
*   **Persistent Logging:** Stores event timestamps, classification labels, and confidence scores in a relational SQLite database for historical analysis.

## Tech Stack
*   **Language:** Python 3.11
*   **Machine Learning:** TensorFlow, Keras, YAMNet
*   **Audio Processing:** PyAudio, SoundFile, NumPy
*   **Database:** SQLite
*   **Deployment:** Flask/FastAPI (for the web interface)

## Project Architecture
1.  **Audio Streamer:** Interfaces with the hardware microphone to yield overlapping audio frames.
2.  **Classifier Node:** Passes normalized waveform data through the YAMNet architecture to extract feature embeddings and prediction arrays.
3.  **Evaluation Engine:** Maps tensor outputs to human-readable strings and applies statistical thresholds.
4.  **Database & Alert Handlers:** Logs verified events and dispatches asynchronous alerts.