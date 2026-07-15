# EchoAlert 🔔
### A real-time sound awareness system for people with hearing loss

---

## Why This Exists

For someone with full hearing, a doorbell, a smoke alarm, or someone calling their name from another room registers instantly and automatically. For someone living with single-sided or partial hearing loss — including the thousands of people affected by Sudden Sensorineural Hearing Loss (SSHL) each year — those same sounds can be missed entirely. That's not just inconvenient. It can be dangerous.

Most existing solutions are either expensive (specialized hearing aids with alert features) or crude (generic vibration alarms that tell you *something* happened, but not *what*). EchoAlert is an attempt at something lighter-weight: a system that listens to a room, recognizes what's making noise, and tells the user clearly and immediately — built entirely on free, open-source tools.

## What It Does

EchoAlert listens to a live microphone feed and classifies ambient sound in real time using **YAMNet**, a pretrained audio classification model from Google. Instead of training a model from scratch, the project's engineering challenge was everything *around* the model:

- Getting live audio reliably into the model, frame by frame, in real time
- Filtering YAMNet's 521 raw sound labels down to the handful that actually matter for accessibility (doorbell, alarm, phone, siren, knock, speech, dog bark)
- Designing detection logic that's responsive without being noisy
- Logging every detection to a real database so the system's performance could actually be *measured*, not just assumed
- Closing the loop with a proper evaluation: precision, recall, and a confusion matrix against manually labeled ground truth

## How It Works

```
Microphone  →  YAMNet (frame-level inference)  →  Label mapping  →  Confirmation logic  →  SQLite  →  Alerts + Dashboard
```

1. **Audio capture** — `sounddevice` streams the microphone in 1-second windows at the 16kHz sample rate YAMNet requires.
2. **Frame-level inference** — Each window is split into ~0.48s frames and classified individually rather than averaged. Averaging was tested early on and found to dilute short, sudden sounds (a single knock, a brief chime) down to near-zero confidence — frame-level analysis preserves them.
3. **Label mapping** — YAMNet's 521 raw labels are mapped to 7 target categories through an explicit, hand-maintained mapping table, with a confidence floor (0.15) to filter out near-zero noise picks.
4. **Confirmation logic** — A detection only counts once two consecutive frames agree on the same category, filtering out single-frame flicker without diluting real sustained sounds.
5. **Persistence** — Every confirmed detection is logged to SQLite with a timestamp, predicted label, and confidence score.
6. **Alerts** — Desktop notifications fire per category, with a cooldown window so one continuous alarm doesn't spam the user with repeat alerts.
7. **Dashboard** — A Streamlit interface shows a live, auto-refreshing feed of detections and a category breakdown, reading directly from the database.

## Evaluation

The core of this project isn't just detection — it's measuring how well the detection actually works. Raw detection rows aren't a fair unit of evaluation on their own: one sustained sound (a doorbell ringing for a few seconds) generates several confirmed rows in a row, which would overcount events and distort precision/recall if compared row-for-row. To fix this, consecutive same-category rows were grouped into discrete **events** before scoring, and each event was manually labeled with its true category from a live test recording.

Two metrics were tracked, since either one alone can be misleading:

- **Event-level accuracy** — did the system correctly identify each real-world sound event
- **Row-level flicker rate** — within a *correctly identified* event, what fraction of individual frames still briefly disagreed with the true label (catches transient misclassification that event-level grouping hides)

### Results from a test recording (8 ground-truth events, 5 categories)

| Category | Precision | Recall | F1 | Events |
|---|---|---|---|---|
| Alarm | 1.00 | 1.00 | 1.00 | 1 |
| Dog bark | 1.00 | 1.00 | 1.00 | 2 |
| Doorbell | 1.00 | 1.00 | 1.00 | 2 |
| Phone | 1.00 | 1.00 | 1.00 | 1 |
| Speech | 1.00 | 1.00 | 1.00 | 2 |

**Event-level accuracy: 8/8 (100%)**
**Row-level flicker rate: 2/32 frames (6.2%)**

The one interesting case: a single continuous phone-ringtone event produced 7 confirmed frames, 2 of which briefly misfired as "alarm" mid-event. Because event-level grouping merged this correctly into one ground-truth event, it still shows up as a perfect detection — but the flicker metric catches the instability underneath the clean-looking headline number. That distinction (*did it fire correctly* vs. *was it stable while doing so*) is exactly why both metrics are reported rather than just one.

## Honest Limitations

A data science project is only as credible as its stated limitations:

- **Small sample.** 8 events across 5 categories is a single test session, not a validated real-world accuracy rate — it demonstrates that the evaluation pipeline works, not that the system is production-ready.
- **Event-boundary ambiguity.** Alarm sounds played back-to-back with gaps shorter than the 4-second grouping threshold got merged into a single ground-truth event, undercounting the true alarm event count in this run.
- **One acoustic environment.** All data comes from one recording session with no variation in background noise, distance from the mic, or hardware.
- **Alarm/phone acoustic overlap.** Ringtone and alarm sounds share enough spectral similarity that transient misclassification between them is the system's most consistent failure mode so far — a targeted next step rather than a solved problem.

## What's Next

- A larger, multi-session test set with deliberate pauses between same-category repetitions, to fix the event-boundary undercounting
- A focused mini-study on the alarm/phone confusion specifically
- A longer or adaptive confirmation window for tonal, sustained sounds to reduce mid-event flicker without slowing down detection of short, percussive sounds like knocks

## Tech Stack

| Layer | Tools |
|---|---|
| Audio capture | `sounddevice`, `numpy` |
| Classification | `TensorFlow`, `TensorFlow Hub` (YAMNet), `librosa` |
| Storage | `SQLite` |
| Evaluation | `pandas`, `scikit-learn` |
| Alerts | `plyer` (desktop notifications) |
| Dashboard | `Streamlit` |

## Repo Structure

```
echoalert/
├── classifier.py          # YAMNet loading + frame-level inference (offline test)
├── label_mapping.py        # Maps YAMNet's 521 labels to 7 target categories
├── mic_stream_test.py       # Live mic capture sanity check
├── mic_yamnet_live.py        # Live detection: mic → YAMNet → confirmation → logging
├── database.py                 # SQLite schema + insert/query helpers
├── alerts.py                     # Desktop notifications with per-category cooldown
├── app.py                          # Streamlit live dashboard
├── group_events.py                  # Collapses raw detections into discrete events
├── evaluate_events.py                 # Confusion matrix, precision/recall, flicker rate
├── requirements.txt
└── events_for_labeling.csv              # Manually labeled ground truth events
```

## Skills Demonstrated

- Applying a pretrained deep learning model to a real-time, streaming problem
- Live audio signal processing and stateful stream handling
- Relational database design and querying
- Rigorous model evaluation: precision, recall, confusion matrices, and awareness of metric blind spots (event-level vs. row-level)
- Building an interactive data dashboard
- End-to-end ownership: problem framing, building, evaluating, and writing up findings — including what didn't work

---

*Built as a personal project applying pretrained ML to a real accessibility problem, with the same rigor a production data science team would bring to measuring whether the thing it built actually works.*