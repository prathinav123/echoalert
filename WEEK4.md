# EchoAlert: Week 4 Evaluation Findings

## What Was Measured

EchoAlert logs a database row every time YAMNet's live audio classification crosses a confidence threshold and a 2-consecutive-frame agreement rule confirms it. In practice, one real-world sound event (a doorbell ring, an alarm going off) can generate several confirmed frame-pairs in quick succession — so evaluating raw rows directly would overcount events and distort precision/recall.

To fix this, detections were grouped into **events**: consecutive same-category rows within a short time gap were collapsed into a single event before comparing against ground truth. Two evaluation metrics were then computed:

- **Event-level accuracy** — did the system correctly identify each real-world sound event, treating a whole sustained sound as one prediction rather than many.
- **Row-level flicker rate** — within a correctly-identified event, what fraction of individual frames still disagreed with the true label. This catches transient misclassification that event-level grouping otherwise hides.

Both numbers matter. Event-level accuracy answers "did the alert fire correctly." Row-level flicker answers "how stable was the model's confidence in that decision" — a system that flickers mid-event but still nets out correct is meaningfully different from one that never wavers, even though both would show up as "correct" at the event level.

## Test Set

A single manually-recorded batch of test sounds was played in sequence and logged by the live system: doorbell, alarm, knock, speech, dog bark, and phone ringtone sounds, plus background noise to test for false positives. Detections were grouped into events (4-second gap threshold, same category), and each event was manually labeled with its true category based on a running log kept during recording.

This produced **8 ground-truth events** across 5 categories present in this run (alarm, dog_bark, doorbell, phone, speech).

## Results

**Confusion Matrix** (rows = actual, columns = predicted):

| | pred_alarm | pred_dog_bark | pred_doorbell | pred_phone | pred_speech |
|---|---|---|---|---|---|
| **actual_alarm** | 1 | 0 | 0 | 0 | 0 |
| **actual_dog_bark** | 0 | 2 | 0 | 0 | 0 |
| **actual_doorbell** | 0 | 0 | 2 | 0 | 0 |
| **actual_phone** | 0 | 0 | 0 | 1 | 0 |
| **actual_speech** | 0 | 0 | 0 | 0 | 2 |

**Precision / Recall / F1 per category:**

| Category | Precision | Recall | F1 | Support (events) |
|---|---|---|---|---|
| alarm | 1.00 | 1.00 | 1.00 | 1 |
| dog_bark | 1.00 | 1.00 | 1.00 | 2 |
| doorbell | 1.00 | 1.00 | 1.00 | 2 |
| phone | 1.00 | 1.00 | 1.00 | 1 |
| speech | 1.00 | 1.00 | 1.00 | 2 |

**Overall event-level accuracy: 8/8 (100.0%)**

**Row-level flicker rate: 2/32 frames (6.2%)**

Per-event breakdown:

| Event Start | Actual Label | Total Frames | Mismatched Frames | Flicker Rate | Mismatched As |
|---|---|---|---|---|---|
| 13:13:01 | doorbell | 1 | 0 | 0% | — |
| 13:13:06 | doorbell | 5 | 0 | 0% | — |
| 13:13:29 | alarm | 9 | 0 | 0% | — |
| 13:14:05 | speech | 4 | 0 | 0% | — |
| 13:14:30 | speech | 2 | 0 | 0% | — |
| 13:14:36 | dog_bark | 3 | 0 | 0% | — |
| 13:14:44 | dog_bark | 1 | 0 | 0% | — |
| 13:15:09 | phone | 7 | 2 | 29% | alarm |

The one notable case: a single continuous phone-ringtone event produced 7 confirmed frames, 2 of which were transiently misclassified as "alarm" mid-event. Because event-level grouping correctly merged these into one ground-truth event (rather than splitting on the misclassified frames), this shows up as a perfect event-level detection — but the row-level flicker metric captures the real, non-trivial instability underneath that clean-looking result.

## Limitations

- **Small sample size.** 8 events across 5 categories is nowhere near enough to make a confident claim about real-world accuracy. The 100% event-level accuracy reflects this specific test run, not a validated production accuracy rate.
- **Alarm event-boundary ambiguity.** During recording, 2-3 separate alarm sounds were played back-to-back with only a 2-3 second gap between them — shorter than the 4-second grouping threshold — so they were merged into a single ground-truth event instead of being counted as 2-3 separate events. This means the alarm category's true event count is undercounted in this evaluation, and the boundary between the individual sounds could not be reliably reconstructed after the fact.
- **Result masking.** As demonstrated by the phone/alarm flicker case, event-level accuracy alone can look artificially strong; it is only meaningful when reported alongside the row-level flicker rate.
- **Single test session.** All data comes from one recording session in one acoustic environment. No variation in background noise, distance from microphone, or device was tested.

## What Would Improve This

- **Larger, more varied test set** — more repetitions per category, recorded across multiple sessions and environments, would give statistically meaningful precision/recall numbers instead of a single-run snapshot.
- **Deliberate pauses within a category, not just between categories** — the alarm-boundary issue was a direct result of playing same-category test sounds close together; leaving a clear gap between individual repetitions (not just between categories) would let event-boundary detection work correctly.
- **Investigate the alarm/phone acoustic overlap directly** — since this is a recurring pattern (also seen earlier in the project), it may be worth a targeted mini-study: collecting several more ringtone and alarm clips specifically to see how often and under what conditions this flicker occurs, rather than treating it as a one-off.
- **A longer or adaptive confirmation window** for tonal, sustained sounds, which may reduce mid-event flicker without slowing down detection of short percussive sounds like knocks.

## Summary

EchoAlert correctly identified all 8 real-world test sound events in this run (100% event-level accuracy), but a closer look at individual frames reveals a 6.2% row-level flicker rate, concentrated entirely in one case where a phone ringtone was briefly misclassified as an alarm mid-event. This distinction — between "did it get the event right" and "was it consistently confident while doing so" — was only visible because detections were evaluated at both the event level and the row level, rather than relying on a single aggregate accuracy number. The test set is small and the alarm category's true event count is likely undercounted due to an event-boundary detection limitation, so the next priority for a more rigorous evaluation is a larger, more carefully paced test recording.