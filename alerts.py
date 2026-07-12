"""
alerts.py

Fires desktop notifications for confirmed EchoAlert detections, with a
per-category cooldown so a single ongoing sound (e.g. an alarm ringing
for 10 seconds) doesn't spam the user with a notification every time a
frame-pair confirms -- just once, then silence until either the sound
stops and restarts, or the cooldown window passes.
"""

import time
from plyer import notification

# Seconds to wait before allowing another notification for the SAME
# category. Tune this once you've lived with it -- 10s is a reasonable
# starting guess: long enough to not spam, short enough that a second
# real event (e.g. the doorbell rings again a bit later) still alerts.
COOLDOWN_SECONDS = 10

# Friendly, human-readable text for each category. Keeps the actual
# notification wording separate from the internal category names, so
# you can adjust tone/wording here without touching detection logic.
ALERT_MESSAGES = {
    "doorbell": "Doorbell",
    "alarm": "Alarm sound detected",
    "siren": "Siren detected",
    "knock": "Knocking detected",
    "speech": "Speech detected",
    "dog_bark": "Dog barking",
    "phone": "Phone ringing",
}

# Tracks the last time each category successfully fired a notification.
# Lives at module level (not inside a function) so it persists across
# repeated calls to maybe_alert() for the life of the running program --
# same reasoning as last_category in mic_yamnet_live.py.
_last_alert_time = {}


def maybe_alert(category, confidence):
    """
    Fires a desktop notification for `category` if enough time has
    passed since the last notification for that same category.
    Returns True if a notification was actually sent, False if it was
    suppressed by the cooldown.

    Call this right alongside insert_detection() -- both should fire
    off the same "confirmed detection" event in mic_yamnet_live.py.
    """
    now = time.time()
    last_time = _last_alert_time.get(category, 0)

    if now - last_time < COOLDOWN_SECONDS:
        return False  # still in cooldown, stay quiet

    message = ALERT_MESSAGES.get(category, category)
    try:
        notification.notify(
            title="EchoAlert",
            message=f"{message} ({confidence:.2f} confidence)",
            timeout=5,  # seconds the notification stays visible
        )
    except Exception as e:
        # Desktop notifications can fail for OS-specific reasons (missing
        # backend, permissions, etc.) -- don't let a notification failure
        # crash the whole detection loop over something non-critical.
        print(f"[alerts] Failed to send notification: {e}")
        return False

    _last_alert_time[category] = now
    return True


if __name__ == "__main__":
    # Quick manual test: fire one alert, then immediately try a second
    # one for the same category -- it should be suppressed by cooldown.
    print("Firing first alert (should show a notification)...")
    sent = maybe_alert("doorbell", 0.85)
    print(f"Sent: {sent}")

    print("Firing second alert immediately (should be suppressed)...")
    sent = maybe_alert("doorbell", 0.90)
    print(f"Sent: {sent}")