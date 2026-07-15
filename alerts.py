"""
alerts.py

Sends desktop notifications for confirmed sound detections. A per-category
cooldown stops a single ongoing sound (e.g. an alarm ringing for 10 seconds)
from spamming the user with a notification every time it's re-confirmed --
just one alert, then silence until the cooldown passes or the sound
category changes.
"""

import time
from plyer import notification

# How many seconds to wait before allowing another notification for the
# SAME category. 10s is a reasonable default: long enough to avoid spam,
# short enough that a genuinely new event still gets through.
COOLDOWN_SECONDS = 10

# Human-readable text shown in the notification for each category.
# Kept separate from detection logic so wording can be changed here
# without touching anything else.
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
# Lives at module level so it persists across repeated calls for as long
# as the program runs.
_last_alert_time = {}


def maybe_alert(category, confidence):
    """
    Fires a desktop notification for `category` if the cooldown for that
    category has passed. Returns True if a notification was sent, False
    if it was suppressed by the cooldown or failed to send.
    """
    now = time.time()
    last_time = _last_alert_time.get(category, 0)

    if now - last_time < COOLDOWN_SECONDS:
        return False  # still in cooldown

    message = ALERT_MESSAGES.get(category, category)
    try:
        notification.notify(
            title="EchoAlert",
            message=f"{message} ({confidence:.2f} confidence)",
            timeout=5,  # seconds the notification stays visible
        )
    except Exception as e:
        # Desktop notifications can fail for OS-specific reasons (missing
        # backend, permissions, etc.) -- don't crash the detection loop
        # over a non-critical failure.
        print(f"[alerts] Failed to send notification: {e}")
        return False

    _last_alert_time[category] = now
    return True


if __name__ == "__main__":
    # Quick manual test: fire one alert, then immediately try a second one
    # for the same category -- the second should be suppressed by cooldown.
    print("Firing first alert (should show a notification)...")
    sent = maybe_alert("doorbell", 0.85)
    print(f"Sent: {sent}")

    print("Firing second alert immediately (should be suppressed)...")
    sent = maybe_alert("doorbell", 0.90)
    print(f"Sent: {sent}")