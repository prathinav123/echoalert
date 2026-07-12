"""
label_mapping.py

Maps YAMNet's ~521 raw class labels down to the handful of target
categories EchoAlert actually cares about (per README: doorbell, alarm,
siren, knock, speech, dog bark).

This is a starting point based on what showed up in your test run plus
the obvious related labels in YAMNet's label set. You WILL want to tweak
this as you test with more real sounds -- treat it as a living document,
not a finished spec. If a chunk keeps getting mis/un-categorized, the fix
is almost always "add the label that's actually showing up to the right
list here," not "make the model more accurate."

Categories are checked in order -- if a frame's labels match more than
one category, the first match wins. Order the most specific/important
categories first (e.g. "alarm" before generic "animal"-adjacent stuff).
"""

TARGET_CATEGORIES = {
    "doorbell": [
        "Doorbell",
        "Bell",
        "Chime",
        "Ding",
    ],
    "alarm": [
        "Alarm",
        "Alarm clock",
        "Buzzer",
        "Smoke detector, smoke alarm",
        "Fire alarm",
        "Siren",
        "Civil defense siren",
        "Ringtone",
        "Telephone",
        "Telephone bell ringing",
    ],
    "siren": [
        "Siren",
        "Civil defense siren",
        "Ambulance (siren)",
        "Police car (siren)",
        "Fire engine, fire truck (siren)",
    ],
    "knock": [
        "Knock",
        "Tap",
        "Mallet percussion",
    ],
    "speech": [
        "Speech",
        "Conversation",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
    ],
    "dog_bark": [
        "Dog",
        "Bark",
        "Howl",
        "Growling",
        "Whimper (dog)",
    ],
}


def categorize(label):
    """
    Given a single raw YAMNet label, return the target category it
    belongs to, or None if it's not one we care about (e.g. "Silence",
    "Music", "Inside, small room" -- background/irrelevant stuff).
    """
    for category, labels in TARGET_CATEGORIES.items():
        if label in labels:
            return category
    return None


def categorize_frame(frame_scores, class_names, top_n=5, min_confidence=0.15):
    """
    Given one frame's full score vector (521 values) and the YAMNet
    class_names list, look at the top_n highest-scoring labels for that
    frame and return the first one that maps to a target category, along
    with its confidence. Returns (None, None) if nothing in the top_n
    matches a target category ABOVE min_confidence.

    min_confidence matters a lot here: a label can sit in the top-5 with
    a near-zero score (e.g. "Speech" at 0.002 during total silence) just
    because it edged out slightly weaker labels, not because the sound
    is actually present. Without a floor, every chunk falsely "detects"
    something. 0.15 is a starting guess -- tune it once you see how real
    target sounds vs. background noise score differently.

    This is the key change from before: instead of only ever looking at
    the single #1 label, we look a few labels deep, since related labels
    (Doorbell/Bell/Chime/Ding, or Dog/Bark/Animal) often trade places
    depending on background noise.
    """
    top_indices = frame_scores.argsort()[::-1][:top_n]

    for idx in top_indices:
        label = class_names[idx]
        category = categorize(label)
        if category is not None and frame_scores[idx] >= min_confidence:
            return category, frame_scores[idx]

    return None, None