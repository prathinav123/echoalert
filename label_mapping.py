"""
label_mapping.py

Maps YAMNet's ~521 raw class labels down to the handful of categories
EchoAlert actually cares about: doorbell, alarm, phone, siren, knock,
speech, dog_bark.

Categories are checked in order -- if a frame's top labels match more
than one category, the first match in TARGET_CATEGORIES wins. The most
specific/important categories should be listed first.

IMPORTANT: each raw label should appear in exactly ONE category's list.
If the same label is listed under two categories, whichever category
comes first in this dict always wins, making the second category
unreachable for that label. Keep every label in exactly one place.
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
        # Telephone/Ringtone labels used to live here, but a ringing
        # phone isn't the same concept as a smoke alarm or alarm clock.
        # They now have their own "phone" category below so they don't
        # get miscategorized as "alarm".
    ],
    "phone": [
        "Telephone",
        "Ringtone",
        "Telephone bell ringing",
        "Telephone dialing, DTMF",
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
    Given one raw YAMNet label, returns the target category it belongs
    to, or None if it's not a category EchoAlert tracks (e.g. "Silence",
    "Music", "Inside, small room").
    """
    for category, labels in TARGET_CATEGORIES.items():
        if label in labels:
            return category
    return None


def categorize_frame(frame_scores, class_names, top_n=5, min_confidence=0.15, debug=False):
    """
    Looks at the top_n highest-scoring labels for one frame and returns
    the first one that maps to a target category, along with its
    confidence. Returns (None, None) if nothing in the top_n matches a
    target category above min_confidence.

    Checking several labels deep (not just the single top label) matters
    because related labels -- e.g. Doorbell/Bell/Chime/Ding, or
    Dog/Bark/Growling -- often trade places depending on background
    noise, and any one of them can indicate the same real-world sound.

    min_confidence filters out labels that only appear in the top-5
    with a near-zero score (e.g. "Speech" at 0.002 during silence)
    just because they narrowly beat out even weaker labels.

    If debug=True, also prints every label in the top_n with its score
    and mapped category, marking whichever one was chosen. This is
    useful for telling apart genuine model ambiguity (two different
    real sounds both scoring high) from a mapping bug (a label missing
    from every category list, or accidentally listed in two).
    """
    top_indices = frame_scores.argsort()[::-1][:top_n]

    if debug:
        print("  raw top-{}:".format(top_n))

    chosen_category = None
    chosen_score = None

    for idx in top_indices:
        label = class_names[idx]
        score = frame_scores[idx]
        category = categorize(label)

        if debug:
            marker = ""
            if chosen_category is None and category is not None and score >= min_confidence:
                marker = "  <- chosen"
            print("    {:<35} {:.3f}  [{}]{}".format(
                label, score, category if category else "-", marker
            ))

        if chosen_category is None and category is not None and score >= min_confidence:
            chosen_category = category
            chosen_score = score

    if not debug:
        return chosen_category, chosen_score

    return chosen_category, chosen_score