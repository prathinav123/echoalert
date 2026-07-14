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

IMPORTANT: each raw label should appear in exactly ONE category's list.
If the same label is listed under two categories, whichever category
comes first in this dict wins EVERY time that label is YAMNet's top
pick -- the second category becomes structurally unreachable for that
label. This bit us once already (Siren / Civil defense siren were
listed under both "alarm" and "siren", which meant "siren" could never
fire from those two labels). Keep it that way: no duplicates across
lists.
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
        # NOTE: "Telephone" / "Ringtone" / "Telephone bell ringing" used
        # to live here. They were pulled out because a ringing phone
        # isn't the "alarm" concept from the README (smoke alarm / alarm
        # clock) -- but leaving them unmapped meant "alarm" still caught
        # them by default whenever they showed up in the top-5 alongside
        # a low-scoring "Alarm" label. They now have their own "phone"
        # category below instead.
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
    Given a single raw YAMNet label, return the target category it
    belongs to, or None if it's not one we care about (e.g. "Silence",
    "Music", "Inside, small room" -- background/irrelevant stuff).
    """
    for category, labels in TARGET_CATEGORIES.items():
        if label in labels:
            return category
    return None


def categorize_frame(frame_scores, class_names, top_n=5, min_confidence=0.15, debug=False):
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

    If debug=True, also prints the full top_n raw label + score list for
    this frame, with the one that got chosen (if any) marked with '->'.
    This is the tool for telling apart genuine model ambiguity (two
    unrelated raw labels for different real categories both scoring
    high) from a mapping bug (a raw label sitting in two category lists,
    or a raw label you forgot to route anywhere). Turn this on whenever
    a category is flickering between two frames for what looks like the
    same real-world sound.
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

# Day 2: live mic streaming + YAMNet inference + label mapping

# - mic_stream_test.py: callback-based sd.InputStream at 16kHz, 1s blocks
# - mic_yamnet_live.py: live YAMNet inference on in-memory audio frames
# - label_mapping.py: maps YAMNet's 521 labels to target categories
#   (doorbell, alarm, siren, knock, speech, dog_bark, phone)
# - Fixed alarm/siren label collision (Siren, Civil defense siren were
#   duplicated across both lists, making siren unreachable)
# - Added phone category so Telephone/Ringtone stop defaulting into alarm
# - Added debug mode to categorize_frame for inspecting raw top-5 labels
#   per frame when categories disagree"