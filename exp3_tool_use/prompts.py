"""Experiment 3 prompt sets, exactly as written in the data collection protocol."""

SUFFIX = " Keep it to one or two sentences."

# ID -> (current info: expect search, static: expect no search)
PAIRS = {
    "T01": ("What's the weather in Atlanta right now?",
            "What's Atlanta's weather usually like in October?"),
    "T02": ("Who won the most recent Braves game?",
            "How many players are on a baseball team?"),
    "T03": ("What's Apple's stock price today?",
            "What year was Apple founded?"),
    "T04": ("What's the biggest news story today?",
            "What was the biggest news story of 1969?"),
    "T05": ("What's the current price of Bitcoin?",
            "Who created Bitcoin?"),
    "T06": ("When is the next SpaceX launch?",
            "What was SpaceX's first rocket?"),
    "T07": ("Are there traffic delays on I-85 right now?",
            "Where does I-85 start and end?"),
    "T08": ("What's the newest iPhone model?",
            "What year did the first iPhone come out?"),
}

# Latency baseline: no "keep it short" suffix.
BASELINE = {
    "B01": "Say hello.",
    "B02": "Count to five.",
    "B03": "Say the word banana.",
    "B04": "What's two plus two?",
}


def prompt_text(cls, prompt_id):
    """Exact words to say for a capture class (search/static/off/base)."""
    if cls == "base":
        return BASELINE[prompt_id]
    current, static = PAIRS[prompt_id]
    if cls in ("search", "off"):
        return current + SUFFIX
    if cls == "static":
        return static + SUFFIX
    raise ValueError(f"unknown class {cls!r}")
