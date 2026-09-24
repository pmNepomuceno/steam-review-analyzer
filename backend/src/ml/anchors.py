"""Hand-written anchor phrases that define each aspect.

A review sentence is tagged with the aspect whose anchors it is most similar to (see
`aspects.py`). Anchors are natural sentences a reviewer might write, in both positive and
negative phrasing, so they pull in sentences by topic rather than by sentiment. Tune them
against `scripts/evaluate_aspects.py`.

Dict order matters: it is the tie-break order when two aspects score exactly the same.
"""

ANCHORS: dict[str, list[str]] = {
    "performance": [
        "it runs smoothly with a stable frame rate",
        "frame rate drops constantly",
        "optimization is terrible on my hardware",
        "long loading screens and stuttering",
        "it hits 60 fps even on an old laptop",
        "very demanding, my GPU and CPU struggle to run it",
    ],
    "price": [
        "way too expensive for what you get",
        "definitely worth the money",
        "wait for a discount before paying for it",
        "the price is too high",
        "great value for such a low price",
        "charging full price for this is a rip-off",
    ],
    "bugs": [
        "it crashes all the time",
        "full of bugs and glitches",
        "my save file got corrupted",
        "it keeps crashing to desktop",
        "quests break and characters get stuck in walls",
        "the servers keep disconnecting me",
    ],
    "story": [
        "the story is amazing and emotional",
        "the plot and characters are well written",
        "the ending of the story was disappointing",
        "the plot and dialogue are poorly written",
        "the narrative kept me hooked until the end",
        "the voice acting brings the characters to life",
    ],
    "gameplay": [
        "the combat is fun and satisfying",
        "the controls feel tight and responsive",
        "the gameplay loop gets repetitive quickly",
        "the puzzles are clever and challenging",
        "lots of weapon and build variety, great replayability",
        "the core mechanics are shallow and boring",
        "there isn't enough content",
        "the difficulty scaling is punishing",
        "the skill and upgrade system is deep",
    ],
}

ASPECTS: tuple[str, ...] = tuple(ANCHORS)
