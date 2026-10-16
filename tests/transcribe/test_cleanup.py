"""Tests for transcript cleanup: energy, phrases, fillers, loops."""

import pytest

from scribe_rpg.config import load_config
from scribe_rpg.transcribe.cleanup import apply_cleanup, is_filler, normalize

from helpers import seg


def cfg(**overrides):
    c = load_config().settings["transcribe"]["cleanup"]
    c.update(overrides)
    return c


def test_normalize():
    assert normalize("© PTA & TVP!") == "pta tvp"
    assert normalize("  Dzięki, za   oglądanie...") == "dzięki za oglądanie"
    assert normalize("snake_case") == "snake case"


def test_energy_drop_reports_percentage():
    s = seg(
        "gm_account", 0, "jeden dwa trzy cztery", voiced=[True, False, False, False]
    )
    kept, dropped = apply_cleanup([s], cfg())
    assert kept == []
    assert (dropped[0].reason, dropped[0].detail) == ("energy", "25% voiced")


def test_energy_ratio_at_threshold_kept():
    s = seg("gm_account", 0, "jeden dwa", voiced=[True, False])
    kept, _ = apply_cleanup([s], cfg())
    assert kept == [s]


def test_segment_without_words_is_dropped_as_energy():
    s = seg("gm_account", 0, "jeden")
    s.words = []
    _, dropped = apply_cleanup([s], cfg())
    assert (dropped[0].reason, dropped[0].detail) == ("energy", "0% voiced")


def test_phrase_dropped_on_short_segment():
    _, dropped = apply_cleanup([seg("gm_account", 0, "Dzięki za oglądanie!")], cfg())
    assert dropped[0].reason == "phrase"


def test_english_phrase_dropped():
    _, dropped = apply_cleanup([seg("gm_account", 0, "Thanks for watching!")], cfg())
    assert dropped[0].reason == "phrase"


def test_phrase_matches_after_normalizing():
    _, dropped = apply_cleanup([seg("gm_account", 0, "© PTA & TVP")], cfg())
    assert dropped[0].reason == "phrase"


def test_thank_you_for_watching_dropped():
    _, dropped = apply_cleanup([seg("gm_account", 0, "Thank you for watching!")], cfg())
    assert dropped[0].reason == "phrase"


@pytest.mark.parametrize(
    "text", ["Thank you.", "Thank you, that was great!", "Oh, thank you so much."]
)
def test_voiced_thanks_kept(text):
    kept, _ = apply_cleanup([seg("player_one", 0, text)], cfg())
    assert len(kept) == 1


def test_phrase_ignored_on_long_segment():
    s = seg("gm_account", 0, "no to thank you mówi bohater do strażnika i odchodzi")
    kept, _ = apply_cleanup([s], cfg())
    assert kept == [s]


@pytest.mark.parametrize(
    "text",
    [
        "Yyyy",
        "Mmmm.",
        "Hmm.",
        "Hm",
        "Eee, yyy.",
        "Y...",
        "yyymmm",
        "eem",
        "Um.",
        "Uh,",
        "Uhm",
        "Erm.",
        "uhhh, um",
    ],
)
def test_fillers_dropped(text):
    assert is_filler(normalize(text))
    _, dropped = apply_cleanup([seg("player_one", 0, text)], cfg())
    assert dropped[0].reason == "filler"


@pytest.mark.parametrize(
    "text",
    [
        "Mhm",
        "Mhm.",
        "Mm-hmm.",
        "Mm-hm.",
        "Mmhmm",
        "Mm-hmm, yes.",
        "Uh, mm-hmm.",
        "Aha.",
        "Tak",
        "Nie.",
        "No.",
        "My.",
        "Yyy, idziemy dalej.",
        "U.",
        "Oh!",
        "Uh-huh.",
        "Yeah",
        "Um, let's go.",
    ],
)
def test_answers_and_real_words_kept(text):
    assert not is_filler(normalize(text))
    kept, _ = apply_cleanup([seg("player_one", 0, text)], cfg())
    assert len(kept) == 1


def test_keep_fillers_option():
    kept, dropped = apply_cleanup(
        [seg("player_one", 0, "Yyyy")], cfg(keep_fillers=True)
    )
    assert len(kept) == 1 and dropped == []


def test_loop_extras_dropped():
    segs = [seg("gm_account", i * 3.0, "idziemy do lasu") for i in range(5)]
    kept, dropped = apply_cleanup(segs, cfg())
    assert kept == segs[:3]
    assert [(d.reason, d.detail) for d in dropped] == [
        ("loop", "repeat 4"),
        ("loop", "repeat 5"),
    ]


def test_loop_counts_per_speaker():
    segs = []
    for i in range(5):
        segs.append(seg("gm_account", i * 4.0, "idziemy do lasu"))
        segs.append(seg("player_one", i * 4.0 + 2.0, "dobrze"))
    _, dropped = apply_cleanup(segs, cfg())
    assert len(dropped) == 2 and all(d.segment.account == "gm_account" for d in dropped)


def test_short_repeats_are_not_loops():
    segs = [seg("player_one", i * 2.0, "Tak tak.") for i in range(6)]
    kept, dropped = apply_cleanup(segs, cfg())
    assert len(kept) == 6 and dropped == []


def test_loop_broken_by_different_text():
    texts = ["idziemy do lasu"] * 3 + ["stop tu stoimy"] + ["idziemy do lasu"] * 3
    segs = [seg("gm_account", i * 3.0, t) for i, t in enumerate(texts)]
    _, dropped = apply_cleanup(segs, cfg())
    assert dropped == []


def test_loop_ignores_punctuation_and_case():
    texts = ["Idziemy do lasu."] * 2 + ["idziemy, do lasu"] * 2
    segs = [seg("gm_account", i * 3.0, t) for i, t in enumerate(texts)]
    _, dropped = apply_cleanup(segs, cfg())
    assert len(dropped) == 1
