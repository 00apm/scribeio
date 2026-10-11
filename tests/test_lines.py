"""Tests for building transcript lines and finding possible bleed."""

from scribe_rpg.transcribe.lines import build_lines, find_bleed
from scribe_rpg.transcribe.session import Segment, Word

from helpers import seg

NAMES = {"gm_account": "Game Master", "player_one": "Player One"}


def words_seg(account, words):
    """Segment from explicit words: [(text, start, end), ...]."""
    ws = [Word(t, s, e, 0.9) for t, s, e in words]
    return Segment(
        account, ws[0].start, ws[-1].end, " ".join(t for t, _, _ in words), ws
    )


def show(lines):
    return [(round(line.start, 2), line.speaker, line.text) for line in lines]


def test_pause_starts_new_line():
    a = seg("gm_account", 0.0, "raz dwa")  # 0.0-0.8
    b = seg("gm_account", 2.3, "trzy cztery")  # gap exactly 1.5
    assert show(build_lines([a, b], NAMES, 1.5)) == [
        (0.0, "Game Master", "raz dwa"),
        (2.3, "Game Master", "trzy cztery"),
    ]


def test_short_gap_joins_segments():
    a = seg("gm_account", 0.0, "raz dwa")
    b = seg("gm_account", 2.2, "trzy cztery")  # gap 1.4
    lines = build_lines([a, b], NAMES, 1.5)
    assert show(lines) == [(0.0, "Game Master", "raz dwa trzy cztery")]


def test_interjection_split_spec_example():
    t0 = 3730.0  # 01:02:10
    texts = ["Wchodzicie", "do", "piwnicy,", "jest", "ciemno", "i", "czuć", "wilgoć..."]
    gm = words_seg(
        "gm_account",
        [(t, t0 + i * 0.5, t0 + (i + 1) * 0.5) for i, t in enumerate(texts)],
    )
    p1 = words_seg("player_one", [("Tak.", t0 + 3.0, t0 + 3.4)])
    assert show(build_lines([gm, p1], NAMES, 1.5)) == [
        (3730.0, "Game Master", "Wchodzicie do piwnicy, jest ciemno i"),
        (3733.0, "Player One", "Tak."),
        (3733.0, "Game Master", "czuć wilgoć..."),
    ]


def test_two_interjections_split_twice():
    gm = seg("gm_account", 0.0, "a b c d e f g h", word_len=1.0)  # 0-8
    p1 = [seg("player_one", 2.0, "Tak."), seg("player_one", 5.0, "Nie.")]
    lines = build_lines([gm, *p1], NAMES, 1.5)
    assert [(line.speaker, line.text) for line in lines] == [
        ("Game Master", "a b"),
        ("Player One", "Tak."),
        ("Game Master", "c d e"),
        ("Player One", "Nie."),
        ("Game Master", "f g h"),
    ]


def test_overlapping_speech():
    a = seg("gm_account", 10.0, "a b c d e f g h i j", word_len=0.6)  # 10.0-16.0
    b = seg("player_one", 12.0, "k l m n o p", word_len=1.0)  # 12.0-18.0
    lines = build_lines([a, b], NAMES, 1.5)
    assert [(line.speaker, line.text) for line in lines] == [
        ("Game Master", "a b c d"),
        ("Player One", "k l m n o p"),
        ("Game Master", "e f g h i j"),
    ]


def test_unknown_account_uses_account_name():
    lines = build_lines([seg("player_two", 0.0, "cześć")], NAMES, 1.5)
    assert lines[0].speaker == "player_two"


def test_estimated_words_are_used_like_others():
    s = seg("gm_account", 0.0, "rok 1923", estimated=True)
    assert show(build_lines([s], NAMES, 1.5)) == [(0.0, "Game Master", "rok 1923")]


def test_ties_sorted_by_end_then_speaker():
    a = seg("player_one", 5.0, "tak tak tak")
    b = seg("gm_account", 5.0, "no")
    c = seg("player_two", 5.0, "no")
    speakers = [line.speaker for line in build_lines([a, b, c], NAMES, 1.5)]
    assert speakers == ["Game Master", "player_two", "Player One"]


def test_no_segments_no_lines():
    assert build_lines([], NAMES, 1.5) == []


SAME = "idziemy dalej korytarzem w stronę światła"


def test_bleed_found():
    a = seg("gm_account", 100.0, SAME, level_db=-18.0)
    b = seg("player_one", 100.1, SAME + ".", level_db=-41.0)
    found = find_bleed([b, a], 0.85, 5, 0.5)
    assert len(found) == 1
    assert (found[0].a.account, found[0].b.account) == ("gm_account", "player_one")
    assert found[0].start == 100.0
    assert found[0].similarity == 1.0


def test_bleed_needs_similar_text():
    a = seg("gm_account", 100.0, SAME)
    b = seg("player_one", 100.1, "zupełnie inne słowa tutaj padły teraz")
    assert find_bleed([a, b], 0.85, 5, 0.5) == []


def test_bleed_needs_min_words():
    a = seg("gm_account", 100.0, "idziemy dalej")
    b = seg("player_one", 100.0, "idziemy dalej")
    assert find_bleed([a, b], 0.85, 5, 0.5) == []


def test_bleed_needs_overlap():
    a = seg("gm_account", 100.0, SAME)  # 100.0-102.4
    b = seg("player_one", 102.0, SAME)  # overlap 0.4 < 0.5 * 2.4
    assert find_bleed([a, b], 0.85, 5, 0.5) == []


def test_bleed_only_across_tracks():
    a = seg("gm_account", 100.0, SAME)
    b = seg("gm_account", 100.5, SAME)
    assert find_bleed([a, b], 0.85, 5, 0.5) == []
