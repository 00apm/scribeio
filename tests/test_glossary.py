"""Tests for the glossary: prompt names and transcript corrections."""

import pytest

from scribe_rpg.errors import ScribeError
from scribe_rpg.transcribe.glossary import Glossary, load_glossary, parse_glossary

TEXT = """\
# invented names only
Vorgrad
Ostromir

Worgrad -> Vorgrad
Ostromir* -> Ostromyr
Czarnej Wierzy -> Czarnej Wieży
"""


def test_names_in_file_order_deduplicated():
    names = parse_glossary(TEXT).names
    assert names == ["Vorgrad", "Ostromir", "Ostromyr", "Czarnej Wieży"]


def test_comments_and_blank_lines_ignored():
    g = parse_glossary("# only a comment\n\n   \n")
    assert g.names == [] and g.corrections == []


def test_whole_word_correction():
    g = parse_glossary("Worgrad -> Vorgrad")
    assert g.correct("Jedziemy do Worgrad.") == "Jedziemy do Vorgrad."
    assert g.correct("W Worgradzie pada.") == "W Worgradzie pada."  # whole word only


def test_stem_correction_keeps_polish_ending():
    g = parse_glossary("Ostromir* -> Ostromyr")
    text = "Spotkali Ostromira przy Ostromirze."
    assert g.correct(text) == "Spotkali Ostromyra przy Ostromyrze."
    assert g.correct("Ostromir") == "Ostromyr"


def test_stem_only_at_word_start():
    g = parse_glossary("mir* -> myr")
    assert g.correct("Ostromir") == "Ostromir"


def test_case_insensitive_match_replacement_as_written():
    g = parse_glossary("worgrad -> Vorgrad")
    assert g.correct("WORGRAD i worgrad") == "Vorgrad i Vorgrad"


def test_multi_word_correction():
    g = parse_glossary("Czarnej Wierzy -> Czarnej Wieży")
    assert g.correct("przy czarnej  wierzy stoi") == "przy Czarnej Wieży stoi"


def test_short_name_does_not_touch_longer_words():
    g = parse_glossary("Ela -> Ella")
    assert g.correct("elastyczny plan, Ela wie") == "elastyczny plan, Ella wie"


def test_english_possessive_whole_word():
    g = parse_glossary("Worgrad -> Vorgrad")
    assert g.correct("Worgrad's gate") == "Vorgrad's gate"


def test_corrections_apply_in_file_order():
    g = parse_glossary("Worgrad -> Vorgrad\nVorgrad -> Vorgród")
    assert g.correct("Worgrad") == "Vorgród"


def test_replacement_with_regex_characters_inserted_literally():
    g = parse_glossary(r"Worgrad -> Vorgrad \1 (city)")
    assert g.correct("Worgrad") == r"Vorgrad \1 (city)"


@pytest.mark.parametrize(
    "line, message",
    [
        ("Worgrad ->", "empty side"),
        ("-> Vorgrad", "empty side"),
        ("a -> b -> c", "more than one"),
        ("Wor*grad -> Vorgrad", r"\*"),
        ("Worgrad -> Vor*", r"\*"),
        ("* -> Vorgrad", "empty side"),
    ],
)
def test_malformed_lines_report_line_number(line, message):
    with pytest.raises(ScribeError, match=rf"names\.txt:3: .*{message}"):
        parse_glossary(f"# header\nVorgrad\n{line}\n", label="names.txt")


def test_load_glossary_with_bom(tmp_path):
    path = tmp_path / "names.txt"
    path.write_bytes("﻿Vorgrad\n".encode())
    assert load_glossary(path).names == ["Vorgrad"]


def test_load_glossary_missing(tmp_path):
    with pytest.raises(ScribeError, match="glossary not found"):
        load_glossary(tmp_path / "missing.txt")


def test_empty_glossary_changes_nothing():
    assert Glossary().correct("bez zmian") == "bez zmian"
