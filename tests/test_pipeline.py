"""Model-free unit tests for the deterministic parts of tasks 2 and 4."""

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "task2_text_pipeline"))
sys.path.insert(0, str(ROOT / "task4_text_to_json"))

from extractor import age_on, clean_answer, grounded, parse_date, validate  # noqa: E402
from pipeline import chunk_sentences, clean_text, remove_bracketed_blocks, split_sentences  # noqa: E402


# ---------------------------------------------------------------- task 2 ---

def test_nested_illustration_block_removed():
    text = "Before.\n[Illustration:\n“caption”\n[_Copyright 1894_]]\nAfter."
    out = remove_bracketed_blocks(text)
    assert "Illustration" not in out and "Copyright" not in out
    assert "Before." in out and "After." in out


def test_clean_text_normalises_characters_and_whitespace():
    raw = ("*** START OF THE PROJECT GUTENBERG EBOOK X ***\nCHAPTER I.\n\n"
           "“_You_ want   to\ntell me,” said café—owner @#$ Mr. Bennet.\n\n\n\nNext   para."
           "\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicence text")
    out = clean_text(raw)
    assert out == '"You want to tell me," said cafe - owner Mr. Bennet.\n\nNext para.'


def test_split_sentences_respects_abbreviations_and_quotes():
    text = '"Is he married?" cried Mrs. Bennet. Mr. Darcy left. "Why?" He said nothing!'
    assert split_sentences(text) == [
        '"Is he married?" cried Mrs. Bennet.',
        "Mr. Darcy left.",
        '"Why?"',
        "He said nothing!",
    ]


def _sentences(n, words):
    return [" ".join([f"s{i}w{j}" for j in range(words - 1)] + [f"end{i}."]) for i in range(n)]


def test_chunks_never_exceed_limit_and_keep_whole_sentences():
    sents = _sentences(50, 17)
    chunks = chunk_sentences(sents, chunk_words=200, overlap_words=40)
    for c in chunks:
        assert c.word_count <= 200
        assert c.text == " ".join(sents[c.first_sentence:c.last_sentence + 1])
    assert chunks[-1].last_sentence == len(sents) - 1  # all text covered


def test_chunks_overlap_by_whole_sentences_within_budget():
    chunks = chunk_sentences(_sentences(50, 17), chunk_words=200, overlap_words=40)
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.first_sentence <= prev.last_sentence      # overlaps
        assert cur.first_sentence > prev.first_sentence      # makes progress
        assert 0 < cur.overlap_words <= 40


def test_oversize_sentence_becomes_its_own_chunk():
    sents = _sentences(2, 10) + [" ".join(["long"] * 250) + "."] + _sentences(2, 10)
    chunks = chunk_sentences(sents, chunk_words=200, overlap_words=20)
    oversize = [c for c in chunks if c.oversize]
    assert len(oversize) == 1 and oversize[0].first_sentence == oversize[0].last_sentence == 2


def test_every_chunk_adds_a_new_sentence():
    # A long sentence after a short tail used to produce a chunk made only of
    # sentences already in the previous chunk (seen once in the full book).
    sents = _sentences(1, 150) + _sentences(1, 30) + _sentences(1, 175) + _sentences(1, 10)
    chunks = chunk_sentences(sents, chunk_words=200, overlap_words=40)
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.last_sentence > prev.last_sentence
    assert all(c.word_count <= 200 for c in chunks)
    assert chunks[-1].last_sentence == len(sents) - 1


def test_invalid_chunk_parameters():
    with pytest.raises(ValueError):
        chunk_sentences(["a."], chunk_words=100, overlap_words=100)


# ---------------------------------------------------------------- task 4 ---

@pytest.mark.parametrize("s,expected", [
    ("14 March 1879", date(1879, 3, 14)),
    ("March 14, 1879", date(1879, 3, 14)),
    ("1987-06-24", date(1987, 6, 24)),
    ("24th Jun 1987", date(1987, 6, 24)),
    ("in 1950", date(1950, 7, 1)),
    ("none", None),
    ("31 February 1900", None),
])
def test_parse_date(s, expected):
    assert parse_date(s) == expected


def test_age_on_handles_birthday_not_yet_reached():
    assert age_on(date(1987, 6, 24), date(2026, 6, 23)) == 38
    assert age_on(date(1987, 6, 24), date(2026, 6, 24)) == 39


def test_grounding_rejects_hallucinations():
    text = "Marie Curie was a Polish physicist."
    assert grounded("Marie Curie", text) == "Marie Curie"
    assert grounded("Polish physicist", text) == "Polish physicist"
    assert grounded("astronaut", text) is None
    assert clean_answer(" None. ") is None


def test_validate_enforces_schema():
    ok = {"name": "A", "age": 3, "occupation": None, "birth_date": None, "is_deceased": False}
    assert validate(ok) == ok
    with pytest.raises(TypeError):
        validate({**ok, "age": "3"})
    with pytest.raises(TypeError):
        validate({**ok, "age": True})
    with pytest.raises(ValueError):
        validate({**ok, "extra": 1})
    with pytest.raises(ValueError):
        validate({**ok, "age": 400})


def test_stated_age_requires_age_context():
    from extractor import stated_age_in_text
    assert stated_age_in_text(34, "Priya Raman, 34, is a software engineer.") == 34
    assert stated_age_in_text(52, "The 52-year-old chef opened a restaurant.") == 52
    assert stated_age_in_text(24, "Lionel Messi (born 24 June 1987) is a footballer.") is None
