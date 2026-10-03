"""Task 4 - Structured data extraction (text -> validated JSON) with Flan-T5.

Why not just ask Flan-T5 to "return JSON"?
    The T5 SentencePiece vocabulary has no "{" or "}" tokens, so Flan-T5 cannot
    reliably emit JSON objects verbatim. Its output also drifts in format between
    inputs. Asking a small seq2seq model for free-form JSON gives brittle results.

Design used here (constrained extraction):
    1. One short, focused prompt per field. Flan-T5 was instruction-tuned on QA, so
       "What is the person's occupation?" works much better than one big prompt.
    2. Output is cleaned, typed, and *grounded*: a string field must actually
       appear in the source text, or it becomes null (stops hallucinations).
    3. Age is often not stated in biographies. If no explicit age is found, the
       model extracts the birth date (and death date, if any) and age is computed
       deterministically in Python.
    4. The final object is validated against a fixed schema and serialised with
       json.dumps, so the output is always valid, machine-readable JSON.

Usage:
    python task4_text_to_json/extractor.py                      # run on the sample bios
    python task4_text_to_json/extractor.py --text "Ada Lovelace (10 December 1815 - 27 November 1852) was an English mathematician."
    python task4_text_to_json/extractor.py --file bio.txt --model google/flan-t5-base
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_bios import SAMPLE_BIOS  # noqa: E402

DEFAULT_MODEL = "google/flan-t5-large"

SCHEMA: dict[str, type | tuple[type, ...]] = {
    "name": (str, type(None)),
    "age": (int, type(None)),
    "occupation": (str, type(None)),
    "birth_date": (str, type(None)),
    "is_deceased": bool,
}

PROMPTS = {
    "name": "Read the text and answer with the full name of the person it is about.\n\nText: {text}\n\nFull name:",
    "occupation": "Read the text and answer with the person's main occupation or profession, in a few words.\n\nText: {text}\n\nOccupation:",
    "stated_age": "Read the text. Does it state how old the person is, as a number of years? If yes, answer with that number only. If not, answer \"none\".\n\nText: {text}\n\nAge:",
    "birth_date": "Read the text and answer with the date the person was born, exactly as written in the text. If it is not mentioned, answer \"none\".\n\nText: {text}\n\nDate of birth:",
    "death_date": "Read the text. If the text says the person has died, answer with the date of death exactly as written. If the person is alive or no death date is given, answer \"none\".\n\nText: {text}\n\nDate of death:",
}

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})

NULL_ANSWERS = {"", "none", "unknown", "n/a", "no", "not mentioned", "not given", "not stated", "null"}


# --------------------------------------------------------- deterministic part ---

def parse_date(s: str | None) -> date | None:
    """Parse '14 March 1879', 'March 14, 1879', '1879-03-14' or a bare year (-> 1 July)."""
    if not s:
        return None
    s = s.strip().lower().replace(",", " ")
    if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s):
        y, mo, d = map(int, m.groups())
        return _safe_date(y, mo, d)
    if m := re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\.?\s+(\d{4})\b", s):
        d, mon, y = m.groups()
        if mon in MONTHS:
            return _safe_date(int(y), MONTHS[mon], int(d))
    if m := re.search(r"\b([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{4})\b", s):
        mon, d, y = m.groups()
        if mon in MONTHS:
            return _safe_date(int(y), MONTHS[mon], int(d))
    if m := re.search(r"\b([a-z]+)\s+(\d{4})\b", s):
        mon, y = m.groups()
        if mon in MONTHS:
            return _safe_date(int(y), MONTHS[mon], 15)
    if m := re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", s):
        return _safe_date(int(m.group(1)), 7, 1)  # year only: mid-year estimate
    return None


def _safe_date(y: int, mo: int, d: int) -> date | None:
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def age_on(birth: date, on: date) -> int:
    return on.year - birth.year - ((on.month, on.day) < (birth.month, birth.day))


def stated_age_in_text(age: int, text: str) -> int | None:
    """Accept a model-proposed age only if the text states it as an age.

    Grounding on the bare number is not enough: for "born 24 June 1987" the model
    answers 24, and "24" does appear in the text.
    """
    n = str(age)
    patterns = [
        rf"\b(aged?|age of)\s+{n}\b",
        rf"\b{n}[- ]years?[- ]old\b",
        rf"^[^,]{{2,60}},\s*{n}\s*,",  # "Priya Raman, 34, is ..."
    ]
    return age if any(re.search(p, text, re.I) for p in patterns) else None


def clean_answer(s: str) -> str | None:
    s = s.strip().strip(" .\"'")
    return None if s.lower() in NULL_ANSWERS else s


def grounded(value: str | None, text: str) -> str | None:
    """Keep a string only if it (or most of its words) really appears in the text."""
    if not value:
        return None
    if value.lower() in text.lower():
        return value
    words = [w for w in re.findall(r"[A-Za-z]+", value.lower()) if len(w) > 2]
    if words and sum(w in text.lower() for w in words) / len(words) >= 0.6:
        return value
    return None


def validate(record: dict) -> dict:
    missing = set(SCHEMA) - set(record)
    extra = set(record) - set(SCHEMA)
    if missing or extra:
        raise ValueError(f"Schema mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    for key, expected in SCHEMA.items():
        if not isinstance(record[key], expected):
            raise TypeError(f"Field {key!r} has type {type(record[key]).__name__}")
    if record["age"] is not None and not 0 <= record["age"] <= 130:
        raise ValueError(f"Implausible age {record['age']}")
    return record


# ---------------------------------------------------------------- model part ---

class BioExtractor:
    def __init__(self, model_name: str = DEFAULT_MODEL, reference_date: date | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        self.reference_date = reference_date or date.today()

    def _ask_all(self, text: str) -> dict[str, str]:
        """Run every field prompt for one text as a single batch, greedy decoding."""
        keys = list(PROMPTS)
        prompts = [PROMPTS[k].format(text=text) for k in keys]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=24, num_beams=1, do_sample=False)
        answers = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return dict(zip(keys, answers))

    def extract(self, text: str) -> dict:
        text = " ".join(text.split())
        raw = self._ask_all(text)

        name = grounded(clean_answer(raw["name"]), text)
        occupation = grounded(clean_answer(raw["occupation"]), text)
        birth_raw = grounded(clean_answer(raw["birth_date"]), text)
        death_raw = grounded(clean_answer(raw["death_date"]), text)

        birth, death = parse_date(birth_raw), parse_date(death_raw)
        if death and birth and death <= birth:
            death = None  # model echoed the birth date as death date

        age = None
        if birth:
            # A birth date is more reliable than a stated age, which may be stale.
            # Deceased -> age at death; living -> age on the reference date.
            age = age_on(birth, death or self.reference_date)
        else:
            stated = clean_answer(raw["stated_age"])
            if stated and (m := re.fullmatch(r"(\d{1,3})(?: years?(?: old)?)?", stated)):
                age = stated_age_in_text(int(m.group(1)), text)

        record = {
            "name": name,
            "age": age if age is not None and 0 <= age <= 130 else None,
            "occupation": occupation,
            "birth_date": birth.isoformat() if birth else None,
            "is_deceased": death is not None,
        }
        return validate(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract name/age/occupation from a biography as JSON.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--text", help="Biography paragraph.")
    src.add_argument("--file", type=Path, help="Text file containing the biography.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--as-of", type=date.fromisoformat, help="Reference date for computing age (default: today).")
    args = parser.parse_args()

    extractor = BioExtractor(args.model, args.as_of)

    if args.text or args.file:
        text = args.text or args.file.read_text(encoding="utf-8")
        print(json.dumps(extractor.extract(text), indent=2))
        return

    results = []
    for bio in SAMPLE_BIOS:
        record = extractor.extract(bio["text"])
        results.append(record)
        print(f"\n# {bio['text'][:90]}...")
        print(json.dumps(record, indent=2))
    print(f"\nAll {len(results)} records passed schema validation.")


if __name__ == "__main__":
    main()
