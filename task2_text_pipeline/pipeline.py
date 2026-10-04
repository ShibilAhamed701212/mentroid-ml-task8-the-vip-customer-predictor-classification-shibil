"""Task 2 - Text cleaning and sentence-aware overlapping chunking.

Stages:
    1. strip_gutenberg_boilerplate  - drop the licence header/footer if present
    2. remove_bracketed_blocks      - drop [Illustration: ...] blocks (they can be nested)
    3. clean_text                   - unicode normalisation, markup removal, special-char
                                      stripping, de-hyphenation, whitespace normalisation
    4. split_sentences              - rule-based splitter aware of abbreviations and quotes
    5. chunk_sentences              - greedy packing of whole sentences into <=200-word
                                      chunks, each starting with the previous chunk's tail

A chunk never cuts a sentence. The one exception is a single sentence longer than the
word limit: it becomes a chunk by itself (oversize) instead of being cut.

Usage:
    python task2_text_pipeline/pipeline.py data/pride_and_prejudice_ch1-3.txt
    python task2_text_pipeline/pipeline.py input.txt --chunk-words 200 --overlap-words 40 --out-dir output/
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)
CHAPTER_HEADING = re.compile(r"^\s*(chapter|book|part|volume)\s+([ivxlcdm]+|\d+)\b\.?\s*$", re.I | re.M)

# Typographic characters mapped to plain ASCII equivalents before stripping.
CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "–": "-", "—": " - ", "―": " - ", "−": "-",
    "…": "...", " ": " ", "·": " ", "•": " ", "﻿": "",
})

# Everything outside this whitelist is treated as a special character and removed.
DISALLOWED = re.compile(r"[^A-Za-z0-9\s.,;:!?'\"()\-]")

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "rev", "gen", "col", "capt",
    "lt", "sgt", "hon", "esq", "vs", "etc", "e.g", "i.e", "no", "vol", "fig", "mt",
}


# ---------------------------------------------------------------- cleaning ---

def strip_gutenberg_boilerplate(text: str) -> str:
    start = GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = GUTENBERG_END.search(text)
    if end:
        text = text[:end.start()]
    return text


def remove_bracketed_blocks(text: str, prefixes: tuple[str, ...] = ("[Illustration", "[_Copyright")) -> str:
    """Remove [...] blocks that start with one of `prefixes`, honouring nested brackets.

    A regex cannot match balanced brackets, and Gutenberg illustration captions often
    contain nested ones, e.g. "[Illustration: ... [_Copyright 1894_]]".
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "[" and text.startswith(prefixes, i):
            depth = 0
            j = i
            while j < n:
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            i = j + 1  # an unclosed block swallows the rest of the text
            out.append("\n\n")  # keep paragraph separation where the block was
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def clean_text(text: str, drop_headings: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_gutenberg_boilerplate(text)
    text = remove_bracketed_blocks(text)
    if drop_headings:
        text = CHAPTER_HEADING.sub("\n", text)

    text = text.translate(CHAR_MAP)
    # NFKD splits accented letters into base + combining mark; dropping the marks
    # turns "café" into "cafe" rather than deleting the whole letter.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = re.sub(r"_(.+?)_", r"\1", text, flags=re.S)  # _emphasis_ markup
    text = DISALLOWED.sub(" ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # words hyphenated across lines
    text = re.sub(r"-{2,}", " - ", text)                  # ASCII "--" dashes

    # Paragraphs are separated by blank lines; single newlines are just line wraps.
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
    paragraphs = [p for p in paragraphs if re.search(r"[A-Za-z]", p)]
    text = "\n\n".join(paragraphs)

    text = re.sub(r"\s+([.,;:!?])", r"\1", text)   # no space before punctuation
    text = re.sub(r"([.,;:!?])(?=[A-Za-z])", r"\1 ", text)  # space after it
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ------------------------------------------------------- sentence splitting ---

_BOUNDARY = re.compile(r"""([.!?]+["')]*)\s+(?=["'(]?[A-Z0-9])""")


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        start = 0
        for m in _BOUNDARY.finditer(paragraph):
            end = m.end(1)
            candidate = paragraph[start:end]
            last_word = candidate.rstrip("\"')").rsplit(None, 1)[-1].rstrip(".").lower() if candidate.strip() else ""
            # "Mr. Bennet" or a single initial like "J. Smith" is not a boundary.
            if last_word in ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha() and m.group(1) == "."):
                continue
            sentences.append(candidate.strip())
            start = m.end()
        tail = paragraph[start:].strip()
        if tail:
            sentences.append(tail)
    return [s for s in sentences if s]


# ----------------------------------------------------------------- chunking ---

@dataclass
class Chunk:
    chunk_id: int
    text: str
    word_count: int
    first_sentence: int
    last_sentence: int  # inclusive
    overlap_words: int  # words shared with the previous chunk
    oversize: bool = False


def word_count(s: str) -> int:
    return len(s.split())


def chunk_sentences(sentences: list[str], chunk_words: int = 200, overlap_words: int = 40) -> list[Chunk]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be in [0, chunk_words)")

    counts = [word_count(s) for s in sentences]
    chunks: list[Chunk] = []
    start = 0
    prev_end = -1  # last sentence index of the previous chunk

    while start < len(sentences):
        end = start
        total = counts[start]
        while end + 1 < len(sentences) and total + counts[end + 1] <= chunk_words:
            end += 1
            total += counts[end]

        shared = sum(counts[start:prev_end + 1]) if prev_end >= start else 0
        chunks.append(Chunk(
            chunk_id=len(chunks),
            text=" ".join(sentences[start:end + 1]),
            word_count=total,
            first_sentence=start,
            last_sentence=end,
            overlap_words=shared,
            oversize=total > chunk_words,
        ))
        if end == len(sentences) - 1:
            break

        # Next chunk starts with the longest run of trailing sentences that fits in
        # the overlap budget, but must start after this chunk's start so we progress.
        # The overlap must also leave room for the next new sentence; otherwise the
        # next chunk would hold only sentences already in this one.
        next_start = end + 1
        budget = min(overlap_words, chunk_words - counts[next_start])
        while next_start - 1 > start and counts[next_start - 1] <= budget:
            next_start -= 1
            budget -= counts[next_start]
        prev_end = end
        start = next_start

    return chunks


# ---------------------------------------------------------------------- CLI ---

def run(input_path: Path, out_dir: Path, chunk_words: int, overlap_words: int) -> list[Chunk]:
    raw = input_path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    sentences = split_sentences(cleaned)
    chunks = chunk_sentences(sentences, chunk_words, overlap_words)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    (out_dir / f"{stem}.clean.txt").write_text(cleaned + "\n", encoding="utf-8")
    with (out_dir / f"{stem}.chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    sizes = [c.word_count for c in chunks]
    print(f"Input:      {input_path} ({word_count(raw):,} raw words, {len(raw):,} chars)")
    print(f"Cleaned:    {word_count(cleaned):,} words, {len(sentences):,} sentences")
    print(f"Chunks:     {len(chunks)} (target <= {chunk_words} words, overlap <= {overlap_words} words)")
    if sizes:
        print(f"Chunk size: min {min(sizes)}, mean {sum(sizes) / len(sizes):.0f}, max {max(sizes)}")
        print(f"Oversize:   {sum(c.oversize for c in chunks)} (single sentences longer than the limit)")
    print(f"Wrote:      {out_dir / f'{stem}.clean.txt'}\n            {out_dir / f'{stem}.chunks.jsonl'}")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean a text file and split it into overlapping sentence-aligned chunks.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    parser.add_argument("--chunk-words", type=int, default=200)
    parser.add_argument("--overlap-words", type=int, default=40)
    parser.add_argument("--show", type=int, default=2, help="Print the first N chunks.")
    args = parser.parse_args()

    chunks = run(args.input, args.out_dir, args.chunk_words, args.overlap_words)
    for c in chunks[:args.show]:
        print(f"\n--- chunk {c.chunk_id} ({c.word_count} words, {c.overlap_words} overlapping) ---\n{c.text}")


if __name__ == "__main__":
    main()
