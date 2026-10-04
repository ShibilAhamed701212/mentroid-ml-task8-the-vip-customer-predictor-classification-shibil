# ML Team Tasks

Four small NLP components, each runnable on its own on a laptop CPU.

> **Note on the repository name.** The repository is named after a "VIP customer predictor" classification task, but it contains no such model. Its actual content is the four NLP tasks below.

| # | Task | Model | Entry point |
|---|------|-------|-------------|
| 1 | Semantic search over 20 movie plots | `sentence-transformers/all-MiniLM-L6-v2` + FAISS | `task1_semantic_search/search.py` |
| 2 | Text cleaning + sentence-aware overlapping chunking | rule-based (no model) | `task2_text_pipeline/pipeline.py` |
| 3 | Zero-shot intent router for support messages | `facebook/bart-large-mnli` | `task3_intent_classifier/classifier.py` |
| 4 | Biography text → validated JSON | `google/flan-t5-large` | `task4_text_to_json/extractor.py` |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Linux, `pip install torch` pulls the CUDA build (several GB). For CPU only, install torch first from the CPU index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Models download from Hugging Face on first run: about 90 MB for task 1, 1.6 GB for task 3 and 3 GB for task 4.
All four tasks were run with Python 3.12, torch 2.14, transformers 5.18, sentence-transformers 6.1 and faiss-cpu 1.15. The unit tests pass on Python 3.10 to 3.13.

## Testing

```bash
pip install pytest ruff
pytest -q tests        # 19 model-free unit tests for tasks 2 and 4
ruff check .           # lint (config in ruff.toml)
```

GitHub Actions (`.github/workflows/ci.yml`) runs ruff, the unit tests and task 2 on the chapters I–III text on Python 3.10–3.13. It does not download the models, so tasks 1, 3 and 4 are only checked by running them locally.

---

## 1. Semantic search engine

```bash
python task1_semantic_search/search.py "A movie about space travel"
python task1_semantic_search/search.py            # interactive
python task1_semantic_search/search.py --numpy "a heist in Las Vegas"
```

**Design**
- Each document is embedded as `title + plot`.
- Embeddings are L2-normalised, so inner product equals cosine similarity. `faiss.IndexFlatIP` therefore returns an exact cosine top-k.
- With 20 documents an exact flat index is the right choice. An approximate index (IVF or HNSW) only pays off at around 100k+ vectors.
- A numpy fallback (`embeddings @ query`) gives identical results when FAISS is not installed.

**Result**

```
Query: "A movie about space travel"
  1. Interstellar  (cosine=0.550)
  2. Apollo 13     (cosine=0.432)
  3. Gravity       (cosine=0.415)
```

## 2. Text chunking and cleaning pipeline

Input: *Pride and Prejudice* from Project Gutenberg (eBook #1342, public domain).
- `data/pride_and_prejudice_ch1-3.txt` is chapters I–III taken from the raw file. It still contains illustration captions, `_italic_` markup, curly quotes and hard line wraps.
- `data/pride_and_prejudice_full.txt` is the full book, including the licence header and footer.

```bash
python task2_text_pipeline/pipeline.py data/pride_and_prejudice_ch1-3.txt
python task2_text_pipeline/pipeline.py data/pride_and_prejudice_full.txt --chunk-words 200 --overlap-words 40
```

Outputs `output/<name>.clean.txt` and `output/<name>.chunks.jsonl`. Each JSONL line has:
- `chunk_id`, `text`, `word_count`
- `first_sentence` and `last_sentence`
- `overlap_words`
- `oversize`

**Cleaning steps**
1. Strip the Gutenberg licence header and footer, using the `*** START/END OF ... ***` markers.
2. Remove `[Illustration ...]` blocks. These contain nested brackets, so a bracket-depth scanner is used instead of a regex.
3. Drop chapter headings.
4. Map curly quotes and dashes to ASCII, and strip accents (`café → cafe`).
5. Remove `_emphasis_` markup and any character outside `A–Z a–z 0–9 .,;:!?'"()-`.
6. Re-join words hyphenated across a line break, unwrap hard-wrapped lines and keep paragraph breaks.
7. Fix the spacing around punctuation.

**Chunking**
- **Sentence splitting** is rule-based. It does not split on abbreviations (`Mr.`, `Mrs.`, `Dr.` …) or on single initials, and it handles closing quotes (`"Why?" he said` stays one sentence). The rules live in code, with no NLTK download step.
- **Packing:** whole sentences are packed greedily until the next sentence would push the chunk past 200 words.
- **Overlap:** each new chunk starts with the longest run of the previous chunk's trailing sentences that fits in the overlap budget (40 words by default). The overlap is shortened, or dropped, when it would leave no room for the next new sentence, so every chunk adds at least one sentence the previous chunk did not have.
- **Long sentences:** a single sentence longer than the limit is never cut. It becomes its own chunk, flagged `oversize: true`.

| Input | Clean words | Sentences | Chunks | Words per chunk (min / mean / max) |
|---|---|---|---|---|
| Chapters I–III | 3,353 | 193 | 21 | 146 / 185 / 199 |
| Full book | 127,258 | 6,122 | 800 | 86 / 183 / 200 |

## 3. Zero-shot intent classifier

```bash
python task3_intent_classifier/classifier.py                       # evaluation report
python task3_intent_classifier/classifier.py "My card was charged twice"
python task3_intent_classifier/classifier.py --json "The app keeps crashing"
```

**How it works.** BART-MNLI treats the message as an NLI premise and each route as a hypothesis, e.g. *"This customer support message is about a technical problem such as a bug, error, crash or login issue."* The entailment probability is the route score, so no training data is needed.

**Design decisions, with measurements on the bundled dummy data**

| Version | Dev set (15) | Held-out set (9) |
|---|---|---|
| Bare label names, softmax across 3 labels | 14/15 | – |
| Descriptive labels, softmax, 0.5 confidence floor | 14/15 | 5/9 |
| **Descriptive labels; General Inquiry as catch-all, threshold 0.7 (final)** | **14/15** | **7/9** |

- "General Inquiry" is not really a topic; it means "none of the others". Scored as one hypothesis among three, it lost to Technical Support on questions like "What languages is your product available in?".
- The final router scores **Billing** and **Technical Support** independently (multi-label entailment).
- A message goes to one of those two only if its entailment probability is at least 0.7. Otherwise it goes to **General Inquiry**.
- Routes live in a single dict, so adding a department is one line.

**Caveat:** these are tiny hand-written sets. The descriptions and threshold were chosen on the dev set. The held-out set wasn't used to tune them, but it did lead to the catch-all redesign, so 7/9 is somewhat optimistic. Before production, measure on a few hundred real tickets. If labelled data exists, fine-tuning a small classifier (e.g. DistilBERT or SetFit) would most likely beat zero-shot.

## 4. Structured data extraction (text → JSON)

```bash
python task4_text_to_json/extractor.py                           # runs the sample biographies
python task4_text_to_json/extractor.py --text "Ada Lovelace (10 December 1815 - 27 November 1852) was an English mathematician."
python task4_text_to_json/extractor.py --file bio.txt --model google/flan-t5-base --as-of 2026-10-03
```

Output schema (always valid JSON, validated before printing):

```json
{"name": "str|null", "age": "int|null", "occupation": "str|null", "birth_date": "YYYY-MM-DD|null", "is_deceased": "bool"}
```

**Why not just prompt "return JSON"?** Flan-T5's SentencePiece vocabulary has no `{` or `}` tokens, so it cannot write a JSON object verbatim, and its free-form output format drifts between inputs. The script uses constrained extraction instead:

1. **One focused prompt per field**, run as a single batch. Flan-T5 is instruction-tuned on QA, so short questions work much better than one big instruction.
2. **Grounding.** A string answer is kept only if it actually appears in the source text. Otherwise it becomes `null`, which blocks hallucinated names and occupations.
3. **Age is computed, not generated.** Wikipedia biographies rarely state an age:
   - If the model finds a birth date (and a death date, if any), Python computes the age deterministically: age at death, or age as of today (or `--as-of`).
   - Otherwise a stated age is accepted only if the text uses it as an age (", 34,", "aged 34", "34-year-old").
   - Why the check is needed: in a first run, the model answered **24** for Messi, taken from "born **24** June". A grounding check on the bare number passed it, because "24" is in the text.
4. **Schema validation.** The script checks types, the exact key set and a plausible age range (0–130), then serialises with `json.dumps`.

**Results** (`--as-of 2026-10-03`, flan-t5-large, CPU). All records pass schema validation.

| Input | name | age | occupation | birth_date | is_deceased |
|---|---|---|---|---|---|
| Albert Einstein bio | Albert Einstein | 76 | physicist | 1879-03-14 | true |
| Marie Curie bio | Marie Curie | 66 | chemist | 1867-11-07 | true |
| Lionel Messi bio | Lionel Andres Messi | 39 | footballer | 1987-06-24 | false |
| Fictional bio, age stated | Priya Raman | 34 | software engineer | null | false |
| Ada Lovelace (unseen, via `--text`) | Ada Lovelace | 36 | mathematician | 1815-12-10 | true |

**Known limitations**
- Only the person's *main* occupation is returned (Curie: "chemist", not "chemist and physicist").
- A bare birth year is treated as 1 July, so the computed age can be off by one.
- Long bios are truncated at 512 tokens. For long input, run task 2's chunker first.

---

## Project layout

```
data/                          Gutenberg source texts
task1_semantic_search/         search.py, movies.py (20 plots)
task2_text_pipeline/           pipeline.py
task3_intent_classifier/       classifier.py, sample_messages.py (dev + held-out)
task4_text_to_json/            extractor.py, sample_bios.py
tests/                         model-free unit tests
output/                        generated by task 2 (git-ignored)
.github/workflows/ci.yml       lint + unit tests
```

## Audit notes (2026-10-04)

Fixed:
- **Task 2 duplicate chunk.** When a long sentence followed the overlap, the next chunk could hold only sentences already in the previous one. This happened once in the full book (a 39-word chunk). The overlap now always leaves room for the next sentence. Regression test: `test_every_chunk_adds_a_new_sentence`.
- **Task 4 schema check.** `validate` accepted `True` as an age because `bool` is a subclass of `int`. It now rejects booleans in the `age` field.
- The task 2 statistics table was out of date and has been re-measured.
- Added a ruff config and a CI workflow; renamed two ambiguous loop variables flagged by ruff (E741).

Known limitations:
- The model-backed tasks (1, 3, 4) have no automated tests; CI covers only the deterministic code.
- `requirements.txt` gives lower bounds only, so a fresh install may pick newer, untested versions.
- Task 2's sentence splitter treats any single letter followed by a period as an initial, so "So do I. Then..." stays one sentence.

## Author
- **Name:** shibilahamed701212
- **Email:** shibilahamed701212@gmail.com
