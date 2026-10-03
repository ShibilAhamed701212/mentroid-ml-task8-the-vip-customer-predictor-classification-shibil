# ML Team Tasks

Four small NLP components, each runnable on its own on a laptop CPU.

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

Models download from Hugging Face on first run: about 90 MB for task 1, 1.6 GB for task 3 and 3 GB for task 4.
Tested with Python 3.13, torch 2.14, transformers 5.18, sentence-transformers 6.1 and faiss-cpu 1.15.

```bash
pytest -q tests        # 18 model-free unit tests for tasks 2 and 4
```

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
- **Overlap:** each new chunk starts with the longest run of the previous chunk's trailing sentences that fits in the overlap budget (40 words by default). Chunks always move forward.
- **Long sentences:** a single sentence longer than the limit is never cut. It becomes its own chunk, flagged `oversize: true`.

| Input | Clean words | Sentences | Chunks | Words per chunk (min / mean / max) |
|---|---|---|---|---|
| Chapters I–III | 3,342 | 193 | 21 | 154 / 186 / 200 |
| Full book | 126,458 | 6,124 | 796 | 39 / 183 / 200 |

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
```

## Author
- **Name:** shibilahamed701212
- **Email:** shibilahamed701212@gmail.com
