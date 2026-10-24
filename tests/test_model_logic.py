"""Tests for the logic around the models in tasks 1, 3 and 4, with the models stubbed out.

The real models are too large for CI, so each test builds the object without calling
its __init__ (which loads the model) and plugs in a small deterministic fake.
"""

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for task in ("task1_semantic_search", "task3_intent_classifier", "task4_text_to_json"):
    sys.path.insert(0, str(ROOT / task))

import classifier  # noqa: E402
import extractor  # noqa: E402

try:  # task 1 needs numpy; tasks 3 and 4 still run without it
    import numpy as np
    import search
except ImportError:
    np = search = None

needs_numpy = pytest.mark.skipif(np is None, reason="numpy not installed")


# ---------------------------------------------------------------- task 1 ---

class FakeEncoder:
    """Bag-of-words over a fixed vocabulary, L2-normalised like the real call."""

    VOCAB = ["space", "astronaut", "heist", "money", "love", "ocean"]

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        rows = []
        for t in texts:
            words = t.lower().replace(".", " ").split()
            v = np.array([words.count(w) for w in self.VOCAB], dtype=np.float32) + 1e-3
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows)


DOCS = [
    {"title": "Orbit", "plot": "An astronaut drifts in space."},
    {"title": "Vault", "plot": "A crew plans a heist to steal money."},
    {"title": "Tide", "plot": "Two people fall in love by the ocean."},
]


def _engine(use_faiss):
    engine = object.__new__(search.SemanticSearchEngine)
    engine.documents = DOCS
    engine.model = FakeEncoder()
    engine.use_faiss = use_faiss
    engine.embeddings = engine._encode([f"{d['title']}. {d['plot']}" for d in DOCS])
    if use_faiss:
        engine.index = search.faiss.IndexFlatIP(engine.embeddings.shape[1])
        engine.index.add(engine.embeddings)
    return engine


BACKENDS = [False] + ([True] if search is not None and search.faiss is not None else [])


@needs_numpy
@pytest.mark.parametrize("use_faiss", BACKENDS)
def test_search_ranks_by_cosine(use_faiss):
    results = _engine(use_faiss).search("a heist for money", top_k=2)
    assert [r.rank for r in results] == [1, 2]
    assert results[0].title == "Vault"
    assert results[0].score >= results[1].score


@needs_numpy
@pytest.mark.parametrize("use_faiss", BACKENDS)
def test_search_clamps_top_k_and_rejects_empty_query(use_faiss):
    engine = _engine(use_faiss)
    assert len(engine.search("space", top_k=50)) == len(DOCS)
    with pytest.raises(ValueError):
        engine.search("   ")


# ---------------------------------------------------------------- task 3 ---

def _router(scores_by_message, threshold=0.7):
    """Router whose 'model' returns the given entailment score per route name."""
    router = object.__new__(classifier.IntentRouter)
    router.routes = classifier.DEFAULT_ROUTES
    router._description_to_route = {d: n for n, d in router.routes.items()}
    router.threshold = threshold

    def fake_pipeline(messages, candidate_labels, **kwargs):
        assert kwargs["multi_label"] is True
        out = []
        for m in messages:
            per_route = scores_by_message[m]
            labels = sorted(candidate_labels, key=lambda d: -per_route[router._description_to_route[d]])
            out.append({"labels": labels, "scores": [per_route[router._description_to_route[d]] for d in labels]})
        return out[0] if len(out) == 1 else out

    router.classifier = fake_pipeline
    return router


def test_router_picks_specific_route_above_threshold():
    router = _router({"refund me": {"Billing": 0.91, "Technical Support": 0.2}})
    [d] = router.classify(["  refund me  "])
    assert (d.message, d.route, d.confidence) == ("refund me", "Billing", 0.91)
    assert d.scores == {"Billing": 0.91, "Technical Support": 0.2}


def test_router_falls_back_to_catch_all_below_threshold():
    router = _router({
        "where is your office?": {"Billing": 0.3, "Technical Support": 0.6},
        "app crashes": {"Billing": 0.1, "Technical Support": 0.7},
    })
    office, crash = router.classify(["where is your office?", "app crashes"])
    assert office.route == classifier.CATCH_ALL_ROUTE
    assert office.confidence == pytest.approx(0.4)
    assert crash.route == "Technical Support"  # threshold is inclusive


def test_router_rejects_empty_messages_and_handles_empty_batch():
    router = _router({})
    assert router.classify([]) == []
    with pytest.raises(ValueError):
        router.classify(["ok", "   "])


# ---------------------------------------------------------------- task 4 ---

def _extractor(answers, as_of=date(2026, 10, 3)):
    ex = object.__new__(extractor.BioExtractor)
    ex.reference_date = as_of
    defaults = {"name": "none", "occupation": "none", "stated_age": "none", "birth_date": "none", "death_date": "none"}
    ex._ask_all = lambda text: {**defaults, **answers}
    return ex


def test_extract_computes_age_at_death():
    text = "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born physicist."
    record = _extractor({
        "name": "Albert Einstein", "occupation": "physicist",
        "birth_date": "14 March 1879", "death_date": "18 April 1955", "stated_age": "24",
    }).extract(text)
    assert record == {"name": "Albert Einstein", "age": 76, "occupation": "physicist",
                      "birth_date": "1879-03-14", "is_deceased": True}


def test_extract_living_person_uses_reference_date():
    # A reference date far from today, so the test fails if extract() reads the clock.
    text = "Lionel Messi (born 24 June 1987) is an Argentine footballer."
    record = _extractor({"name": "Lionel Messi", "birth_date": "24 June 1987"}, as_of=date(2040, 6, 23)).extract(text)
    assert record["age"] == 52 and record["is_deceased"] is False


def test_extract_ignores_death_date_that_echoes_birth_date():
    text = "Lionel Messi (born 24 June 1987) is an Argentine footballer."
    record = _extractor({"birth_date": "24 June 1987", "death_date": "24 June 1987"}).extract(text)
    assert record["is_deceased"] is False


def test_extract_drops_ungrounded_answers_and_unstated_age():
    text = "Lionel Messi (born in Rosario) is a footballer."
    record = _extractor({"name": "Diego Maradona", "occupation": "astronaut", "stated_age": "24"}).extract(text)
    assert record == {"name": None, "age": None, "occupation": None, "birth_date": None, "is_deceased": False}


def test_extract_accepts_age_stated_in_text():
    text = "Priya Raman, 34, is a software engineer."
    record = _extractor({"name": "Priya Raman", "stated_age": "34 years old"}).extract(text)
    assert record["age"] == 34 and record["birth_date"] is None
