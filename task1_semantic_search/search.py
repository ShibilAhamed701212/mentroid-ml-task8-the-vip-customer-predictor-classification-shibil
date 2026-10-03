"""Task 1 - Semantic search over movie plots.

Pipeline:
    plots --(sentence-transformers)--> L2-normalised embeddings --> FAISS IndexFlatIP
    query --(same encoder)----------> normalised vector ---------> top-k by cosine similarity

Because every vector is unit length, inner product == cosine similarity, so an exact
inner-product index gives exact cosine ranking. If FAISS is not installed the same
search runs as a single numpy matrix-vector product.

Usage:
    python task1_semantic_search/search.py "A movie about space travel"
    python task1_semantic_search/search.py            # interactive mode
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from movies import MOVIES  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

try:
    import faiss
except ImportError:  # numpy fallback keeps the script usable without faiss
    faiss = None


@dataclass
class SearchResult:
    rank: int
    score: float
    title: str
    plot: str


class SemanticSearchEngine:
    def __init__(self, documents: list[dict], model_name: str = DEFAULT_MODEL, use_faiss: bool = True):
        from sentence_transformers import SentenceTransformer

        if not documents:
            raise ValueError("Need at least one document to index.")
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.use_faiss = use_faiss and faiss is not None

        # Embed title + plot: the title adds useful signal ("Interstellar" -> space).
        texts = [f"{d['title']}. {d['plot']}" for d in documents]
        self.embeddings = self._encode(texts)

        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")
        top_k = max(1, min(top_k, len(self.documents)))
        q = self._encode([query])

        if self.use_faiss:
            scores, idx = self.index.search(q, top_k)
            pairs = zip(idx[0].tolist(), scores[0].tolist())
        else:
            sims = self.embeddings @ q[0]
            order = np.argsort(-sims)[:top_k]
            pairs = ((int(i), float(sims[i])) for i in order)

        return [
            SearchResult(rank=r, score=s, title=self.documents[i]["title"], plot=self.documents[i]["plot"])
            for r, (i, s) in enumerate(pairs, start=1)
        ]


def print_results(query: str, results: list[SearchResult]) -> None:
    print(f'\nQuery: "{query}"')
    for r in results:
        print(f"  {r.rank}. {r.title}  (cosine={r.score:.3f})")
        print(f"     {r.plot}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over 20 movie plots.")
    parser.add_argument("query", nargs="*", help="Natural-language query. Omit for interactive mode.")
    parser.add_argument("-k", "--top-k", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--numpy", action="store_true", help="Use numpy instead of FAISS.")
    args = parser.parse_args()

    engine = SemanticSearchEngine(MOVIES, model_name=args.model, use_faiss=not args.numpy)
    backend = "FAISS IndexFlatIP" if engine.use_faiss else "numpy"
    print(f"Indexed {len(MOVIES)} plots with {args.model} ({engine.embeddings.shape[1]}-d, {backend}).")

    if args.query:
        query = " ".join(args.query)
        print_results(query, engine.search(query, args.top_k))
        return

    print("Type a query (empty line to quit).")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        print_results(query, engine.search(query, args.top_k))


if __name__ == "__main__":
    main()
