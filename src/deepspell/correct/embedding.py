"""Embedding-space correction (successor of the kd-tree token lookup space).

A trained :class:`TokenEncoder` maps the query into the embedding space; the
nearest corpus tokens by cosine similarity are re-ranked by Damerau-
Levenshtein distance. Persistence is a plain ``.npz`` (vectors) plus
``.tokens`` text file — no pickle. Search is exact brute-force numpy (fast
into the low millions of tokens); ``hnswlib`` is used transparently when
installed and the space is large.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from rapidfuzz.distance import DamerauLevenshtein

from ..charset import CharVocab
from ..models.encoder import TokenEncoder

logger = logging.getLogger(__name__)

_HNSW_THRESHOLD = 1_000_000

try:
    import hnswlib
except ImportError:
    hnswlib = None


class EmbeddingCorrector:
    def __init__(self, encoder: TokenEncoder, char_vocab: CharVocab, tokens: list[str], vectors: np.ndarray):
        if len(tokens) != vectors.shape[0]:
            raise ValueError("token list and vector matrix disagree")
        self.encoder = encoder
        self.char_vocab = char_vocab
        self.tokens = tokens
        self.vectors = vectors.astype(np.float32)
        self._index = None
        if hnswlib is not None and len(tokens) >= _HNSW_THRESHOLD:
            self._index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
            self._index.init_index(max_elements=len(tokens), ef_construction=200, M=16)
            self._index.add_items(self.vectors)
            self._index.set_ef(64)
            logger.info("hnswlib index over %d tokens", len(tokens))

    @classmethod
    def build(
        cls, encoder: TokenEncoder, char_vocab: CharVocab, names: list[str], batch_size: int = 512
    ) -> EmbeddingCorrector:
        normalized = sorted({char_vocab.normalize(n) for n in names if char_vocab.normalize(n)})
        vectors = encoder.encode_strings(normalized, char_vocab, batch_size=batch_size)
        return cls(encoder, char_vocab, normalized, vectors)

    def save(self, base_path: str | Path) -> Path:
        base = Path(base_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(base.with_suffix(".npz"), vectors=self.vectors)
        base.with_suffix(".tokens").write_text("\n".join(self.tokens), encoding="utf-8")
        return base

    @classmethod
    def load(cls, encoder: TokenEncoder, char_vocab: CharVocab, base_path: str | Path) -> EmbeddingCorrector:
        base = Path(base_path)
        vectors = np.load(base.with_suffix(".npz"))["vectors"]
        tokens = base.with_suffix(".tokens").read_text(encoding="utf-8").splitlines()
        return cls(encoder, char_vocab, tokens, vectors)

    def match(self, token: str, k: int = 3) -> list[tuple[str, float]]:
        if not self.tokens:
            return []
        query = self.char_vocab.normalize(token)
        vector = self.encoder.encode_strings([query], self.char_vocab)[0]
        fetch = min(max(k * 4, k), len(self.tokens))
        if self._index is not None:
            labels, _ = self._index.knn_query(vector[None, :], k=fetch)
            candidate_ids = labels[0]
        else:
            similarity = self.vectors @ vector
            candidate_ids = np.argpartition(-similarity, fetch - 1)[:fetch]
        ranked = sorted(
            (
                (self.tokens[int(i)], float(DamerauLevenshtein.distance(query, self.tokens[int(i)])))
                for i in candidate_ids
            ),
            key=lambda pair: pair[1],
        )
        return ranked[:k]
