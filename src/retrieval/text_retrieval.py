"""Instruction-only cosine retrieval. Images are never read by this module."""
import numpy as np


class TextMemoryRetriever:
    def __init__(self, encoder):
        self.encoder = encoder

    def shortlist(self, memory, embeddings, instruction, candidates):
        if candidates < 1:
            raise ValueError('Text retrieval count must be positive')
        if not memory:
            return [], {}, []
        query = np.asarray(self.encoder.encode([instruction]), dtype=np.float32)
        vectors = np.asarray(embeddings['Instruction'], dtype=np.float32)
        if query.ndim != 2 or query.shape[0] != 1 or vectors.ndim != 2 or vectors.shape != (len(memory), query.shape[1]):
            raise ValueError('Instruction embedding dimensions do not match memory')
        norms = np.linalg.norm(vectors, axis=1)
        query_norm = np.linalg.norm(query[0])
        if not np.isfinite(query).all() or not np.isfinite(vectors).all() or query_norm <= 0 or np.any(norms <= 0):
            raise ValueError('Invalid instruction embedding')
        scores = (vectors @ query[0]) / (norms * query_norm)
        indexes = sorted(range(len(memory)), key=lambda i: (-float(scores[i]), i))
        ranked = [{'index': i, 'Id': memory[i]['Id'], 'score': float(scores[i]), 'rank': n}
                  for n, i in enumerate(indexes, 1)]
        chosen = indexes[:candidates]
        subset = {key: [values[i] for i in chosen] if isinstance(values, list) else values[chosen]
                  for key, values in embeddings.items() if key not in ('VisualScores', 'RetrievalScores')}
        subset['RetrievalScores'] = scores[chosen]
        return [memory[i] for i in chosen], subset, ranked
