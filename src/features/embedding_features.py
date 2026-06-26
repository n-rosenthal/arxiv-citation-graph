# ============================================================================
# FEATURES DE EMBEDDING — paper_rows (id, title, abstract) -> embeddings densos
# via sentence-transformers (modelo leve, roda em CPU)
# ============================================================================
import torch

from src.features.tfidf_features import build_corpus

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dims, ~80MB, CPU-friendly

_model_cache = {}


def _get_model(model_name: str = EMBEDDING_MODEL_NAME):
    """Carrega (e cacheia) o modelo de sentence-transformers."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def build_embedding_features(paper_rows, model_name: str = EMBEDDING_MODEL_NAME,
                              batch_size: int = 32) -> torch.Tensor:
    """Gera embeddings densos (já normalizados L2) a partir de título+abstract.
    Retorna um tensor float de shape (num_papers, embedding_dim)."""
    corpus = build_corpus(paper_rows)
    model = _get_model(model_name)
    embeddings = model.encode(
        corpus, batch_size=batch_size, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    return torch.tensor(embeddings, dtype=torch.float)


def build_combined_features(paper_rows, tfidf_features: torch.Tensor = None,
                             model_name: str = EMBEDDING_MODEL_NAME,
                             batch_size: int = 32) -> torch.Tensor:
    """Concatena features TF-IDF + embeddings densos.
    Se tfidf_features não for fornecido, é calculado a partir de paper_rows."""
    if tfidf_features is None:
        from src.features.tfidf_features import build_tfidf_features
        tfidf_features = build_tfidf_features(paper_rows)

    emb_features = build_embedding_features(paper_rows, model_name=model_name, batch_size=batch_size)
    return torch.cat([tfidf_features, emb_features], dim=1)