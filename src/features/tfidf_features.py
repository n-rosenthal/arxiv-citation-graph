# ============================================================================
# FEATURES TF-IDF — paper_rows (id, title, abstract, categories) -> tensores
# ============================================================================
from typing import Dict, List, Tuple

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import TFIDF_MAX_FEATURES

DEFAULT_CATEGORY = 'cs.OH'


def build_corpus(paper_rows) -> List[str]:
    """Concatena título + abstract de cada paper. paper_rows: lista de
    namedtuples com atributos .title e .abstract (ex: de build_graph_from_db)."""
    return [f"{r.title or ''} {r.abstract or ''}" for r in paper_rows]


def extract_primary_categories(paper_rows, default: str = DEFAULT_CATEGORY) -> List[str]:
    """Extrai a categoria primária (primeira da lista) de cada paper.
    paper_rows: lista de namedtuples com atributo .categories (string
    separada por vírgulas, ex: 'cs.LG, cs.AI')."""
    return [
        r.categories.split(',')[0].strip() if r.categories else default
        for r in paper_rows
    ]


def build_tfidf_features(paper_rows, max_features: int = TFIDF_MAX_FEATURES) -> torch.Tensor:
    """Gera features TF-IDF normalizadas (L2) a partir de paper_rows.
    Retorna um tensor float de shape (num_papers, max_features)."""
    corpus = build_corpus(paper_rows)
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    x = vectorizer.fit_transform(corpus)
    x = normalize(x, norm='l2')
    return torch.tensor(x.toarray(), dtype=torch.float)


def extract_macro_categories(paper_rows, default: str = DEFAULT_CATEGORY) -> List[str]:
    """Extrai a macro-área (prefixo antes do ponto) da categoria primária de
    cada paper. Ex: 'cs.LG, cs.AI' -> 'cs'; 'stat.ML' -> 'stat'.

    Reduz drasticamente o número de classes (de ~80 subcategorias arXiv
    para ~6-10 macro-áreas: cs, stat, math, eess, physics, etc.), o que dá
    amostras suficientes por classe para treino/avaliação confiáveis."""
    primary = extract_primary_categories(paper_rows, default=default)
    return [cat.split('.')[0] for cat in primary]


def build_category_labels(paper_rows, macro: bool = True) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Converte categorias em labels inteiras.

    Se macro=True (padrão), usa extract_macro_categories (ex: 'cs', 'stat',
    'math' — poucas classes, balanceadas). Se macro=False, usa a
    subcategoria completa (ex: 'cs.LG', 'cs.AI' — muitas classes, esparsas).

    Retorna (y, cat_to_idx), onde y é um tensor long de shape (num_papers,)
    e cat_to_idx mapeia nome da categoria -> índice da classe."""
    categories = extract_macro_categories(paper_rows) if macro else extract_primary_categories(paper_rows)
    unique_cats = sorted(set(categories))
    cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
    y = torch.tensor([cat_to_idx[cat] for cat in categories], dtype=torch.long)
    return y, cat_to_idx