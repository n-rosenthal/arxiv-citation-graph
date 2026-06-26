# ============================================================================
# CONSTRUÇÃO DO GRAFO — papers/citations (banco) -> estrutura de grafo
# ============================================================================
import random
from collections import defaultdict
from typing import Dict, Optional

from src.db import Paper, Citation


def build_graph_from_db(SessionLocal) -> Dict:
    """Retorna dados do grafo completamente desacoplados da sessão ORM.

    SessionLocal: factory de sessões (ex: o retorno de get_session_factory()).

    Retorna um dict com:
        num_nodes, num_edges, edges (lista de tuplas (idx_src, idx_tgt)),
        id_to_idx (mapa paper_id -> índice), paper_rows (lista de namedtuples
        com id, title, abstract, categories, ordenada por id)."""
    session = SessionLocal()
    try:
        # Serializa tudo em memória antes de fechar a sessão
        paper_rows = session.query(
            Paper.id, Paper.title, Paper.abstract, Paper.categories
        ).all()
        citation_rows = session.query(Citation.source_id, Citation.target_id).all()
    finally:
        session.close()

    # Indexa na mesma ordem fixa e determinística
    paper_rows = sorted(paper_rows, key=lambda r: r[0])
    id_to_idx = {row[0]: idx for idx, row in enumerate(paper_rows)}
    edges = [
        (id_to_idx[src], id_to_idx[tgt])
        for src, tgt in citation_rows
        if src in id_to_idx and tgt in id_to_idx
    ]
    return {
        'num_nodes': len(paper_rows),
        'num_edges': len(edges),
        'edges': edges,
        'id_to_idx': id_to_idx,
        'paper_rows': paper_rows,   # lista de namedtuples (id, title, abstract, categories)
    }


def filter_by_min_degree(graph_data: Dict, min_degree: int = 2,
                          mode: str = 'total') -> Dict:
    """Filtra nós com grau total (ou in/out) abaixo de min_degree, retendo
    apenas o subgrafo induzido pelos nós que satisfazem o critério.

    mode: 'total' (in+out), 'in' (apenas in-degree), 'out' (apenas out-degree).

    Útil para remover nós isolados (grau 0) ou com muito poucas conexões,
    onde a agregação de vizinhança das GNNs não contribui nada.
    Nós com abstract vazio também se beneficiam (features TF-IDF nulas),
    mas o filtro aqui é puramente estrutural.

    Retorna novo dict no mesmo formato de build_graph_from_db."""
    from collections import Counter

    paper_rows = graph_data['paper_rows']
    edges = graph_data['edges']

    # Calcula grau de cada nó
    out_deg = Counter(src for src, _ in edges)
    in_deg  = Counter(tgt for _, tgt in edges)

    def degree(idx):
        if mode == 'in':    return in_deg[idx]
        if mode == 'out':   return out_deg[idx]
        return in_deg[idx] + out_deg[idx]  # total

    keep_set = {idx for idx in range(len(paper_rows)) if degree(idx) >= min_degree}
    keep_indices = sorted(keep_set)
    old_to_new = {old: new for new, old in enumerate(keep_indices)}

    new_paper_rows = [paper_rows[i] for i in keep_indices]
    new_edges = [
        (old_to_new[src], old_to_new[tgt])
        for src, tgt in edges
        if src in old_to_new and tgt in old_to_new
    ]
    new_id_to_idx = {row[0]: idx for idx, row in enumerate(new_paper_rows)}

    removed = len(paper_rows) - len(new_paper_rows)
    print(f"   🔪 Filtro min_degree={min_degree} ({mode}): "
          f"{len(paper_rows)} → {len(new_paper_rows)} nós "
          f"({removed} removidos, {len(new_edges)} arestas restantes)")

    return {
        'num_nodes': len(new_paper_rows),
        'num_edges': len(new_edges),
        'edges': new_edges,
        'id_to_idx': new_id_to_idx,
        'paper_rows': new_paper_rows,
    }

 
def subsample_by_macro_category(graph_data: Dict, max_ratio: float = 3.0, seed: int = 42) -> Dict:
    """Subamostra nós da(s) macro-área(s) majoritária(s) para reduzir
    desbalanceamento de classes, preservando a estrutura do grafo entre os
    nós remanescentes (arestas remapeadas, não apenas filtradas).
 
    max_ratio: tamanho máximo permitido da maior classe em relação à
    SEGUNDA maior classe (ex: 3.0 → maior classe pode ter no máximo 3x o
    tamanho da segunda maior). Classes menores que a segunda maior não são
    afetadas.
 
    Retorna um novo dict no mesmo formato de build_graph_from_db, com
    paper_rows, edges e id_to_idx recalculados para o subconjunto."""
    from src.features import extract_macro_categories
 
    paper_rows = graph_data['paper_rows']
    macro_cats = extract_macro_categories(paper_rows)
 
    # Agrupa índices originais por macro-área
    by_macro = defaultdict(list)
    for idx, cat in enumerate(macro_cats):
        by_macro[cat].append(idx)
 
    sizes = sorted((len(v) for v in by_macro.values()), reverse=True)
    if len(sizes) < 2:
        return graph_data  # só uma classe — nada a balancear
 
    second_largest = sizes[1]
    target_max = int(second_largest * max_ratio)
 
    rng = random.Random(seed)
    keep_indices = set()
    for cat, indices in by_macro.items():
        if len(indices) > target_max:
            keep_indices.update(rng.sample(indices, target_max))
        else:
            keep_indices.update(indices)
 
    keep_indices = sorted(keep_indices)
    old_to_new = {old: new for new, old in enumerate(keep_indices)}
 
    new_paper_rows = [paper_rows[i] for i in keep_indices]
    new_edges = [
        (old_to_new[src], old_to_new[tgt])
        for src, tgt in graph_data['edges']
        if src in old_to_new and tgt in old_to_new
    ]
    new_id_to_idx = {row[0]: idx for idx, row in enumerate(new_paper_rows)}
 
    print(f"   📉 Subamostragem: {graph_data['num_nodes']} → {len(new_paper_rows)} nós "
          f"(maior classe limitada a {target_max}, baseado na 2ª maior = {second_largest})")
 
    return {
        'num_nodes': len(new_paper_rows),
        'num_edges': len(new_edges),
        'edges': new_edges,
        'id_to_idx': new_id_to_idx,
        'paper_rows': new_paper_rows,
    }
