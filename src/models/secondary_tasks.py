# ============================================================================
# TAREFAS SECUNDÁRIAS DE CLASSIFICAÇÃO
# Tarefas onde a estrutura do grafo carrega informação que o texto não tem,
# favorecendo GNNs sobre classificadores de conteúdo puro.
# ============================================================================
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.db import Paper, Citation


def build_popularity_labels(graph_data: Dict, papers_df: pd.DataFrame,
                             threshold: int = None) -> Tuple[torch.Tensor, Dict]:
    """Tarefa 1: predição de popularidade por in-degree.

    Label binária: paper está no quartil superior de in-degree (citado muitas
    vezes dentro do grafo) vs. abaixo disso. O in-degree é estrutural —
    invisível no texto, mas capturado diretamente pela GNN através da
    agregação de vizinhança reversa.

    threshold: in-degree mínimo para label=1 (padrão: percentil 75).
    Retorna (y, label_info)."""
    from collections import Counter
    in_degrees = Counter(tgt for _, tgt in graph_data['edges'])
    n_nodes = graph_data['num_nodes']
    degrees = [in_degrees.get(i, 0) for i in range(n_nodes)]

    if threshold is None:
        threshold = int(np.percentile(degrees, 75))

    y = torch.tensor([1 if d > threshold else 0 for d in degrees], dtype=torch.long)
    label_info = {
        'task': 'popularity',
        'description': f'In-degree > {threshold} (percentil 75)',
        'num_classes': 2,
        'class_names': ['baixa citação', 'alta citação'],
        'positive_rate': y.float().mean().item(),
    }
    return y, label_info


def build_subcategory_cs_labels(papers_df: pd.DataFrame,
                                 graph_data: Dict) -> Tuple[torch.Tensor, Dict]:
    """Tarefa 2: classificação de subcategoria dentro de cs.*.

    Distingue cs.LG, cs.AI, cs.CL, cs.CV, cs.IR (e 'cs.other' para demais)
    apenas entre papers de cs.*. Muito mais difícil pelo texto (vocabulário
    similar entre subcomunidades), mas diferenciável pela rede de citações
    (cada subcomunidade cita principalmente a si mesma).

    Retorna (y, label_info). Nós fora de cs.* recebem label = num_classes-1
    (classe 'outro') — pode-se mascarar esses nós no treino."""
    MAIN_CS_SUBS = ['cs.LG', 'cs.AI', 'cs.CL', 'cs.CV', 'cs.IR']

    idx_to_id = {idx: pid for pid, idx in graph_data['id_to_idx'].items()}

    def get_primary(categories: str) -> str:
        if not categories:
            return 'other'
        primary = categories.split(',')[0].strip()
        return primary if primary in MAIN_CS_SUBS else ('cs.other' if primary.startswith('cs.') else 'other')

    all_cats = ['other'] + MAIN_CS_SUBS + ['cs.other']
    cat_to_idx = {c: i for i, c in enumerate(all_cats)}

    labels = []
    for node_idx in range(graph_data['num_nodes']):
        pid = idx_to_id[node_idx]
        row = papers_df[papers_df['id'] == pid]
        cat = get_primary(row['categories'].values[0] if len(row) else '')
        labels.append(cat_to_idx[cat])

    y = torch.tensor(labels, dtype=torch.long)
    label_info = {
        'task': 'cs_subcategory',
        'description': 'Subcategoria dentro de cs.* (LG, AI, CL, CV, IR, other)',
        'num_classes': len(all_cats),
        'class_names': all_cats,
        'cat_to_idx': cat_to_idx,
    }
    return y, label_info


def build_interdisciplinary_labels(papers_df: pd.DataFrame,
                                    graph_data: Dict) -> Tuple[torch.Tensor, Dict]:
    """Tarefa 3: detecção de papers interdisciplinares.

    Label binária: paper cita artigos de pelo menos 2 macro-áreas distintas
    (ex: cs + stat) vs. cita apenas sua própria área. Papers interdisciplinares
    são pontes no grafo — identificáveis pela estrutura de vizinhança, não
    pelo texto.

    Requer o graph_data completo com edges e id_to_idx."""
    from collections import defaultdict
    from src.features import extract_macro_categories

    idx_to_id = {idx: pid for pid, idx in graph_data['id_to_idx'].items()}
    id_to_macro = {}
    for _, row in papers_df.iterrows():
        cat = row['categories'] or ''
        primary = cat.split(',')[0].strip()
        id_to_macro[row['id']] = primary.split('.')[0] if primary else 'unknown'

    # Para cada nó, coleta as macro-áreas dos seus vizinhos outgoing
    neighbor_macros = defaultdict(set)
    for src_idx, tgt_idx in graph_data['edges']:
        tgt_id = idx_to_id[tgt_idx]
        macro = id_to_macro.get(tgt_id, 'unknown')
        neighbor_macros[src_idx].add(macro)

    labels = []
    for node_idx in range(graph_data['num_nodes']):
        macros = neighbor_macros[node_idx]
        macros.discard('unknown')
        # Interdisciplinar = cita pelo menos 2 macro-áreas distintas
        labels.append(1 if len(macros) >= 2 else 0)

    y = torch.tensor(labels, dtype=torch.long)
    pos_rate = y.float().mean().item()
    label_info = {
        'task': 'interdisciplinary',
        'description': 'Paper cita ≥2 macro-áreas distintas',
        'num_classes': 2,
        'class_names': ['disciplinar', 'interdisciplinar'],
        'positive_rate': pos_rate,
    }
    return y, label_info