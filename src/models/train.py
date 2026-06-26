# ============================================================================
# TREINAMENTO — GNNs (GCN, GraphSAGE, GAT) + baseline clássico, com checkpoints
# ============================================================================
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from src.config import (
    DB_URL, MODELS_DIR,
    GNN_HIDDEN_DIM, GNN_NUM_LAYERS,
    GAT_HIDDEN_DIM, GAT_HEADS,
    DROPOUT, GAT_DROPOUT, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    TRAIN_SPLIT, VAL_SPLIT,
    MIN_NODES_FOR_TRAINING, MIN_EDGES_FOR_TRAINING,
)
from src.db import get_session_factory, ModelCheckpoint
from src.graph import build_graph_from_db, subsample_by_macro_category, filter_by_min_degree
from src.features import build_tfidf_features, build_category_labels
from src.features.embedding_features import build_embedding_features, build_combined_features
from src.models.gnn_models import GCN, GraphSAGE, GAT
from src.models.baseline_models import train_baseline_models, tensors_to_numpy

FEATURE_BUILDERS = {
    'tfidf': build_tfidf_features,
    'embeddings': build_embedding_features,
    'combined': build_combined_features,
}


def make_split_masks(n_nodes: int, train_split: float = TRAIN_SPLIT, val_split: float = VAL_SPLIT):
    """Gera máscaras booleanas train/val/test via permutação aleatória."""
    indices = torch.randperm(n_nodes)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[indices[:int(train_split * n_nodes)]] = True
    val_mask[indices[int(train_split * n_nodes):int(val_split * n_nodes)]] = True
    test_mask[indices[int(val_split * n_nodes):]] = True
    return train_mask, val_mask, test_mask


def get_next_version(SessionLocal, model_name: str) -> int:
    session = SessionLocal()
    try:
        latest = (session.query(ModelCheckpoint)
                  .filter_by(model_name=model_name)
                  .order_by(ModelCheckpoint.version.desc())
                  .first())
        return (latest.version + 1) if latest else 1
    finally:
        session.close()


def build_dataset(db_url: str = DB_URL, feature_mode: str = 'tfidf',
                   split_masks=None, graph_data=None,
                   balance_classes: bool = True, max_class_ratio: float = 3.0,
                   min_degree: int = 1):
    """Monta o objeto Data (PyG) + máscaras a partir do banco.

    min_degree: grau mínimo (in+out) para um nó entrar no dataset. Nós com
    grau < min_degree são descartados antes do balanceamento. Padrão=1
    (remove apenas nós completamente isolados). Use 2 ou 3 para um grafo
    mais denso onde a agregação de vizinhança das GNNs é mais informativa.

    Retorna (data, graph_data, cat_to_idx) ou None se dados insuficientes."""
    if feature_mode not in FEATURE_BUILDERS:
        raise ValueError(f"feature_mode inválido: {feature_mode!r}. Use um de {list(FEATURE_BUILDERS)}")

    SessionLocal = get_session_factory(db_url)
    if graph_data is None:
        graph_data = build_graph_from_db(SessionLocal)
        if min_degree > 0:
            graph_data = filter_by_min_degree(graph_data, min_degree=min_degree)
        if balance_classes:
            graph_data = subsample_by_macro_category(graph_data, max_ratio=max_class_ratio)

    if graph_data['num_nodes'] < MIN_NODES_FOR_TRAINING or graph_data['num_edges'] < MIN_EDGES_FOR_TRAINING:
        print(f"   ⚠️ Dados insuficientes: {graph_data['num_nodes']} nós, {graph_data['num_edges']} arestas")
        return None

    paper_rows = graph_data['paper_rows']
    x = FEATURE_BUILDERS[feature_mode](paper_rows)
    y, cat_to_idx = build_category_labels(paper_rows)

    edges = graph_data['edges']
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                   if edges else torch.zeros((2, 0), dtype=torch.long))

    n_nodes = graph_data['num_nodes']
    if split_masks is None:
        split_masks = make_split_masks(n_nodes)
    train_mask, val_mask, test_mask = split_masks

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data, graph_data, cat_to_idx


def compute_class_weights(y: torch.Tensor, train_mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Calcula pesos por classe (inverse frequency, normalizado) a partir das
    labels de treino — equivalente a class_weight='balanced' do sklearn:
    weight[c] = n_samples / (n_classes * count[c])."""
    train_y = y[train_mask]
    counts = torch.bincount(train_y, minlength=num_classes).float()
    counts = counts.clamp(min=1)  # evita divisão por zero em classes ausentes no treino
    n_samples = train_y.shape[0]
    weights = n_samples / (num_classes * counts)
    return weights


def train_gnn(name: str, model, data, device, class_weights: torch.Tensor = None) -> dict:
    """Treina um modelo GNN por EPOCHS épocas. Retorna métricas finais.

    class_weights: tensor opcional de shape (num_classes,) — pesos por
    classe para compensar desbalanceamento na cross_entropy (equivalente a
    class_weight='balanced' do sklearn). Se None, todas as classes têm peso 1."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    weight = class_weights.to(device) if class_weights is not None else None

    best_val_acc = torch.tensor(0.0)
    final_loss = torch.tensor(0.0)
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weight)
        loss.backward()
        optimizer.step()
        final_loss = loss.detach()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

    print(f"      ✓ {name.upper()}: test_acc={test_acc:.4f}")
    return {
        'model': model,
        'test_acc': test_acc.item(),
        'best_val_acc': best_val_acc.item(),
        'loss': final_loss.item(),
    }


def train_models(db_url: str = DB_URL, models_dir: Path = MODELS_DIR,
                  feature_mode: str = 'tfidf', split_masks=None, graph_data=None,
                  balanced: bool = True, min_degree: int = 1) -> dict:
    """Treina GCN, GraphSAGE, GAT e baseline clássico com o feature_mode
    indicado ('tfidf', 'embeddings' ou 'combined'). Salva checkpoints no
    banco e os state_dicts em models_dir (prefixados pelo feature_mode).

    min_degree: remove nós com grau total < min_degree antes do treino.
    Padrão=1 (só remove isolados). Use 2+ para focar em nós com vizinhança
    real, onde as GNNs têm vantagem estrutural.

    balanced: se True (padrão), aplica pesos de classe balanceados.

    Retorna dict com resultados."""
    dataset = build_dataset(db_url, feature_mode=feature_mode, split_masks=split_masks,
                             graph_data=graph_data, min_degree=min_degree)
    if dataset is None:
        return {}
    data, graph_data, cat_to_idx = dataset

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    SessionLocal = get_session_factory(db_url)
    n_nodes = graph_data['num_nodes']
    num_classes = data.y.max().item() + 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    class_weights = compute_class_weights(data.y.cpu(), data.train_mask.cpu(), num_classes) if balanced else None
    if class_weights is not None:
        print(f"   ⚖️  Pesos de classe (balanceamento): {class_weights.round(decimals=2).tolist()}")

    gnn_models = {
        'gcn': GCN(data.num_features, GNN_HIDDEN_DIM, num_classes,
                   dropout=DROPOUT, num_layers=GNN_NUM_LAYERS),
        'graphsage': GraphSAGE(data.num_features, GNN_HIDDEN_DIM, num_classes,
                               dropout=DROPOUT, num_layers=GNN_NUM_LAYERS),
        'gat': GAT(data.num_features, GAT_HIDDEN_DIM, num_classes,
                   heads=GAT_HEADS, dropout=GAT_DROPOUT, num_layers=GNN_NUM_LAYERS),
    }

    results = {}
    for name, model in gnn_models.items():
        print(f"   📊 Treinando {name.upper()} ({feature_mode})...")
        res = train_gnn(name, model, data, device, class_weights=class_weights)
        results[name] = res

        version = get_next_version(SessionLocal, f"{name}_{feature_mode}")
        file_path = str(models_dir / f"{name}_{feature_mode}_v{version}.pt")

        session = SessionLocal()
        try:
            checkpoint = ModelCheckpoint(
                model_name=f"{name}_{feature_mode}", version=version,
                accuracy=res['test_acc'], loss=res['loss'],
                num_nodes=n_nodes, num_edges=graph_data['num_edges'], num_classes=num_classes,
                file_path=file_path,
                metrics={'accuracy': res['test_acc'], 'best_val_accuracy': res['best_val_acc'],
                         'feature_mode': feature_mode}
            )
            session.add(checkpoint)
            session.commit()
        finally:
            session.close()

        torch.save(res['model'].state_dict(), file_path)

    # Baseline clássico sobre as mesmas features/labels/split
    print(f"   📊 Treinando baselines clássicos ({feature_mode})...")
    x_np, y_np, train_mask_np, val_mask_np, test_mask_np = tensors_to_numpy(
        data.x.cpu(), data.y.cpu(), data.train_mask.cpu(), data.val_mask.cpu(), data.test_mask.cpu()
    )
    baseline_results = train_baseline_models(x_np, y_np, train_mask_np, val_mask_np, test_mask_np)
    for name, res in baseline_results.items():
        print(f"      ✓ {name}: test_acc={res['test_acc']:.4f}")
    results['baselines'] = baseline_results

    print(f"   ✅ Treinamento concluído ({feature_mode})!")
    return results


def train_models_compare(db_url: str = DB_URL, models_dir: Path = MODELS_DIR,
                          feature_modes=('tfidf', 'combined'),
                          balance_classes: bool = True, max_class_ratio: float = 3.0,
                          min_degree: int = 1) -> dict:
    """Treina e compara modelos sob diferentes feature_modes, usando o MESMO
    split e grafo (já filtrado e balanceado) para garantir comparação justa.

    min_degree: remove nós com grau total < min_degree antes do treino.
    Aplicado ANTES do balanceamento — define o universo de nós elegíveis.

    Retorna dict: {feature_mode: resultados_de_train_models}.
    Se dados insuficientes, retorna {}."""
    SessionLocal = get_session_factory(db_url)
    graph_data = build_graph_from_db(SessionLocal)
    if min_degree > 0:
        graph_data = filter_by_min_degree(graph_data, min_degree=min_degree)
    if balance_classes:
        graph_data = subsample_by_macro_category(graph_data, max_ratio=max_class_ratio)

    if graph_data['num_nodes'] < MIN_NODES_FOR_TRAINING or graph_data['num_edges'] < MIN_EDGES_FOR_TRAINING:
        print(f"   ⚠️ Dados insuficientes: {graph_data['num_nodes']} nós, {graph_data['num_edges']} arestas")
        return {}

    split_masks = make_split_masks(graph_data['num_nodes'])

    all_results = {}
    for mode in feature_modes:
        print(f"\n=== Feature mode: {mode} ===")
        all_results[mode] = train_models(db_url, models_dir, feature_mode=mode,
                                           split_masks=split_masks, graph_data=graph_data)
    return all_results


if __name__ == "__main__":
    train_models()