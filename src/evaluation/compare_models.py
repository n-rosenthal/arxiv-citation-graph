# ============================================================================
# COMPARAÇÃO DE MODELOS — GNNs vs baselines clássicos
# ============================================================================
from pathlib import Path
from typing import Dict

import torch
from sklearn.metrics import classification_report

from src.config import DB_URL, MODELS_DIR
from src.models.train import build_dataset, train_models, train_models_compare


def evaluate_gnn(model, data) -> Dict:
    """Avalia um modelo GNN já treinado no conjunto de teste.
    Retorna accuracy e classification_report (texto)."""
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    report = classification_report(y_true, y_pred, zero_division=0)
    return {'test_acc': test_acc, 'report': report}


def run_comparison(db_url: str = DB_URL, models_dir: Path = MODELS_DIR, retrain: bool = True,
                    feature_mode: str = 'tfidf') -> Dict:
    """Treina (se retrain=True) ou carrega os modelos e monta uma tabela
    comparativa de acurácia entre GCN, GraphSAGE, GAT e os baselines clássicos,
    usando o feature_mode indicado ('tfidf', 'embeddings' ou 'combined').

    Retorna dict no formato:
        {'gcn': {'test_acc': ..., 'report': ...},
         'graphsage': {...}, 'gat': {...},
         'logistic_regression': {...}, 'random_forest': {...}, 'gradient_boosting': {...}}
    """
    if retrain:
        results = train_models(db_url=db_url, models_dir=models_dir, feature_mode=feature_mode)
        if not results:
            return {}

        comparison = {}
        for name in ('gcn', 'graphsage', 'gat'):
            comparison[name] = {
                'test_acc': results[name]['test_acc'],
                'best_val_acc': results[name]['best_val_acc'],
                'loss': results[name]['loss'],
            }
        for name, res in results['baselines'].items():
            comparison[name] = {
                'test_acc': res['test_acc'],
                'val_acc': res['val_acc'],
                'report': res['report'],
            }
        return comparison

    # Modo "apenas avaliação" — útil se já existem checkpoints e o dataset
    # não mudou desde o último treino. Reconstrói o dataset (para ter as
    # mesmas máscaras seria necessário persistir os índices; aqui assumimos
    # retrain=True como caminho padrão para garantir consistência).
    dataset = build_dataset(db_url, feature_mode=feature_mode)
    if dataset is None:
        return {}
    raise NotImplementedError(
        "Avaliação sem retreino requer persistir as máscaras train/val/test "
        "junto do checkpoint. Use retrain=True (padrão) para um resultado consistente."
    )


def run_feature_comparison(db_url: str = DB_URL, models_dir: Path = MODELS_DIR,
                            feature_modes=('tfidf', 'combined'),
                            balance_classes: bool = True, max_class_ratio: float = 3.0,
                            min_degree: int = 1) -> Dict:
    """Compara o desempenho dos modelos sob diferentes representações de features.

    min_degree: remove nós com grau total < min_degree antes do treino.
    """
    all_results = train_models_compare(db_url=db_url, models_dir=models_dir, feature_modes=feature_modes,
                                        balance_classes=balance_classes, max_class_ratio=max_class_ratio,
                                        min_degree=min_degree)
    if not all_results:
        return {}

    comparison = {}
    for mode, results in all_results.items():
        mode_comparison = {}
        for name in ('gcn', 'graphsage', 'gat'):
            mode_comparison[name] = {
                'test_acc': results[name]['test_acc'],
                'best_val_acc': results[name]['best_val_acc'],
                'loss': results[name]['loss'],
            }
        for name, res in results['baselines'].items():
            mode_comparison[name] = {
                'test_acc': res['test_acc'],
                'val_acc': res['val_acc'],
                'report': res['report'],
            }
        comparison[mode] = mode_comparison
    return comparison


def print_feature_comparison_table(comparison: Dict):
    """Imprime uma tabela comparando test_acc de cada modelo sob cada
    feature_mode, lado a lado (ex: colunas 'tfidf' e 'combined')."""
    if not comparison:
        print("   ⚠️ Nenhum resultado para comparar.")
        return

    modes = list(comparison.keys())
    all_models = sorted({name for mode_res in comparison.values() for name in mode_res})

    header = f"   {'modelo':<22}" + "".join(f"{m:>14}" for m in modes)
    print("\n📊 Comparação por feature_mode (acurácia no conjunto de teste)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for name in all_models:
        row = f"   {name:<22}"
        for mode in modes:
            acc = comparison.get(mode, {}).get(name, {}).get('test_acc')
            row += f"{acc:>14.4f}" if acc is not None else f"{'N/A':>14}"
        print(row)
    print("-" * len(header))


def print_comparison_table(comparison: Dict):
    """Imprime uma tabela simples comparando test_acc de todos os modelos."""
    if not comparison:
        print("   ⚠️ Nenhum resultado para comparar.")
        return

    print("\n📊 Comparação de modelos (acurácia no conjunto de teste)")
    print("-" * 45)
    for name, res in sorted(comparison.items(), key=lambda kv: kv[1].get('test_acc', 0), reverse=True):
        acc = res.get('test_acc')
        print(f"   {name:<22} {acc:.4f}" if acc is not None else f"   {name:<22} N/A")
    print("-" * 45)


if __name__ == "__main__":
    comparison = run_comparison()
    print_comparison_table(comparison)