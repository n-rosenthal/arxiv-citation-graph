# ============================================================================
# BASELINE CLÁSSICO — classificação sobre features TF-IDF (sem usar o grafo)
# Cumpre o requisito da disciplina de "ao menos um modelo de classificação
# ou regressão" treinado com pipeline padrão (sklearn).
# ============================================================================
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score


BASELINE_MODELS = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000,
        class_weight="balanced" if True else None,
        n_jobs=-1
    ),

    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced" if True else None,
        random_state=42,
        n_jobs=-1
    ),

    "gradient_boosting": lambda: GradientBoostingClassifier(
        random_state=42
    ),
}


def train_baseline_models(x: np.ndarray, y: np.ndarray,
                           train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray
                           ) -> Dict[str, Dict]:
    """Treina cada modelo em BASELINE_MODELS sobre as features x (TF-IDF) e
    labels y (categoria primária), usando as mesmas máscaras train/val/test
    do split do grafo.

    Retorna dict: {nome_modelo: {'model': estimator, 'test_acc': float,
                                  'report': str (classification_report)}}"""
    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    results = {}
    for name, factory in BASELINE_MODELS.items():
        model = factory()
        model.fit(x_train, y_train)

        val_pred = model.predict(x_val)
        val_acc = accuracy_score(y_val, val_pred)

        test_pred = model.predict(x_test)
        test_acc = accuracy_score(y_test, test_pred)
        report = classification_report(y_test, test_pred, zero_division=0)

        results[name] = {
            'model': model,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'report': report,
        }
    return results


def tensors_to_numpy(x, y, train_mask, val_mask, test_mask) -> Tuple[np.ndarray, ...]:
    """Converte tensores PyTorch (x, y, masks) para numpy, para uso com sklearn."""
    return (
        x.detach().cpu().numpy(),
        y.detach().cpu().numpy(),
        train_mask.detach().cpu().numpy(),
        val_mask.detach().cpu().numpy(),
        test_mask.detach().cpu().numpy(),
    )