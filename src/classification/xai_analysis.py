"""
xai_analysis.py
===============
Explainable AI (XAI) per BodyPoseRec - analisi di importanza delle feature.

Obiettivo
---------
Un classificatore non deve essere una "scatola nera". All'esame orale e nel
documento tecnico bisogna saper rispondere alla domanda: "Perche' il modello
ha predetto questa posa?". Questo modulo lo spiega in modo quantitativo.

Tecnica: Permutation Feature Importance
---------------------------------------
E' un metodo XAI model-agnostic (funziona sia su SVM che su MLP):
1. Si misura l'accuratezza del modello sul test set.
2. Per ogni feature, si mescola CASUALMENTE quella sola colonna tra i
   campioni (permutazione), distruggendone il contenuto informativo ma
   mantenendone la distribuzione.
3. Si rimisura l'accuratezza. Il CALO di accuratezza e' l'importanza di
   quella feature: se permutarla peggiora molto il modello, allora il
   modello dipendeva fortemente da essa.
4. Si ripete piu' volte e si media (la permutazione e' casuale).

Il vantaggio rispetto ai coefficienti interni del modello e' che questo
metodo misura l'importanza in termini di IMPATTO SULLE PRESTAZIONI reali,
ed e' confrontabile tra modelli diversi.

Output prodotti:
- results/feature_importance_svm.png  (grafico a barre)
- results/feature_importance_mlp.png
- results/feature_importance.txt       (ranking testuale)

Esecuzione:
    python -m src.classification.xai_analysis

Riferimenti didattici:
- Blueprint BodyPoseRec, sezione "XAI - Interpretabilita' del modello".
- Modulo 3, valutazione e analisi critica dei modelli.
"""

import os
import pickle
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non interattivo: salva i grafici su file
import matplotlib.pyplot as plt

from src.config import (
    FEATURES_DIR, SVM_MODEL_PATH, MLP_MODEL_PATH, SCALER_PATH,
    FEATURE_NAMES_PATH, RESULTS_DIR, RANDOM_SEED, ensure_dirs,
)

# Numero di ripetizioni della permutazione per ogni feature: piu' alto =
# stima piu' stabile ma piu' lento. 10 e' un buon compromesso.
N_REPEATS = 10


def _load_test_data():
    """Carica il test set (feature gia' estratte) e i nomi delle feature."""
    X_test = np.load(FEATURES_DIR / "X_test.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return X_test, y_test, feature_names


def _load_models():
    """Carica SVM, MLP e lo scaler addestrati."""
    import tensorflow as tf

    with open(SVM_MODEL_PATH, "rb") as f:
        svm = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    mlp = tf.keras.models.load_model(MLP_MODEL_PATH)
    return svm, mlp, scaler


def _accuracy_svm(svm, scaler, X, y):
    """Accuratezza dell'SVM su un set di feature grezze (le standardizza)."""
    return float(np.mean(svm.predict(scaler.transform(X)) == y))


def _accuracy_mlp(mlp, scaler, X, y):
    """Accuratezza dell'MLP su un set di feature grezze (le standardizza)."""
    probs = mlp.predict(scaler.transform(X), verbose=0)
    return float(np.mean(np.argmax(probs, axis=1) == y))


def permutation_importance(accuracy_fn, X, y, feature_names,
                           n_repeats=N_REPEATS, seed=RANDOM_SEED):
    """Calcola la permutation importance di ogni feature.

    Parametri
    ---------
    accuracy_fn : callable
        Funzione che, dato (X, y), restituisce l'accuratezza del modello.
    X, y : np.ndarray
        Test set (feature grezze ed etichette vere).
    feature_names : list[str]
        Nomi delle feature (per il report).
    n_repeats : int
        Numero di permutazioni casuali per feature.

    Ritorna
    -------
    importances_mean, importances_std : np.ndarray
        Calo medio e deviazione standard di accuratezza per ogni feature.
    """
    rng = np.random.default_rng(seed)
    baseline = accuracy_fn(X, y)
    n_features = X.shape[1]

    importances = np.zeros((n_features, n_repeats), dtype=np.float64)
    for feat_idx in range(n_features):
        for rep in range(n_repeats):
            X_permuted = X.copy()
            # Mescola la sola colonna feat_idx tra i campioni.
            perm = rng.permutation(X.shape[0])
            X_permuted[:, feat_idx] = X_permuted[perm, feat_idx]
            permuted_acc = accuracy_fn(X_permuted, y)
            # Importanza = quanto cala l'accuratezza rispetto alla baseline.
            importances[feat_idx, rep] = baseline - permuted_acc

    return importances.mean(axis=1), importances.std(axis=1), baseline


def _plot_importance(names, means, stds, model_label, out_path):
    """Disegna e salva il grafico a barre dell'importanza delle feature."""
    # Ordina le feature dalla piu' importante alla meno importante.
    order = np.argsort(means)
    sorted_names = [names[i] for i in order]
    sorted_means = means[order]
    sorted_stds = stds[order]

    plt.figure(figsize=(9, 8))
    y_pos = np.arange(len(sorted_names))
    plt.barh(y_pos, sorted_means, xerr=sorted_stds,
             color="#4C72B0", edgecolor="black", alpha=0.85)
    plt.yticks(y_pos, sorted_names, fontsize=8)
    plt.xlabel("Calo di accuratezza dopo permutazione (importanza)")
    plt.title(f"Permutation Feature Importance - {model_label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def run_xai_analysis():
    """Esegue l'analisi XAI completa su SVM e MLP e salva grafici e report."""
    ensure_dirs()
    print("=== BodyPoseRec - Analisi XAI (Permutation Importance) ===\n")

    X_test, y_test, feature_names = _load_test_data()
    svm, mlp, scaler = _load_models()

    # --- SVM ---
    print("Calcolo importanza feature per l'SVM...")
    svm_mean, svm_std, svm_base = permutation_importance(
        lambda X, y: _accuracy_svm(svm, scaler, X, y),
        X_test, y_test, feature_names,
    )
    _plot_importance(feature_names, svm_mean, svm_std, "SVM",
                     RESULTS_DIR / "feature_importance_svm.png")

    # --- MLP ---
    print("Calcolo importanza feature per l'MLP...")
    mlp_mean, mlp_std, mlp_base = permutation_importance(
        lambda X, y: _accuracy_mlp(mlp, scaler, X, y),
        X_test, y_test, feature_names,
    )
    _plot_importance(feature_names, mlp_mean, mlp_std, "MLP",
                     RESULTS_DIR / "feature_importance_mlp.png")

    # --- Report testuale ---
    report_path = RESULTS_DIR / "feature_importance.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BodyPoseRec - Permutation Feature Importance\n")
        f.write("=" * 60 + "\n\n")
        f.write("L'importanza e' il calo di accuratezza (media su "
                f"{N_REPEATS} permutazioni) quando la feature viene "
                "mescolata casualmente.\n\n")

        for label, means, base in [("SVM", svm_mean, svm_base),
                                   ("MLP", mlp_mean, mlp_base)]:
            f.write(f"--- {label} (accuratezza baseline = {base:.4f}) ---\n")
            order = np.argsort(means)[::-1]
            for rank, idx in enumerate(order, 1):
                f.write(f"  {rank:2d}. {feature_names[idx]:24s} "
                        f"importanza = {means[idx]:+.4f}\n")
            f.write("\n")

    # Riepilogo a console: le 5 feature piu' importanti per ciascun modello.
    for label, means in [("SVM", svm_mean), ("MLP", mlp_mean)]:
        top5 = np.argsort(means)[::-1][:5]
        print(f"\nTop-5 feature piu' importanti ({label}):")
        for rank, idx in enumerate(top5, 1):
            print(f"  {rank}. {feature_names[idx]:24s} "
                  f"({means[idx]:+.4f})")

    print(f"\nGrafici salvati in: {RESULTS_DIR}")
    print(f"Report salvato in : {report_path}")


if __name__ == "__main__":
    run_xai_analysis()
