"""
evaluate.py
===========
Valutazione comparativa dei due modelli sul test set "blind".

Questo script misura le prestazioni finali di SVM e MLP sul TEST SET, il
sottoinsieme di dati che non e' mai stato usato ne' per l'addestramento
ne' per il tuning. E' l'unico modo per ottenere una stima onesta e non
distorta della capacita' di generalizzazione.

Per ciascun modello produce tutti gli artefatti di valutazione richiesti
dal documento d'esame ("Performance Evaluation"):
  - accuracy, precision, recall, F1 (macro);
  - classification report per-classe;
  - matrice di confusione (assoluta e normalizzata);
  - curve ROC One-vs-Rest con AUC.

Genera infine un confronto diretto SVM vs MLP, utile per la sezione
"Experimental Results" del documento tecnico.

Uso da terminale:
    python -m src.classification.evaluate

Riferimenti didattici:
- Modulo 3, lab-m03-14-4.py (confusion matrix, classification report, ROC).
- Blueprint BodyPoseRec, sezione "Metodologie di Valutazione".
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import pickle
import numpy as np

from src.config import (
    ensure_dirs, FEATURES_DIR, RESULTS_DIR, ROC_DIR,
    SVM_MODEL_PATH, MLP_MODEL_PATH, SCALER_PATH,
    POSE_CLASSES, POSE_DISPLAY_NAMES, NUM_CLASSES,
)
from src.utils.metrics import (
    compute_summary_metrics, save_classification_report,
    plot_confusion_matrix, plot_normalized_confusion_matrix, plot_roc_curves,
)

# Nomi leggibili delle classi, nell'ordine corretto, per grafici e report.
DISPLAY_NAMES = [POSE_DISPLAY_NAMES[c] for c in POSE_CLASSES]


def load_test_set():
    """Carica feature ed etichette del test set."""
    X_test = np.load(FEATURES_DIR / "X_test.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")
    return X_test, y_test


def load_scaler():
    """Carica lo StandardScaler addestrato."""
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


def evaluate_svm(X_test_scaled, y_test):
    """Valuta l'SVM sul test set. Ritorna (predizioni, probabilita', metriche)."""
    print("\n--- Valutazione SVM ---")
    with open(SVM_MODEL_PATH, "rb") as f:
        svm = pickle.load(f)

    y_pred = svm.predict(X_test_scaled)
    y_prob = svm.predict_proba(X_test_scaled)
    metrics = compute_summary_metrics(y_test, y_pred)

    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")

    # Classification report per-classe.
    save_classification_report(
        y_test, y_pred, DISPLAY_NAMES,
        RESULTS_DIR / "classification_report_svm.txt",
        extra_header="BodyPoseRec - Classification Report SVM (test set)",
    )
    # Matrici di confusione.
    plot_confusion_matrix(
        y_test, y_pred, DISPLAY_NAMES,
        RESULTS_DIR / "confusion_matrix_svm.png",
        title="Matrice di Confusione - SVM (test set)",
    )
    plot_normalized_confusion_matrix(
        y_test, y_pred, DISPLAY_NAMES,
        RESULTS_DIR / "confusion_matrix_svm_normalized.png",
        title="Matrice di Confusione normalizzata - SVM",
    )
    # Curve ROC One-vs-Rest.
    auc_scores = plot_roc_curves(y_test, y_prob, DISPLAY_NAMES, ROC_DIR,
                                 model_tag="svm")
    metrics["macro_auc"] = auc_scores["MACRO_AVG"]
    print(f"  Macro AUC: {metrics['macro_auc']:.4f}")

    return y_pred, y_prob, metrics


def evaluate_mlp(X_test_scaled, y_test):
    """Valuta l'MLP sul test set. Ritorna (predizioni, probabilita', metriche)."""
    print("\n--- Valutazione MLP ---")
    # Import locale di TensorFlow: solo qui serve.
    from tensorflow import keras

    mlp = keras.models.load_model(MLP_MODEL_PATH)
    y_prob = mlp.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = compute_summary_metrics(y_test, y_pred)

    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")

    save_classification_report(
        y_test, y_pred, DISPLAY_NAMES,
        RESULTS_DIR / "classification_report_mlp.txt",
        extra_header="BodyPoseRec - Classification Report MLP (test set)",
    )
    plot_confusion_matrix(
        y_test, y_pred, DISPLAY_NAMES,
        RESULTS_DIR / "confusion_matrix_mlp.png",
        title="Matrice di Confusione - MLP (test set)",
        cmap="Greens",
    )
    plot_normalized_confusion_matrix(
        y_test, y_pred, DISPLAY_NAMES,
        RESULTS_DIR / "confusion_matrix_mlp_normalized.png",
        title="Matrice di Confusione normalizzata - MLP",
    )
    auc_scores = plot_roc_curves(y_test, y_prob, DISPLAY_NAMES, ROC_DIR,
                                 model_tag="mlp")
    metrics["macro_auc"] = auc_scores["MACRO_AVG"]
    print(f"  Macro AUC: {metrics['macro_auc']:.4f}")

    return y_pred, y_prob, metrics


def write_comparison(svm_metrics, mlp_metrics):
    """Scrive il confronto diretto SVM vs MLP in un file riassuntivo."""
    comparison_path = RESULTS_DIR / "model_comparison.txt"
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write("BodyPoseRec - Confronto modelli sul test set\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Metrica':<22s} {'SVM':>12s} {'MLP':>12s}\n")
        f.write("-" * 48 + "\n")
        for key, label in [("accuracy", "Accuracy"),
                           ("macro_precision", "Macro Precision"),
                           ("macro_recall", "Macro Recall"),
                           ("macro_f1", "Macro F1-score"),
                           ("macro_auc", "Macro AUC")]:
            sv = svm_metrics.get(key, 0.0)
            ml = mlp_metrics.get(key, 0.0)
            f.write(f"{label:<22s} {sv:>12.4f} {ml:>12.4f}\n")
        f.write("\n")
        # Indichiamo quale modello e' migliore in macro F1.
        if mlp_metrics["macro_f1"] > svm_metrics["macro_f1"]:
            f.write("Modello con Macro F1 piu' alto: MLP\n")
        elif svm_metrics["macro_f1"] > mlp_metrics["macro_f1"]:
            f.write("Modello con Macro F1 piu' alto: SVM\n")
        else:
            f.write("I due modelli hanno lo stesso Macro F1.\n")

    # Salviamo anche una versione JSON, comoda per il documento tecnico.
    json_path = RESULTS_DIR / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"svm": svm_metrics, "mlp": mlp_metrics}, f, indent=2)

    return comparison_path


def main():
    """Esegue la valutazione completa di entrambi i modelli."""
    ensure_dirs()
    print("=" * 65)
    print("  BodyPoseRec - Valutazione sul test set (blind)")
    print("=" * 65)

    # Caricamento test set e standardizzazione (stesso scaler dei training).
    X_test, y_test = load_test_set()
    scaler = load_scaler()
    X_test_scaled = scaler.transform(X_test)
    print(f"\nTest set: {X_test.shape[0]} campioni, "
          f"{X_test.shape[1]} feature ciascuno.")

    # Valutazione dei due modelli.
    _, _, svm_metrics = evaluate_svm(X_test_scaled, y_test)
    _, _, mlp_metrics = evaluate_mlp(X_test_scaled, y_test)

    # Confronto finale.
    comparison_path = write_comparison(svm_metrics, mlp_metrics)

    print("\n" + "=" * 65)
    print("  Riepilogo del confronto (test set)")
    print("=" * 65)
    print(f"{'Metrica':<22s} {'SVM':>12s} {'MLP':>12s}")
    print("-" * 48)
    for key, label in [("accuracy", "Accuracy"),
                       ("macro_precision", "Macro Precision"),
                       ("macro_recall", "Macro Recall"),
                       ("macro_f1", "Macro F1-score"),
                       ("macro_auc", "Macro AUC")]:
        print(f"{label:<22s} {svm_metrics[key]:>12.4f} "
              f"{mlp_metrics[key]:>12.4f}")

    print(f"\nTutti i grafici e i report sono in: {RESULTS_DIR}")
    print(f"Confronto salvato in: {comparison_path}")
    print("\nValutazione completata con successo.")


if __name__ == "__main__":
    main()
