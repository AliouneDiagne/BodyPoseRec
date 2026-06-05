"""
metrics.py
==========
Utility per la valutazione delle prestazioni dei classificatori.

Raccoglie le funzioni che producono le metriche richieste esplicitamente
dal documento d'esame ("Performance Evaluation"): matrice di confusione,
classification report (precision / recall / F1) e curve ROC con AUC.

Lo stile ricalca il lab del Modulo 3 (lab-m03-14-4.py), che usa
scikit-learn per le metriche e seaborn per le heatmap.

Riferimenti didattici:
- Modulo 3, lab-m03-14-4.py (confusion_matrix, classification_report,
  roc_curve, roc_auc_score, heatmap seaborn).
- Blueprint BodyPoseRec, sezione "Metodologie di Valutazione delle
  Prestazioni" (Macro-F1 per dataset sbilanciati, ROC/AUC).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend non interattivo: salva su file senza display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc, f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import label_binarize


def compute_summary_metrics(y_true, y_pred):
    """Calcola le metriche scalari principali in un unico dizionario.

    Usiamo la media 'macro' (media aritmetica tra le classi, senza pesare
    per la numerosita') perche', come spiega il blueprint, costringe il
    modello a comportarsi bene anche sulle pose meno rappresentate.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_classification_report(y_true, y_pred, class_names, out_path,
                                extra_header=""):
    """Genera il classification report testuale e lo salva su file.

    Il report per-classe (One-vs-Rest) mostra precision, recall e F1 di
    ciascuna posa: e' lo strumento per capire QUALI pose il modello
    gestisce bene e quali no.
    """
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, digits=4
    )
    summary = compute_summary_metrics(y_true, y_pred)

    with open(out_path, "w", encoding="utf-8") as f:
        if extra_header:
            f.write(extra_header + "\n")
            f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy globale : {summary['accuracy']:.4f}\n")
        f.write(f"Macro Precision  : {summary['macro_precision']:.4f}\n")
        f.write(f"Macro Recall     : {summary['macro_recall']:.4f}\n")
        f.write(f"Macro F1-score   : {summary['macro_f1']:.4f}\n\n")
        f.write("Classification report per-classe (One-vs-Rest):\n\n")
        f.write(report)

    return report, summary


def plot_confusion_matrix(y_true, y_pred, class_names, out_path,
                          title="Confusion Matrix", cmap="Blues"):
    """Disegna e salva la matrice di confusione come heatmap.

    Valori alti sulla diagonale = classificazioni corrette. Valori alti
    fuori diagonale rivelano le confusioni sistematiche (es. Front Lat
    Spread scambiato per Back Lat Spread).
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "Numero di campioni"})
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("Classe reale (Ground Truth)")
    plt.xlabel("Classe predetta")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    return cm


def plot_normalized_confusion_matrix(y_true, y_pred, class_names, out_path,
                                     title="Confusion Matrix (normalizzata)"):
    """Variante della matrice di confusione normalizzata per riga.

    Ogni riga somma a 1: mostra la *percentuale* di campioni di una classe
    classificati in ciascuna categoria. Utile quando le classi non sono
    perfettamente bilanciate.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    # Normalizzazione riga per riga (proteggendo da righe tutte a zero).
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                vmin=0.0, vmax=1.0,
                cbar_kws={"label": "Frazione di campioni"})
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("Classe reale (Ground Truth)")
    plt.xlabel("Classe predetta")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    return cm_norm


def plot_roc_curves(y_true, y_score, class_names, out_dir, model_tag=""):
    """Disegna una curva ROC One-vs-Rest per ciascuna delle 7 classi.

    La ROC plotta True Positive Rate contro False Positive Rate al variare
    della soglia decisionale. L'area sotto la curva (AUC) misura il potere
    discriminativo: 1.0 = perfetto, 0.5 = caso casuale.

    Parametri
    ---------
    y_true : array (N,)
        Etichette vere intere.
    y_score : array (N, num_classes)
        Probabilita' / punteggi predetti per ciascuna classe.
    """
    n_classes = len(class_names)
    # Binarizza le etichette: y_true_bin[:, k] = 1 se il campione e' classe k.
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    auc_scores = {}

    # --- Una figura per ogni classe ---
    for k in range(n_classes):
        # Se in test la classe k non comparisse mai, salta (ROC non definita).
        if y_true_bin[:, k].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, k], y_score[:, k])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_names[k]] = roc_auc

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--",
                 label="Classificatore casuale")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC - {class_names[k]}", fontsize=12, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fname = f"{model_tag}_roc_class_{k}.png" if model_tag else f"roc_class_{k}.png"
        plt.savefig(out_dir / fname, dpi=120)
        plt.close()

    # --- Figura riassuntiva con tutte le curve sovrapposte ---
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap("tab10", n_classes)
    for k in range(n_classes):
        if y_true_bin[:, k].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, k], y_score[:, k])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.8, color=cmap(k),
                 label=f"{class_names[k]} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    title = "Curve ROC One-vs-Rest"
    if model_tag:
        title += f" - {model_tag.upper()}"
    plt.title(title, fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    summary_name = f"{model_tag}_roc_all.png" if model_tag else "roc_all.png"
    plt.savefig(out_dir / summary_name, dpi=120)
    plt.close()

    # AUC media (macro): un singolo numero che riassume il potere
    # discriminativo del modello su tutte le classi.
    macro_auc = float(np.mean(list(auc_scores.values()))) if auc_scores else 0.0
    auc_scores["MACRO_AVG"] = macro_auc
    return auc_scores


def plot_training_history(history_dict, out_path):
    """Disegna le curve di loss e accuracy dell'addestramento dell'MLP.

    Confrontare training e validation permette di diagnosticare l'overfitting:
    se la validation loss risale mentre la training loss continua a scendere,
    il modello sta memorizzando il rumore.

    Parametri
    ---------
    history_dict : dict
        Dizionario history.history restituito da model.fit() di Keras.
    """
    plt.figure(figsize=(12, 5))

    # --- Pannello sinistro: accuracy ---
    plt.subplot(1, 2, 1)
    plt.plot(history_dict["accuracy"], label="Training accuracy", lw=2)
    if "val_accuracy" in history_dict:
        plt.plot(history_dict["val_accuracy"], label="Validation accuracy", lw=2)
    plt.title("Accuracy durante l'addestramento", fontweight="bold")
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    # --- Pannello destro: loss ---
    plt.subplot(1, 2, 2)
    plt.plot(history_dict["loss"], label="Training loss", lw=2)
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="Validation loss", lw=2)
    plt.title("Loss durante l'addestramento", fontweight="bold")
    plt.xlabel("Epoca")
    plt.ylabel("Categorical Cross-Entropy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
