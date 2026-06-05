"""
train_svm.py
============
Addestramento del classificatore classico: Support Vector Machine (SVM).

L'SVM rappresenta il modello "classico" richiesto dal documento d'esame
(la slide 2 chiede esplicitamente "classical and deep learning"). Le SVM
funzionano cercando l'iperpiano che separa le classi con il margine
massimo; grazie al kernel RBF possono gestire confini decisionali non
lineari, utili qui perche' alcune pose hanno geometrie simili (es. Front
e Back Double Biceps).

Questo script ricalca fedelmente la metodologia del lab del Modulo 3
(lab-m03-15-4.py):
  - StandardScaler per portare tutte le feature sulla stessa scala;
  - GridSearchCV per il tuning sistematico di C e gamma;
  - K-fold cross-validation per una stima non distorta;
  - valutazione finale su validation e salvataggio del modello.

PERCHE' LO STANDARDSCALER
-------------------------
Il vettore di feature mescola grandezze eterogenee: angoli in gradi
(0-180), distanze adimensionali (~0-2), visibilita' (0-1). Senza
standardizzazione le feature con valori grandi (gli angoli) dominerebbero
il calcolo delle distanze del kernel RBF. Lo scaler sottrae la media e
divide per la deviazione standard, mettendo tutte le feature "alla pari".
Lo scaler viene fittato SOLO sul train e poi riusato identico su val e
test, per non far "trapelare" informazioni del test nell'addestramento.

Uso da terminale:
    python -m src.classification.train_svm

Riferimenti didattici:
- Modulo 3, lab-m03-15-4.py (SVM + GridSearchCV + StandardScaler).
- Blueprint BodyPoseRec, sezione "Classificazione Statistica: SVM".
"""

import json
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from src.config import (
    ensure_dirs, FEATURES_DIR, MODELS_DIR, RESULTS_DIR,
    SVM_PARAM_GRID, SVM_CV_FOLDS, RANDOM_SEED,
    SVM_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
    POSE_CLASSES,
)
from src.feature_extraction.feature_engineering import FEATURE_NAMES
from src.utils.metrics import compute_summary_metrics


def load_split(split):
    """Carica feature ed etichette di uno split (train / val / test)."""
    X = np.load(FEATURES_DIR / f"X_{split}.npy")
    y = np.load(FEATURES_DIR / f"y_{split}.npy")
    return X, y


def train():
    """Addestra l'SVM con tuning degli iperparametri e salva gli artefatti."""
    ensure_dirs()
    print("=" * 65)
    print("  BodyPoseRec - Addestramento SVM (modello classico)")
    print("=" * 65)

    # -----------------------------------------------------------------
    # PASSO 1: caricamento dei dati
    # -----------------------------------------------------------------
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    print(f"\n[1/5] Dati caricati:")
    print(f"      Train: {X_train.shape}, Validation: {X_val.shape}")

    # -----------------------------------------------------------------
    # PASSO 2: standardizzazione delle feature
    # -----------------------------------------------------------------
    # Lo scaler "impara" media e deviazione standard SOLO dal train.
    print(f"\n[2/5] Standardizzazione delle feature (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # -----------------------------------------------------------------
    # PASSO 3: tuning degli iperparametri con GridSearchCV
    # -----------------------------------------------------------------
    # GridSearchCV prova ogni combinazione (C, gamma) della griglia e, per
    # ognuna, esegue una K-fold cross-validation sul train; sceglie poi la
    # combinazione con la miglior accuracy media. probability=True abilita
    # predict_proba (serve per le curve ROC e per lo smoothing temporale).
    print(f"\n[3/5] Tuning con GridSearchCV "
          f"({SVM_CV_FOLDS}-fold cross-validation)...")
    print(f"      Griglia di ricerca: {SVM_PARAM_GRID}")

    base_svm = SVC(probability=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        estimator=base_svm,
        param_grid=SVM_PARAM_GRID,
        cv=SVM_CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,        # usa tutti i core disponibili
        verbose=1,
    )
    grid.fit(X_train_scaled, y_train)

    best_svm = grid.best_estimator_
    print(f"\n      Migliori iperparametri trovati: {grid.best_params_}")
    print(f"      Accuracy media in cross-validation: "
          f"{grid.best_score_:.4f}")

    # -----------------------------------------------------------------
    # PASSO 4: valutazione sul validation set
    # -----------------------------------------------------------------
    print(f"\n[4/5] Valutazione sul validation set...")
    y_val_pred = best_svm.predict(X_val_scaled)
    val_metrics = compute_summary_metrics(y_val, y_val_pred)
    print(f"      Accuracy validation : {val_metrics['accuracy']:.4f}")
    print(f"      Macro F1 validation : {val_metrics['macro_f1']:.4f}")

    # -----------------------------------------------------------------
    # PASSO 5: salvataggio degli artefatti
    # -----------------------------------------------------------------
    # Salviamo tre cose: il modello SVM, lo scaler (indispensabile: la
    # demo deve standardizzare gli input nello stesso identico modo) e i
    # nomi delle feature (utili per l'XAI).
    print(f"\n[5/5] Salvataggio degli artefatti...")
    with open(SVM_MODEL_PATH, "wb") as f:
        pickle.dump(best_svm, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    print(f"      Modello SVM salvato : {SVM_MODEL_PATH}")
    print(f"      Scaler salvato      : {SCALER_PATH}")
    print(f"      Nomi feature salvati: {FEATURE_NAMES_PATH}")

    # Salviamo anche un piccolo log testuale dell'addestramento.
    log_path = RESULTS_DIR / "svm_training_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("BodyPoseRec - Log addestramento SVM\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Migliori iperparametri: {grid.best_params_}\n")
        f.write(f"Accuracy CV media: {grid.best_score_:.4f}\n\n")
        f.write(f"Accuracy validation: {val_metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision val: {val_metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall val   : {val_metrics['macro_recall']:.4f}\n")
        f.write(f"Macro F1 val       : {val_metrics['macro_f1']:.4f}\n")

    print(f"      Log salvato         : {log_path}")
    print("\nAddestramento SVM completato con successo.")
    return best_svm, scaler, grid.best_params_


if __name__ == "__main__":
    train()
