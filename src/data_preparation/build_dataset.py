"""
build_dataset.py
================
Script di preparazione del dataset: genera, divide e salva su disco.

Questo e' il PRIMO script da eseguire nella pipeline. Svolge tre compiti:
  1. genera il dataset sintetico di keypoint (33 landmark per le 7 pose);
  2. calcola il vettore di feature geometriche per ogni campione;
  3. suddivide i dati in train / validation / test in modo stratificato e
     salva tutto su disco in formato .npy.

La suddivisione e' STRATIFICATA: ogni split mantiene la stessa proporzione
di classi del dataset completo. E' una regola d'oro del Modulo 3 (i lab
usano `train_test_split(..., stratify=...)`), perche' garantisce che il
test set sia rappresentativo e che la valutazione non sia falsata.

Il test set, una volta creato, NON deve mai essere usato per addestrare o
per il tuning: e' il set "blind" su cui si misurano le prestazioni finali.

Uso da terminale:
    python -m src.data_preparation.build_dataset

Riferimenti didattici:
- Modulo 3, lab-m03-15-x.py (pipeline progetto: load -> split -> train).
- Blueprint BodyPoseRec, cartella data/ (raw, keypoints, features, split).
"""

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    ensure_dirs, SAMPLES_PER_CLASS, RANDOM_SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    KEYPOINTS_DIR, FEATURES_DIR, SPLIT_DIR,
    POSE_CLASSES, NUM_CLASSES,
)
from src.data_preparation.synthetic_pose_generator import generate_dataset
from src.feature_extraction.feature_engineering import (
    extract_features_batch, NUM_FEATURES,
)


def build():
    """Esegue l'intera preparazione del dataset e salva i file su disco."""
    ensure_dirs()
    print("=" * 65)
    print("  BodyPoseRec - Preparazione del dataset sintetico")
    print("=" * 65)

    # -----------------------------------------------------------------
    # PASSO 1: generazione degli scheletri sintetici
    # -----------------------------------------------------------------
    print(f"\n[1/4] Generazione di {SAMPLES_PER_CLASS} campioni x "
          f"{NUM_CLASSES} classi...")
    X_kps, y = generate_dataset(SAMPLES_PER_CLASS, seed=RANDOM_SEED)
    print(f"      Totale campioni generati: {len(y)}")
    print(f"      Shape keypoints: {X_kps.shape}  (N, 33 landmark, 4 valori)")

    # -----------------------------------------------------------------
    # PASSO 2: estrazione delle feature geometriche
    # -----------------------------------------------------------------
    print(f"\n[2/4] Estrazione del vettore di feature geometriche...")
    X_feat = extract_features_batch(X_kps)
    print(f"      Shape feature: {X_feat.shape}  (N, {NUM_FEATURES} feature)")

    # -----------------------------------------------------------------
    # PASSO 3: split stratificato train / validation / test
    # -----------------------------------------------------------------
    print(f"\n[3/4] Suddivisione stratificata "
          f"({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})...")

    # Primo split: separiamo il TEST dal resto (train+val).
    # stratify=y garantisce la stessa proporzione di classi in ogni parte.
    idx_all = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=TEST_RATIO, stratify=y, random_state=RANDOM_SEED,
    )

    # Secondo split: dal train+val ricaviamo TRAIN e VALIDATION.
    # La frazione di validation va ricalcolata rispetto al sottoinsieme.
    val_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_relative, stratify=y[idx_trainval],
        random_state=RANDOM_SEED,
    )

    print(f"      Train      : {len(idx_train)} campioni")
    print(f"      Validation : {len(idx_val)} campioni")
    print(f"      Test       : {len(idx_test)} campioni")

    # -----------------------------------------------------------------
    # PASSO 4: salvataggio su disco
    # -----------------------------------------------------------------
    print(f"\n[4/4] Salvataggio dei file...")

    # Keypoint grezzi (per la demo, l'XAI e l'analisi qualitativa).
    np.save(KEYPOINTS_DIR / "keypoints_all.npy", X_kps)
    np.save(KEYPOINTS_DIR / "labels_all.npy", y)

    # Feature gia' calcolate, divise per split (input diretto di SVM e MLP).
    np.save(FEATURES_DIR / "X_train.npy", X_feat[idx_train])
    np.save(FEATURES_DIR / "y_train.npy", y[idx_train])
    np.save(FEATURES_DIR / "X_val.npy", X_feat[idx_val])
    np.save(FEATURES_DIR / "y_val.npy", y[idx_val])
    np.save(FEATURES_DIR / "X_test.npy", X_feat[idx_test])
    np.save(FEATURES_DIR / "y_test.npy", y[idx_test])

    # Salviamo anche i keypoint del test set: servono alla demo offline e
    # all'analisi degli errori (per ricostruire lo scheletro dei fallimenti).
    np.save(FEATURES_DIR / "X_test_keypoints.npy", X_kps[idx_test])

    # File .txt con gli indici di ogni split: documentano in modo umano e
    # verificabile la suddivisione (come la cartella dataset_split/ del
    # blueprint).
    np.savetxt(SPLIT_DIR / "train.txt", idx_train, fmt="%d")
    np.savetxt(SPLIT_DIR / "val.txt", idx_val, fmt="%d")
    np.savetxt(SPLIT_DIR / "test.txt", idx_test, fmt="%d")

    print(f"      Keypoint salvati in : {KEYPOINTS_DIR}")
    print(f"      Feature salvate in  : {FEATURES_DIR}")
    print(f"      Indici split in     : {SPLIT_DIR}")

    # -----------------------------------------------------------------
    # Riepilogo finale e verifica del bilanciamento
    # -----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Bilanciamento delle classi per split")
    print("=" * 65)
    print(f"{'Classe':24s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
    for class_id, class_name in enumerate(POSE_CLASSES):
        n_tr = int(np.sum(y[idx_train] == class_id))
        n_va = int(np.sum(y[idx_val] == class_id))
        n_te = int(np.sum(y[idx_test] == class_id))
        print(f"{class_name:24s} {n_tr:8d} {n_va:8d} {n_te:8d}")

    print("\nPreparazione del dataset completata con successo.")
    print("Prossimo passo: addestrare i modelli con")
    print("  python -m src.classification.train_svm")
    print("  python -m src.classification.train_mlp")


if __name__ == "__main__":
    build()
