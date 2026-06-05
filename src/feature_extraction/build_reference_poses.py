"""
build_reference_poses.py
========================
Costruisce il file di riferimento models/reference_poses.json.

Per ognuna delle 7 pose genera lo scheletro "prototipo pulito" (senza
rumore ne' variazione antropometrica), ne estrae il vettore di feature e
salva i valori di riferimento. Questi valori rappresentano la posa eseguita
"alla perfezione" secondo i descrittori cinematici del blueprint.

Il file prodotto viene poi usato da pose_scoring.py per assegnare a un
atleta un punteggio di qualita' (0-100): si confrontano le sue feature
angolari con quelle del prototipo della posa riconosciuta.

Esecuzione:
    python -m src.feature_extraction.build_reference_poses
"""

import json

import numpy as np

from src.config import POSE_CLASSES, REFERENCE_POSES_PATH, ensure_dirs
from src.data_preparation.synthetic_pose_generator import generate_clean_prototype
from src.feature_extraction.feature_engineering import (
    extract_features, FEATURE_NAMES,
)

# Sottoinsieme di feature "interpretabili" usate per il punteggio di qualita'.
# Sono gli angoli articolari e le inclinazioni: grandezze biomeccaniche che
# un giudice di gara valuterebbe davvero. Le feature di profondita' o
# visibilita' non hanno senso come "obiettivo da raggiungere".
SCORING_FEATURES = [
    "elbow_angle_max", "elbow_angle_min",
    "shoulder_abduction_max", "shoulder_abduction_min",
    "knee_angle_max", "knee_angle_min",
    "hip_angle_max", "hip_angle_min",
    "trunk_inclination", "shoulder_line_tilt",
]


def build_reference_poses():
    """Genera e salva su disco il dizionario dei prototipi di riferimento."""
    ensure_dirs()
    reference = {}

    for class_name in POSE_CLASSES:
        # Scheletro prototipo pulito (deterministico, nessun rumore).
        prototype_kps = generate_clean_prototype(class_name)
        # Vettore di feature completo (26 dimensioni).
        feature_vector = extract_features(prototype_kps)

        # Salviamo solo le feature di scoring, indicizzate per nome cosi'
        # il file resta leggibile e robusto a future aggiunte di feature.
        feature_dict = {}
        for name in SCORING_FEATURES:
            idx = FEATURE_NAMES.index(name)
            feature_dict[name] = float(feature_vector[idx])

        reference[class_name] = feature_dict

    # Scrittura del JSON, indentato per leggibilita' umana.
    with open(REFERENCE_POSES_PATH, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2)

    print(f"Prototipi di riferimento salvati in: {REFERENCE_POSES_PATH}")
    print(f"Pose di riferimento generate: {len(reference)}")
    for class_name, feats in reference.items():
        angoli = ", ".join(f"{k}={v:.1f}" for k, v in list(feats.items())[:3])
        print(f"  {class_name:24s} | {angoli} ...")
    return reference


if __name__ == "__main__":
    build_reference_poses()
