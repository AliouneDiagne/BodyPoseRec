"""
pose_scoring.py
===============
Punteggio di qualita' dell'esecuzione di una posa (0-100).

Questo modulo realizza uno dei miglioramenti consigliati dal blueprint:
oltre a RICONOSCERE quale delle 7 pose sta eseguendo l'atleta, ne valuta
quanto bene la sta eseguendo. Il sistema passa cosi' da semplice
classificatore a strumento di coaching.

Come funziona
-------------
1. build_reference_poses.py ha precalcolato, per ogni posa, i valori
   angolari "ideali" (prototipo pulito) in models/reference_poses.json.
2. Dato lo scheletro dell'atleta, si estraggono le sue feature angolari.
3. Si confronta ogni angolo dell'atleta con il corrispondente angolo
   ideale: piccoli scarti -> punteggio alto, grandi scarti -> punteggio basso.
4. Il punteggio complessivo e' la media dei punteggi dei singoli angoli;
   vengono inoltre restituiti dei suggerimenti testuali sugli angoli peggiori.

La conversione scarto -> punteggio usa un decadimento esponenziale: e'
morbido (nessun salto netto) e limitato in [0, 100].

Riferimenti didattici:
- Blueprint BodyPoseRec, sezione "Miglioramento: Pose Scoring".
- Modulo 2, feature geometriche come descrittori interpretabili.
"""

import json

import numpy as np

from src.config import REFERENCE_POSES_PATH, display_name
from src.feature_extraction.feature_engineering import extract_features, FEATURE_NAMES

# Tolleranza angolare (in gradi): uno scarto pari a TOLERANCE_DEG rispetto
# all'angolo ideale fa scendere il punteggio del singolo angolo a circa il
# 37% (e^-1). Valore scelto come compromesso ragionevole per il bodybuilding,
# dove anche pochi gradi contano ma serve tolleranza alla variabilita' umana.
TOLERANCE_DEG = 25.0

# Nomi "leggibili" degli angoli, per i suggerimenti mostrati all'utente.
ANGLE_LABELS = {
    "elbow_angle_max":        "apertura gomito (braccio piu' esteso)",
    "elbow_angle_min":        "apertura gomito (braccio piu' flesso)",
    "shoulder_abduction_max": "abduzione spalla (lato piu' alto)",
    "shoulder_abduction_min": "abduzione spalla (lato piu' basso)",
    "knee_angle_max":         "flessione ginocchio (gamba piu' tesa)",
    "knee_angle_min":         "flessione ginocchio (gamba piu' piegata)",
    "hip_angle_max":          "angolo dell'anca (lato piu' aperto)",
    "hip_angle_min":          "angolo dell'anca (lato piu' chiuso)",
    "trunk_inclination":      "inclinazione del tronco",
    "shoulder_line_tilt":     "inclinazione della linea delle spalle",
}


def load_reference_poses(path=REFERENCE_POSES_PATH):
    """Carica il dizionario dei prototipi di riferimento da JSON.

    Solleva FileNotFoundError con un messaggio chiaro se il file non esiste
    ancora (va generato prima con build_reference_poses.py).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"File dei prototipi non trovato: {path}\n"
            "Generalo prima con:\n"
            "    python -m src.feature_extraction.build_reference_poses"
        ) from exc


def _angle_score(observed, reference):
    """Converte lo scarto angolare in un punteggio 0-100 (decadimento esponenziale).

    score = 100 * exp( -|observed - reference| / TOLERANCE_DEG )

    Scarto 0 gradi   -> 100 punti.
    Scarto 25 gradi  -> ~37 punti.
    Scarto 50 gradi  -> ~13 punti.
    """
    error = abs(observed - reference)
    return 100.0 * float(np.exp(-error / TOLERANCE_DEG))


class PoseQualityScorer:
    """Assegna un punteggio di qualita' confrontando l'atleta col prototipo."""

    def __init__(self, reference_path=REFERENCE_POSES_PATH):
        self.reference = load_reference_poses(reference_path)

    def score(self, keypoints, class_name):
        """Valuta la qualita' di una posa.

        Parametri
        ---------
        keypoints : np.ndarray (33, 4)
            Scheletro dell'atleta (sintetico o da MediaPipe).
        class_name : str
            Nome interno della posa riconosciuta dal classificatore
            (es. "side_chest"). Determina quale prototipo usare.

        Ritorna
        -------
        dict con:
            overall_score : punteggio complessivo 0-100 (media degli angoli).
            per_angle     : dizionario {nome_angolo: punteggio}.
            suggestions   : lista di stringhe con i consigli di correzione
                            (ordinati dal problema piu' grave).
        """
        if class_name not in self.reference:
            raise ValueError(f"Posa sconosciuta nei riferimenti: {class_name}")

        ref_angles = self.reference[class_name]
        observed = extract_features(keypoints)

        per_angle = {}
        deviations = {}  # scarto in gradi, per ordinare i suggerimenti
        for name, ref_value in ref_angles.items():
            idx = FEATURE_NAMES.index(name)
            obs_value = float(observed[idx])
            per_angle[name] = _angle_score(obs_value, ref_value)
            deviations[name] = abs(obs_value - ref_value)

        overall = float(np.mean(list(per_angle.values())))

        # Suggerimenti: prendiamo i tre angoli con scarto maggiore, ma solo
        # se il loro punteggio e' sotto 75 (altrimenti la posa va gia' bene).
        worst = sorted(deviations.items(), key=lambda kv: kv[1], reverse=True)
        suggestions = []
        for name, dev in worst[:3]:
            if per_angle[name] < 75.0:
                label = ANGLE_LABELS.get(name, name)
                suggestions.append(
                    f"Correggi la {label}: scarto di {dev:.0f} gradi "
                    f"rispetto al riferimento."
                )
        if not suggestions:
            suggestions.append("Esecuzione molto vicina al modello di riferimento.")

        return {
            "overall_score": overall,
            "per_angle": per_angle,
            "suggestions": suggestions,
        }


def quality_label(score):
    """Traduce un punteggio numerico in un giudizio qualitativo testuale."""
    if score >= 85:
        return "Eccellente"
    if score >= 70:
        return "Buona"
    if score >= 50:
        return "Sufficiente"
    return "Da migliorare"


if __name__ == "__main__":
    # Demo rapida: valuta un prototipo pulito (deve dare ~100) e un
    # prototipo a cui aggiungiamo rumore (deve dare un punteggio piu' basso).
    from src.data_preparation.synthetic_pose_generator import (
        generate_clean_prototype, generate_sample,
    )

    scorer = PoseQualityScorer()
    print("=== Test del Pose Quality Scorer ===\n")
    for class_name in ["front_double_biceps", "side_chest", "back_lat_spread"]:
        clean = generate_clean_prototype(class_name)
        result = scorer.score(clean, class_name)
        print(f"{display_name(class_name)} (prototipo pulito)")
        print(f"  Punteggio: {result['overall_score']:.1f}/100 "
              f"-> {quality_label(result['overall_score'])}")
        rng = np.random.default_rng(0)
        noisy = generate_sample(class_name, rng, allow_flip=False)
        result_noisy = scorer.score(noisy, class_name)
        print(f"  Con variazione realistica: "
              f"{result_noisy['overall_score']:.1f}/100 "
              f"-> {quality_label(result_noisy['overall_score'])}")
        for s in result_noisy["suggestions"]:
            print(f"    - {s}")
        print()
