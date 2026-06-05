"""
feature_engineering.py
======================
Trasformazione dei landmark grezzi in un vettore di feature geometriche.

Questo modulo implementa il cuore concettuale del progetto: invece di dare
a SVM/MLP le coordinate grezze dei landmark (che porterebbero a overfitting,
come spiega il blueprint), costruiamo un vettore di descrittori geometrici
"semanticamente esplicito" e spazialmente invariante.

Il vettore di feature finale combina quattro famiglie di descrittori:
  1. ANGOLI ARTICOLARI  - gomiti, spalle, anche, ginocchia, inclinazione
                          tronco. Sono la codifica piu' potente della postura.
  2. DISTANZE NORMALIZZATE - distanza tra polsi, tra gomiti, mani-fianchi,
                          tra caviglie. Catturano apertura/chiusura.
  3. RAPPORTI ADIMENSIONALI - larghezza spalle/bacino (V-taper), ecc.
  4. PROFONDITA' E VISIBILITA' - statistiche sull'asse z e sulla visibilita',
                          fondamentali per distinguere pose frontali da dorsali.

INVARIANZA ASSIALE (sinistra/destra)
------------------------------------
Come richiesto dal blueprint, le feature speculari (es. angolo gomito sx vs
dx) NON vengono inserite in ordine fisso, ma aggregate con operatori
simmetrici max() e min(). Cosi' il vettore resta identico se l'immagine
viene specchiata: il modello non deve imparare due volte la stessa posa.

Riferimenti didattici:
- Blueprint BodyPoseRec, sezione "Estrazione Topologica e Ingegnerizzazione
  delle Feature" (angoli, distanze, rapporti, invarianza assiale).
"""

import numpy as np

from src.config import (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, NOSE,
)
from src.utils.geometry import (
    angle_between, euclidean, midpoint, normalize_skeleton,
    vector_angle_with_vertical, safe_ratio,
)


# ===========================================================================
# 1. NOMI DELLE FEATURE
# ===========================================================================
# L'ordine di questa lista DEFINISCE l'ordine delle colonne nel vettore di
# feature. E' fondamentale che extract_features() produca i valori
# esattamente in questo ordine: i nomi servono poi all'analisi XAI per dire
# "quale feature ha pesato di piu'".
FEATURE_NAMES = [
    # --- Angoli articolari aggregati (invarianti al lato) ---
    "elbow_angle_max",        # gomito piu' esteso
    "elbow_angle_min",        # gomito piu' flesso
    "shoulder_abduction_max", # abduzione spalla massima
    "shoulder_abduction_min", # abduzione spalla minima
    "knee_angle_max",         # ginocchio piu' esteso
    "knee_angle_min",         # ginocchio piu' flesso
    "hip_angle_max",          # angolo anca massimo
    "hip_angle_min",          # angolo anca minimo
    # --- Angoli del tronco ---
    "trunk_inclination",      # inclinazione del tronco rispetto alla verticale
    "shoulder_line_tilt",     # inclinazione della linea delle spalle
    # --- Distanze normalizzate (invarianti, gia' divise per H_torso) ---
    "wrist_distance",         # distanza tra i due polsi
    "elbow_distance",         # distanza tra i due gomiti
    "ankle_distance",         # distanza tra le due caviglie
    "hand_to_hip_max",        # mano-fianco: lato piu' lontano
    "hand_to_hip_min",        # mano-fianco: lato piu' vicino
    "wrist_height_max",       # altezza del polso piu' alto (rispetto al tronco)
    "wrist_height_min",       # altezza del polso piu' basso
    # --- Rapporti adimensionali ---
    "shoulder_hip_ratio",     # V-taper: larghezza spalle / larghezza bacino
    "arm_span_ratio",         # apertura braccia / altezza tronco
    "stance_width_ratio",     # larghezza stance / larghezza bacino
    # --- Profondita' e orientamento (asse z) ---
    "shoulder_depth_spread",  # |z spalla sx - z spalla dx|: alto = profilo
    "hip_depth_spread",       # |z anca sx - z anca dx|
    "wrist_depth_mean",       # z medio dei polsi: segno = davanti/dietro
    "body_depth_range",       # escursione totale lungo z
    # --- Visibilita' ---
    "nose_visibility",        # bassa nelle pose di schiena
    "core_visibility_mean",   # visibilita' media dei landmark core
]

NUM_FEATURES = len(FEATURE_NAMES)   # dimensione del vettore di feature


# ===========================================================================
# 2. ESTRAZIONE DEL VETTORE DI FEATURE
# ===========================================================================
def extract_features(keypoints):
    """Trasforma un array di 33 landmark nel vettore di feature geometriche.

    Parametri
    ---------
    keypoints : np.ndarray, shape (33, 4)
        Landmark grezzi in formato MediaPipe [x, y, z, visibility], con
        x, y in [0, 1].

    Ritorna
    -------
    np.ndarray, shape (NUM_FEATURES,)
        Il vettore di feature, nello stesso ordine di FEATURE_NAMES.
    """
    kps = np.asarray(keypoints, dtype=np.float64)

    # --- Passo 1: normalizzazione spaziale ---
    # Rende lo scheletro invariante a traslazione e scala. Da qui in poi
    # lavoriamo su coordinate ricentrate sul bacino e scalate sul torso.
    norm, torso_h = normalize_skeleton(kps)

    # Estraiamo le coordinate (x, y) dei landmark core: per gli angoli nel
    # piano immagine usiamo le 2D; per la profondita' useremo la z separata.
    def p2d(idx):
        return norm[idx, :2]

    # --- Passo 2: punti notevoli derivati ---
    pelvis = midpoint(p2d(L_HIP), p2d(R_HIP))
    shoulder_center = midpoint(p2d(L_SHOULDER), p2d(R_SHOULDER))

    features = []

    # ----- FAMIGLIA 1: ANGOLI ARTICOLARI -----
    # Gomito = angolo spalla-gomito-polso. Invarianza assiale: max e min
    # tra lato sinistro e destro (operatori simmetrici).
    elbow_l = angle_between(p2d(L_SHOULDER), p2d(L_ELBOW), p2d(L_WRIST))
    elbow_r = angle_between(p2d(R_SHOULDER), p2d(R_ELBOW), p2d(R_WRIST))
    features.append(max(elbow_l, elbow_r))   # elbow_angle_max
    features.append(min(elbow_l, elbow_r))   # elbow_angle_min

    # Abduzione della spalla = angolo anca-spalla-gomito (quanto e' alzato
    # il braccio rispetto al tronco).
    abd_l = angle_between(p2d(L_HIP), p2d(L_SHOULDER), p2d(L_ELBOW))
    abd_r = angle_between(p2d(R_HIP), p2d(R_SHOULDER), p2d(R_ELBOW))
    features.append(max(abd_l, abd_r))       # shoulder_abduction_max
    features.append(min(abd_l, abd_r))       # shoulder_abduction_min

    # Ginocchio = angolo anca-ginocchio-caviglia (gamba estesa o flessa).
    knee_l = angle_between(p2d(L_HIP), p2d(L_KNEE), p2d(L_ANKLE))
    knee_r = angle_between(p2d(R_HIP), p2d(R_KNEE), p2d(R_ANKLE))
    features.append(max(knee_l, knee_r))     # knee_angle_max
    features.append(min(knee_l, knee_r))     # knee_angle_min

    # Anca = angolo spalla-anca-ginocchio (flessione dell'anca).
    hip_l = angle_between(p2d(L_SHOULDER), p2d(L_HIP), p2d(L_KNEE))
    hip_r = angle_between(p2d(R_SHOULDER), p2d(R_HIP), p2d(R_KNEE))
    features.append(max(hip_l, hip_r))       # hip_angle_max
    features.append(min(hip_l, hip_r))       # hip_angle_min

    # ----- ANGOLI DEL TRONCO -----
    # Inclinazione del tronco: angolo del segmento pelvi->centro spalle
    # rispetto alla verticale.
    features.append(vector_angle_with_vertical(pelvis, shoulder_center))

    # Inclinazione della linea delle spalle: angolo del segmento che unisce
    # le due spalle rispetto all'orizzontale. Lo otteniamo come 90 - angolo
    # con la verticale, in valore assoluto.
    sh_tilt = abs(90.0 - vector_angle_with_vertical(p2d(L_SHOULDER),
                                                    p2d(R_SHOULDER)))
    features.append(sh_tilt)

    # ----- FAMIGLIA 2: DISTANZE NORMALIZZATE -----
    # (norm e' gia' scalato su H_torso, quindi le distanze sono adimensionali)
    features.append(euclidean(p2d(L_WRIST), p2d(R_WRIST)))     # wrist_distance
    features.append(euclidean(p2d(L_ELBOW), p2d(R_ELBOW)))     # elbow_distance
    features.append(euclidean(p2d(L_ANKLE), p2d(R_ANKLE)))     # ankle_distance

    # Distanza mano-fianco (proiezione): quanto la mano e' lontana dal
    # proprio fianco. Bassa nei Lat Spread (mani sui fianchi).
    h2h_l = euclidean(p2d(L_WRIST), p2d(L_HIP))
    h2h_r = euclidean(p2d(R_WRIST), p2d(R_HIP))
    features.append(max(h2h_l, h2h_r))       # hand_to_hip_max
    features.append(min(h2h_l, h2h_r))       # hand_to_hip_min

    # Altezza dei polsi: la coordinata y (ricordando che y cresce verso il
    # basso, la neghiamo cosi' "alto" = valore grande). Distingue le pose a
    # braccia alte (double biceps, abs) da quelle a braccia basse (lat spread).
    wrist_h_l = -norm[L_WRIST, 1]
    wrist_h_r = -norm[R_WRIST, 1]
    features.append(max(wrist_h_l, wrist_h_r))   # wrist_height_max
    features.append(min(wrist_h_l, wrist_h_r))   # wrist_height_min

    # ----- FAMIGLIA 3: RAPPORTI ADIMENSIONALI -----
    shoulder_width = euclidean(p2d(L_SHOULDER), p2d(R_SHOULDER))
    hip_width = euclidean(p2d(L_HIP), p2d(R_HIP))
    features.append(safe_ratio(shoulder_width, hip_width))   # shoulder_hip_ratio

    # Apertura delle braccia (polso-polso) rapportata all'altezza del tronco.
    arm_span = euclidean(p2d(L_WRIST), p2d(R_WRIST))
    features.append(safe_ratio(arm_span, 1.0))   # arm_span_ratio (torso=1)

    # Larghezza dello stance rispetto al bacino.
    stance_width = euclidean(p2d(L_ANKLE), p2d(R_ANKLE))
    features.append(safe_ratio(stance_width, hip_width))   # stance_width_ratio

    # ----- FAMIGLIA 4: PROFONDITA' E ORIENTAMENTO (asse z) -----
    # Differenza di profondita' tra le spalle: nelle pose laterali una
    # spalla e' molto piu' avanti dell'altra, quindi questo valore e' alto.
    features.append(abs(norm[L_SHOULDER, 2] - norm[R_SHOULDER, 2]))  # shoulder_depth_spread
    features.append(abs(norm[L_HIP, 2] - norm[R_HIP, 2]))            # hip_depth_spread

    # z medio dei polsi: il segno indica se le mani sono davanti (z<0) o
    # dietro (z>0) il corpo -> distingue Side Chest da Side Triceps.
    features.append((norm[L_WRIST, 2] + norm[R_WRIST, 2]) / 2.0)     # wrist_depth_mean

    # Escursione totale lungo z dei landmark core: pose laterali hanno un
    # range di profondita' molto maggiore delle frontali.
    core_z = norm[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_WRIST, R_WRIST,
                   L_ANKLE, R_ANKLE], 2]
    features.append(float(core_z.max() - core_z.min()))              # body_depth_range

    # ----- FAMIGLIA 5: VISIBILITA' -----
    # La visibility del naso crolla nelle pose di schiena: e' un indizio
    # potente per separare le 2 pose frontali dalle 2 dorsali.
    features.append(float(kps[NOSE, 3]))                             # nose_visibility

    core_idx = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
                L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
    features.append(float(np.mean(kps[core_idx, 3])))                # core_visibility_mean

    feature_vector = np.array(features, dtype=np.float32)

    # Verifica di coerenza: il numero di feature prodotte deve corrispondere
    # esattamente ai nomi dichiarati. Se fallisce, c'e' un bug da correggere.
    assert feature_vector.shape[0] == NUM_FEATURES, (
        f"Mismatch feature: prodotte {feature_vector.shape[0]}, "
        f"attese {NUM_FEATURES}"
    )
    return feature_vector


def extract_features_batch(keypoints_array):
    """Applica extract_features a un intero batch di scheletri.

    Parametri
    ---------
    keypoints_array : np.ndarray, shape (N, 33, 4)

    Ritorna
    -------
    np.ndarray, shape (N, NUM_FEATURES)
    """
    return np.array([extract_features(kp) for kp in keypoints_array],
                    dtype=np.float32)


if __name__ == "__main__":
    # Test: genera un campione per ogni posa e stampa il vettore di feature.
    from src.data_preparation.synthetic_pose_generator import generate_sample
    from src.config import POSE_CLASSES

    rng = np.random.default_rng(0)
    print(f"Numero di feature per campione: {NUM_FEATURES}\n")
    for name in POSE_CLASSES:
        sample = generate_sample(name, rng, allow_flip=False)
        feats = extract_features(sample)
        print(f"{name:24s} -> feature vector shape {feats.shape}, "
              f"primi 4 valori: {np.round(feats[:4], 2)}")
    print("\nFeature engineering funzionante.")
