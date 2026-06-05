"""
geometry.py
===========
Utility matematiche per l'analisi geometrica dei landmark corporei.

Questo modulo contiene le funzioni "atomiche" su cui si basa l'intero
feature engineering del progetto: calcolo di angoli articolari, distanze
euclidee e normalizzazione spaziale dello scheletro.

Tutte le funzioni operano su array NumPy e sono prive di stato (pure
functions): questo le rende facili da testare e riutilizzare sia in fase
di training sia in fase di inferenza real-time.

Riferimenti didattici:
- Blueprint BodyPoseRec, sezione "Calcolo dei Gradi di Liberta' e Angoli
  Articolari" (formula dell'angolo via prodotto scalare).
- Blueprint BodyPoseRec, sezione "Invarianza di Traslazione e
  Normalizzazione di Scala".
"""

import numpy as np

# Costante "epsilon": un numero piccolissimo che si somma ai denominatori
# per evitare divisioni per zero quando due vettori sono colineari o
# coincidenti. E' la stessa tecnica indicata nel blueprint.
EPS = 1e-8


def angle_between(a, b, c):
    """Calcola l'angolo (in gradi) nel vertice B del triangolo A-B-C.

    E' la forma piu' potente di codifica della postura: misura, ad esempio,
    quanto e' flesso un gomito (vertice = gomito, A = spalla, C = polso).

    Implementa la formula del blueprint:
        theta = arccos( (u . v) / (||u|| ||v|| + eps) ) * 180/pi
    dove u = A - B e v = C - B sono i due segmenti ossei uscenti dal vertice.

    Parametri
    ---------
    a, b, c : np.ndarray
        Coordinate dei tre punti (2D o 3D). B e' il vertice/fulcro.

    Ritorna
    -------
    float
        L'angolo in gradi, nell'intervallo [0, 180].
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    u = a - b   # segmento osseo B->A
    v = c - b   # segmento osseo B->C

    # Prodotto scalare normalizzato = coseno dell'angolo.
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + EPS)
    # np.clip evita errori numerici (valori tipo 1.0000001 fuori dal dominio
    # di arccos).
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_theta)))


def euclidean(p1, p2):
    """Distanza euclidea tra due punti (norma L2 della differenza)."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    return float(np.linalg.norm(p1 - p2))


def midpoint(p1, p2):
    """Punto medio tra due punti. Usato per pelvi e centro spalle."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    return (p1 + p2) / 2.0


def vector_angle_with_vertical(p_from, p_to):
    """Angolo (in gradi) tra il segmento p_from->p_to e l'asse verticale.

    Utile per misurare l'inclinazione di un segmento corporeo (es. quanto
    e' inclinato il tronco) in modo indipendente dalla lunghezza.

    Nota sul sistema di coordinate: in un'immagine l'asse Y cresce verso il
    basso. Qui usiamo il vettore "verticale" (0, -1) (verso l'alto), cosi'
    un tronco eretto risulta vicino a 0 gradi.
    """
    p_from = np.asarray(p_from, dtype=np.float64)
    p_to = np.asarray(p_to, dtype=np.float64)

    seg = p_to - p_from
    vertical = np.array([0.0, -1.0])
    # Usiamo solo le componenti (x, y) anche se i punti fossero 3D.
    seg2d = seg[:2]
    cos_theta = np.dot(seg2d, vertical) / (np.linalg.norm(seg2d) + EPS)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def normalize_skeleton(keypoints):
    """Rende lo scheletro invariante a traslazione e scala.

    Senza questa normalizzazione un classificatore "memorizzerebbe" la
    posizione assoluta del soggetto nel frame (overfitting) invece della
    sua POSTURA. Il blueprint la rende obbligatoria.

    Procedimento (sezione "Normalizzazione Spaziale" del blueprint):
      1. Invarianza di traslazione: si sposta l'origine sul punto medio
         pelvico P_pelvis = (anca_sx + anca_dx) / 2.
      2. Invarianza di scala: si divide tutto per H_torso, la distanza
         euclidea tra il centro delle spalle e il centro del bacino.

    Parametri
    ---------
    keypoints : np.ndarray, shape (33, 4)
        I 33 landmark BlazePose. Ogni riga e' [x, y, z, visibility].

    Ritorna
    -------
    norm_kps : np.ndarray, shape (33, 4)
        Landmark normalizzati. Le colonne x, y, z sono ricentrate e scalate;
        la colonna visibility resta invariata.
    torso_height : float
        Il fattore di scala H_torso usato (utile per normalizzare anche le
        distanze calcolate altrove).
    """
    # Importazione locale per evitare dipendenze circolari con config.py.
    from src.config import L_HIP, R_HIP, L_SHOULDER, R_SHOULDER

    kps = np.asarray(keypoints, dtype=np.float64).copy()

    # --- 1. Invarianza di traslazione ---
    pelvis = midpoint(kps[L_HIP, :3], kps[R_HIP, :3])
    kps[:, :3] = kps[:, :3] - pelvis   # ogni landmark ricentrato sul bacino

    # --- 2. Invarianza di scala ---
    shoulder_center = midpoint(kps[L_SHOULDER, :3], kps[R_SHOULDER, :3])
    # Il bacino e' ora nell'origine, quindi H_torso = norma del centro spalle.
    torso_height = np.linalg.norm(shoulder_center)
    if torso_height < EPS:
        # Caso degenere (scheletro collassato): evita divisione per zero.
        torso_height = 1.0
    kps[:, :3] = kps[:, :3] / torso_height

    return kps, torso_height


def safe_ratio(numerator, denominator):
    """Rapporto numericamente stabile (denominatore protetto da epsilon)."""
    return float(numerator / (denominator + EPS))
