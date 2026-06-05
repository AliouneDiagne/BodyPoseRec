"""
viz.py
======
Utility di visualizzazione per BodyPoseRec.

Contiene le funzioni che disegnano elementi grafici sui frame: lo
scheletro a partire dai 33 landmark, l'etichetta della posa riconosciuta
con la barra di confidenza, e l'overlay di confronto con una posa di
riferimento (la funzionalita' "pose scoring" suggerita come miglioria).

Tutte le funzioni lavorano su immagini in formato BGR di OpenCV e
restituiscono una copia annotata, senza modificare l'immagine originale.
"""

import cv2
import numpy as np

from src.config import (
    NUM_LANDMARKS, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, NOSE,
)

# Connessioni "ossee" tra landmark: ogni coppia e' un segmento da disegnare.
# E' un sottoinsieme essenziale della topologia BlazePose, sufficiente a
# rendere lo scheletro leggibile per il dominio bodybuilding.
SKELETON_EDGES = [
    (L_SHOULDER, R_SHOULDER),   # cingolo scapolare
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),     # braccio sinistro
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),     # braccio destro
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),      # tronco
    (L_HIP, R_HIP),                                # bacino
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),            # gamba sinistra
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),            # gamba destra
]


def draw_skeleton(image, keypoints, color=(0, 255, 0), point_color=(0, 0, 255),
                  thickness=2, radius=4):
    """Disegna lo scheletro sull'immagine a partire dai landmark normalizzati [0,1].

    Parametri
    ---------
    image : np.ndarray
        Immagine BGR su cui disegnare.
    keypoints : np.ndarray, shape (33, >=2)
        Landmark con coordinate x, y in [0, 1] (formato MediaPipe grezzo).
    color : tuple
        Colore BGR dei segmenti ossei.
    point_color : tuple
        Colore BGR dei punti articolari.

    Ritorna
    -------
    np.ndarray
        Copia dell'immagine con lo scheletro disegnato.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Le coordinate MediaPipe sono normalizzate: vanno riscalate ai pixel.
    pts_px = []
    for i in range(NUM_LANDMARKS):
        x = int(keypoints[i, 0] * w)
        y = int(keypoints[i, 1] * h)
        pts_px.append((x, y))

    # Disegna prima i segmenti ossei...
    for a, b in SKELETON_EDGES:
        cv2.line(annotated, pts_px[a], pts_px[b], color, thickness)

    # ...poi i punti articolari (sopra i segmenti, cosi' restano visibili).
    for idx in [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST,
                R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]:
        cv2.circle(annotated, pts_px[idx], radius, point_color, -1)

    return annotated


def draw_pose_label(image, pose_name, confidence, position=(15, 40)):
    """Scrive il nome della posa riconosciuta e una barra di confidenza.

    La barra colorata (verde = confidenza alta, rosso = bassa) da' un
    feedback visivo immediato all'atleta o al giudice.
    """
    annotated = image.copy()
    x, y = position

    # Sfondo semitrasparente per rendere il testo leggibile su ogni immagine.
    overlay = annotated.copy()
    cv2.rectangle(overlay, (x - 10, y - 30), (x + 360, y + 45),
                  (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)

    # Testo principale con il nome della posa.
    cv2.putText(annotated, pose_name, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Barra di confidenza: lunghezza proporzionale, colore variabile.
    bar_x, bar_y = x, y + 22
    bar_w_max = 300
    bar_w = int(bar_w_max * float(np.clip(confidence, 0.0, 1.0)))
    # Verde se confidente, giallo se medio, rosso se incerto.
    if confidence >= 0.75:
        bar_color = (0, 200, 0)
    elif confidence >= 0.5:
        bar_color = (0, 200, 200)
    else:
        bar_color = (0, 0, 220)
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + 14),
                  (90, 90, 90), 1)
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14),
                  bar_color, -1)
    cv2.putText(annotated, f"{confidence * 100:.1f}%",
                (bar_x + bar_w_max + 10, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


def draw_quality_score(image, score, position=(15, 130)):
    """Scrive il punteggio di qualita' della posa (0-100).

    Il "pose scoring" e' una delle migliorie suggerite: confronta gli
    angoli dell'atleta con quelli ideali della posa di riferimento.
    """
    annotated = image.copy()
    x, y = position

    if score >= 80:
        color = (0, 200, 0)
        verdict = "Ottima"
    elif score >= 60:
        color = (0, 200, 200)
        verdict = "Buona"
    else:
        color = (0, 0, 220)
        verdict = "Da correggere"

    cv2.putText(annotated, f"Qualita': {score:.0f}/100 ({verdict})",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return annotated


def draw_fps(image, fps, position=None):
    """Scrive il valore di FPS nell'angolo in alto a destra del frame."""
    annotated = image.copy()
    h, w = annotated.shape[:2]
    if position is None:
        position = (w - 140, 30)
    cv2.putText(annotated, f"FPS: {fps:.1f}", position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return annotated


def render_skeleton_canvas(keypoints, size=400, color=(0, 200, 0),
                           background=(20, 20, 20), title=None):
    """Disegna uno scheletro normalizzato su una tela vuota (per confronti).

    A differenza di draw_skeleton, qui non c'e' un'immagine di sfondo: si
    parte da una tela monocroma. Usato per mostrare fianco a fianco la posa
    dell'atleta e quella di riferimento.

    Parametri
    ---------
    keypoints : np.ndarray, shape (33, >=2)
        Landmark normalizzati (ricentrati sul bacino, scala torso = 1).
    size : int
        Lato in pixel della tela quadrata.
    """
    canvas = np.full((size, size, 3), background, dtype=np.uint8)

    # I landmark normalizzati hanno l'origine sul bacino: li mappiamo al
    # centro della tela, con un fattore di zoom che li rende ben visibili.
    cx, cy = size // 2, size // 2
    zoom = size * 0.28

    def to_px(idx):
        x = int(cx + keypoints[idx, 0] * zoom)
        y = int(cy + keypoints[idx, 1] * zoom)
        return (x, y)

    for a, b in SKELETON_EDGES:
        cv2.line(canvas, to_px(a), to_px(b), color, 2)
    for idx in [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST,
                R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]:
        cv2.circle(canvas, to_px(idx), 4, (0, 0, 230), -1)

    if title:
        cv2.putText(canvas, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return canvas


def side_by_side(img_left, img_right):
    """Affianca orizzontalmente due immagini della stessa altezza.

    Usato dalla demo per mostrare la posa dell'atleta accanto a quella di
    riferimento di un campione.
    """
    h = max(img_left.shape[0], img_right.shape[0])

    def pad_to(img, target_h):
        if img.shape[0] == target_h:
            return img
        pad = np.zeros((target_h - img.shape[0], img.shape[1], 3),
                       dtype=img.dtype)
        return np.vstack([img, pad])

    return np.hstack([pad_to(img_left, h), pad_to(img_right, h)])
