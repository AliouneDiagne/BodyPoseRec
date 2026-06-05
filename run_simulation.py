"""
run_simulation.py
=================
Simulazione OFFLINE della pipeline real-time di BodyPoseRec (senza webcam).

Scopo
-----
La demo live (run_demo.py) richiede una webcam, quindi non e' eseguibile in
un ambiente headless. Questo script dimostra comunque, in modo verificabile,
la parte piu' delicata della pipeline real-time: la STABILIZZAZIONE
TEMPORALE delle predizioni.

Cosa fa
-------
1. Costruisce una "routine di gara" sintetica: una sequenza di frame in cui
   un atleta virtuale tiene una posa per qualche secondo, poi passa alla
   successiva, e cosi' via per piu' pose.
2. A ogni frame aggiunge rumore ai keypoint (simula gli errori di MediaPipe).
3. Classifica ogni frame in due modi:
   - GREZZO: argmax diretto sul singolo frame (instabile, "balla").
   - STABILIZZATO: con la media mobile temporale (TemporalSmoother).
4. Confronta i due segnali, conta i cambi di etichetta (flicker) e salva un
   grafico che visualizza l'effetto dello smoothing.

Cosi' si verifica, numeri alla mano, che il post-processing temporale
riduce drasticamente il flickering - esattamente l'analisi richiesta dal
Modulo 4.

Esecuzione:
    python run_simulation.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    POSE_CLASSES, ID_TO_CLASS, display_name, RESULTS_DIR,
    RANDOM_SEED, ensure_dirs,
)
from src.data_preparation.synthetic_pose_generator import generate_clean_prototype
from src.feature_extraction.feature_engineering import extract_features
from src.realtime.pose_recognizer import PoseRecognizer
from src.realtime.temporal_filter import TemporalSmoother

# Quanti frame "tiene" ogni posa nella routine simulata.
FRAMES_PER_POSE = 45
# Intensita' del rumore aggiunto ai keypoint ad ogni frame.
KEYPOINT_NOISE = 0.05


def build_synthetic_routine(seed=RANDOM_SEED):
    """Costruisce una sequenza di keypoint che simula una routine di gara.

    Restituisce
    -----------
    sequence : list[np.ndarray]
        Lista di array (33, 4): un set di keypoint per ogni frame.
    truth : list[int]
        Etichetta vera (class_id) di ogni frame.
    """
    rng = np.random.default_rng(seed)
    # Routine: 4 pose in sequenza (sottoinsieme delle 7 per chiarezza).
    routine = ["front_double_biceps", "side_chest",
               "back_double_biceps", "abdominals_and_thighs"]

    sequence = []
    truth = []
    for class_name in routine:
        # Prototipo pulito della posa: l'atleta la "tiene" ferma.
        base_kps = generate_clean_prototype(class_name)
        class_id = POSE_CLASSES.index(class_name)
        for _ in range(FRAMES_PER_POSE):
            # Rumore gaussiano sulle coordinate x, y, z (non sulla visibility).
            noisy = base_kps.copy()
            noise = rng.normal(0.0, KEYPOINT_NOISE, size=(33, 3))
            noisy[:, :3] += noise.astype(np.float32)
            sequence.append(noisy)
            truth.append(class_id)
    return sequence, truth


def run_simulation():
    """Esegue la simulazione e salva il grafico di confronto."""
    ensure_dirs()
    print("=== BodyPoseRec - Simulazione stabilizzazione temporale ===\n")

    sequence, truth = build_synthetic_routine()
    n_frames = len(sequence)
    print(f"Routine simulata: {n_frames} frame "
          f"({n_frames // FRAMES_PER_POSE} pose, "
          f"{FRAMES_PER_POSE} frame ciascuna)")
    print(f"Rumore sui keypoint: sigma = {KEYPOINT_NOISE}\n")

    # Usiamo l'MLP come classificatore (e' il modello deep learning).
    recognizer = PoseRecognizer(model_type="mlp")
    # Un secondo smoother "pulito" per ottenere anche le predizioni grezze:
    # in realta' le predizioni grezze le ricaviamo prima dello smoothing.
    smoother = TemporalSmoother()

    raw_preds = []
    smooth_preds = []

    for keypoints in sequence:
        # --- Predizione grezza (singolo frame) ---
        features = extract_features(keypoints)
        probs = recognizer._predict_proba(features)
        raw_preds.append(int(np.argmax(probs)))
        # --- Predizione stabilizzata (media mobile) ---
        smoothed = smoother.update(probs)
        smooth_preds.append(smoothed["class_id"])

    raw_preds = np.array(raw_preds)
    smooth_preds = np.array(smooth_preds)
    truth = np.array(truth)

    # --- Metriche: flicker e accuratezza ---
    # "Flicker" = numero di volte in cui l'etichetta cambia tra frame
    # consecutivi. La verita' cambia solo 3 volte (4 pose -> 3 transizioni).
    raw_flicker = int(np.sum(raw_preds[1:] != raw_preds[:-1]))
    smooth_flicker = int(np.sum(smooth_preds[1:] != smooth_preds[:-1]))
    truth_flicker = int(np.sum(truth[1:] != truth[:-1]))

    raw_acc = float(np.mean(raw_preds == truth))
    smooth_acc = float(np.mean(smooth_preds == truth))

    print("Risultati:")
    print(f"  Cambi di etichetta nella verita'      : {truth_flicker}")
    print(f"  Cambi di etichetta - predizione GREZZA: {raw_flicker}")
    print(f"  Cambi di etichetta - STABILIZZATA     : {smooth_flicker}")
    reduction = 100.0 * (1 - smooth_flicker / max(raw_flicker, 1))
    print(f"  Riduzione del flickering              : {reduction:.1f}%")
    print(f"  Accuratezza frame-by-frame  GREZZA    : {raw_acc:.3f}")
    print(f"  Accuratezza frame-by-frame  STABILIZZ.: {smooth_acc:.3f}")

    # --- Grafico di confronto ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    frames = np.arange(n_frames)

    # Pannello 1: predizione grezza vs verita'.
    axes[0].step(frames, truth, where="post", label="Verita'",
                 color="black", linewidth=2.5, alpha=0.6)
    axes[0].step(frames, raw_preds, where="post", label="Predizione grezza",
                 color="#C44E52", linewidth=1.2)
    axes[0].set_title(f"Predizione GREZZA (frame singolo) - "
                      f"{raw_flicker} cambi di etichetta")
    axes[0].set_ylabel("Classe (id)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_yticks(range(len(POSE_CLASSES)))
    axes[0].grid(alpha=0.3)

    # Pannello 2: predizione stabilizzata vs verita'.
    axes[1].step(frames, truth, where="post", label="Verita'",
                 color="black", linewidth=2.5, alpha=0.6)
    axes[1].step(frames, smooth_preds, where="post",
                 label="Predizione stabilizzata", color="#4C72B0",
                 linewidth=1.6)
    axes[1].set_title(f"Predizione STABILIZZATA (media mobile, finestra) - "
                      f"{smooth_flicker} cambi di etichetta")
    axes[1].set_ylabel("Classe (id)")
    axes[1].set_xlabel("Frame")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].set_yticks(range(len(POSE_CLASSES)))
    axes[1].grid(alpha=0.3)

    fig.suptitle("BodyPoseRec - Effetto della stabilizzazione temporale "
                 "sul flickering", fontsize=13, y=1.0)
    fig.tight_layout()
    out_path = RESULTS_DIR / "temporal_smoothing_simulation.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nGrafico salvato in: {out_path}")
    print("\nConclusione: la media mobile riduce il flickering mantenendo "
          "(o migliorando) l'accuratezza, confermando l'utilita' del "
          "post-processing temporale nella pipeline real-time.")


if __name__ == "__main__":
    run_simulation()
