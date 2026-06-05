"""
robustness_analysis.py
======================
Stress test di robustezza e analisi degli errori (failure analysis).

PERCHE' QUESTO MODULO E' NECESSARIO
-----------------------------------
Sul test set "pulito" entrambi i modelli raggiungono prestazioni quasi
perfette: e' una conseguenza onesta del fatto che il dataset sintetico,
generato da prototipi ben separati, e' linearmente separabile nello spazio
delle feature. Una matrice di confusione tutta diagonale, pero', non
insegna NULLA sull'analisi dei fallimenti, che il documento d'esame
richiede esplicitamente ("Failure Analysis").

Questo modulo crea quindi condizioni avverse controllate e misura DOVE e
QUANDO i modelli si rompono. E' la versione rigorosa e quantitativa della
failure analysis: invece di limitarci a dire "il modello e' perfetto",
mostriamo la sua curva di degrado e identifichiamo le confusioni che
emergono per prime.

I TRE STRESS TEST
-----------------
  1. RUMORE GAUSSIANO: si aggiunge rumore crescente alle coordinate dei
     keypoint, simulando una stima MediaPipe sempre piu' imprecisa
     (videocamere scadenti, bassa risoluzione, compressione aggressiva).
  2. OCCLUSIONE: si azzera la visibility e si perturbano alcuni landmark,
     simulando arti nascosti dal corpo o usciti dal campo visivo.
  3. PERTURBAZIONE DELLA PROFONDITA': si degrada il solo asse z, perche'
     il blueprint indica che la profondita' e' il segnale critico per
     distinguere le pose frontali da quelle dorsali.

Uso da terminale:
    python -m src.classification.robustness_analysis

Riferimenti didattici:
- Blueprint BodyPoseRec, sezione "Analisi degli Errori (Failure Analysis)".
- Fase 44 del piano (Benchmarking di Sistema / Stress Test).
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    ensure_dirs, FEATURES_DIR, RESULTS_DIR, MODELS_DIR,
    SVM_MODEL_PATH, MLP_MODEL_PATH, SCALER_PATH,
    POSE_CLASSES, POSE_DISPLAY_NAMES, RANDOM_SEED,
    L_WRIST, R_WRIST, L_ELBOW, R_ELBOW,
)
from src.feature_extraction.feature_engineering import extract_features_batch
from src.utils.metrics import compute_summary_metrics, plot_confusion_matrix

DISPLAY_NAMES = [POSE_DISPLAY_NAMES[c] for c in POSE_CLASSES]


# ===========================================================================
# 1. FUNZIONI DI DEGRADO CONTROLLATO
# ===========================================================================
def add_keypoint_noise(keypoints, sigma, rng):
    """Aggiunge rumore gaussiano di deviazione standard 'sigma' ai keypoint.

    Sigma e' espresso nelle stesse unita' delle coordinate normalizzate
    MediaPipe [0,1]: sigma=0.05 significa uno spostamento tipico del 5%
    della dimensione del frame, gia' molto consistente.
    """
    noisy = keypoints.copy()
    noise = rng.normal(0.0, sigma, size=noisy[:, :, :3].shape)
    noisy[:, :, :3] += noise
    return noisy


def apply_occlusion(keypoints, n_occluded, rng):
    """Simula l'occlusione di 'n_occluded' landmark per ogni scheletro.

    Un landmark occluso ha visibility azzerata e coordinate fortemente
    perturbate: e' cio' che accade realmente quando MediaPipe "perde" un
    giunto nascosto dal corpo (es. un polso dietro la schiena).
    """
    occluded = keypoints.copy()
    n_samples, n_landmarks, _ = occluded.shape
    # Diamo priorita' all'occlusione dei landmark distali degli arti
    # superiori (polsi, gomiti): sono quelli che il blueprint indica come
    # piu' soggetti a sparizione.
    candidate_joints = [L_WRIST, R_WRIST, L_ELBOW, R_ELBOW]

    for i in range(n_samples):
        chosen = rng.choice(candidate_joints,
                            size=min(n_occluded, len(candidate_joints)),
                            replace=False)
        for j in chosen:
            occluded[i, j, 3] = 0.0   # visibility -> 0
            # Coordinate perturbate pesantemente (giunto "fuori controllo").
            occluded[i, j, :3] += rng.normal(0.0, 0.15, size=3)
    return occluded


def degrade_depth(keypoints, sigma, rng):
    """Degrada selettivamente il solo asse z (profondita').

    Il blueprint indica la profondita' come il segnale chiave per separare
    pose frontali e dorsali. Degradando solo z verifichiamo quanto i
    modelli dipendano da questo asse, notoriamente il piu' rumoroso
    nell'output 2D->pseudo3D di MediaPipe.
    """
    degraded = keypoints.copy()
    noise = rng.normal(0.0, sigma, size=degraded[:, :, 2].shape)
    degraded[:, :, 2] += noise
    return degraded


# ===========================================================================
# 2. CARICAMENTO MODELLI E DATI
# ===========================================================================
def load_models_and_data():
    """Carica i modelli addestrati, lo scaler e i keypoint del test set."""
    from tensorflow import keras

    with open(SVM_MODEL_PATH, "rb") as f:
        svm = pickle.load(f)
    mlp = keras.models.load_model(MLP_MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Usiamo i KEYPOINT del test set (non le feature gia' calcolate): il
    # degrado va applicato sui keypoint, poi le feature si ricalcolano.
    X_test_kps = np.load(FEATURES_DIR / "X_test_keypoints.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")
    return svm, mlp, scaler, X_test_kps, y_test


def evaluate_on_keypoints(svm, mlp, scaler, keypoints, y_true):
    """Ricalcola le feature dai keypoint (eventualmente degradati) e valuta.

    Ritorna un dizionario con accuracy e macro-F1 di entrambi i modelli.
    """
    # Pipeline identica a quella reale: keypoint -> feature -> scaling.
    X_feat = extract_features_batch(keypoints)
    X_scaled = scaler.transform(X_feat)

    # SVM.
    svm_pred = svm.predict(X_scaled)
    svm_m = compute_summary_metrics(y_true, svm_pred)

    # MLP.
    mlp_prob = mlp.predict(X_scaled, verbose=0)
    mlp_pred = np.argmax(mlp_prob, axis=1)
    mlp_m = compute_summary_metrics(y_true, mlp_pred)

    return {
        "svm_acc": svm_m["accuracy"], "svm_f1": svm_m["macro_f1"],
        "mlp_acc": mlp_m["accuracy"], "mlp_f1": mlp_m["macro_f1"],
        "svm_pred": svm_pred, "mlp_pred": mlp_pred,
    }


# ===========================================================================
# 3. ESECUZIONE DEGLI STRESS TEST
# ===========================================================================
def run_noise_sweep(svm, mlp, scaler, X_kps, y_true, rng):
    """Stress test 1: accuracy al crescere del rumore gaussiano."""
    print("\n[Test 1/3] Sweep del rumore gaussiano sui keypoint...")
    sigmas = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    rows = []
    for sigma in sigmas:
        noisy = add_keypoint_noise(X_kps, sigma, rng)
        r = evaluate_on_keypoints(svm, mlp, scaler, noisy, y_true)
        rows.append((sigma, r["svm_acc"], r["mlp_acc"]))
        print(f"  sigma={sigma:.2f}  ->  SVM acc={r['svm_acc']:.3f}  "
              f"MLP acc={r['mlp_acc']:.3f}")
    return sigmas, rows


def run_occlusion_sweep(svm, mlp, scaler, X_kps, y_true, rng):
    """Stress test 2: accuracy al crescere del numero di landmark occlusi."""
    print("\n[Test 2/3] Sweep dell'occlusione dei landmark...")
    n_occluded_levels = [0, 1, 2, 3, 4]
    rows = []
    for n in n_occluded_levels:
        occluded = apply_occlusion(X_kps, n, rng)
        r = evaluate_on_keypoints(svm, mlp, scaler, occluded, y_true)
        rows.append((n, r["svm_acc"], r["mlp_acc"]))
        print(f"  landmark occlusi={n}  ->  SVM acc={r['svm_acc']:.3f}  "
              f"MLP acc={r['mlp_acc']:.3f}")
    return n_occluded_levels, rows


def run_depth_sweep(svm, mlp, scaler, X_kps, y_true, rng):
    """Stress test 3: accuracy al crescere del rumore sul solo asse z."""
    print("\n[Test 3/3] Sweep del degrado della profondita' (asse z)...")
    sigmas = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75]
    rows = []
    for sigma in sigmas:
        degraded = degrade_depth(X_kps, sigma, rng)
        r = evaluate_on_keypoints(svm, mlp, scaler, degraded, y_true)
        rows.append((sigma, r["svm_acc"], r["mlp_acc"]))
        print(f"  sigma_z={sigma:.2f}  ->  SVM acc={r['svm_acc']:.3f}  "
              f"MLP acc={r['mlp_acc']:.3f}")
    return sigmas, rows


def plot_degradation_curves(noise_data, occ_data, depth_data):
    """Disegna le tre curve di degrado in un unico grafico a tre pannelli."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Pannello 1: rumore gaussiano ---
    sigmas, rows = noise_data
    svm_acc = [r[1] for r in rows]
    mlp_acc = [r[2] for r in rows]
    axes[0].plot(sigmas, svm_acc, "o-", label="SVM", lw=2)
    axes[0].plot(sigmas, mlp_acc, "s-", label="MLP", lw=2)
    axes[0].set_xlabel("Sigma del rumore (coord. normalizzate)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Robustezza al rumore gaussiano", fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # --- Pannello 2: occlusione ---
    levels, rows = occ_data
    svm_acc = [r[1] for r in rows]
    mlp_acc = [r[2] for r in rows]
    axes[1].plot(levels, svm_acc, "o-", label="SVM", lw=2)
    axes[1].plot(levels, mlp_acc, "s-", label="MLP", lw=2)
    axes[1].set_xlabel("Numero di landmark occlusi")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Robustezza all'occlusione", fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # --- Pannello 3: degrado profondita' ---
    sigmas, rows = depth_data
    svm_acc = [r[1] for r in rows]
    mlp_acc = [r[2] for r in rows]
    axes[2].plot(sigmas, svm_acc, "o-", label="SVM", lw=2)
    axes[2].plot(sigmas, mlp_acc, "s-", label="MLP", lw=2)
    axes[2].set_xlabel("Sigma del rumore sull'asse z")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Robustezza al degrado della profondita'",
                      fontweight="bold")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    out_path = RESULTS_DIR / "robustness_curves.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    return out_path


def analyze_failure_confusions(svm, mlp, scaler, X_kps, y_true, rng):
    """Identifica QUALI confusioni emergono per prime sotto stress.

    Applichiamo un rumore moderato-alto (sigma=0.10), abbastanza da far
    sbagliare i modelli, e generiamo la matrice di confusione risultante.
    Cosi' la failure analysis ha una matrice NON diagonale da commentare.
    """
    print("\n[Analisi] Confusioni sotto rumore moderato (sigma=0.10)...")
    noisy = add_keypoint_noise(X_kps, 0.10, rng)
    r = evaluate_on_keypoints(svm, mlp, scaler, noisy, y_true)

    # Matrice di confusione dell'MLP sotto stress (di solito piu' istruttiva).
    plot_confusion_matrix(
        y_true, r["mlp_pred"], DISPLAY_NAMES,
        RESULTS_DIR / "confusion_matrix_stress_mlp.png",
        title="Matrice di Confusione sotto stress (MLP, rumore sigma=0.10)",
        cmap="Oranges",
    )

    # Estraiamo le coppie (vera -> predetta) piu' frequenti tra gli errori.
    errors = {}
    for true_c, pred_c in zip(y_true, r["mlp_pred"]):
        if true_c != pred_c:
            key = (int(true_c), int(pred_c))
            errors[key] = errors.get(key, 0) + 1

    top_confusions = sorted(errors.items(), key=lambda kv: kv[1],
                            reverse=True)[:5]
    print("  Confusioni piu' frequenti (vera -> predetta):")
    confusion_list = []
    for (tc, pc), count in top_confusions:
        line = (f"{POSE_DISPLAY_NAMES[POSE_CLASSES[tc]]} -> "
                f"{POSE_DISPLAY_NAMES[POSE_CLASSES[pc]]}: {count} casi")
        print(f"    {line}")
        confusion_list.append(line)
    return r["mlp_acc"], confusion_list


# ===========================================================================
# 4. MAIN
# ===========================================================================
def main():
    """Esegue tutti gli stress test e salva curve, matrici e report."""
    ensure_dirs()
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 65)
    print("  BodyPoseRec - Stress Test di Robustezza e Failure Analysis")
    print("=" * 65)

    svm, mlp, scaler, X_kps, y_true = load_models_and_data()
    print(f"\nTest set caricato: {X_kps.shape[0]} scheletri.")

    # Esecuzione dei tre stress test.
    noise_data = run_noise_sweep(svm, mlp, scaler, X_kps, y_true, rng)
    occ_data = run_occlusion_sweep(svm, mlp, scaler, X_kps, y_true, rng)
    depth_data = run_depth_sweep(svm, mlp, scaler, X_kps, y_true, rng)

    # Grafico riassuntivo delle curve di degrado.
    curves_path = plot_degradation_curves(noise_data, occ_data, depth_data)
    print(f"\nCurve di degrado salvate in: {curves_path}")

    # Analisi delle confusioni sotto stress.
    stress_acc, confusions = analyze_failure_confusions(
        svm, mlp, scaler, X_kps, y_true, rng)

    # Report testuale completo della failure analysis.
    report_path = RESULTS_DIR / "robustness_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BodyPoseRec - Report di Robustezza e Failure Analysis\n")
        f.write("=" * 60 + "\n\n")

        f.write("STRESS TEST 1 - Rumore gaussiano sui keypoint\n")
        f.write("-" * 60 + "\n")
        for sigma, sv, ml in noise_data[1]:
            f.write(f"  sigma={sigma:.2f}  SVM acc={sv:.3f}  "
                    f"MLP acc={ml:.3f}\n")

        f.write("\nSTRESS TEST 2 - Occlusione dei landmark\n")
        f.write("-" * 60 + "\n")
        for n, sv, ml in occ_data[1]:
            f.write(f"  landmark occlusi={n}  SVM acc={sv:.3f}  "
                    f"MLP acc={ml:.3f}\n")

        f.write("\nSTRESS TEST 3 - Degrado della profondita' (asse z)\n")
        f.write("-" * 60 + "\n")
        for sigma, sv, ml in depth_data[1]:
            f.write(f"  sigma_z={sigma:.2f}  SVM acc={sv:.3f}  "
                    f"MLP acc={ml:.3f}\n")

        f.write("\nCONFUSIONI PIU' FREQUENTI SOTTO STRESS (rumore 0.10)\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Accuracy MLP sotto stress: {stress_acc:.3f}\n")
        for line in confusions:
            f.write(f"  - {line}\n")

    # Salviamo anche i dati numerici in JSON per il documento tecnico.
    json_path = RESULTS_DIR / "robustness_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "noise_sweep": [list(r) for r in noise_data[1]],
            "occlusion_sweep": [list(r) for r in occ_data[1]],
            "depth_sweep": [list(r) for r in depth_data[1]],
            "stress_accuracy_mlp": stress_acc,
            "top_confusions": confusions,
        }, f, indent=2)

    print(f"Report salvato in: {report_path}")
    print("\nStress test completato con successo.")


if __name__ == "__main__":
    main()
