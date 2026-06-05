"""
run_demo.py
===========
Punto di ingresso della demo live di BodyPoseRec.

Comandi:
    python run_demo.py                 # MLP, webcam indice 0
    python run_demo.py --model svm     # usa SVM
    python run_demo.py --camera 1      # webcam alternativa

Durante la demo:
    q -> esci
    r -> resetta il filtro temporale

NOTA: se MediaPipe non funziona, usa run_simulation.py per
dimostrare la stessa logica su dati sintetici.
"""

import argparse
import sys

from src.config import (
    SVM_MODEL_PATH, MLP_MODEL_PATH, SCALER_PATH, REFERENCE_POSES_PATH,
)


def _check_prerequisites(model_type):
    problems = []
    if not SCALER_PATH.exists():
        problems.append(f"Manca lo scaler: {SCALER_PATH}")
    if model_type == "svm" and not SVM_MODEL_PATH.exists():
        problems.append(f"Manca il modello SVM: {SVM_MODEL_PATH}")
    if model_type == "mlp" and not MLP_MODEL_PATH.exists():
        problems.append(f"Manca il modello MLP: {MLP_MODEL_PATH}")
    if not REFERENCE_POSES_PATH.exists():
        problems.append(f"Mancano i prototipi: {REFERENCE_POSES_PATH}")
    return problems


def main():
    parser = argparse.ArgumentParser(
        description="Demo live BodyPoseRec"
    )
    parser.add_argument("--model", choices=["svm", "mlp"], default="mlp")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    problems = _check_prerequisites(args.model)
    if problems:
        print("Prerequisiti mancanti:")
        for p in problems:
            print(f"  - {p}")
        print("\nEsegui prima:")
        print("  python -m src.data_preparation.build_dataset")
        print("  python -m src.classification.train_svm")
        print("  python -m src.classification.train_mlp")
        print("  python -m src.feature_extraction.build_reference_poses")
        sys.exit(1)

    # Test MediaPipe prima di avviare la demo
    print("Verifica MediaPipe...")
    try:
        from src.pose_estimation.mediapipe_pose import _load_mediapipe
        _, _, _, use_legacy = _load_mediapipe()
        api = "legacy (mp.solutions)" if use_legacy else "Tasks API"
        print(f"MediaPipe OK - usando {api}")
    except ImportError as e:
        print(f"\nMediaPipe non disponibile: {e}")
        print("\nAlternativa: dimostra la pipeline con dati sintetici:")
        print("  python run_simulation.py")
        sys.exit(1)

    from src.realtime.pose_recognizer import run_webcam_demo
    run_webcam_demo(model_type=args.model, camera_index=args.camera)


if __name__ == "__main__":
    main()
