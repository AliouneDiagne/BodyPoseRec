"""
pose_recognizer.py
==================
Riconoscitore in tempo reale delle 7 pose obbligatorie da webcam.

Questo modulo assembla l'INTERA pipeline di BodyPoseRec in un'unica
applicazione live:

    Webcam --> MediaPipe (33 keypoint) --> feature engineering (26 feature)
           --> StandardScaler --> classificatore (SVM o MLP)
           --> stabilizzazione temporale (media mobile) --> overlay a schermo

Include anche il pose scoring: oltre al nome della posa, mostra un
punteggio di qualita' 0-100 confrontando l'atleta col prototipo ideale.

IMPORTANTE - ESECUZIONE
-----------------------
Questo file richiede una webcam fisica e le librerie native di MediaPipe e
OpenCV, percio' NON viene eseguito nell'ambiente di sviluppo headless.
E' codice completo e commentato, pronto per essere lanciato in locale dallo
studente tramite `python run_demo.py`.

Per verificare la LOGICA di stabilizzazione temporale senza webcam, usare
invece `python run_simulation.py`, che alimenta la stessa catena di filtri
con una sequenza di keypoint sintetici.

Riferimenti didattici:
- Modulo 4, pipeline di video analysis in tempo reale.
- Blueprint BodyPoseRec, fasi 21-22 (real-time e smoothing temporale).
"""

import time
import pickle
import json

import numpy as np

from src.config import (
    SVM_MODEL_PATH, MLP_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
    display_name, ID_TO_CLASS,
)
from src.feature_extraction.feature_engineering import extract_features
from src.feature_extraction.pose_scoring import PoseQualityScorer, quality_label
from src.realtime.temporal_filter import TemporalSmoother


class PoseRecognizer:
    """Incapsula il modello di classificazione e il pose scorer.

    Tiene insieme: scaler, classificatore (SVM o MLP), filtro temporale e
    scorer di qualita'. Espone un solo metodo, predict(), che data una
    matrice di keypoint restituisce la posa stabilizzata e il suo punteggio.
    """

    def __init__(self, model_type="mlp"):
        """model_type: "svm" oppure "mlp" (default mlp, il modello deep)."""
        self.model_type = model_type.lower()

        # --- Caricamento dello scaler (StandardScaler addestrato) ---
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        # --- Caricamento del classificatore scelto ---
        if self.model_type == "svm":
            with open(SVM_MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
        elif self.model_type == "mlp":
            self._load_mlp_model()
        else:
            raise ValueError("model_type deve essere 'svm' o 'mlp'")

        # --- Filtro temporale e scorer di qualita' ---
        self.smoother = TemporalSmoother()
        self.scorer = PoseQualityScorer()

    def _load_mlp_model(self):
        """Carica MLP: prima prova ONNX (no TF a runtime), poi fallback Keras."""
        import os
        onnx_path = str(MLP_MODEL_PATH).replace(".keras", ".onnx")
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                self._onnx_session = ort.InferenceSession(onnx_path)
                self._onnx_input_name = self._onnx_session.get_inputs()[0].name
                self._use_onnx = True
                print("MLP caricato via ONNX (no TensorFlow richiesto a runtime)")
                return
            except ImportError:
                pass
        import tensorflow as tf
        self.model = tf.keras.models.load_model(str(MLP_MODEL_PATH))
        self._use_onnx = False

    def _predict_proba(self, feature_vector):
        """Restituisce il vettore di probabilita' softmax per un campione."""
        X = self.scaler.transform(feature_vector.reshape(1, -1)).astype("float32")
        if self.model_type == "svm":
            return self.model.predict_proba(X)[0]
        if getattr(self, "_use_onnx", False):
            return self._onnx_session.run(None, {self._onnx_input_name: X})[0][0]
        return self.model.predict(X, verbose=0)[0]

    def predict(self, keypoints):
        """Esegue un passo di riconoscimento su un frame.

        Parametri
        ---------
        keypoints : np.ndarray (33, 4)
            Keypoint del frame corrente (da MediaPipe).

        Ritorna
        -------
        dict con: class_id, class_name, display, confidence, is_certain,
        quality_score, quality_label, suggestions.
        """
        # 1) Feature engineering: da 33 keypoint a 26 feature geometriche.
        features = extract_features(keypoints)
        # 2) Classificazione: probabilita' softmax sulle 7 classi.
        probs = self._predict_proba(features)
        # 3) Stabilizzazione temporale (media mobile + soglia confidenza).
        smoothed = self.smoother.update(probs)

        class_id = smoothed["class_id"]
        class_name = ID_TO_CLASS[class_id]

        # 4) Pose scoring: qualita' dell'esecuzione rispetto al prototipo.
        score_result = self.scorer.score(keypoints, class_name)

        return {
            "class_id": class_id,
            "class_name": class_name,
            "display": display_name(class_id),
            "confidence": smoothed["confidence"],
            "is_certain": smoothed["is_certain"],
            "quality_score": score_result["overall_score"],
            "quality_label": quality_label(score_result["overall_score"]),
            "suggestions": score_result["suggestions"],
        }

    def reset(self):
        """Azzera il filtro temporale (es. quando l'atleta esce dal campo)."""
        self.smoother.reset()


def run_webcam_demo(model_type="mlp", camera_index=0):
    """Avvia la demo live da webcam.

    Questo e' il cuore della Fase 21 del blueprint. Il loop:
    1. acquisisce un frame dalla webcam;
    2. estrae i keypoint con MediaPipe;
    3. li passa al PoseRecognizer (feature -> modello -> smoothing);
    4. disegna lo scheletro e gli overlay informativi;
    5. mostra il frame e calcola gli FPS.

    Premere 'q' per uscire, 'r' per azzerare il filtro temporale.

    NOTA: richiede una webcam; non eseguibile in ambiente headless.
    """
    # Import locali: cosi' l'assenza di webcam/OpenCV non blocca l'import
    # del modulo (necessario perche' run_simulation.py importa da qui).
    import cv2
    from src.pose_estimation.mediapipe_pose import MediaPipePoseEstimator
    from src.utils.viz import draw_pose_label, draw_quality_score, draw_fps

    print(f"Avvio demo BodyPoseRec (modello: {model_type.upper()})")
    print("Premere 'q' per uscire, 'r' per resettare il filtro temporale.")

    recognizer = PoseRecognizer(model_type=model_type)
    estimator = MediaPipePoseEstimator(static_image_mode=False)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Impossibile aprire la webcam (indice {camera_index}). "
            "Verifica che una webcam sia collegata e disponibile."
        )

    prev_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame non disponibile, interruzione.")
                break

            # Effetto specchio: piu' naturale per l'utente che si guarda.
            frame = cv2.flip(frame, 1)

            # --- Estrazione keypoint ---
            keypoints = estimator.extract(frame)

            if keypoints is not None:
                # --- Riconoscimento + scoring ---
                result = recognizer.predict(keypoints)
                # --- Disegno scheletro e overlay ---
                frame = estimator.draw(frame)
                if result["is_certain"]:
                    label = result["display"]
                else:
                    label = "Incerto / In transizione"
                frame = draw_pose_label(frame, label, result["confidence"])
                frame = draw_quality_score(frame, result["quality_score"])
            else:
                # Nessuna persona: azzeriamo il filtro per non "ricordare"
                # una posa vecchia quando l'atleta rientra in campo.
                recognizer.reset()
                cv2.putText(frame, "Nessun atleta rilevato", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # --- FPS ---
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            frame = draw_fps(frame, fps)

            cv2.imshow("BodyPoseRec - Demo Live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                recognizer.reset()
    finally:
        # Rilascio risorse SEMPRE, anche in caso di errore.
        cap.release()
        cv2.destroyAllWindows()
        estimator.close()
        print("Demo terminata, risorse rilasciate.")


if __name__ == "__main__":
    run_webcam_demo()
