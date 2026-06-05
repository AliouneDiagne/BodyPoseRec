"""
mediapipe_pose.py
=================
Wrapper robusto attorno a MediaPipe Pose (BlazePose).

Compatibile con MediaPipe 0.10.x sia legacy (soluzioni classiche)
che nuova Task API. Gestisce automaticamente le differenze tra versioni.

NOTA: se la demo non parte a causa di MediaPipe, il progetto funziona
lo stesso tramite: python run_simulation.py
"""

import numpy as np
import logging
from typing import Optional

from src.config import (
    NUM_LANDMARKS,
    MEDIAPIPE_MODEL_COMPLEXITY,
    MEDIAPIPE_MIN_DETECTION_CONF,
    MEDIAPIPE_MIN_TRACKING_CONF,
)

logger = logging.getLogger(__name__)


def _load_mediapipe():
    """
    Carica MediaPipe gestendo le differenze tra versioni.
    Ritorna (pose_class, drawing_utils, drawing_styles, use_legacy).
    use_legacy=True  -> usa mp.solutions.pose (versioni <= 0.10.21)
    use_legacy=False -> usa Tasks API (versioni >= 0.10.30)
    """
    try:
        import mediapipe as mp

        # --- Tentativo 1: API legacy (mp.solutions.pose) ---
        # Funziona con mediapipe <= 0.10.21
        try:
            pose_mod = mp.solutions.pose
            drawing_utils = mp.solutions.drawing_utils
            drawing_styles = mp.solutions.drawing_styles
            # Test rapido per verificare che funzioni davvero
            _ = pose_mod.POSE_CONNECTIONS
            return pose_mod, drawing_utils, drawing_styles, True
        except AttributeError:
            pass

        # --- Tentativo 2: import diretto dal sottomodulo ---
        # Alcune versioni richiedono import esplicito
        try:
            from mediapipe.python.solutions import pose as pose_mod
            from mediapipe.python.solutions import drawing_utils
            from mediapipe.python.solutions import drawing_styles
            return pose_mod, drawing_utils, drawing_styles, True
        except (ImportError, AttributeError):
            pass

        # --- Tentativo 3: Tasks API (mediapipe >= 0.10.30) ---
        # Richiede download del modello al primo avvio
        try:
            import urllib.request, os
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision

            model_path = "pose_landmarker_lite.task"
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_lite/float16/latest/"
                "pose_landmarker_lite.task"
            )

            if not os.path.exists(model_path):
                print(f"Download modello MediaPipe Tasks: {model_url}")
                urllib.request.urlretrieve(model_url, model_path)
                print("Download completato.")

            return vision, None, None, False   # tasks API: no drawing_utils classici

        except Exception as e:
            raise ImportError(
                f"Nessuna API MediaPipe funzionante trovata ({e}).\n"
                "Soluzione raccomandata:\n"
                "  pip uninstall mediapipe -y\n"
                "  pip install mediapipe==0.10.14\n"
                "Nota: il progetto funziona senza webcam tramite:\n"
                "  python run_simulation.py"
            )

    except ImportError:
        raise ImportError(
            "MediaPipe non e' installato.\n"
            "  pip install mediapipe==0.10.14\n"
            "Nota: il progetto funziona senza webcam tramite:\n"
            "  python run_simulation.py"
        )


class MediaPipePoseEstimator:
    """Wrapper attorno a MediaPipe Pose per l'estrazione dei 33 landmark.

    Uso tipico::

        with MediaPipePoseEstimator() as estimator:
            kps = estimator.extract(frame_bgr)  # -> (33, 4) o None
    """

    def __init__(self,
                 static_image_mode: bool = False,
                 model_complexity: int = MEDIAPIPE_MODEL_COMPLEXITY,
                 min_detection_confidence: float = MEDIAPIPE_MIN_DETECTION_CONF,
                 min_tracking_confidence: float = MEDIAPIPE_MIN_TRACKING_CONF):

        pose_mod, drawing_utils, drawing_styles, use_legacy = _load_mediapipe()

        self._use_legacy = use_legacy
        self._last_results = None

        if use_legacy:
            # API classica: mp.solutions.pose.Pose
            self._mp_pose = pose_mod
            self._mp_drawing = drawing_utils
            self._mp_styles = drawing_styles

            self.pose = pose_mod.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            # Tasks API (mediapipe >= 0.10.30)
            import os
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision

            model_path = "pose_landmarker_lite.task"
            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._vision = vision
            self.pose = vision.PoseLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Estrae i 33 landmark. Ritorna (33,4) o None."""
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        if self._use_legacy:
            results = self.pose.process(frame_rgb)
            self._last_results = results
            if not results.pose_landmarks:
                return None
            kps = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
            for i, lm in enumerate(results.pose_landmarks.landmark):
                kps[i] = [lm.x, lm.y, lm.z, lm.visibility]
            return kps
        else:
            # Tasks API
            import mediapipe as mp
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb.copy()
            )
            results = self.pose.detect(mp_image)
            self._last_results = results
            if not results.pose_landmarks:
                return None
            kps = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
            lms = results.pose_landmarks[0]
            for i, lm in enumerate(lms):
                kps[i] = [lm.x, lm.y, lm.z, lm.visibility]
            return kps

    def draw(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Disegna lo scheletro sul frame."""
        annotated = frame_bgr.copy()
        if not self._last_results:
            return annotated

        if self._use_legacy:
            if self._last_results.pose_landmarks:
                self._mp_drawing.draw_landmarks(
                    annotated,
                    self._last_results.pose_landmarks,
                    self._mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self._mp_styles
                    .get_default_pose_landmarks_style(),
                )
        else:
            # Con Tasks API: disegno manuale semplice
            if self._last_results.pose_landmarks:
                import cv2
                h, w = annotated.shape[:2]
                for lm in self._last_results.pose_landmarks[0]:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)
        return annotated

    def close(self):
        if hasattr(self, 'pose') and self.pose is not None:
            self.pose.close()
            self.pose = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def extract_keypoints_from_image(image_path: str) -> Optional[np.ndarray]:
    """Estrae i 33 landmark da file immagine."""
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Impossibile leggere: {image_path}")
        return None
    with MediaPipePoseEstimator(static_image_mode=True) as estimator:
        return estimator.extract(image)
