"""
app_dashboard.py
================
Dashboard interattiva di BodyPoseRec (Streamlit).

E' uno dei miglioramenti consigliati dal blueprint e una delle "advanced
features" che la traccia d'esame premia (deployment come web app). Permette
di esplorare il sistema senza webcam, in modo visuale e interattivo:

- si sceglie una delle 7 pose e si genera uno scheletro sintetico;
- si regola il livello di rumore con uno slider;
- il sistema classifica la posa (SVM o MLP), mostra le probabilita',
  il punteggio di qualita' e il confronto con il prototipo di riferimento.

Avvio:
    streamlit run app_dashboard.py

NOTA: richiede il pacchetto 'streamlit' (incluso in requirements.txt) e che
la pipeline di addestramento sia gia' stata eseguita (modelli presenti in
models/). Non viene eseguito nell'ambiente di sviluppo headless.
"""

import pickle
import numpy as np

import streamlit as st

from src.config import (
    POSE_CLASSES, ID_TO_CLASS, display_name, POSE_DISPLAY_NAMES,
    SVM_MODEL_PATH, MLP_MODEL_PATH, SCALER_PATH,
)
from src.data_preparation.synthetic_pose_generator import (
    generate_clean_prototype, generate_sample,
)
from src.feature_extraction.feature_engineering import extract_features
from src.feature_extraction.pose_scoring import PoseQualityScorer, quality_label
from src.utils.viz import render_skeleton_canvas


@st.cache_resource
def load_models():
    """Carica scaler, SVM e MLP una sola volta (cache di Streamlit).

    Per l'MLP usa ONNX Runtime se disponibile (evita il conflitto
    tensorflow/protobuf/mediapipe), con fallback a Keras.
    """
    import os
    import pickle

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(SVM_MODEL_PATH, "rb") as f:
        svm = pickle.load(f)

    onnx_path = str(MLP_MODEL_PATH).replace(".keras", ".onnx")
    if os.path.exists(onnx_path):
        import onnxruntime as ort
        mlp = ort.InferenceSession(onnx_path)
        mlp._is_onnx = True
        mlp._onnx_input_name = mlp.get_inputs()[0].name
    else:
        import tensorflow as tf
        mlp = tf.keras.models.load_model(str(MLP_MODEL_PATH))
        mlp._is_onnx = False
    return scaler, svm, mlp


@st.cache_resource
def load_scorer():
    """Carica il pose scorer (cache di Streamlit)."""
    return PoseQualityScorer()


def predict(model, model_type, scaler, features):
    """Restituisce il vettore di probabilita' del modello scelto."""
    X = scaler.transform(features.reshape(1, -1)).astype("float32")
    if model_type == "svm":
        return model.predict_proba(X)[0]
    if getattr(model, "_is_onnx", False):
        return model.run(None, {model._onnx_input_name: X})[0][0]
    return model.predict(X, verbose=0)[0]


def main():
    st.set_page_config(page_title="BodyPoseRec Dashboard", layout="wide")
    st.title("BodyPoseRec - Riconoscimento delle 7 Pose Obbligatorie")
    st.caption("Dashboard interattiva per esplorare la pipeline di "
               "Computer Vision (pose estimation + classificazione).")

    # --- Barra laterale: controlli ---
    st.sidebar.header("Controlli")
    pose_label = st.sidebar.selectbox(
        "Posa da generare",
        options=POSE_CLASSES,
        format_func=lambda c: POSE_DISPLAY_NAMES[c],
    )
    model_type = st.sidebar.radio(
        "Classificatore", options=["mlp", "svm"],
        format_func=lambda m: "MLP (deep learning)" if m == "mlp"
        else "SVM (classico)",
    )
    noise = st.sidebar.slider(
        "Livello di rumore / variazione", 0.0, 0.12, 0.03, 0.01,
        help="Simula l'imprecisione di MediaPipe e la variabilita' "
             "antropometrica tra atleti diversi.",
    )
    seed = st.sidebar.number_input("Seed casuale", 0, 9999, 42)

    # --- Generazione dello scheletro sintetico ---
    rng = np.random.default_rng(int(seed))
    if noise <= 0.0:
        keypoints = generate_clean_prototype(pose_label)
    else:
        # Generiamo un campione e vi aggiungiamo rumore extra controllato
        # dallo slider, per rendere l'effetto visibile e interattivo.
        keypoints = generate_sample(pose_label, rng, allow_flip=False)
        keypoints = keypoints.copy()
        keypoints[:, :3] += rng.normal(0.0, noise, size=(33, 3)).astype(
            np.float32)

    # --- Inferenza ---
    scaler, svm, mlp = load_models()
    scorer = load_scorer()
    model = svm if model_type == "svm" else mlp

    features = extract_features(keypoints)
    probs = predict(model, model_type, scaler, features)
    pred_id = int(np.argmax(probs))
    pred_name = ID_TO_CLASS[pred_id]
    score_result = scorer.score(keypoints, pred_name)

    # --- Layout a tre colonne ---
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Scheletro generato")
        canvas = render_skeleton_canvas(keypoints)
        # render_skeleton_canvas restituisce un'immagine BGR: convertiamo.
        st.image(canvas[:, :, ::-1], use_column_width=True)

    with col2:
        st.subheader("Classificazione")
        is_correct = (pred_name == pose_label)
        st.metric(
            label="Posa riconosciuta",
            value=display_name(pred_id),
            delta="corretta" if is_correct else "ERRATA",
            delta_color="normal" if is_correct else "inverse",
        )
        st.write("**Probabilita' per classe:**")
        prob_dict = {POSE_DISPLAY_NAMES[POSE_CLASSES[i]]: float(probs[i])
                     for i in range(len(POSE_CLASSES))}
        st.bar_chart(prob_dict)

    with col3:
        st.subheader("Qualita' dell'esecuzione")
        score = score_result["overall_score"]
        st.metric("Punteggio", f"{score:.1f} / 100",
                  delta=quality_label(score))
        st.progress(min(int(score), 100))
        st.write("**Suggerimenti di correzione:**")
        for s in score_result["suggestions"]:
            st.write(f"- {s}")

    st.divider()
    st.info(
        "Questa dashboard usa dati sintetici per garantire riproducibilita' "
        "e rispetto della privacy. La stessa pipeline (feature engineering "
        "+ classificatore) alimenta la demo live da webcam (run_demo.py)."
    )


if __name__ == "__main__":
    main()
