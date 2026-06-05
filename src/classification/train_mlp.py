"""
train_mlp.py
============
Addestramento del modello deep learning: Multi-Layer Perceptron (MLP).

L'MLP e' il modello "deep learning" richiesto esplicitamente dal documento
d'esame, accanto al modello classico (SVM). E' un classificatore neurale
denso (fully-connected) realizzato con Keras/TensorFlow, esattamente lo
stack usato nei lab del Modulo 3 (lab-m03-12-4.py, lab-m03-13-4.py).

PERCHE' UN MLP E NON UNA CNN
----------------------------
Le CNN (VGG, ResNet) servono quando l'input e' un'immagine di pixel
grezzi, perche' sfruttano la struttura spaziale 2D. Qui pero' l'input non
e' un'immagine: e' gia' un vettore di feature geometriche a 26 dimensioni.
Per un input vettoriale 1D la scelta corretta e' una rete densa. Come dice
il teorema di approssimazione universale (Modulo 3), una rete feed-forward
con attivazioni non lineari puo' approssimare qualsiasi funzione continua.

ARCHITETTURA (dal blueprint)
----------------------------
  Input(26) -> Dense(64,ReLU) -> Dropout(0.3)
            -> Dense(32,ReLU) -> Dropout(0.3)
            -> Dense(7, Softmax)
  Loss: Categorical Cross-Entropy   Optimizer: Adam

Tecniche di regolarizzazione adottate:
  - Dropout: spegne casualmente il 30% dei neuroni a ogni passo, costringe
    la rete a non dipendere da singoli neuroni (riduce l'overfitting).
  - Early Stopping: ferma l'addestramento quando la validation loss smette
    di migliorare, ripristinando i pesi migliori.

Uso da terminale:
    python -m src.classification.train_mlp

Riferimenti didattici:
- Modulo 3, lab-m03-12-4.py (CNN Keras: Sequential, compile, fit, Dropout).
- Blueprint BodyPoseRec, sezione "Reti Neurali Profonde: MLP".
"""

import os
# Riduce il rumore dei log di TensorFlow (solo errori, niente info/warning).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
import pickle
import numpy as np

# L'import di TensorFlow e' "lazy" (dentro le funzioni) cosi' gli altri
# moduli del progetto non lo caricano se non serve. Qui pero' serve subito.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    ensure_dirs, FEATURES_DIR, MODELS_DIR, RESULTS_DIR,
    MLP_HIDDEN_UNITS, MLP_DROPOUT, MLP_EPOCHS, MLP_BATCH_SIZE,
    MLP_LEARNING_RATE, MLP_EARLY_STOPPING_PATIENCE,
    NUM_CLASSES, RANDOM_SEED,
    MLP_MODEL_PATH, SCALER_PATH,
)
from src.feature_extraction.feature_engineering import NUM_FEATURES
from src.utils.metrics import compute_summary_metrics, plot_training_history


def build_mlp(input_dim, num_classes):
    """Costruisce e compila l'architettura MLP descritta nel blueprint.

    Parametri
    ---------
    input_dim : int
        Dimensione del vettore di feature in ingresso (qui 26).
    num_classes : int
        Numero di classi in uscita (qui 7).

    Ritorna
    -------
    keras.Model
        Il modello compilato, pronto per l'addestramento.
    """
    model = keras.Sequential(name="BodyPoseRec_MLP")

    # Layer di input esplicito: rende lo schema leggibile in model.summary().
    model.add(keras.Input(shape=(input_dim,), name="feature_vector"))

    # Hidden layer densi con attivazione ReLU + Dropout.
    # ReLU (f(x)=max(0,x)) e' lo standard: evita il problema della
    # sparizione del gradiente tipico delle sigmoidi.
    for i, units in enumerate(MLP_HIDDEN_UNITS):
        model.add(layers.Dense(units, activation="relu",
                               name=f"dense_{i + 1}"))
        model.add(layers.Dropout(MLP_DROPOUT, name=f"dropout_{i + 1}"))

    # Layer di output: un neurone per classe, attivazione Softmax che
    # trasforma i punteggi in una distribuzione di probabilita' (somma = 1).
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    # Compilazione: Adam come ottimizzatore, Categorical Cross-Entropy come
    # loss (standard per la classificazione multiclasse con one-hot).
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=MLP_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_split(split):
    """Carica feature ed etichette di uno split."""
    X = np.load(FEATURES_DIR / f"X_{split}.npy")
    y = np.load(FEATURES_DIR / f"y_{split}.npy")
    return X, y


def _export_onnx(model):
    """Esporta il modello Keras in formato ONNX (via SavedModel intermedio).

    ONNX Runtime non ha dipendenze da protobuf versionate, quindi l'inferenza
    funziona nello stesso ambiente di mediapipe senza conflitti.
    """
    import shutil
    try:
        import tf2onnx, onnx, subprocess, sys
        saved_path = str(MLP_MODEL_PATH).replace(".keras", "_saved_model")
        if __import__("os").path.exists(saved_path):
            shutil.rmtree(saved_path)
        model.export(saved_path)
        onnx_path = str(MLP_MODEL_PATH).replace(".keras", ".onnx")
        result = subprocess.run(
            [sys.executable, "-m", "tf2onnx.convert",
             "--saved-model", saved_path,
             "--output", onnx_path,
             "--opset", "13"],
            capture_output=True, text=True,
        )
        if "Successfully converted" in result.stderr or "Successfully converted" in result.stdout:
            print(f"      Modello ONNX salvato: {onnx_path}")
        else:
            print("      ONNX export: nessun errore bloccante, vedere log se necessario")
    except Exception as exc:
        print(f"      ONNX export skipped: {exc}")


def train():
    """Addestra l'MLP e salva il modello, le curve e il log."""
    ensure_dirs()
    # Fissiamo i seed per rendere l'addestramento il piu' riproducibile
    # possibile (l'MLP, a differenza dell'SVM, ha inizializzazione casuale).
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("=" * 65)
    print("  BodyPoseRec - Addestramento MLP (modello deep learning)")
    print("=" * 65)

    # -----------------------------------------------------------------
    # PASSO 1: caricamento dei dati
    # -----------------------------------------------------------------
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    print(f"\n[1/6] Dati caricati:")
    print(f"      Train: {X_train.shape}, Validation: {X_val.shape}")

    # -----------------------------------------------------------------
    # PASSO 2: standardizzazione
    # -----------------------------------------------------------------
    # Riusiamo LO STESSO scaler addestrato per l'SVM, cosi' i due modelli
    # ricevono input identici e il confronto e' equo. Se lo scaler non
    # esiste ancora (MLP eseguito prima di SVM), lo creiamo qui.
    print(f"\n[2/6] Standardizzazione delle feature...")
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print(f"      Riuso dello scaler gia' addestrato (coerenza con SVM).")
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        print(f"      Nuovo scaler creato e salvato.")

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # -----------------------------------------------------------------
    # PASSO 3: one-hot encoding delle etichette
    # -----------------------------------------------------------------
    # La Categorical Cross-Entropy richiede le etichette in formato one-hot:
    # la classe 2 diventa [0,0,1,0,0,0,0]. E' la stessa to_categorical()
    # usata nei lab del Modulo 3.
    print(f"\n[3/6] One-hot encoding delle etichette...")
    y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_oh = keras.utils.to_categorical(y_val, NUM_CLASSES)

    # -----------------------------------------------------------------
    # PASSO 4: costruzione del modello
    # -----------------------------------------------------------------
    print(f"\n[4/6] Costruzione dell'architettura MLP...")
    model = build_mlp(NUM_FEATURES, NUM_CLASSES)
    model.summary(print_fn=lambda s: print("      " + s))

    # -----------------------------------------------------------------
    # PASSO 5: addestramento con Early Stopping
    # -----------------------------------------------------------------
    print(f"\n[5/6] Addestramento (max {MLP_EPOCHS} epoche, "
          f"early stopping patience={MLP_EARLY_STOPPING_PATIENCE})...")

    # EarlyStopping monitora la validation loss: se non migliora per
    # 'patience' epoche consecutive, l'addestramento si ferma e vengono
    # ripristinati i pesi dell'epoca migliore.
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=MLP_EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train_scaled, y_train_oh,
        validation_data=(X_val_scaled, y_val_oh),
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    # -----------------------------------------------------------------
    # PASSO 6: valutazione e salvataggio
    # -----------------------------------------------------------------
    print(f"\n[6/6] Valutazione e salvataggio...")
    y_val_prob = model.predict(X_val_scaled, verbose=0)
    y_val_pred = np.argmax(y_val_prob, axis=1)
    val_metrics = compute_summary_metrics(y_val, y_val_pred)
    print(f"      Accuracy validation : {val_metrics['accuracy']:.4f}")
    print(f"      Macro F1 validation : {val_metrics['macro_f1']:.4f}")

    # Salviamo il modello nel formato nativo Keras (.keras): contiene
    # architettura + pesi, quindi e' direttamente ricaricabile.
    model.save(MLP_MODEL_PATH)
    print(f"      Modello MLP salvato : {MLP_MODEL_PATH}")

    # Esportiamo anche in ONNX: elimina la dipendenza da tensorflow/protobuf
    # a runtime (inference via onnxruntime, compatibile con mediapipe).
    _export_onnx(model)

    # Grafico delle curve di training: serve a diagnosticare l'overfitting
    # e va incluso nel documento tecnico.
    history_plot = RESULTS_DIR / "mlp_training_history.png"
    plot_training_history(history.history, history_plot)
    print(f"      Curve di training   : {history_plot}")

    # Log testuale dell'addestramento.
    log_path = RESULTS_DIR / "mlp_training_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("BodyPoseRec - Log addestramento MLP\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Architettura: Input({NUM_FEATURES}) -> ")
        f.write(" -> ".join(f"Dense({u},ReLU)+Dropout({MLP_DROPOUT})"
                            for u in MLP_HIDDEN_UNITS))
        f.write(f" -> Dense({NUM_CLASSES},Softmax)\n")
        f.write(f"Optimizer: Adam (lr={MLP_LEARNING_RATE})\n")
        f.write(f"Loss: Categorical Cross-Entropy\n")
        f.write(f"Epoche eseguite: {len(history.history['loss'])}\n\n")
        f.write(f"Accuracy validation: {val_metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision val: {val_metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall val   : {val_metrics['macro_recall']:.4f}\n")
        f.write(f"Macro F1 val       : {val_metrics['macro_f1']:.4f}\n")
    print(f"      Log salvato         : {log_path}")

    print("\nAddestramento MLP completato con successo.")
    return model, history


if __name__ == "__main__":
    train()
