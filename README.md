
---

# BodyPoseRec — Riconoscimento delle 7 Pose Obbligatorie del Bodybuilding Maschile

Sistema di Computer Vision che riconosce in tempo reale, a partire dai keypoint
del corpo, quale delle **7 pose obbligatorie** del bodybuilding maschile sta
eseguendo un atleta. Progetto finale del corso di Computer Vision (EPICODE).

---

## 1. Panoramica

Nelle competizioni di bodybuilding maschile la valutazione si articola su sette
pose cardine. BodyPoseRec automatizza il riconoscimento di queste pose con una
pipeline completa di Computer Vision che combina **pose estimation**,
**feature engineering geometrico** e **classificazione** (sia classica che deep
learning), arricchita da **stabilizzazione temporale**, **pose scoring** e
**analisi di interpretabilità (XAI)**.

Le 7 classi riconosciute (mutuamente esclusive):

| ID | Classe | Nome italiano |
|----|--------|---------------|
| 0 | `front_double_biceps` | Doppio bicipite frontale |
| 1 | `front_lat_spread` | Dorsali frontali |
| 2 | `side_chest` | Espansione toracica (di profilo) |
| 3 | `side_triceps` | Tricipiti laterali (di profilo) |
| 4 | `back_double_biceps` | Doppio bicipite di schiena |
| 5 | `back_lat_spread` | Dorsali di schiena |
| 6 | `abdominals_and_thighs` | Addominali e quadricipiti |

### Scelta del dataset: keypoint sintetici

Il sistema è addestrato su un **dataset sintetico di keypoint** generato a
partire dai descrittori cinematici delle 7 pose. Questa scelta è motivata e
coerente con il corso:

- **Privacy ed etica**: nessuna immagine reale di persone, nessun dato
  biometrico — il problema di privacy è eliminato alla radice.
- **Riproducibilità**: chiunque cloni il repository ottiene esattamente lo
  stesso dataset eseguendo un solo comando (nessun download, nessuna
  annotazione manuale).
- **Allineamento didattico**: gli script di laboratorio del Modulo 3 usano la
  stessa tecnica (`generate_synthetic_data()`) quando un dataset reale non è
  disponibile.

Il generatore produce sequenze di 33 landmark MediaPipe BlazePose per ogni
posa, applicando **variabilità antropometrica** (altezza, larghezza spalle,
proporzioni), **rumore articolare** e **flip orizzontale**.

---

## 2. Architettura e pipeline

La pipeline rispetta le quattro fasi richieste dalla traccia d'esame:

```
  [1] DATA ACQUISITION & PREPROCESSING
      Generatore sintetico di keypoint (33 landmark BlazePose)
      + variabilità antropometrica, rumore, flip orizzontale
                         |
                         v
  [2] FEATURE ENGINEERING / REPRESENTATION
      Normalizzazione scheletro (invarianza a traslazione e scala)
      -> 26 feature geometriche: angoli articolari, distanze
         normalizzate, rapporti, profondità, visibilità
                         |
                         v
  [3] CORE LOGIC — CLASSIFICAZIONE (due modelli)
      - SVM (RBF kernel) ........ modello classico, GridSearchCV
      - MLP (Keras) ............. modello deep learning
                         |
                         v
  [4] POST-PROCESSING
      Stabilizzazione temporale (media mobile sulle probabilità)
      + soglia di confidenza (gestione dell'incertezza)
```

Moduli aggiuntivi: **pose scoring** (qualità dell'esecuzione 0–100 rispetto a
un prototipo di riferimento), **analisi XAI** (permutation feature importance),
**analisi di robustezza** (stress test con rumore, occlusione, degrado della
profondità) e **dashboard Streamlit** interattiva.

### Struttura del repository

```
BodyPoseRec/
├── src/
│   ├── config.py                      # Configurazione centrale (costanti, path)
│   ├── utils/
│   │   ├── geometry.py                # Funzioni geometriche (angoli, normalizzazione)
│   │   ├── metrics.py                 # Metriche e grafici (confusion matrix, ROC)
│   │   └── viz.py                     # Visualizzazione (scheletro, overlay)
│   ├── data_preparation/
│   │   ├── synthetic_pose_generator.py # Generatore di keypoint sintetici
│   │   └── build_dataset.py           # Costruzione e split del dataset
│   ├── pose_estimation/
│   │   └── mediapipe_pose.py          # Wrapper MediaPipe (demo live)
│   ├── feature_extraction/
│   │   ├── feature_engineering.py     # Estrazione delle 26 feature geometriche
│   │   ├── build_reference_poses.py   # Prototipi di riferimento per lo scoring
│   │   └── pose_scoring.py            # Punteggio di qualità dell'esecuzione
│   ├── classification/
│   │   ├── train_svm.py               # Addestramento SVM (modello classico)
│   │   ├── train_mlp.py               # Addestramento MLP (deep learning)
│   │   ├── evaluate.py                # Valutazione comparativa sul test set
│   │   ├── robustness_analysis.py     # Stress test e failure analysis
│   │   └── xai_analysis.py            # Interpretabilità (feature importance)
│   └── realtime/
│       ├── temporal_filter.py         # Filtri di stabilizzazione temporale
│       ├── pose_recognizer.py         # Riconoscitore real-time da webcam
│       └── config.yaml                # Parametri della demo
├── data/                              # Dataset generato (non versionato)
├── models/                            # Modelli addestrati (non versionati)
├── results/                           # Grafici e report (non versionati)
├── docs/
│   └── technical_analysis.pdf         # Documento tecnico (deliverable)
├── run_demo.py                        # Demo live da webcam
├── run_simulation.py                  # Simulazione offline (senza webcam)
├── app_dashboard.py                   # Dashboard Streamlit interattiva
├── requirements.txt
└── README.md
```

---

## 3. Setup

### Prerequisiti
- Python 3.10 o superiore (consigliata 3.11).

### Gestione delle dipendenze e conflitto TensorFlow/MediaPipe

**Problema noto:** TensorFlow 2.21+ richiede `protobuf >= 6`, mentre MediaPipe richiede `protobuf < 5`. I due pacchetti **non possono coesistere** nello stesso ambiente virtuale.

**Soluzione adottata:**
- **Addestramento e valutazione** → ambiente principale (`venv`) con TensorFlow e Keras.
- **Inferenza MLP** → il modello viene esportato in **ONNX** (`mlp_model.onnx`). ONNX Runtime non ha dipendenze da protobuf, quindi può funzionare insieme a MediaPipe.
- **Demo live da webcam** → ambiente **separato** consigliato, oppure uso del modello SVM (che non richiede TensorFlow).

Il codice di `train_mlp.py` esporta automaticamente il modello ONNX. `pose_recognizer.py` e `app_dashboard.py` caricano prima il modello ONNX e solo come fallback caricano il modello Keras.

### Installazione

```bash
# 1. Clona il repository
git clone <URL-del-tuo-repository>
cd BodyPoseRec

# 2. Crea e attiva un ambiente virtuale (per training + valutazione + dashboard)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Installa le dipendenze
pip install -r requirements.txt
```

> **Nota per la demo live da webcam (run_demo.py):**  
> Se vuoi usare l’MLP con webcam, crea un ambiente separato (es. `venv_demo`) e installa solo:
> ```
> pip install opencv-python mediapipe==0.10.14 onnxruntime numpy
> ```
> Poi copia il modello `mlp_model.onnx` in quella cartella.  
> In alternativa, usa la demo con SVM (`python run_demo.py --model svm`) che funziona nell’ambiente principale senza conflitti (SVM non richiede TensorFlow a runtime).

---

## 4. Esecuzione

Eseguire i comandi **in quest'ordine** dalla cartella radice del progetto (nell’ambiente principale, con TensorFlow).

```bash
# --- FASE 1: genera il dataset sintetico ---
python -m src.data_preparation.build_dataset

# --- FASE 2: addestra i due modelli ---
python -m src.classification.train_svm     # modello classico
python -m src.classification.train_mlp     # modello deep learning + esportazione ONNX

# --- FASE 3: valutazione comparativa sul test set ---
python -m src.classification.evaluate

# --- FASE 4: analisi di robustezza (failure analysis) ---
python -m src.classification.robustness_analysis

# --- FASE 5: analisi di interpretabilità (XAI) ---
python -m src.classification.xai_analysis

# --- FASE 6: prototipi di riferimento per il pose scoring ---
python -m src.feature_extraction.build_reference_poses

# --- FASE 7: simulazione della stabilizzazione temporale (senza webcam) ---
python run_simulation.py
```

### Demo live da webcam

**Raccomandazione per evitare conflitti:** usa un ambiente separato per la demo live.

```bash
# Crea un nuovo ambiente (fuori dal progetto o in una sottocartella)
cd ..
python -m venv venv_demo
venv_demo\Scripts\activate     # Windows
# oppure source venv_demo/bin/activate

# Installa solo i pacchetti necessari per la demo
pip install opencv-python mediapipe==0.10.14 onnxruntime numpy

# Copia i file necessari dal progetto principale:
# - mlp_model.onnx (generato durante l'addestramento)
# - svm_best.pkl, scaler.pkl
# - il codice sorgente src/ e run_demo.py
# (oppure esegui la demo direttamente nella cartella del progetto
#  con l'ambiente principale se usi --model svm)

# Avvia la demo
python run_demo.py                 # usa l'MLP (richiede ONNX)
python run_demo.py --model svm     # usa l'SVM (nessun conflitto)
```

Comandi durante la demo: `q` per uscire, `r` per resettare il filtro temporale.

### Dashboard interattiva (nell’ambiente principale)

```bash
streamlit run app_dashboard.py
```

---

## 5. Riepilogo dei risultati

Valutazione sul **test set cieco** (630 campioni, 90 per classe), dati sintetici:

| Metrica | SVM | MLP |
|---------|-----|-----|
| Accuracy | 1.000 | 1.000 |
| Macro Precision | 1.000 | 1.000 |
| Macro Recall | 1.000 | 1.000 |
| Macro F1-score | 1.000 | 1.000 |
| Macro AUC | 1.000 | 1.000 |

Su dati sintetici puliti entrambi i modelli sono perfettamente accurati: le 7
pose sono separabili **by design**. Per ottenere una *failure analysis*
significativa è stata condotta un'**analisi di robustezza** sotto degrado
controllato:

- **Rumore sui keypoint**: l'accuratezza scende gradualmente; l'MLP è più
  robusto dell'SVM (es. a σ=0.05 → MLP 0.82 vs SVM 0.69).
- **Occlusione dei landmark**: con 4 landmark occlusi l'accuratezza scende a
  circa 0.60 per entrambi.
- **Stabilizzazione temporale**: nella simulazione riduce il flickering del
  **~92%** (da 36 a 3 cambi di etichetta) e migliora l'accuratezza
  frame-by-frame da 0.89 a 0.96.

Dettagli completi, tabelle e grafici nel documento `docs/technical_analysis.pdf`.

---

## 6. Note

- **Colab**: non utilizzato. L'intero progetto gira in locale; il dataset
  sintetico rende superfluo l'uso di risorse cloud.
- **Modello pre-addestrato**: MediaPipe è usato esclusivamente come estrattore
  di keypoint nella demo live (un problema di regressione già risolto). Tutta
  la logica di classificazione, feature engineering, scoring e
  stabilizzazione è sviluppata su misura per il progetto.
- **ONNX**: il modello MLP viene esportato in ONNX per l’inferenza, eliminando
  la dipendenza da TensorFlow a runtime e risolvendo il conflitto con
  `protobuf` richiesto da MediaPipe.
- **Ambiente headless**: `run_demo.py` e `app_dashboard.py` richiedono una
  webcam / un browser; `run_simulation.py` verifica la stessa logica senza
  hardware aggiuntivo.

---

## 7. Licenza

Progetto realizzato a scopo didattico per l'esame finale del corso di
Computer Vision (EPICODE Institute of Technology).