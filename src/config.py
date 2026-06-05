"""
config.py
=========
Configurazione centrale del progetto BodyPoseRec.

Questo modulo raccoglie in un unico punto TUTTE le costanti del progetto:
la tassonomia delle 7 pose, gli indici anatomici dei landmark MediaPipe
BlazePose, i percorsi delle cartelle e gli iperparametri dei modelli.

Centralizzare la configurazione e' una pratica di "reproducible research":
ogni script (data generation, feature extraction, training, valutazione,
demo) importa da qui, quindi non esistono "valori magici" sparsi nel codice
e una modifica si propaga in modo coerente all'intera pipeline.

Riferimenti didattici:
- Blueprint BodyPoseRec, sezione "Tassonomia Cinematica" (7 classi).
- Blueprint BodyPoseRec, sezione "Infrastruttura MediaPipe BlazePose"
  (topologia a 33 landmark, indici anatomici).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PERCORSI DEL PROGETTO
# ---------------------------------------------------------------------------
# PROJECT_ROOT viene calcolato in modo relativo a questo file (src/config.py),
# quindi il progetto funziona indipendentemente dalla cartella di lavoro.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "extracted_frames"
KEYPOINTS_DIR = DATA_DIR / "keypoints"          # array .npy dei 33 landmark grezzi
FEATURES_DIR = DATA_DIR / "features"            # vettori .npy di feature geometriche
ANNOTATIONS_DIR = DATA_DIR / "annotations"
SPLIT_DIR = DATA_DIR / "dataset_split"          # train.txt / val.txt / test.txt

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
ROC_DIR = RESULTS_DIR / "roc_curves"
SAMPLE_PRED_DIR = RESULTS_DIR / "sample_predictions"
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_dirs():
    """Crea tutte le cartelle del progetto se non esistono ancora.

    Viene chiamata all'inizio degli script che scrivono file su disco,
    cosi' l'utente non deve creare manualmente la struttura.
    """
    for d in [DATA_DIR, RAW_VIDEOS_DIR, FRAMES_DIR, KEYPOINTS_DIR,
              FEATURES_DIR, ANNOTATIONS_DIR, SPLIT_DIR, MODELS_DIR,
              RESULTS_DIR, ROC_DIR, SAMPLE_PRED_DIR, DOCS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. TASSONOMIA DELLE POSE (le 7 classi mutuamente esclusive)
# ---------------------------------------------------------------------------
# L'ordine di questa lista DEFINISCE la corrispondenza id <-> nome.
# L'indice nella lista e' la "class_id" usata da SVM e MLP.
POSE_CLASSES = [
    "front_double_biceps",   # 0 - Doppio bicipite frontale
    "front_lat_spread",      # 1 - Dorsali frontali
    "side_chest",            # 2 - Espansione toracica (di profilo)
    "side_triceps",          # 3 - Tricipiti laterali (di profilo)
    "back_double_biceps",    # 4 - Doppio bicipite di schiena
    "back_lat_spread",       # 5 - Dorsali di schiena
    "abdominals_and_thighs", # 6 - Addominali e quadricipiti
]

# Nomi "leggibili" per i grafici, la demo e il documento tecnico.
POSE_DISPLAY_NAMES = {
    "front_double_biceps":   "Front Double Biceps",
    "front_lat_spread":      "Front Lat Spread",
    "side_chest":            "Side Chest",
    "side_triceps":          "Side Triceps",
    "back_double_biceps":    "Back Double Biceps",
    "back_lat_spread":       "Back Lat Spread",
    "abdominals_and_thighs": "Abdominals and Thighs",
}

NUM_CLASSES = len(POSE_CLASSES)               # 7
CLASS_TO_ID = {name: i for i, name in enumerate(POSE_CLASSES)}
ID_TO_CLASS = {i: name for i, name in enumerate(POSE_CLASSES)}


def display_name(class_identifier):
    """Restituisce il nome leggibile data una class_id (int) o un nome interno (str)."""
    if isinstance(class_identifier, int):
        class_identifier = ID_TO_CLASS[class_identifier]
    return POSE_DISPLAY_NAMES.get(class_identifier, class_identifier)


# ---------------------------------------------------------------------------
# 3. INDICI DEI LANDMARK MEDIAPIPE BLAZEPOSE (topologia a 33 punti)
# ---------------------------------------------------------------------------
# MediaPipe Pose restituisce 33 landmark. Il blueprint indica che per il
# bodybuilding servono i centri di rotazione biomeccanica (spalle, gomiti,
# polsi, anche, ginocchia, caviglie), scartando i dettagli del volto.
# Manteniamo anche il naso (0) come unico riferimento di orientamento.
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28
NOSE = 0

NUM_LANDMARKS = 33   # topologia completa BlazePose

# Sottoinsieme di landmark "core" effettivamente usati nel feature engineering.
CORE_LANDMARKS = [
    NOSE,
    L_SHOULDER, R_SHOULDER,
    L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST,
    L_HIP, R_HIP,
    L_KNEE, R_KNEE,
    L_ANKLE, R_ANKLE,
]

# Coppie speculari sinistra/destra: servono al flip orizzontale (data
# augmentation) per scambiare correttamente i landmark dei due lati.
MIRROR_PAIRS = [
    (L_SHOULDER, R_SHOULDER),
    (L_ELBOW, R_ELBOW),
    (L_WRIST, R_WRIST),
    (L_HIP, R_HIP),
    (L_KNEE, R_KNEE),
    (L_ANKLE, R_ANKLE),
]


# ---------------------------------------------------------------------------
# 4. PARAMETRI DEL DATASET SINTETICO
# ---------------------------------------------------------------------------
# Quanti campioni generare per ogni classe. 600 e' un buon compromesso:
# abbastanza per addestrare SVM e MLP senza overfitting, ma veloce da
# generare. Le 7 classi sono bilanciate by-design.
SAMPLES_PER_CLASS = 600

# Suddivisione train / validation / test (la somma deve fare 1.0).
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Seed globale per la riproducibilita' (numpy, train_test_split, ecc.).
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# 5. IPERPARAMETRI DEI MODELLI
# ---------------------------------------------------------------------------
# --- SVM (modello classico) ---
# Griglia di ricerca per GridSearchCV: gli stessi valori usati nel lab del
# Modulo 3 (lab-m03-15-4.py).
SVM_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf"],
}
SVM_CV_FOLDS = 5   # K-fold cross-validation

# --- MLP (modello deep learning, Keras) ---
# Architettura coerente con il blueprint: due hidden layer (64 e 32 neuroni),
# ReLU, Dropout ~30%, output softmax a 7 neuroni.
MLP_HIDDEN_UNITS = [64, 32]
MLP_DROPOUT = 0.3
MLP_EPOCHS = 120
MLP_BATCH_SIZE = 32
MLP_LEARNING_RATE = 1e-3
MLP_EARLY_STOPPING_PATIENCE = 15


# ---------------------------------------------------------------------------
# 6. PARAMETRI DELLA PIPELINE REAL-TIME
# ---------------------------------------------------------------------------
# Ampiezza della finestra FIFO per la media mobile sulle probabilita' softmax
# (stabilizzazione temporale anti-flickering). Il blueprint suggerisce N=5..10.
TEMPORAL_WINDOW = 7

# Sotto questa confidenza la predizione viene marcata come "Uncertain".
CONFIDENCE_THRESHOLD = 0.55

# Parametri di inizializzazione di MediaPipe Pose.
MEDIAPIPE_MODEL_COMPLEXITY = 1      # 0=lite, 1=full, 2=heavy
MEDIAPIPE_MIN_DETECTION_CONF = 0.5
MEDIAPIPE_MIN_TRACKING_CONF = 0.5


# ---------------------------------------------------------------------------
# 7. PERCORSI DEI FILE MODELLO (artefatti serializzati)
# ---------------------------------------------------------------------------
SVM_MODEL_PATH = MODELS_DIR / "svm_best.pkl"
MLP_MODEL_PATH = MODELS_DIR / "mlp_model.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
REFERENCE_POSES_PATH = MODELS_DIR / "reference_poses.json"


if __name__ == "__main__":
    # Eseguendo direttamente questo file si stampa un riepilogo della
    # configurazione: utile come verifica rapida dell'ambiente.
    ensure_dirs()
    print("=== Configurazione BodyPoseRec ===")
    print(f"Root del progetto : {PROJECT_ROOT}")
    print(f"Numero di classi  : {NUM_CLASSES}")
    for i, name in enumerate(POSE_CLASSES):
        print(f"  [{i}] {name:24s} -> {display_name(i)}")
    print(f"Campioni per classe: {SAMPLES_PER_CLASS}")
    print(f"Split train/val/test: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print("Cartelle del progetto create/verificate con successo.")
