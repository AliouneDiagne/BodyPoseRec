"""
synthetic_pose_generator.py
===========================
Generatore di scheletri sintetici per le 7 pose obbligatorie del bodybuilding.

PERCHE' UN GENERATORE SINTETICO
-------------------------------
Il blueprint prevede l'addestramento su video reali di gare IFBB. Quei
video pero' (a) non sono disponibili in questo ambiente, (b) sono dati
biometrici di persone reali, con i problemi di privacy che il blueprint
stesso evidenzia (GDPR/BIPA). La soluzione adottata e' la stessa del lab
del Modulo 3 (lab-m03-13-4.py), dove la funzione `generate_synthetic_data()`
crea un dataset sintetico quando quello reale manca.

Qui non generiamo immagini ma direttamente i 33 landmark BlazePose: la
pose estimation di MediaPipe trasforma comunque ogni frame in keypoint,
quindi partire dai keypoint e' equivalente e rende l'intera pipeline
(feature engineering, SVM, MLP, valutazione, smoothing, XAI) eseguibile
end-to-end senza alcun download.

COME FUNZIONA
-------------
Per ogni posa definiamo un "prototipo": le coordinate (x, y, z) ideali dei
13 landmark core, costruite a partire dai descrittori cinematici del
blueprint (es. gomiti a ~90 gradi nel Front Double Biceps, mani sui
fianchi nel Lat Spread, profilo a 90 gradi nelle pose Side, ecc.).

Ogni campione del dataset si ottiene dal prototipo applicando:
  1. variabilita' antropometrica  -> arti piu' lunghi/corti, spalle piu'
     larghe/strette (simula atleti con corporature diverse);
  2. rumore gaussiano sui giunti  -> simula l'imprecisione di MediaPipe;
  3. jitter di posa               -> piccole variazioni dell'esecuzione;
  4. eventuale flip orizzontale   -> per le pose simmetriche/laterali.

Il risultato e' un array (33, 4) per campione: [x, y, z, visibility],
identico nel formato all'output reale di MediaPipe Pose.

Riferimenti didattici:
- Modulo 3, lab-m03-13-4.py, funzione generate_synthetic_data().
- Blueprint BodyPoseRec, tabella "Tassonomia Cinematica" (descrittori
  cinematici di ciascuna delle 7 pose).
- Blueprint BodyPoseRec, sezione "Simulazione di Degrado e Data Augmentation".
"""

import numpy as np

from src.config import (
    NUM_LANDMARKS, POSE_CLASSES, RANDOM_SEED,
    NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
)


# ===========================================================================
# 1. PROTOTIPI DELLE POSE
# ===========================================================================
# Convenzione del sistema di coordinate (stile MediaPipe, ma qui in uno
# spazio "comodo" centrato sul corpo, che verra' poi spostato e scalato):
#   - x : cresce verso destra dell'immagine
#   - y : cresce verso il BASSO (come nelle immagini); quindi le spalle
#         hanno y piccolo, le caviglie y grande
#   - z : profondita' relativa; z negativo = piu' vicino alla camera
# I valori sono in "unita' corpo": l'altezza del tronco e' circa 1.0.
#
# Ogni prototipo e' un dizionario {indice_landmark: (x, y, z)}.
# Definiamo solo i 13 landmark core; gli altri 20 verranno riempiti con
# valori plausibili e bassa visibility (il dominio bodybuilding li ignora).


def _base_torso():
    """Restituisce i landmark del tronco/bacino comuni a tutte le pose.

    Spalle, anche e naso costituiscono lo "scheletro portante" che cambia
    poco tra le pose frontali; le pose laterali lo modificheranno ruotandolo.
    """
    return {
        NOSE:       (0.00, -1.15, 0.00),
        L_SHOULDER: (-0.32, -0.85, 0.00),
        R_SHOULDER: (0.32, -0.85, 0.00),
        L_HIP:      (-0.20, 0.00, 0.00),
        R_HIP:      (0.20, 0.00, 0.00),
    }


def _proto_front_double_biceps():
    """Posa 0 - Front Double Biceps.

    Descrittore (blueprint): atleta frontale, abduzione della spalla,
    gomiti flessi a ~90 gradi, polsi alti per mostrare il picco bicipitale,
    una gamba leggermente avanzata.
    """
    p = _base_torso()
    # Braccia sollevate ai lati: gomiti all'altezza delle spalle, polsi
    # piu' in alto -> tipica forma "a doppia W" del double biceps.
    p[L_ELBOW] = (-0.70, -0.85, -0.05)
    p[R_ELBOW] = (0.70, -0.85, -0.05)
    p[L_WRIST] = (-0.55, -1.20, -0.10)
    p[R_WRIST] = (0.55, -1.20, -0.10)
    # Gambe: una avanzata (z negativo) per evidenziare il quadricipite.
    p[L_KNEE] = (-0.22, 0.95, -0.15)
    p[R_KNEE] = (0.22, 1.00, 0.05)
    p[L_ANKLE] = (-0.24, 1.90, -0.15)
    p[R_ANKLE] = (0.26, 1.95, 0.05)
    return p


def _proto_front_lat_spread():
    """Posa 1 - Front Lat Spread.

    Descrittore: mani sui fianchi/cresta iliaca, gomiti spinti verso
    l'esterno per aprire il "ventaglio" dorsale. Polsi bassi, vicino alle
    anche; gomiti larghi.
    """
    p = _base_torso()
    # Gomiti spinti molto in fuori (apertura dorsale massima).
    p[L_ELBOW] = (-0.78, -0.30, 0.10)
    p[R_ELBOW] = (0.78, -0.30, 0.10)
    # Polsi appoggiati sui fianchi: vicini alle anche.
    p[L_WRIST] = (-0.30, -0.05, 0.05)
    p[R_WRIST] = (0.30, -0.05, 0.05)
    # Gambe quasi parallele, stance neutra.
    p[L_KNEE] = (-0.21, 0.98, 0.00)
    p[R_KNEE] = (0.21, 0.98, 0.00)
    p[L_ANKLE] = (-0.22, 1.93, 0.00)
    p[R_ANKLE] = (0.22, 1.93, 0.00)
    return p


def _proto_side_chest():
    """Posa 2 - Side Chest (di profilo).

    Descrittore: tronco ruotato di ~90 gradi rispetto alla camera, mani
    unite davanti, un braccio attraversa il torso flesso a ~90 gradi.
    La rotazione del busto si simula COMPRIMENDO la x (le due spalle
    appaiono quasi sovrapposte) e differenziando fortemente la z.
    """
    p = _base_torso()
    # Profilo: le spalle si proiettano quasi sullo stesso punto x, ma una
    # e' davanti (z negativo) e una dietro (z positivo).
    p[L_SHOULDER] = (-0.10, -0.85, 0.28)   # spalla posteriore
    p[R_SHOULDER] = (0.10, -0.85, -0.28)   # spalla anteriore
    p[L_HIP] = (-0.08, 0.00, 0.20)
    p[R_HIP] = (0.08, 0.00, -0.20)
    p[NOSE] = (0.18, -1.15, -0.30)         # il viso guarda di lato
    # Braccia: mani unite al centro davanti al petto.
    p[L_ELBOW] = (0.05, -0.45, -0.30)
    p[R_ELBOW] = (0.20, -0.40, -0.45)
    p[L_WRIST] = (0.12, -0.55, -0.55)
    p[R_WRIST] = (0.14, -0.52, -0.55)      # polsi vicini -> mani giunte
    # Gambe di profilo, una flessa.
    p[L_KNEE] = (-0.05, 0.98, 0.25)
    p[R_KNEE] = (0.10, 1.00, -0.20)
    p[L_ANKLE] = (-0.10, 1.93, 0.30)
    p[R_ANKLE] = (0.18, 1.92, -0.25)
    return p


def _proto_side_triceps():
    """Posa 3 - Side Triceps (di profilo).

    Descrittore: di profilo, entrambe le braccia portate dietro il tronco,
    contrazione del tricipite con braccio posteriore parzialmente esteso.
    Si distingue dalla Side Chest soprattutto per l'angolo del gomito
    posteriore (piu' esteso) e per i polsi spostati DIETRO il corpo.
    """
    p = _base_torso()
    # Stessa rotazione di profilo della Side Chest.
    p[L_SHOULDER] = (-0.10, -0.85, 0.28)
    p[R_SHOULDER] = (0.10, -0.85, -0.28)
    p[L_HIP] = (-0.08, 0.00, 0.20)
    p[R_HIP] = (0.08, 0.00, -0.20)
    p[NOSE] = (0.18, -1.15, -0.30)
    # Braccia portate DIETRO: polsi con z positivo (dietro il corpo),
    # gomito posteriore quasi esteso.
    p[L_ELBOW] = (-0.18, -0.30, 0.45)
    p[R_ELBOW] = (-0.05, -0.35, 0.30)
    p[L_WRIST] = (-0.10, 0.10, 0.55)
    p[R_WRIST] = (-0.08, 0.05, 0.50)
    # Gambe di profilo, una in tensione.
    p[L_KNEE] = (-0.05, 0.98, 0.25)
    p[R_KNEE] = (0.12, 1.00, -0.18)
    p[L_ANKLE] = (-0.10, 1.93, 0.30)
    p[R_ANKLE] = (0.20, 1.92, -0.22)
    return p


def _proto_back_double_biceps():
    """Posa 4 - Back Double Biceps (di schiena).

    Descrittore: come la classe 0 ma di spalle. La geometria angolare e'
    quasi speculare al Front Double Biceps; la differenza chiave e' il
    segno della profondita' z (il corpo e' girato) e l'assenza di feature
    facciali frontali. Una gamba e' tipicamente estesa indietro sulle punte.
    """
    p = _base_torso()
    # Corpo di schiena: il naso (volto) e' lontano dalla camera (z positivo,
    # molto ridotto in visibility piu' avanti).
    p[NOSE] = (0.00, -1.15, 0.35)
    # Braccia sollevate e flesse come nel front, ma con z invertito.
    p[L_ELBOW] = (-0.70, -0.85, 0.05)
    p[R_ELBOW] = (0.70, -0.85, 0.05)
    p[L_WRIST] = (-0.55, -1.20, 0.10)
    p[R_WRIST] = (0.55, -1.20, 0.10)
    # Una gamba estesa indietro, in appoggio sulle punte (caviglia alta).
    p[L_KNEE] = (-0.22, 1.00, 0.05)
    p[R_KNEE] = (0.24, 0.92, 0.25)
    p[L_ANKLE] = (-0.24, 1.95, 0.05)
    p[R_ANKLE] = (0.30, 1.75, 0.40)
    return p


def _proto_back_lat_spread():
    """Posa 5 - Back Lat Spread (di schiena).

    Descrittore: analoga alla classe 1 ma di schiena; mani sui fianchi,
    gomiti spinti in avanti/fuori per mostrare l'ampiezza della schiena.
    Si distingue dalla classe 1 per il segno della profondita' z.
    """
    p = _base_torso()
    p[NOSE] = (0.00, -1.15, 0.35)
    # Gomiti spinti in fuori, mani sui fianchi (come front lat ma z invertito).
    p[L_ELBOW] = (-0.78, -0.30, -0.10)
    p[R_ELBOW] = (0.78, -0.30, -0.10)
    p[L_WRIST] = (-0.30, -0.05, -0.05)
    p[R_WRIST] = (0.30, -0.05, -0.05)
    # Gambe in stance neutra.
    p[L_KNEE] = (-0.21, 0.98, 0.05)
    p[R_KNEE] = (0.21, 0.98, 0.05)
    p[L_ANKLE] = (-0.22, 1.93, 0.05)
    p[R_ANKLE] = (0.22, 1.93, 0.05)
    return p


def _proto_abdominals_and_thighs():
    """Posa 6 - Abdominals and Thighs.

    Descrittore: posa frontale, mani incrociate dietro la nuca/collo,
    addome contratto, una gamba estesa in avanti per isolare il quadricipite.
    Caratteristica unica: i polsi sono ALTI, dietro la testa (y molto
    piccolo, vicino o sopra il naso).
    """
    p = _base_torso()
    # Mani dietro la nuca: gomiti larghi e alti, polsi vicini alla testa.
    p[L_ELBOW] = (-0.55, -1.05, 0.10)
    p[R_ELBOW] = (0.55, -1.05, 0.10)
    p[L_WRIST] = (-0.18, -1.25, 0.25)
    p[R_WRIST] = (0.18, -1.25, 0.25)
    # Una gamba tesa in avanti (z negativo marcato), l'altra di appoggio.
    p[L_KNEE] = (-0.20, 0.95, -0.35)
    p[R_KNEE] = (0.22, 1.00, 0.05)
    p[L_ANKLE] = (-0.18, 1.85, -0.55)
    p[R_ANKLE] = (0.24, 1.95, 0.05)
    return p


# Dizionario che mappa il nome della classe alla funzione che ne costruisce
# il prototipo. L'ordine corrisponde a POSE_CLASSES in config.py.
_PROTOTYPE_BUILDERS = {
    "front_double_biceps":   _proto_front_double_biceps,
    "front_lat_spread":      _proto_front_lat_spread,
    "side_chest":            _proto_side_chest,
    "side_triceps":          _proto_side_triceps,
    "back_double_biceps":    _proto_back_double_biceps,
    "back_lat_spread":       _proto_back_lat_spread,
    "abdominals_and_thighs": _proto_abdominals_and_thighs,
}

# Visibility tipica dei landmark core: alta per quelli "core", piu' bassa
# per il naso nelle pose di schiena (gestito in _build_full_skeleton).
_CORE_VISIBILITY = 0.97


# ===========================================================================
# 2. COSTRUZIONE DELLO SCHELETRO COMPLETO (33 landmark)
# ===========================================================================
def _build_full_skeleton(proto, class_name):
    """Espande un prototipo di 13 landmark core in un array completo (33, 4).

    I 20 landmark non-core (dettagli del volto, mani, piedi fini) vengono
    riempiti con valori plausibili e bassa visibility: cosi' il formato e'
    identico all'output reale di MediaPipe, ma il feature engineering
    (che usa solo i landmark core) non ne e' influenzato.
    """
    skeleton = np.zeros((NUM_LANDMARKS, 4), dtype=np.float64)

    # Landmark core: copiamo le coordinate dal prototipo.
    for idx, (x, y, z) in proto.items():
        skeleton[idx, 0] = x
        skeleton[idx, 1] = y
        skeleton[idx, 2] = z
        skeleton[idx, 3] = _CORE_VISIBILITY

    # Nelle pose di schiena il volto non e' visibile: la visibility del
    # naso scende, come accadrebbe realmente con MediaPipe.
    if class_name in ("back_double_biceps", "back_lat_spread"):
        skeleton[NOSE, 3] = 0.35

    # Landmark non-core: li poniamo in posizioni plausibili interpolando
    # tra punti core vicini, con visibility bassa.
    # Occhi/orecchie (1..10): vicino al naso.
    for idx in range(1, 11):
        skeleton[idx, :3] = skeleton[NOSE, :3] + np.array([0.0, -0.03, 0.0])
        skeleton[idx, 3] = 0.30
    # Punti delle mani (17..22): vicino ai polsi.
    for idx in (17, 19, 21):
        skeleton[idx, :3] = skeleton[L_WRIST, :3] + np.array([0.0, 0.05, 0.0])
        skeleton[idx, 3] = 0.40
    for idx in (18, 20, 22):
        skeleton[idx, :3] = skeleton[R_WRIST, :3] + np.array([0.0, 0.05, 0.0])
        skeleton[idx, 3] = 0.40
    # Punti dei piedi (29..32): vicino alle caviglie.
    for idx in (29, 31):
        skeleton[idx, :3] = skeleton[L_ANKLE, :3] + np.array([0.0, 0.08, 0.0])
        skeleton[idx, 3] = 0.45
    for idx in (30, 32):
        skeleton[idx, :3] = skeleton[R_ANKLE, :3] + np.array([0.0, 0.08, 0.0])
        skeleton[idx, 3] = 0.45

    return skeleton


# ===========================================================================
# 3. AUGMENTATION: VARIABILITA' ANTROPOMETRICA E RUMORE
# ===========================================================================
def _apply_anthropometric_variation(skeleton, rng):
    """Modifica le proporzioni corporee per simulare atleti diversi.

    Senza questa variazione tutti i campioni di una classe sarebbero
    identici e il modello non imparerebbe a generalizzare tra fisici
    diversi (rischio di "representation bias", citato nel blueprint).

    Applica tre trasformazioni indipendenti:
      - allungamento/accorciamento degli arti superiori;
      - allungamento/accorciamento degli arti inferiori;
      - allargamento/restringimento del cingolo scapolare.
    """
    sk = skeleton.copy()

    # Fattori di scala campionati attorno a 1.0 (1.0 = corporatura "media").
    arm_scale = rng.normal(1.0, 0.10)    # +/- 10% sulla lunghezza braccia
    leg_scale = rng.normal(1.0, 0.10)    # +/- 10% sulla lunghezza gambe
    shoulder_scale = rng.normal(1.0, 0.08)  # +/- 8% sull'apertura spalle

    # --- Arti superiori: scaliamo gomito e polso rispetto alla spalla ---
    for shoulder, elbow, wrist in [(L_SHOULDER, L_ELBOW, L_WRIST),
                                   (R_SHOULDER, R_ELBOW, R_WRIST)]:
        s = sk[shoulder, :3]
        sk[elbow, :3] = s + (sk[elbow, :3] - s) * arm_scale
        sk[wrist, :3] = s + (sk[wrist, :3] - s) * arm_scale

    # --- Arti inferiori: scaliamo ginocchio e caviglia rispetto all'anca ---
    for hip, knee, ankle in [(L_HIP, L_KNEE, L_ANKLE),
                             (R_HIP, R_KNEE, R_ANKLE)]:
        h = sk[hip, :3]
        sk[knee, :3] = h + (sk[knee, :3] - h) * leg_scale
        sk[ankle, :3] = h + (sk[ankle, :3] - h) * leg_scale

    # --- Apertura del cingolo scapolare: scaliamo la x delle spalle ---
    sk[L_SHOULDER, 0] *= shoulder_scale
    sk[R_SHOULDER, 0] *= shoulder_scale

    return sk


def _apply_joint_noise(skeleton, rng, sigma=0.025):
    """Aggiunge rumore gaussiano alle coordinate dei giunti.

    Simula l'imprecisione intrinseca della stima dei landmark di MediaPipe
    (i keypoint reali "tremano" leggermente da un frame all'altro). E' la
    versione "sui keypoint" dell'iniezione di rumore descritta dal
    blueprint nella sezione di data augmentation.
    """
    sk = skeleton.copy()
    noise = rng.normal(0.0, sigma, size=(NUM_LANDMARKS, 3))
    sk[:, :3] += noise
    return sk


def _apply_pose_jitter(skeleton, rng):
    """Applica piccole variazioni "di esecuzione" della posa.

    Mentre il rumore e' del sensore, il jitter rappresenta il fatto che
    nessun atleta esegue la posa in modo identico: una rotazione globale
    lieve del corpo e una piccola traslazione.
    """
    sk = skeleton.copy()

    # Rotazione globale di un piccolo angolo attorno all'asse verticale
    # (simula il soggetto leggermente girato rispetto alla camera).
    theta = rng.normal(0.0, np.radians(7.0))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Rotazione nel piano x-z (l'asse y, verticale, resta invariato).
    x = sk[:, 0].copy()
    z = sk[:, 2].copy()
    sk[:, 0] = x * cos_t - z * sin_t
    sk[:, 2] = x * sin_t + z * cos_t

    # Piccola traslazione globale.
    sk[:, :3] += rng.normal(0.0, 0.02, size=3)
    return sk


def _apply_horizontal_flip(skeleton):
    """Esegue il flip orizzontale scambiando i landmark sinistra/destra.

    Per le pose laterali (Side Chest, Side Triceps) l'atleta puo' esibire
    indifferentemente il lato destro o sinistro: il flip insegna al modello
    l'invarianza assiale richiesta dal blueprint. Per le pose frontali e
    posteriori il flip raddoppia semplicemente la varieta' dei campioni.
    """
    from src.config import MIRROR_PAIRS

    sk = skeleton.copy()
    # Specchiare significa invertire il segno della coordinata x...
    sk[:, 0] *= -1.0
    # ...e scambiare i landmark dei due lati (la spalla sx diventa la dx).
    for left, right in MIRROR_PAIRS:
        sk[[left, right]] = sk[[right, left]]
    return sk


def _to_mediapipe_frame(skeleton, rng):
    """Converte lo scheletro nello spazio coordinate "stile MediaPipe".

    MediaPipe restituisce x, y normalizzati nell'intervallo [0, 1] rispetto
    al frame. Mappiamo lo scheletro (centrato sul corpo) dentro questo
    intervallo, posizionandolo in un punto casuale del "frame virtuale":
    cosi' il dataset contiene soggetti in posizioni e scale diverse, e il
    feature engineering dovra' davvero normalizzare (non e' un test banale).
    """
    sk = skeleton.copy()

    # Scala globale: il corpo occupa una frazione variabile del frame
    # (soggetto piu' vicino o piu' lontano dalla camera).
    body_scale = rng.uniform(0.18, 0.28)
    sk[:, :3] *= body_scale

    # Posizione del centro del corpo nel frame [0,1], con margini di
    # sicurezza per non far uscire i landmark dal frame.
    center_x = rng.uniform(0.40, 0.60)
    center_y = rng.uniform(0.45, 0.62)
    sk[:, 0] += center_x
    sk[:, 1] += center_y

    # I valori x, y vengono confinati in [0,1]; z resta relativo.
    sk[:, 0] = np.clip(sk[:, 0], 0.0, 1.0)
    sk[:, 1] = np.clip(sk[:, 1], 0.0, 1.0)
    return sk


# ===========================================================================
# 4. API PUBBLICA DEL GENERATORE
# ===========================================================================
def generate_sample(class_name, rng, allow_flip=True):
    """Genera UN singolo campione sintetico per la classe indicata.

    Pipeline di generazione (prototipo -> campione realistico):
      prototipo ideale
        -> variazione antropometrica
        -> jitter di posa
        -> (eventuale) flip orizzontale
        -> rumore sui giunti
        -> mappatura nello spazio coordinate MediaPipe [0,1]

    Parametri
    ---------
    class_name : str
        Nome della posa (deve essere in POSE_CLASSES).
    rng : np.random.Generator
        Generatore di numeri casuali (passato dall'esterno per garantire
        la riproducibilita').
    allow_flip : bool
        Se True, con probabilita' 0.5 il campione viene specchiato.

    Ritorna
    -------
    np.ndarray, shape (33, 4)
        Lo scheletro sintetico in formato MediaPipe [x, y, z, visibility].
    """
    if class_name not in _PROTOTYPE_BUILDERS:
        raise ValueError(f"Classe sconosciuta: {class_name}")

    # 1. Prototipo ideale -> scheletro completo a 33 landmark.
    proto = _PROTOTYPE_BUILDERS[class_name]()
    skeleton = _build_full_skeleton(proto, class_name)

    # 2. Variabilita' antropometrica (corporature diverse).
    skeleton = _apply_anthropometric_variation(skeleton, rng)

    # 3. Jitter di posa (esecuzione non identica).
    skeleton = _apply_pose_jitter(skeleton, rng)

    # 4. Flip orizzontale casuale (invarianza al lato).
    if allow_flip and rng.random() < 0.5:
        skeleton = _apply_horizontal_flip(skeleton)

    # 5. Rumore sui giunti (imprecisione del sensore).
    skeleton = _apply_joint_noise(skeleton, rng)

    # 6. Mappatura nello spazio MediaPipe [0,1].
    skeleton = _to_mediapipe_frame(skeleton, rng)

    return skeleton.astype(np.float32)


def generate_dataset(samples_per_class, seed=RANDOM_SEED, verbose=True):
    """Genera l'intero dataset sintetico bilanciato.

    Parametri
    ---------
    samples_per_class : int
        Numero di campioni da generare per ciascuna delle 7 classi.
    seed : int
        Seed per la riproducibilita'.

    Ritorna
    -------
    X : np.ndarray, shape (N, 33, 4)
        Tutti gli scheletri generati.
    y : np.ndarray, shape (N,)
        Le etichette intere corrispondenti (0..6).
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for class_id, class_name in enumerate(POSE_CLASSES):
        for _ in range(samples_per_class):
            sample = generate_sample(class_name, rng, allow_flip=True)
            X.append(sample)
            y.append(class_id)
        if verbose:
            print(f"  [{class_id}] {class_name:24s}: "
                  f"{samples_per_class} campioni generati")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Mescola il dataset: importante per non avere le classi in blocchi
    # ordinati (cosa che puo' falsare alcuni split o batch).
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def generate_clean_prototype(class_name):
    """Restituisce lo scheletro IDEALE di una posa, senza rumore.

    Usato per costruire le pose di riferimento (pose scoring): rappresenta
    l'esecuzione "da manuale" con cui confrontare quella dell'atleta.
    """
    proto = _PROTOTYPE_BUILDERS[class_name]()
    skeleton = _build_full_skeleton(proto, class_name)
    # Mappiamo al centro del frame, scala fissa, senza alcuna aleatorieta'.
    skeleton[:, :3] *= 0.23
    skeleton[:, 0] += 0.5
    skeleton[:, 1] += 0.53
    return skeleton.astype(np.float32)


if __name__ == "__main__":
    # Test rapido: genera un mini-dataset e ne stampa le dimensioni.
    print("Test del generatore sintetico...")
    X, y = generate_dataset(samples_per_class=5, verbose=True)
    print(f"\nDataset di test: X={X.shape}, y={y.shape}")
    print(f"Classi presenti: {np.unique(y)}")
    print(f"Range coordinate x: [{X[:,:,0].min():.3f}, {X[:,:,0].max():.3f}]")
    print(f"Range coordinate y: [{X[:,:,1].min():.3f}, {X[:,:,1].max():.3f}]")
    print("Generatore funzionante.")
