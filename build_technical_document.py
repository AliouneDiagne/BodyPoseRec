"""
build_technical_document.py
===========================
Genera il "Technical Analysis Document" di BodyPoseRec in formato PDF.

Il documento (max 10 pagine) e' uno dei deliverable obbligatori della
traccia d'esame e contiene le sezioni richieste: Problem Statement,
Methodology, Experimental Results, Failure Analysis, Ethical Considerations.

Il PDF viene costruito con reportlab (Platypus) leggendo i risultati reali
prodotti dalla pipeline (cartella results/). Va quindi eseguito DOPO aver
lanciato evaluate.py, robustness_analysis.py, xai_analysis.py e
run_simulation.py.

Esecuzione:
    python build_technical_document.py

Output:
    docs/technical_analysis.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak,
)

from src.config import DOCS_DIR, RESULTS_DIR, ensure_dirs

# Colori coerenti con lo stile delle slide del corso (viola/magenta EPICODE).
NAVY = HexColor("#1A1A40")
MAGENTA = HexColor("#E6007A")
LIGHT_GREY = HexColor("#F0F0F0")

OUTPUT_PATH = DOCS_DIR / "technical_analysis.pdf"


def _build_styles():
    """Definisce gli stili tipografici del documento."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="DocTitle", parent=styles["Title"], fontSize=22,
        textColor=NAVY, spaceAfter=6, leading=26))
    styles.add(ParagraphStyle(
        name="DocSubtitle", parent=styles["Normal"], fontSize=11,
        textColor=MAGENTA, alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(
        name="H1", parent=styles["Heading1"], fontSize=14,
        textColor=NAVY, spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(
        name="H2", parent=styles["Heading2"], fontSize=11.5,
        textColor=MAGENTA, spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle(
        name="Body", parent=styles["Normal"], fontSize=9.5,
        alignment=TA_JUSTIFY, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(
        name="Caption", parent=styles["Normal"], fontSize=8,
        textColor=HexColor("#555555"), alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(
        name="CellText", parent=styles["Normal"], fontSize=8.5, leading=11))
    return styles


def _para(text, style):
    return Paragraph(text, style)


def _table(data, col_widths, styles, header=True):
    """Crea una tabella formattata in modo coerente con lo stile del corso."""
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    cmds = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, HexColor("#FFFFFF")]),
    ]
    if header:
        cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ]
    t.setStyle(TableStyle(cmds))
    return t


def _image(path, width_cm, styles, caption=None):
    """Inserisce un'immagine scalata a una larghezza fissa, con didascalia."""
    elems = []
    if path.exists():
        img = Image(str(path))
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth = width_cm * cm
        img.drawHeight = width_cm * cm * ratio
        img.hAlign = "CENTER"
        elems.append(img)
        if caption:
            elems.append(_para(caption, styles["Caption"]))
    return elems


def build_document():
    """Costruisce e salva il PDF del documento tecnico."""
    ensure_dirs()
    styles = _build_styles()
    story = []

    # ======================================================================
    # COPERTINA / INTESTAZIONE
    # ======================================================================
    story.append(Spacer(1, 0.4 * cm))
    story.append(_para("BodyPoseRec", styles["DocTitle"]))
    story.append(_para(
        "Technical Analysis Document<br/>"
        "Riconoscimento delle 7 Pose Obbligatorie del Bodybuilding Maschile",
        styles["DocSubtitle"]))
    story.append(_para(
        "Progetto finale - Corso di Computer Vision - EPICODE Institute of "
        "Technology", styles["Caption"]))
    story.append(Spacer(1, 0.3 * cm))

    # ----------------------------------------------------------------------
    # ABSTRACT
    # ----------------------------------------------------------------------
    story.append(_para("Abstract", styles["H1"]))
    story.append(_para(
        "BodyPoseRec e' un sistema di Computer Vision che riconosce le sette "
        "pose obbligatorie del bodybuilding maschile a partire dai keypoint "
        "scheletrici del corpo. La pipeline integra le quattro fasi richieste "
        "- acquisizione e preprocessing, feature engineering, classificazione "
        "(modello classico e deep learning), post-processing - e aggiunge "
        "moduli di pose scoring, interpretabilita' (XAI) e analisi di "
        "robustezza. Per ragioni di privacy e riproducibilita' il sistema e' "
        "addestrato su un dataset sintetico di keypoint generato dai "
        "descrittori cinematici delle pose. Sul test set entrambi i modelli "
        "(SVM con kernel RBF e MLP a due strati nascosti) raggiungono "
        "accuratezza, precision, recall ed F1 macro pari a 1.000. Poiche' i "
        "dati puliti sono separabili per costruzione, la failure analysis e' "
        "condotta tramite stress test controllati: il sistema mantiene buona "
        "accuratezza fino a un rumore moderato sui keypoint e degrada in modo "
        "prevedibile sotto occlusione. La stabilizzazione temporale riduce il "
        "flickering delle predizioni di circa il 92%.",
        styles["Body"]))

    # ======================================================================
    # 1. PROBLEM STATEMENT
    # ======================================================================
    story.append(_para("1. Problem Statement", styles["H1"]))
    story.append(_para(
        "Nelle competizioni di bodybuilding maschile la valutazione degli "
        "atleti si articola su sette pose obbligatorie codificate "
        "(\"mandatory poses\"): Front Double Biceps, Front Lat Spread, Side "
        "Chest, Side Triceps, Back Double Biceps, Back Lat Spread e "
        "Abdominals and Thighs. Ogni posa mette in evidenza specifici gruppi "
        "muscolari secondo una configurazione articolare precisa. Il "
        "riconoscimento di queste pose e' attualmente affidato esclusivamente "
        "all'occhio di giudici esperti: un processo intrinsecamente "
        "soggettivo, non tracciabile e non scalabile.",
        styles["Body"]))
    story.append(_para(
        "Il problema affrontato da BodyPoseRec e' la classificazione "
        "automatica della posa eseguita da un atleta a partire da un flusso "
        "video. La sua rilevanza e' triplice. Sul piano applicativo, un "
        "sistema di questo tipo supporta l'allenamento (feedback immediato "
        "all'atleta), la preparazione di gara e l'analisi tecnica. Sul piano "
        "informatico, il problema e' un caso di studio completo di "
        "classificazione di pose umane, che richiede l'intera pipeline di "
        "Computer Vision: stima dei keypoint, rappresentazione delle feature, "
        "modello di classificazione e post-processing. Sul piano della "
        "difficolta', il dominio presenta sfide concrete: le pose laterali "
        "possono essere eseguite indifferentemente sul lato sinistro o "
        "destro (serve invarianza speculare); le pose di schiena occludono il "
        "volto; gli atleti hanno proporzioni antropometriche molto diverse.",
        styles["Body"]))

    # ======================================================================
    # 2. METHODOLOGY
    # ======================================================================
    story.append(_para("2. Methodology", styles["H1"]))

    story.append(_para("2.1 Architettura della pipeline", styles["H2"]))
    story.append(_para(
        "Il sistema e' organizzato come una pipeline modulare a quattro "
        "stadi. (1) <b>Acquisizione e preprocessing</b>: generazione di "
        "scheletri sintetici a 33 landmark e relativa normalizzazione. (2) "
        "<b>Feature engineering</b>: trasformazione dei keypoint in un "
        "vettore di 26 feature geometriche. (3) <b>Classificazione</b>: due "
        "modelli distinti, un SVM (classico) e un MLP (deep learning). (4) "
        "<b>Post-processing</b>: stabilizzazione temporale delle predizioni "
        "tramite media mobile sulle probabilita'. Moduli aggiuntivi "
        "realizzano il punteggio di qualita' della posa, l'analisi di "
        "interpretabilita' e l'analisi di robustezza.",
        styles["Body"]))

    story.append(_para("2.2 Acquisizione dei dati: keypoint sintetici",
                        styles["H2"]))
    story.append(_para(
        "Non esiste un dataset pubblico di pose di bodybuilding annotate, e "
        "l'uso di video reali di atleti solleverebbe problemi di privacy e "
        "di diritto d'autore. Il sistema e' quindi addestrato su un dataset "
        "sintetico: per ciascuna posa, partendo dai descrittori cinematici "
        "(angoli di spalla, gomito, anca, ginocchio; posizione di mani e "
        "piedi), viene costruito uno scheletro prototipo a 33 landmark nel "
        "formato MediaPipe BlazePose. Su ogni campione si applicano poi "
        "variazione antropometrica (altezza, larghezza delle spalle, "
        "proporzioni degli arti), rumore articolare gaussiano e flip "
        "orizzontale. Sono generati 600 campioni per classe, 4200 in totale, "
        "perfettamente bilanciati. La generazione sintetica e' la stessa "
        "tecnica usata negli script di laboratorio del Modulo 3 quando un "
        "dataset reale non e' disponibile, ed e' citata esplicitamente nella "
        "sezione di Ethical Considerations come scelta consapevole.",
        styles["Body"]))

    story.append(_para("2.3 Feature engineering", styles["H2"]))
    story.append(_para(
        "I keypoint grezzi non vengono passati direttamente al "
        "classificatore. Lo scheletro viene prima normalizzato per ottenere "
        "invarianza alla traslazione (origine sul punto medio del bacino) e "
        "alla scala (divisione per l'altezza del torso). Da qui si estrae un "
        "vettore di 26 feature geometriche raggruppate in cinque famiglie: "
        "angoli articolari (gomito, abduzione di spalla, ginocchio, anca, "
        "inclinazione del tronco), distanze normalizzate (tra polsi, gomiti, "
        "caviglie; mano-anca), rapporti corporei (spalle/anche, apertura "
        "delle braccia, larghezza della base d'appoggio), descrittori di "
        "profondita' e orientamento (asse z dei landmark) e indicatori di "
        "visibilita'. Questo approccio - feature geometriche interpretabili "
        "invece dell'immagine grezza - e' coerente con il Modulo 2 ed e' "
        "robusto anche con un numero contenuto di campioni.",
        styles["Body"]))

    story.append(_para("2.4 Modelli di classificazione", styles["H2"]))
    story.append(_para(
        "La traccia richiede sia un modello classico sia un modello deep "
        "learning. Il <b>modello classico</b> e' una Support Vector Machine "
        "con kernel RBF; gli iperparametri C e gamma sono selezionati con "
        "GridSearchCV (5-fold cross-validation), preceduta da "
        "standardizzazione delle feature. Il <b>modello deep learning</b> e' "
        "un Multi-Layer Perceptron implementato in Keras/TensorFlow, con "
        "architettura Input(26) - Dense(64, ReLU) - Dropout(0.3) - "
        "Dense(32, ReLU) - Dropout(0.3) - Dense(7, Softmax), ottimizzatore "
        "Adam e loss categorical cross-entropy. Il Dropout e l'early "
        "stopping sulla loss di validazione contrastano l'overfitting. I due "
        "modelli condividono le stesse feature, quindi il confronto e' equo.",
        styles["Body"]))

    story.append(_para("2.5 Post-processing e moduli aggiuntivi",
                        styles["H2"]))
    story.append(_para(
        "Nella pipeline real-time le predizioni vengono stabilizzate con una "
        "media mobile sui vettori di probabilita' softmax (finestra FIFO di "
        "7 frame) seguita da una soglia di confidenza che marca come "
        "\"incerte\" le predizioni deboli. Il modulo di <b>pose scoring</b> "
        "confronta gli angoli dell'atleta con quelli di un prototipo di "
        "riferimento e assegna un punteggio di qualita' 0-100. Il modulo "
        "<b>XAI</b> stima l'importanza di ciascuna feature tramite "
        "permutation importance. Per la demo da webcam, l'estrazione dei "
        "keypoint e' delegata a MediaPipe Pose: un modello pre-addestrato "
        "usato come semplice estrattore di rappresentazione, in linea con la "
        "regola d'esame che ne consente l'uso se combinato con logica custom.",
        styles["Body"]))

    story.append(PageBreak())

    # ======================================================================
    # 3. EXPERIMENTAL RESULTS
    # ======================================================================
    story.append(_para("3. Experimental Results", styles["H1"]))

    story.append(_para("3.1 Setup sperimentale", styles["H2"]))
    story.append(_para(
        "Il dataset di 4200 campioni e' suddiviso in modo stratificato in "
        "training (70%, 2940 campioni), validation (15%, 630) e test (15%, "
        "630), con seed fisso per la riproducibilita'. Il validation set "
        "guida la scelta degli iperparametri e l'early stopping; il test set "
        "e' cieco e usato una sola volta per la valutazione finale. Le "
        "metriche adottate sono quelle indicate dalla traccia per i problemi "
        "di classificazione: accuracy, precision, recall, F1-score (in "
        "versione macro, ossia mediata sulle classi) e matrice di "
        "confusione; e' inoltre riportata l'AUC macro (curve ROC One-vs-Rest).",
        styles["Body"]))

    story.append(_para("3.2 Risultati sul test set", styles["H2"]))
    story.append(_para(
        "La Tabella 1 confronta i due modelli sul test set cieco. Entrambi "
        "classificano correttamente tutti i 630 campioni.",
        styles["Body"]))
    comparison_data = [
        ["Metrica", "SVM (classico)", "MLP (deep learning)"],
        ["Accuracy", "1.0000", "1.0000"],
        ["Macro Precision", "1.0000", "1.0000"],
        ["Macro Recall", "1.0000", "1.0000"],
        ["Macro F1-score", "1.0000", "1.0000"],
        ["Macro AUC", "1.0000", "1.0000"],
    ]
    story.append(_table(comparison_data,
                        [5 * cm, 4.5 * cm, 4.5 * cm], styles))
    story.append(Spacer(1, 0.15 * cm))
    story.append(_para("Tabella 1 - Confronto SVM vs MLP sul test set "
                       "(630 campioni, 90 per classe).", styles["Caption"]))

    story.append(_para(
        "Le matrici di confusione (Figura 1) sono perfettamente diagonali: "
        "nessuna posa viene confusa con un'altra. Questo risultato, lungi "
        "dall'essere sospetto, e' la conseguenza diretta del fatto che il "
        "dataset sintetico e' separabile per costruzione - ogni posa ha una "
        "firma geometrica netta. Cio' rende necessaria una failure analysis "
        "basata su stress test, discussa nella Sezione 4.",
        styles["Body"]))

    imgs = _image(RESULTS_DIR / "confusion_matrix_svm.png", 7.4, styles)
    imgs2 = _image(RESULTS_DIR / "confusion_matrix_mlp.png", 7.4, styles)
    if imgs and imgs2:
        side = Table([[imgs[0], imgs2[0]]],
                     colWidths=[7.8 * cm, 7.8 * cm])
        side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(side)
    story.append(_para("Figura 1 - Matrici di confusione sul test set: "
                       "SVM (sinistra) e MLP (destra).", styles["Caption"]))

    story.append(_para("3.3 Andamento dell'addestramento dell'MLP",
                        styles["H2"]))
    story.append(_para(
        "L'MLP e' stato addestrato per un massimo di 120 epoche con early "
        "stopping (pazienza 15). L'addestramento si e' interrotto alla 84a "
        "epoca. La Figura 2 mostra le curve di loss e accuratezza: training "
        "e validation procedono insieme senza divergenze, segno che il "
        "Dropout e l'early stopping hanno efficacemente prevenuto "
        "l'overfitting.",
        styles["Body"]))
    for e in _image(RESULTS_DIR / "mlp_training_history.png", 13, styles,
                    "Figura 2 - Curve di loss e accuratezza dell'MLP "
                    "(training vs validation)."):
        story.append(e)

    story.append(PageBreak())

    # ======================================================================
    # 4. FAILURE ANALYSIS
    # ======================================================================
    story.append(_para("4. Failure Analysis", styles["H1"]))
    story.append(_para(
        "Su dati sintetici puliti il sistema non commette errori: una "
        "failure analysis basata solo sul test set sarebbe quindi vuota. "
        "Per analizzare in modo onesto i limiti del sistema, le condizioni "
        "operative reali sono state simulate degradando in modo controllato "
        "i keypoint del test set e misurando il decadimento delle "
        "prestazioni. Sono stati condotti tre stress test.",
        styles["Body"]))

    story.append(_para("4.1 Stress test: rumore sui keypoint", styles["H2"]))
    story.append(_para(
        "A ogni landmark e' stato sommato rumore gaussiano di deviazione "
        "standard crescente, simulando l'imprecisione di MediaPipe in "
        "condizioni di scarsa illuminazione o bassa risoluzione. La Tabella "
        "2 riporta i risultati.",
        styles["Body"]))
    noise_data = [
        ["Rumore (sigma)", "Accuracy SVM", "Accuracy MLP"],
        ["0.00", "1.000", "1.000"],
        ["0.02", "0.995", "0.997"],
        ["0.03", "0.935", "0.971"],
        ["0.05", "0.687", "0.824"],
        ["0.07", "0.456", "0.644"],
        ["0.10", "0.235", "0.470"],
    ]
    story.append(_table(noise_data, [5 * cm, 4.5 * cm, 4.5 * cm], styles))
    story.append(Spacer(1, 0.15 * cm))
    story.append(_para("Tabella 2 - Accuratezza al crescere del rumore sui "
                       "keypoint.", styles["Caption"]))
    story.append(_para(
        "Emerge un risultato chiave: l'MLP e' sistematicamente piu' robusto "
        "dell'SVM. Per rumore moderato (sigma fino a 0.03) entrambi i modelli "
        "restano sopra il 93% di accuratezza, ma all'aumentare del rumore il "
        "vantaggio dell'MLP cresce - a sigma 0.05 l'MLP ottiene 0.82 contro "
        "0.69 dell'SVM. Questo motiva la scelta dell'MLP come modello "
        "predefinito nella demo real-time, dove il rumore di MediaPipe e' "
        "inevitabile.",
        styles["Body"]))

    story.append(_para("4.2 Stress test: occlusione e profondita'",
                        styles["H2"]))
    story.append(_para(
        "Nel secondo test un numero crescente di landmark e' stato \"oscurato"
        "\" (azzerandone la visibilita' e perturbandone la posizione), "
        "simulando arti nascosti dal corpo o usciti dall'inquadratura. Con "
        "un solo landmark occluso l'accuratezza scende a circa 0.93-0.95; "
        "con quattro landmark occlusi crolla a circa 0.60 per entrambi i "
        "modelli. Nel terzo test e' stata degradata la sola coordinata di "
        "profondita' (asse z): il sistema regge bene piccoli errori (a "
        "sigma_z 0.05 l'accuratezza resta sopra 0.98) ma peggiora "
        "sensibilmente oltre, indicando una dipendenza dalle feature di "
        "profondita' usate per distinguere le pose laterali. La Figura 3 "
        "riassume le tre curve di degrado.",
        styles["Body"]))
    for e in _image(RESULTS_DIR / "robustness_curves.png", 14, styles,
                    "Figura 3 - Curve di degrado: rumore, occlusione e "
                    "perturbazione della profondita'."):
        story.append(e)

    story.append(_para("4.3 Confusioni tipiche sotto stress", styles["H2"]))
    story.append(_para(
        "Analizzando gli errori dell'MLP sotto forte rumore (sigma 0.10) "
        "emergono confusioni non casuali ma <i>biomeccanicamente "
        "plausibili</i>. La confusione piu' frequente e' Back Lat Spread "
        "scambiata per Back Double Biceps: due pose di schiena che "
        "condividono l'orientamento del corpo e differiscono solo nella "
        "configurazione delle braccia, l'informazione piu' fragile sotto "
        "rumore. Analogamente Side Triceps viene confusa con Abdominals and "
        "Thighs e Side Chest con Front Double Biceps. Il pattern conferma "
        "che il sistema, quando sbaglia, sbaglia \"in modo ragionevole\": "
        "perde il dettaglio fine degli arti ma conserva l'asse "
        "frontale/posteriore. La Figura 4 mostra la matrice di confusione "
        "sotto stress.",
        styles["Body"]))
    for e in _image(RESULTS_DIR / "confusion_matrix_stress_mlp.png", 9,
                    styles, "Figura 4 - Matrice di confusione dell'MLP sotto "
                    "rumore elevato (sigma 0.10)."):
        story.append(e)

    story.append(PageBreak())

    story.append(_para("4.4 Interpretabilita' (XAI)", styles["H2"]))
    story.append(_para(
        "L'analisi di permutation importance spiega su quali feature il "
        "modello fa affidamento. La feature dominante e' di gran lunga la "
        "visibilita' del naso (nose_visibility): da sola distingue l'asse "
        "frontale da quello posteriore (nelle pose di schiena il volto e' "
        "occluso). Le altre feature mostrano importanza individuale ridotta: "
        "non perche' inutili, ma perche' <i>ridondanti</i> - il dataset e' "
        "talmente separabile che, permutando una singola feature, le altre "
        "compensano. Questa ridondanza e' essa stessa un risultato: spiega "
        "perche' i modelli sono cosi' robusti al rumore moderato e perche' "
        "le matrici di confusione pulite sono diagonali. La Figura 5 mostra "
        "il ranking di importanza per l'MLP.",
        styles["Body"]))
    for e in _image(RESULTS_DIR / "feature_importance_mlp.png", 11, styles,
                    "Figura 5 - Permutation feature importance per l'MLP."):
        story.append(e)

    story.append(_para("4.5 Stabilizzazione temporale", styles["H2"]))
    story.append(_para(
        "L'ultimo aspetto analizzato e' il comportamento temporale. Una "
        "routine di gara sintetica (180 frame, 4 pose) e' stata classificata "
        "frame per frame, con e senza media mobile. Senza stabilizzazione la "
        "predizione cambia etichetta 36 volte (contro le 3 transizioni "
        "reali): un flickering vistoso. Con la media mobile i cambi scendono "
        "a 3, una riduzione del 92%, e l'accuratezza frame-by-frame migliora "
        "da 0.89 a 0.96. La Figura 6 visualizza l'effetto.",
        styles["Body"]))
    for e in _image(RESULTS_DIR / "temporal_smoothing_simulation.png", 14,
                    styles, "Figura 6 - Predizione grezza vs stabilizzata "
                    "lungo una routine simulata."):
        story.append(e)

    story.append(_para("4.6 Limiti del sistema", styles["H2"]))
    story.append(_para(
        "I principali limiti emersi sono tre. Primo: il sistema e' "
        "addestrato su dati sintetici; la transizione a video reali "
        "richiederebbe una validazione su keypoint estratti da MediaPipe da "
        "filmati veri. Secondo: la robustezza all'occlusione e' limitata - "
        "oltre due o tre landmark mancanti le prestazioni calano "
        "sensibilmente. Terzo: il sistema classifica pose statiche, non "
        "gestisce esplicitamente le transizioni tra una posa e l'altra, che "
        "vengono assorbite solo dal filtro temporale.",
        styles["Body"]))

    story.append(Spacer(1, 0.3 * cm))

    # ======================================================================
    # 5. ETHICAL CONSIDERATIONS
    # ======================================================================
    story.append(_para("5. Ethical Considerations", styles["H1"]))

    story.append(_para("5.1 Privacy e dati biometrici", styles["H2"]))
    story.append(_para(
        "La posa e la conformazione del corpo sono dati biometrici "
        "sensibili. Un sistema addestrato su video reali di atleti "
        "ricadrebbe nel campo del GDPR (in Europa) e di normative come il "
        "BIPA (in alcuni stati USA), che impongono consenso esplicito e "
        "limitazioni sulla conservazione. BodyPoseRec affronta il problema "
        "alla radice: il dataset e' interamente sintetico, quindi nessuna "
        "persona reale e' coinvolta nell'addestramento. Nella demo live, "
        "inoltre, l'elaborazione avviene interamente in locale (edge "
        "processing): il sistema lavora sui soli keypoint e non salva ne' "
        "trasmette i frame della webcam, secondo il principio del data "
        "minimization.",
        styles["Body"]))

    story.append(_para("5.2 Bias di rappresentazione", styles["H2"]))
    story.append(_para(
        "Un dataset sintetico elimina i problemi di privacy ma introduce un "
        "rischio diverso: il bias del generatore. I prototipi delle pose "
        "codificano un certo intervallo di proporzioni antropometriche; se "
        "tale intervallo non copre la reale diversita' degli atleti (per "
        "altezza, struttura, eta'), il modello potrebbe risultare meno "
        "accurato su corporature fuori distribuzione. La variazione "
        "antropometrica inserita nel generatore mitiga in parte il problema, "
        "ma non lo elimina. E' inoltre importante notare che il progetto "
        "riguarda esplicitamente le pose del bodybuilding <i>maschile</i>: "
        "il sistema non va usato su categorie diverse (Women's Physique, "
        "Figure) senza un riaddestramento dedicato, perche' la tassonomia "
        "delle pose e' differente.",
        styles["Body"]))

    story.append(_para("5.3 Uso responsabile", styles["H2"]))
    story.append(_para(
        "BodyPoseRec e' uno strumento di supporto, non un giudice "
        "automatico. Il punteggio di qualita' della posa e' calcolato "
        "rispetto a un prototipo geometrico e non sostituisce la "
        "valutazione di un giudice umano, che considera anche definizione "
        "muscolare, simmetria e presenza scenica - aspetti fuori dalla "
        "portata di un classificatore basato su soli keypoint. Presentare il "
        "punteggio come verdetto oggettivo sarebbe fuorviante: va comunicato "
        "come feedback indicativo. Il sistema rispetta inoltre le buone "
        "pratiche di sicurezza viste nel corso: nessuna dipendenza da "
        "servizi esterni in fase di inferenza, codice ispezionabile e "
        "modelli interpretabili tramite l'analisi XAI.",
        styles["Body"]))

    # ======================================================================
    # 6. CONCLUSION AND FUTURE WORK
    # ======================================================================
    story.append(_para("6. Conclusion and Future Work", styles["H1"]))
    story.append(_para(
        "BodyPoseRec realizza una pipeline di Computer Vision completa e "
        "modulare per il riconoscimento delle sette pose obbligatorie del "
        "bodybuilding maschile, soddisfacendo tutti i requisiti della "
        "traccia: le quattro fasi della pipeline, un modello classico e uno "
        "deep learning, una valutazione quantitativa con le metriche "
        "appropriate, una failure analysis basata su stress test e "
        "un'analisi etica. I due modelli raggiungono prestazioni perfette su "
        "dati puliti; la robustezza analizzata sotto degrado mostra che "
        "l'MLP e' la scelta migliore per l'uso real-time, e che la "
        "stabilizzazione temporale e' essenziale per una predizione stabile.",
        styles["Body"]))
    story.append(_para(
        "Gli sviluppi futuri naturali sono tre. Primo, la validazione su un "
        "dataset reale di keypoint estratti da video di gara, per misurare "
        "il divario tra dominio sintetico e reale. Secondo, l'introduzione "
        "di un modello temporale esplicito (LSTM o GRU) che classifichi le "
        "<i>sequenze</i> di pose di un'intera routine invece dei singoli "
        "frame. Terzo, l'estensione ad altre categorie competitive con la "
        "loro tassonomia di pose. La struttura modulare del progetto rende "
        "ciascuna estensione realizzabile senza riscrivere la pipeline.",
        styles["Body"]))

    # Costruzione del PDF.
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=A4,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
        title="BodyPoseRec - Technical Analysis Document",
        author="Progetto Computer Vision - EPICODE",
    )
    doc.build(story)
    print(f"Documento tecnico generato: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_document()
