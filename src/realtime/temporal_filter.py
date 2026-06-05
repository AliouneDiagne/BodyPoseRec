"""
temporal_filter.py
==================
Post-processing temporale per la pipeline real-time di BodyPoseRec.

Problema risolto da questo modulo
---------------------------------
Un classificatore applicato in modo indipendente a ogni singolo frame
"balla" (flickering): basta un keypoint rumoroso per far oscillare la
predizione tra due classi per un solo frame, producendo un'etichetta
instabile e fastidiosa a schermo.

La soluzione, descritta nel Modulo 4 (post-processing nelle pipeline di
video analysis), e' sfruttare la COERENZA TEMPORALE: la posa di un atleta
non cambia 30 volte al secondo, quindi predizioni di frame vicini devono
essere "fuse" prima di mostrare un risultato.

Questo file implementa tre filtri complementari:
1. MovingAverageFilter   - media mobile sui vettori di probabilita' softmax.
2. MajorityVoteFilter    - voto di maggioranza sulle ultime N etichette.
3. ConfidenceGate        - marca come "Uncertain" le predizioni a bassa
                           confidenza (gestione dell'incertezza).

Il filtro principale usato nella demo e' MovingAverageFilter, perche'
mediare le PROBABILITA' (e non le etichette gia' decise) conserva piu'
informazione: una sequenza [0.51, 0.49] e [0.49, 0.51] viene gestita in
modo morbido invece di alternare bruscamente le classi.

Riferimenti didattici:
- Modulo 4, "Post-processing temporale (smoothing delle predizioni)".
- Blueprint BodyPoseRec, sezione "Stabilizzazione Temporale (anti-jitter)".
"""

from collections import deque, Counter

import numpy as np

from src.config import NUM_CLASSES, TEMPORAL_WINDOW, CONFIDENCE_THRESHOLD


class MovingAverageFilter:
    """Media mobile sulle probabilita' softmax (finestra FIFO di ampiezza N).

    Mantiene una coda degli ultimi N vettori di probabilita'. A ogni nuovo
    frame restituisce la classe con probabilita' MEDIA piu' alta sulla
    finestra. E' il filtro raccomandato perche' fonde l'informazione
    probabilistica completa, non solo l'argmax.
    """

    def __init__(self, window_size=TEMPORAL_WINDOW, num_classes=NUM_CLASSES):
        self.window_size = window_size
        self.num_classes = num_classes
        # deque con maxlen: quando e' piena, l'inserimento di un nuovo
        # elemento scarta automaticamente il piu' vecchio (comportamento FIFO).
        self.buffer = deque(maxlen=window_size)

    def reset(self):
        """Svuota la finestra (es. quando l'atleta esce dall'inquadratura)."""
        self.buffer.clear()

    def update(self, prob_vector):
        """Aggiunge un nuovo vettore di probabilita' e restituisce la stima fusa.

        Parametri
        ---------
        prob_vector : np.ndarray di forma (num_classes,)
            Probabilita' softmax prodotte dal classificatore sul frame corrente.

        Ritorna
        -------
        (class_id, confidence, avg_vector)
            class_id   : indice della classe vincente sulla media.
            confidence : probabilita' media della classe vincente.
            avg_vector : vettore di probabilita' mediato sull'intera finestra.
        """
        prob_vector = np.asarray(prob_vector, dtype=np.float32).ravel()
        self.buffer.append(prob_vector)

        # Media aritmetica di tutti i vettori attualmente in finestra.
        avg_vector = np.mean(self.buffer, axis=0)
        class_id = int(np.argmax(avg_vector))
        confidence = float(avg_vector[class_id])
        return class_id, confidence, avg_vector


class MajorityVoteFilter:
    """Voto di maggioranza sulle ultime N etichette (classi gia' decise).

    Filtro piu' semplice della media mobile: invece delle probabilita'
    accumula gli argmax e restituisce l'etichetta piu' frequente. Lo
    forniamo per completezza didattica e come confronto.
    """

    def __init__(self, window_size=TEMPORAL_WINDOW):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def reset(self):
        self.buffer.clear()

    def update(self, class_id):
        """Aggiunge un'etichetta e restituisce (etichetta_maggioritaria, frazione)."""
        self.buffer.append(int(class_id))
        counts = Counter(self.buffer)
        winner, votes = counts.most_common(1)[0]
        fraction = votes / len(self.buffer)
        return winner, fraction


class ConfidenceGate:
    """Gestione dell'incertezza tramite soglia di confidenza.

    Se la probabilita' della classe vincente e' sotto la soglia, la
    predizione viene marcata come "incerta". Questo evita di mostrare con
    sicurezza una posa quando l'atleta e' in transizione tra due pose o non
    sta affatto posando (input out-of-distribution).
    """

    def __init__(self, threshold=CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    def check(self, confidence):
        """Restituisce True se la confidenza e' sufficiente per fidarsi."""
        return confidence >= self.threshold


class TemporalSmoother:
    """Filtro temporale completo: media mobile + soglia di confidenza.

    E' il componente effettivamente usato dalla demo real-time. Combina
    MovingAverageFilter e ConfidenceGate dietro un'unica interfaccia
    semplice: si passa il vettore di probabilita' del frame corrente e si
    ottiene una decisione gia' stabilizzata e validata.
    """

    def __init__(self, window_size=TEMPORAL_WINDOW,
                 confidence_threshold=CONFIDENCE_THRESHOLD,
                 num_classes=NUM_CLASSES):
        self.ma_filter = MovingAverageFilter(window_size, num_classes)
        self.gate = ConfidenceGate(confidence_threshold)

    def reset(self):
        self.ma_filter.reset()

    def update(self, prob_vector):
        """Elabora il vettore di probabilita' del frame corrente.

        Ritorna un dizionario con:
            class_id    : classe stabilizzata (int).
            confidence  : confidenza media della finestra (float).
            is_certain  : True se la confidenza supera la soglia.
            avg_vector  : vettore di probabilita' mediato.
        """
        class_id, confidence, avg_vector = self.ma_filter.update(prob_vector)
        is_certain = self.gate.check(confidence)
        return {
            "class_id": class_id,
            "confidence": confidence,
            "is_certain": is_certain,
            "avg_vector": avg_vector,
        }
