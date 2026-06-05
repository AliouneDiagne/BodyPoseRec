# Installazione su Windows — Guida Completa

## Prerequisiti
- **Python 3.11** (non 3.12 o 3.13 — incompatibili con alcune librerie CV)
- Git installato

## Setup completo (prima volta)

```powershell
# Attiva l'ambiente virtuale
venv\Scripts\activate

# Installa NELL'ORDINE CORRETTO per evitare conflitti
pip install numpy scikit-learn matplotlib seaborn scipy pyyaml reportlab
pip install tensorflow
pip install opencv-python
pip install mediapipe==0.10.14
```

> **IMPORTANTE:** Non usare `pip install -r requirements.txt` tutto insieme
> se hai avuto problemi. Installa in quest'ordine per evitare conflitti.

## Ordine esecuzione pipeline

```powershell
python -m src.data_preparation.build_dataset
python -m src.classification.train_svm
python -m src.classification.train_mlp
python -m src.classification.evaluate
python -m src.classification.robustness_analysis
python -m src.classification.xai_analysis
python -m src.feature_extraction.build_reference_poses
python run_simulation.py          # demo senza webcam
python build_technical_document.py
```

## Demo live con webcam

```powershell
python run_demo.py
```

## Problemi comuni e soluzioni

### Errore: `mp.solutions` non trovato
```powershell
pip uninstall mediapipe -y
pip install mediapipe==0.10.14
```

### Errore: numpy incompatibile con scipy/sklearn
```powershell
pip install "numpy>=2.0"
```
> NON installare numpy 1.24.x: rompe scipy e scikit-learn

### Errore: tensorflow non si avvia
```powershell
pip uninstall tensorflow tensorflow-intel -y
pip install tensorflow
```
> NON installare tensorflow==2.13.0: richiede numpy 1.24 (incompatibile)

### Webcam non rilevata
```powershell
# Prova con indice diverso
python run_demo.py --camera 1
python run_demo.py --camera 2
```
Oppure modifica `src/realtime/config.yaml`: `camera_index: 1`

### Messaggi WARNING TensorFlow (normali, non bloccanti)
```
WARNING: oneDNN custom operations are on
WARNING: TensorFlow GPU support is not available on native Windows
```
Questi sono normali: TensorFlow non usa GPU su Windows nativo,
ma funziona perfettamente su CPU.

### `ModuleNotFoundError: No module named 'src'`
Assicurarsi di eseguire sempre dalla cartella radice del progetto:
```powershell
cd C:\...\BodyPoseRec    # cartella radice
python -m src.classification.train_svm    # con -m, non come file
```

## Combinazione versioni testata (funzionante)
| Libreria | Versione testata |
|----------|-----------------|
| Python | 3.11 |
| numpy | 2.4.6 |
| scikit-learn | 1.8.0 |
| tensorflow | 2.21.0 |
| scipy | 1.17.1 |
| matplotlib | 3.10.9 |
| mediapipe | 0.10.14 |
| opencv-python | 4.13.0 |
