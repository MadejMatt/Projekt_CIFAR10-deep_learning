# Klasyfikacja obrazów CIFAR-10 — głębokie sieci konwolucyjne

Projekt klasyfikacji obrazów ze zbioru **CIFAR-10** (10 klas, 32×32 px, RGB) zrealizowany w **PyTorch**.
Całość prowadzona jest jako spójna **historia badawcza**: od najprostszego modelu bazowego, przez dobór
sposobu treningu i strojenie hiperparametrów, aż po nowoczesne architektury głębokie (VGG, ResNet). Każdą
decyzję projektową uzasadniamy wynikami eksperymentów, a nie z góry przyjętym „najlepszym" modelem.

**Autorzy:** Piotr Dyga, Wiktor Bojarski, Mateusz Madej
**Stos:** Python 3.12 · PyTorch (CUDA 12.8) · trening na NVIDIA RTX 5070

---

## Najważniejsze wyniki

Pełna ścieżka jakości — od surowego baseline'u do finalnej sieci rezydualnej:

| Model | Test accuracy | F1 macro | Liczba parametrów |
|---|---|---|---|
| Baseline CNN (surowy) | 0.781 | 0.782 | ≈ 0,62 mln |
| VGG-like (głęboka, bez skip connections) | 0.938 | 0.938 | ≈ 3,97 mln |
| **ResNet-like (połączenia rezydualne)** | **0.952** | **0.952** | ≈ 11,17 mln |

**Wnioski w skrócie:**
- **Harmonogram `learning rate` ważył więcej niż wybór optymalizatora** — `CosineAnnealingLR` podniósł SGD o ~3,6 pp i pozwolił mu wyprzedzić Adama.
- **Augmentacja danych była pojedynczo najsilniejszym czynnikiem** w strojeniu (+6–7 pp), bo bezpośrednio leczyła przeuczenie baseline'u.
- **Głębokość architektury** dała kolejny duży skok (VGG: +~6 pp), a **połączenia rezydualne** jeszcze +1,4 pp — ale już znacznie mniejszym kosztem zysku do liczby parametrów.
- Najtrudniejszą klasą na każdym etapie pozostaje **kot** (mylony głównie z psem); najłatwiejsze są pojazdy.

---

## Struktura repozytorium

```
Projekt_CIFAR10/
├── src/
│   └── cifar10_pipeline.py        # cały pipeline: dane, modele, trening, ewaluacja, grid search
├── notebooks/
│   ├── EDA & Preprocessing.ipynb  # eksploracja danych i preprocessing
│   └── CIFAR10_Modeling.ipynb     # notebook badawczy — 6 etapów modelowania
├── experiments/                   # artefakty eksperymentów (tworzone automatycznie przez pipeline)
│   ├── results_summary.csv        # zbiorcze wyniki (1 wiersz = 1 eksperyment)
│   ├── baseline/{nazwa}/          # config.json, metrics.json, history.csv, *.png
│   ├── tuning/                    # grid_results.csv, best_config.json
│   ├── vgg_like/{nazwa}/
│   └── resnet_like/{nazwa}/
├── reports/
│   ├── Notatki_modelowanie_do_raportu.md  # notatki merytoryczne do rozdziału „Modelowanie"
│   └── Raport z Projektu.docx
├── data/                          # NIE w repo — zbyt duże (patrz niżej)
│   └── cifar10_preprocessed.npz
├── requirements.txt
└── README.md
```

> **Uwaga:** plik z danymi (`data/cifar10_preprocessed.npz`, ~229 MB) oraz wagi modeli
> (`experiments/**/model.pt`) są celowo **wykluczone z repozytorium** (`.gitignore`) ze względu na rozmiar.
> W repo zostają lekkie artefakty: metryki (`metrics.json`), konfiguracje, pliki CSV i wykresy (`*.png`).

---

## Podejście badawcze (6 etapów)

Notebook `notebooks/CIFAR10_Modeling.ipynb` prowadzi przez kolejne etapy, z których każdy wynika z wniosków poprzedniego:

1. **Model bazowy (Baseline CNN)** — prosta sieć (3 bloki Conv-BN-ReLU-MaxPool + głowa FC) jako uczciwy punkt odniesienia.
2. **Adam vs SGD i rola harmonogramu LR** — kontrolowana ablacja 2×2 (`{Adam, SGD} × {stały LR, cosine}`) izolująca wpływ harmonogramu od wyboru optymalizatora.
3. **Strojenie hiperparametrów (Grid Search)** — 36 kombinacji `lr × batch_size × dropout × augmentacja`, selekcja wyłącznie na zbiorze walidacyjnym.
4. **Architektura głęboka typu VGG** — sprawdzenie, ile daje sama głębokość (bez skip connections).
5. **Architektura z połączeniami rezydualnymi (ResNet)** — czy „autostrada dla gradientu" przełamuje ograniczenia głębokich sieci.
6. **Porównanie modeli i analiza błędów** — zestawienie zbiorcze + macierze pomyłek.

Metodyka: podział **train / validation / test** (zbiór testowy nietknięty do oceny końcowej), ustalone ziarno losowości dla powtarzalności, wybór najlepszego modelu po najniższej stracie walidacyjnej. Metryki: `accuracy`, `F1-score (macro)`, macierz pomyłek.

---

## Środowisko i instalacja

1. Utwórz i aktywuj wirtualne środowisko (Python 3.12):
   ```bash
   python3.12 -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate     # Linux / WSL
   ```

2. Zainstaluj PyTorch pod **CUDA 12.8** (wymagane dla RTX 5070 / Blackwell `sm_120`):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

3. Zainstaluj pozostałe zależności:
   ```bash
   pip install -r requirements.txt
   ```

4. Sprawdź dostępność GPU (powinno zwrócić `True` i nazwę karty):
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

---

## Jak uruchomić

Otwórz `notebooks/CIFAR10_Modeling.ipynb` i wykonuj komórki etap po etapie. Notebook importuje funkcje
z `src/cifar10_pipeline.py`; foldery w `experiments/` tworzą się automatycznie. **Grid search jest
wznawialny** — po przerwaniu wystarczy uruchomić komórkę ponownie, a ukończone konfiguracje zostaną pominięte.

Konwencja nazewnictwa eksperymentów:
```
{architektura}_bs{batch_size}_lr{lr}_drop{dropout}_aug{0|1}
```
Przykłady: `baseline_bs128_lr0.001_drop0.5_aug0`, `vgg_like_bs128_lr0.05_drop0.5_aug1`.

---

## Główne elementy `src/cifar10_pipeline.py`

| Obszar | Funkcje / klasy |
|---|---|
| Reprodukowalność | `set_seed`, `get_device` |
| Dane | `load_data`, `stratified_split`, `CIFAR10Dataset`, `build_transforms`, `make_dataloaders` |
| Architektury | `build_baseline_cnn`, `build_vgg_like`, `build_resnet_like`, `count_parameters` |
| Trening | `build_optimizer`, `build_scheduler`, `build_criterion`, `EarlyStopping`, `train_model` |
| Ewaluacja | `evaluate_model`, `plot_history`, `plot_confusion_matrix` |
| Eksperymenty | `make_experiment_name`, `save_experiment`, `update_results_summary`, `run_experiment` |
| Grid search | `run_grid_search` |

---

## Transparentność — wykorzystanie narzędzi AI

Do zbudowania wysokiej jakości kodu źródłowego pipeline'u (`src/cifar10_pipeline.py`) wykorzystaliśmy
asystenta **Claude Opus 4.8** (Anthropic).

Cała **praca koncepcyjna i badawcza** — projekt eksperymentów, dobór architektur i hiperparametrów,
interpretacja wyników oraz wnioski — została wykonana **samodzielnie**. Łączny nakład pracy: **ok. 15–20 godzin**.
