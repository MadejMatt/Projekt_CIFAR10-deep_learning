# Struktura repozytorium — Projekt CIFAR-10

Klasyfikacja obrazów CIFAR-10 (10 klas, 32×32 RGB) w PyTorch.
Autorzy: Piotr Dyga, Wiktor Bojarski, Mateusz Madej.

## Drzewo katalogów

```
Projekt_CIFAR10/
├── data/                                   # wykluczone z git (zbyt duze)
│   └── cifar10_preprocessed.npz            # x_train, y_train, x_test, y_test (float32 [0,1])
├── src/
│   └── cifar10_pipeline.py                 # caly pipeline: dane, modele, trening, ewaluacja, grid search
├── experiments/                            # tworzone automatycznie przez pipeline
│   ├── results_summary.csv                 # 1 wiersz = 1 eksperyment, append-only
│   ├── baseline/{nazwa}/                    # config.json, metrics.json, history.csv, model.pt, *.png
│   ├── tuning/                              # grid_results.csv, best_config.json
│   ├── vgg_like/{nazwa}/ ...
│   └── resnet_like/{nazwa}/ ...
├── notebooks/
│   ├── EDA & Preprocessing.ipynb           # nie ruszamy (EDA + preprocessing)
│   └── CIFAR10_Modeling.ipynb              # notebook badawczy (6 etapow)
├── reports/
│   └── Raport z Projektu.docx              # nie ruszamy
├── Opus_masterPrompt.md                    # opis intencji pipeline (+ komentarze opus_comm)
├── repo_schema.md
├── .gitignore
└── requirements.txt
```

> Uwaga: pliki `cifar10_preprocessed.npz`, `EDA & Preprocessing.ipynb` oraz
> `Raport z Projektu.docx` znajduja sie obecnie w korzeniu projektu. Sciezki w
> notebooku sa zdefiniowane wzgledem `PROJECT_ROOT` i mozna je latwo dostosowac,
> jesli zdecydujecie sie przeniesc pliki do `notebooks/` i `reports/`.

## Konwencja nazewnictwa eksperymentow

`{architektura}_bs{batch_size}_lr{lr}_drop{dropout}_aug{0|1}`

Przyklady: `baseline_bs128_lr0.001_drop0.5_aug0`, `vgg_like_bs128_lr0.05_drop0.5_aug1`.

## Srodowisko

1. Utworz i aktywuj venv (Python 3.12):

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate        # WSL/Linux
   ```

2. Zainstaluj PyTorch pod CUDA 12.8 (wymagane dla RTX 5070 / Blackwell sm_120):

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

3. Zainstaluj reszte zaleznosci:

   ```bash
   pip install -r requirements.txt
   ```

4. Sprawdz GPU (powinno zwrocic `True` i nazwe karty):

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

## Jak uruchamiac

Otworz `notebooks/CIFAR10_Modeling.ipynb` i wykonuj komorki etap po etapie.
Notebook importuje funkcje z `src/cifar10_pipeline.py`; foldery w `experiments/`
tworza sie automatycznie. Grid search jest resumowalny — po przerwaniu wystarczy
uruchomic komorke ponownie, a ukonczone konfiguracje zostana pominiete.

## Glowne elementy `src/cifar10_pipeline.py`

| Obszar | Funkcje / klasy |
|---|---|
| Reprodukowalnosc | `set_seed`, `get_device` |
| Dane | `load_data`, `stratified_split`, `CIFAR10Dataset`, `build_transforms`, `make_dataloaders` |
| Architektury | `build_baseline_cnn`, `build_vgg_like`, `build_resnet_like`, `count_parameters` |
| Trening | `build_optimizer`, `build_scheduler`, `build_criterion`, `EarlyStopping`, `train_model` |
| Ewaluacja | `evaluate_model`, `plot_history`, `plot_confusion_matrix` |
| Eksperymenty | `make_experiment_name`, `save_experiment`, `update_results_summary`, `run_experiment` |
| Grid search | `run_grid_search` |
