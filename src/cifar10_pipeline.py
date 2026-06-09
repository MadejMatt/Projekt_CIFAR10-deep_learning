"""Pipeline modelowania dla CIFAR-10.

Samodzielny, pozbawiony globalnego stanu zestaw narzędzi do trenowania,
ewaluacji i zarządzania eksperymentami deep learningowymi na CIFAR-10 (PyTorch).

Założenia projektowe
--------------------
* Brak zmiennego stanu globalnego: każda funkcja dostaje wszystko przez argumenty.
* Wszystkie ścieżki to :class:`pathlib.Path`; katalogi wynikowe tworzone są w razie potrzeby.
* Eksperymenty są reprodukowalne (``set_seed``) i wznawialne (grid search pomija
  foldery, które już zawierają wyniki).

Wczytane dane (``data/cifar10_preprocessed.npz``) zawierają tablice float32
przeskalowane do ``[0, 1]`` w układzie channels-last ``(N, 32, 32, 3)``. Nie
stosujemy tu dodatkowego skalowania; opcjonalna standaryzacja per-kanał jest
dostępna jako konfigurowalny transform (patrz ``normalize`` w konfiguracji).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# --------------------------------------------------------------------------- #
# Stałe
# --------------------------------------------------------------------------- #

#: Kanoniczna kolejność klas CIFAR-10 (indeksy 0-9).
CIFAR10_CLASSES = [
    "samolot", "samochod", "ptak", "kot", "jelen",
    "pies", "zaba", "kon", "statek", "ciezarowka",
]

#: Średnia/odchylenie per-kanał zbioru treningowego (w [0, 1]), z analizy EDA.
#: Używane tylko gdy w konfiguracji ``normalize=True``.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

#: Stałe ziarno podziału train/val, aby każdy eksperyment dzielił ten sam zbiór
#: walidacyjny, niezależnie od ziarna treningu danego eksperymentu.
SPLIT_SEED = 42


# --------------------------------------------------------------------------- #
# Reprodukowalność i urządzenie
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    """Ustawia wszystkie istotne generatory liczb losowych dla reprodukowalności."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Zwraca urządzenie CUDA jeśli dostępne, w przeciwnym razie CPU.

    Włącza ``cudnn.benchmark`` dla szybkości (wejście ma stały rozmiar 32x32).
    To kompromis: rezygnujemy z bitowej powtarzalności na rzecz przepustowości,
    co ma sens przy 36 runach grid searcha; ziarna i tak czynią runy praktycznie
    reprodukowalnymi.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_worker(worker_id: int) -> None:
    """Seeduje workery DataLoadera, aby augmentacja była reprodukowalna."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
# Dane
# --------------------------------------------------------------------------- #

def load_data(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wczytuje przetworzony ``.npz`` i zwraca ``(x_train, y_train, x_test, y_test)``.

    Etykiety spłaszczane do 1-D int64; piksele pozostają float32 w ``[0, 1]``.
    """
    with np.load(Path(npz_path)) as data:
        x_train = data["x_train"].astype("float32")
        y_train = data["y_train"].reshape(-1).astype("int64")
        x_test = data["x_test"].astype("float32")
        y_test = data["y_test"].reshape(-1).astype("int64")
    return x_train, y_train, x_test, y_test


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.1,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dzieli na train/val z zachowaniem proporcji klas. Deterministyczne przez ``seed``."""
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=val_size, stratify=y, random_state=seed,
    )
    return x_tr, y_tr, x_val, y_val


class CIFAR10Dataset(Dataset):
    """Dataset CIFAR-10 trzymany w pamięci, stosujący transformacje w locie.

    Dane przechowywane są jako channels-last float32 ``[0, 1]`` (numpy).
    ``__getitem__`` konwertuje każdą próbkę do tensora ``(C, H, W)`` i aplikuje
    transform, więc augmentacja jest losowana na nowo w każdej epoce (nigdy nie
    jest prekomputowana).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, transform: Callable | None = None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # HWC [0,1] -> tensor CHW
        img = torch.from_numpy(self.x[idx]).permute(2, 0, 1).contiguous()
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[idx])


def build_transforms(augment: bool, normalize: bool) -> tuple[Callable | None, Callable | None]:
    """Buduje ``(train_transform, eval_transform)``.

    Augmentacja (tylko train): RandomCrop(32, padding=4) + RandomHorizontalFlip.
    Normalizacja (oba zbiory, opcjonalna): standaryzacja per-kanał.
    Wszystkie transformacje działają na tensorach float już w ``[0, 1]``.
    """
    train_ops: list[Callable] = []
    eval_ops: list[Callable] = []

    if augment:
        train_ops.append(transforms.RandomCrop(32, padding=4))
        train_ops.append(transforms.RandomHorizontalFlip())

    if normalize:
        norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        train_ops.append(norm)
        eval_ops.append(norm)

    train_t = transforms.Compose(train_ops) if train_ops else None
    eval_t = transforms.Compose(eval_ops) if eval_ops else None
    return train_t, eval_t


def make_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    augment: bool,
    normalize: bool,
    num_workers: int = 0,
    seed: int = SPLIT_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Tworzy DataLoadery train/val/test. Augmentacja stosowana tylko do train."""
    train_t, eval_t = build_transforms(augment, normalize)

    train_ds = CIFAR10Dataset(x_train, y_train, transform=train_t)
    val_ds = CIFAR10Dataset(x_val, y_val, transform=eval_t)
    test_ds = CIFAR10Dataset(x_test, y_test, transform=eval_t)

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        worker_init_fn=_seed_worker, generator=generator, **common,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# Architektury
# --------------------------------------------------------------------------- #

def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Blok podstawowy: Conv 3x3 -> BatchNorm -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BaselineCNN(nn.Module):
    """Prosta sieć 3-blokowa (Conv-BN-ReLU-MaxPool) + głowa FC z dropoutem.

    Kanały rosną 32 -> 64 -> 128; rozmiar przestrzenny 32 -> 16 -> 8 -> 4.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn_relu(3, 32), nn.MaxPool2d(2),
            _conv_bn_relu(32, 64), nn.MaxPool2d(2),
            _conv_bn_relu(64, 128), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class VGGLike(nn.Module):
    """Głębsza sieć w stylu VGG: grupy konwolucji 3x3 przed każdym MaxPoolem.

    Bloki: [64,64] -> [128,128] -> [256,256,256], bez skip connections.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn_relu(3, 64), _conv_bn_relu(64, 64), nn.MaxPool2d(2),       # 16
            _conv_bn_relu(64, 128), _conv_bn_relu(128, 128), nn.MaxPool2d(2),   # 8
            _conv_bn_relu(128, 256), _conv_bn_relu(256, 256),
            _conv_bn_relu(256, 256), nn.MaxPool2d(2),                           # 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    """Blok resztkowy: dwie konwolucje 3x3 Conv-BN z połączeniem skip."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            # Projekcja 1x1, aby ścieżka skip miała zgodny kształt/liczbę kanałów.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # połączenie resztkowe ("autostrada gradientu")
        return torch.relu(out)


class ResNetLike(nn.Module):
    """ResNet dla wejść 32x32: lekki stem (bez agresywnego downsamplingu) + 4 grupy.

    Stem zachowuje 32x32; grupy zmniejszają 32 -> 32 -> 16 -> 8 -> 4; GlobalAvgPool
    zasila klasyfikator. BatchNorm regularyzuje, więc dropout domyślnie 0.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.0,
        blocks_per_group: int = 2,
        channels: tuple[int, int, int, int] = (64, 128, 256, 512),
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = _conv_bn_relu(3, c1)  # 3x3 stride 1, pozostaje 32x32
        self.layer1 = self._make_group(c1, c1, blocks_per_group, stride=1)
        self.layer2 = self._make_group(c1, c2, blocks_per_group, stride=2)
        self.layer3 = self._make_group(c2, c3, blocks_per_group, stride=2)
        self.layer4 = self._make_group(c3, c4, blocks_per_group, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )

    @staticmethod
    def _make_group(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        # Pierwszy blok grupy realizuje downsampling (stride), kolejne stride=1.
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.classifier(x)


def build_baseline_cnn(num_classes: int = 10, dropout: float = 0.5, **_) -> nn.Module:
    """Buduje baseline CNN. Nadmiarowe kwargs ignorowane dla jednolitego API builderów."""
    return BaselineCNN(num_classes=num_classes, dropout=dropout)


def build_vgg_like(num_classes: int = 10, dropout: float = 0.5, **_) -> nn.Module:
    """Buduje sieć VGG-like."""
    return VGGLike(num_classes=num_classes, dropout=dropout)


def build_resnet_like(num_classes: int = 10, dropout: float = 0.0, **_) -> nn.Module:
    """Buduje sieć ResNet-inspired (dropout domyślnie wyłączony)."""
    return ResNetLike(num_classes=num_classes, dropout=dropout)


#: Mapa, aby konfiguracja mogła wskazać builder po nazwie architektury.
MODEL_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "baseline": build_baseline_cnn,
    "vgg_like": build_vgg_like,
    "resnet_like": build_resnet_like,
}


def count_parameters(model: nn.Module) -> int:
    """Zwraca liczbę trenowalnych parametrów modelu."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Elementy składowe treningu
# --------------------------------------------------------------------------- #

def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Tworzy optimizer z konfiguracji (``optimizer`` w {adam, sgd})."""
    name = config.get("optimizer", "sgd").lower()
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=weight_decay, nesterov=config.get("nesterov", False),
        )
    raise ValueError(f"Nieznany optimizer: {name!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Tworzy scheduler LR z konfiguracji (``scheduler`` w {cosine, step, none})."""
    name = config.get("scheduler", "none")
    name = (name or "none").lower()
    if name in ("none", ""):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["max_epochs"]
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 30),
            gamma=config.get("gamma", 0.1),
        )
    raise ValueError(f"Nieznany scheduler: {name!r}")


def build_criterion(config: dict) -> nn.Module:
    """Funkcja straty: cross-entropy z opcjonalnym label smoothing."""
    return nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))


class EarlyStopping:
    """Śledzi najlepszą (najniższą) wartość metryki i zatrzymuje po ``patience`` epokach.

    Przy poprawie ``state_dict`` modelu jest zapisywany do ``checkpoint_path``.
    """

    def __init__(self, patience: int, checkpoint_path: Path):
        self.patience = patience
        self.checkpoint_path = Path(checkpoint_path)
        self.best_value = float("inf")
        self.best_epoch = -1
        self.counter = 0
        self.should_stop = False

    def step(self, value: float, model: nn.Module, epoch: int) -> bool:
        """Aktualizuje stan wartością z epoki. Zwraca True jeśli zapisano nowy najlepszy."""
        if value < self.best_value:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: "torch.amp.GradScaler | None" = None,
) -> tuple[float, float]:
    """Wykonuje jedną epokę. Trenuje gdy podano ``optimizer``, inaczej ewaluuje.

    Zwraca ``(srednia_strata, accuracy)``.
    """
    is_train = optimizer is not None
    model.train(is_train)
    use_amp = scaler is not None and device.type == "cuda"

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            # Mixed precision (AMP) przyspiesza trening na GPU bez utraty jakości.
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += inputs.size(0)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    checkpoint_path: Path,
    use_amp: bool = True,
    verbose: bool = True,
) -> dict:
    """Trenuje z early stoppingiem na val_loss; na końcu przywraca najlepsze wagi.

    Gdy ``config["early_stopping"]`` jest False, trening przechodzi pełny budżet
    ``max_epochs`` (bez przedwczesnego stopu), ale wagi z najniższym ``val_loss``
    są nadal zapisywane. Konieczne dla harmonogramu CosineAnnealingLR, który musi
    dobiec do końca, by zadziałał annealing — early stopping ucinałby go za wcześnie.

    Zwraca słownik historii z metrykami per epoka oraz informacją o zatrzymaniu.
    """
    model.to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = build_criterion(config)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None
    stopper = EarlyStopping(config.get("patience", 15), checkpoint_path)
    use_early_stopping = config.get("early_stopping", True)

    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "lr")}
    max_epochs = config["max_epochs"]

    for epoch in range(1, max_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        improved = stopper.step(val_loss, model, epoch)
        if verbose:
            flag = " *" if improved else ""
            print(
                f"Epoka {epoch:3d}/{max_epochs} | "
                f"lr {current_lr:.4f} | "
                f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val_loss {val_loss:.4f} acc {val_acc:.4f}{flag}"
            )
        if use_early_stopping and stopper.should_stop:
            if verbose:
                print(f"Early stopping w epoce {epoch} "
                      f"(najlepsza epoka {stopper.best_epoch}).")
            break

    # Przywracamy do modelu najlepszy checkpoint.
    model.load_state_dict(torch.load(stopper.checkpoint_path, map_location=device))

    return {
        **history,
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best_value,
        "epochs_trained": len(history["train_loss"]),
        "early_stopped": stopper.should_stop,
    }


# --------------------------------------------------------------------------- #
# Ewaluacja i wykresy
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str] = CIFAR10_CLASSES,
) -> dict:
    """Ewaluuje model: accuracy, F1 macro, F1 per klasa i surowe predykcje."""
    model.eval().to(device)
    y_true, y_pred = [], []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        y_pred.append(outputs.argmax(1).cpu().numpy())
        y_true.append(targets.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    accuracy = float((y_true == y_pred).mean())
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_per = f1_score(y_true, y_pred, average=None)
    f1_per_class = {name: float(v) for name, v in zip(class_names, f1_per)}

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def load_trained_model(
    checkpoint_path: Path,
    arch: str,
    dropout: float,
    device: torch.device,
    num_classes: int = 10,
) -> nn.Module:
    """Odtwarza architekturę i wczytuje zapisane wagi (tryb eval).

    Przydatne do ponownej ewaluacji ukończonego eksperymentu (np. najlepszego
    modelu z gridu na zbiorze testowym) bez przetrenowywania.
    """
    model = MODEL_BUILDERS[arch](num_classes=num_classes, dropout=dropout)
    model.load_state_dict(torch.load(Path(checkpoint_path), map_location=device))
    return model.to(device).eval()


def plot_history(history: dict, save_path: Path | None = None, title: str = ""):
    """Rysuje krzywe straty i accuracy z zaznaczoną najlepszą epoką."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    epochs = range(1, len(history["train_loss"]) + 1)
    best = history.get("best_epoch", -1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Strata (loss)")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Dokladnosc (accuracy)")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Accuracy")

    for ax in axes:
        if best > 0:
            ax.axvline(best, color="green", linestyle="--", alpha=0.7,
                       label=f"najlepsza (ep. {best})")
        ax.legend()
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CIFAR10_CLASSES,
    save_path: Path | None = None,
    title: str = "Macierz pomylek (znormalizowana)",
):
    """Rysuje znormalizowaną wierszami macierz pomyłek z wartościami procentowymi."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        cm_norm * 100, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "% wiersza"}, ax=ax,
    )
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawdziwa klasa")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Zarządzanie eksperymentami
# --------------------------------------------------------------------------- #

def make_experiment_name(config: dict) -> str:
    """Buduje unikalną nazwę eksperymentu z konfiguracji.

    Format: ``{arch}_bs{batch}_lr{lr}_drop{dropout}_aug{0|1}``.
    """
    return (
        f"{config['arch']}"
        f"_bs{config['batch_size']}"
        f"_lr{config['lr']}"
        f"_drop{config['dropout']}"
        f"_aug{int(config['augment'])}"
    )


def save_experiment(
    exp_dir: Path,
    config: dict,
    metrics: dict,
    history: dict,
) -> None:
    """Zapisuje ``config.json``, ``metrics.json`` i ``history.csv`` do ``exp_dir``."""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Przed serializacją metryk usuwamy obszerne surowe predykcje.
    metrics_to_save = {k: v for k, v in metrics.items() if k not in ("y_true", "y_pred")}
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)

    hist_cols = ["train_loss", "val_loss", "train_acc", "val_acc", "lr"]
    hist_df = pd.DataFrame({k: history[k] for k in hist_cols})
    hist_df.index.name = "epoch"
    hist_df.index += 1
    hist_df.to_csv(exp_dir / "history.csv")


def update_results_summary(summary_path: Path, row: dict) -> None:
    """Dopisuje jeden wiersz do zbiorczego CSV (append-only; nigdy nie nadpisuje pliku)."""
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    header = not summary_path.exists()
    pd.DataFrame([row]).to_csv(summary_path, mode="a", header=header, index=False)


def _summary_row(config: dict, history: dict, num_params: int,
                 val_metrics: dict, test_metrics: dict | None) -> dict:
    """Składa jeden płaski wiersz do tabeli zbiorczej."""
    row = {
        "experiment_name": make_experiment_name(config),
        "arch": config["arch"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "dropout": config["dropout"],
        "augment": int(config["augment"]),
        "optimizer": config.get("optimizer", "sgd"),
        "scheduler": config.get("scheduler", "none"),
        "normalize": int(config.get("normalize", False)),
        "label_smoothing": config.get("label_smoothing", 0.0),
        "weight_decay": config.get("weight_decay", 0.0),
        "num_params": num_params,
        "best_epoch": history["best_epoch"],
        "epochs_trained": history["epochs_trained"],
        "early_stopped": history["early_stopped"],
        "best_val_loss": round(history["best_val_loss"], 4),
        "val_accuracy": round(val_metrics["accuracy"], 4),
        "val_f1_macro": round(val_metrics["f1_macro"], 4),
        "test_accuracy": round(test_metrics["accuracy"], 4) if test_metrics else None,
        "test_f1_macro": round(test_metrics["f1_macro"], 4) if test_metrics else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return row


# --------------------------------------------------------------------------- #
# Orkiestracja wysokiego poziomu
# --------------------------------------------------------------------------- #

def _reconstruct_result(
    data: tuple,
    device: torch.device,
    exp_dir: Path,
    evaluate_test: bool,
) -> dict:
    """Odtwarza pełny wynik z zapisanych artefaktów (bez przetrenowywania).

    Wczytuje ``config.json`` i ``history.csv``, ładuje ``model.pt`` i ponownie
    liczy metryki (val oraz opcjonalnie test). Pozwala wznawiać pracę po
    restarcie kernela i mieć komplet danych do wykresów.
    """
    with open(exp_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    with open(exp_dir / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    hist_df = pd.read_csv(exp_dir / "history.csv")
    history = {c: hist_df[c].tolist()
               for c in ("train_loss", "val_loss", "train_acc", "val_acc", "lr")}
    history["best_epoch"] = int(metrics.get("best_epoch",
                                            int(hist_df["val_loss"].idxmin()) + 1))
    history["best_val_loss"] = float(hist_df["val_loss"].min())
    history["epochs_trained"] = len(hist_df)
    history["early_stopped"] = None

    _, val_loader, test_loader = make_dataloaders(
        *data, batch_size=config["batch_size"],
        augment=config["augment"], normalize=config.get("normalize", False),
        num_workers=config.get("num_workers", 0),
    )
    model = load_trained_model(
        exp_dir / "model.pt", config["arch"], config["dropout"], device,
        num_classes=config.get("num_classes", 10),
    )
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device) if evaluate_test else None

    return {
        "config": config,
        "exp_dir": exp_dir,
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "num_params": metrics.get("num_params"),
        "cached": True,
    }


def run_experiment(
    config: dict,
    data: tuple,
    device: torch.device,
    experiments_dir: Path,
    category: str,
    model_builder: Callable[..., nn.Module] | None = None,
    evaluate_test: bool = False,
    make_plots: bool = True,
    resume: bool = True,
    verbose: bool = True,
) -> dict:
    """Uruchamia pojedynczy eksperyment od początku do końca.

    Kroki: seed -> budowa modelu -> dataloadery -> trening -> ewaluacja na val
    (oraz na test gdy ``evaluate_test``) -> zapis artefaktów -> aktualizacja zbiorczej tabeli.

    ``data`` to ``(x_train, y_train, x_val, y_val, x_test, y_test)``.
    Gdy ``resume`` i folder eksperymentu ma już ``metrics.json``, zwracany jest
    wynik z cache bez przetrenowywania.

    Zwraca słownik z ``config``, ``history``, ``val_metrics``, ``test_metrics``,
    ``num_params`` oraz ``exp_dir``.
    """
    name = make_experiment_name(config)
    exp_dir = Path(experiments_dir) / category / name
    summary_path = Path(experiments_dir) / "results_summary.csv"

    if resume and (exp_dir / "metrics.json").exists():
        if verbose:
            print(f"[pomijam trening] {category}/{name} juz istnieje — wczytuje wyniki.")
        return _reconstruct_result(data, device, exp_dir, evaluate_test)

    set_seed(config.get("seed", 0))
    x_train, y_train, x_val, y_val, x_test, y_test = data

    if model_builder is None:
        model_builder = MODEL_BUILDERS[config["arch"]]
    model = model_builder(
        num_classes=config.get("num_classes", 10), dropout=config["dropout"]
    )
    num_params = count_parameters(model)

    train_loader, val_loader, test_loader = make_dataloaders(
        x_train, y_train, x_val, y_val, x_test, y_test,
        batch_size=config["batch_size"],
        augment=config["augment"],
        normalize=config.get("normalize", False),
        num_workers=config.get("num_workers", 0),
    )

    if verbose:
        print(f"\n=== {category}/{name} | parametry: {num_params:,} ===")

    history = train_model(
        model, train_loader, val_loader, config, device,
        checkpoint_path=exp_dir / "model.pt", verbose=verbose,
    )

    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device) if evaluate_test else None

    # metrics.json zawiera metryki val oraz test, jeśli zostały policzone.
    metrics_out = {"num_params": num_params, "val": {k: v for k, v in val_metrics.items()
                                                     if k not in ("y_true", "y_pred")}}
    if test_metrics is not None:
        metrics_out["test"] = {k: v for k, v in test_metrics.items()
                               if k not in ("y_true", "y_pred")}
        metrics_out["best_epoch"] = history["best_epoch"]
    save_experiment(exp_dir, config, metrics_out, history)

    if make_plots:
        plot_history(history, save_path=exp_dir / "history.png", title=name)
        eval_for_cm = test_metrics if test_metrics is not None else val_metrics
        plot_confusion_matrix(
            eval_for_cm["y_true"], eval_for_cm["y_pred"],
            save_path=exp_dir / "confusion_matrix.png",
        )
        import matplotlib.pyplot as plt
        plt.close("all")

    update_results_summary(
        summary_path, _summary_row(config, history, num_params, val_metrics, test_metrics)
    )

    return {
        "config": config,
        "exp_dir": exp_dir,
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "num_params": num_params,
        "cached": False,
    }


# --------------------------------------------------------------------------- #
# Grid search
# --------------------------------------------------------------------------- #

def _iter_grid(param_grid: dict[str, Iterable]) -> Iterable[dict]:
    """Generuje każdą kombinację z ``param_grid`` jako słownik."""
    import itertools

    keys = list(param_grid)
    for values in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, values))


def run_grid_search(
    model_builder: Callable[..., nn.Module],
    param_grid: dict[str, list],
    fixed_config: dict,
    data: tuple,
    device: torch.device,
    experiments_dir: Path,
    category: str = "baseline",
    verbose: bool = True,
) -> pd.DataFrame:
    """Wznawialny grid search; selekcja modelu wyłącznie na walidacji (test nietknięty).

    Dla każdej kombinacji: buduje, trenuje, ewaluuje na walidacji, zapisuje
    eksperyment i dopisuje do tabeli zbiorczej. Foldery z gotowymi wynikami są
    pomijane, więc wyszukiwanie wznawia się czysto po crashu.

    Zapisuje ``tuning/grid_results.csv`` i ``tuning/best_config.json``; zwraca
    DataFrame posortowany malejąco po accuracy walidacyjnym.
    """
    records = []
    combos = list(_iter_grid(param_grid))
    if verbose:
        print(f"Grid search: {len(combos)} kombinacji.")

    for i, combo in enumerate(combos, start=1):
        config = {**fixed_config, **combo}
        if verbose:
            print(f"\n--- [{i}/{len(combos)}] {combo} ---")
        result = run_experiment(
            config, data, device, experiments_dir, category=category,
            model_builder=model_builder, evaluate_test=False,
            make_plots=False, resume=True, verbose=verbose,
        )

        # val_metrics jest dostępne zarówno dla świeżego runu, jak i z cache.
        records.append({
            **combo,
            "val_accuracy": result["val_metrics"]["accuracy"],
            "val_f1_macro": result["val_metrics"]["f1_macro"],
            "experiment_name": make_experiment_name(config),
        })

    results_df = pd.DataFrame(records).sort_values("val_accuracy", ascending=False)

    tuning_dir = Path(experiments_dir) / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tuning_dir / "grid_results.csv", index=False)

    best_combo = results_df.iloc[0].to_dict()
    best_config = {**fixed_config,
                   **{k: best_combo[k] for k in param_grid if k in best_combo}}
    with open(tuning_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\nNajlepsza konfiguracja (val_acc={best_combo['val_accuracy']:.4f}): "
              f"{ {k: best_config[k] for k in param_grid} }")
    return results_df
