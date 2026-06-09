# Notatki merytoryczne do raportu — rozdział „Modelowanie"

Materiał pomocniczy do napisania części o modelowaniu (kontynuacja po sekcji „3. Przygotowanie
danych"). Zawiera: (A) proponowaną strukturę rozdziału, (B) słowniczek pojęć z przystępnymi
definicjami, (C) opis sekcja-po-sekcji — **co** robimy, **gdzie** (który eksperyment / plik z wynikami),
**dlaczego**, oraz jakie rysunki/tabele wstawić. Liczby pochodzą z zakończonych już treningów
(źródła: `experiments/results_summary.csv` oraz pliki `metrics.json` poszczególnych eksperymentów).

Konwencja terminologiczna: proza po polsku, ale stosujemy **utrwalone terminy** z polskiego uczenia
maszynowego — przy pojęciach, które po polsku zwyczajowo zostają w oryginale (`learning rate`, `batch`,
`dropout`, `skip connections`), nie tłumaczymy ich na siłę, tylko podajemy w nawiasie naturalny opis.

---

## A. Proponowana struktura rozdziału

Numeracja kontynuuje istniejący raport (1. Wstęp, 2. EDA, 3. Przygotowanie danych):

- **4. Modelowanie i eksperymenty**
  - 4.1. Podejście badawcze (od prostego do złożonego)
  - 4.2. Metodyka eksperymentów (podział danych, reprodukowalność, metryki oceny)
  - 4.3. Etap 1 — model bazowy (Baseline CNN)
  - 4.4. Etap 2 — dobór optymalizatora i harmonogramu uczenia (Adam vs SGD)
  - 4.5. Etap 3 — strojenie hiperparametrów (Grid Search)
  - 4.6. Etap 4 — architektura głęboka typu VGG
  - 4.7. Etap 5 — architektura z połączeniami rezydualnymi (ResNet)
- **5. Podsumowanie i wnioski**
  - 5.1. Porównanie modeli
  - 5.2. Analiza błędów (macierz pomyłek)
  - 5.3. Wnioski końcowe i możliwe kierunki rozwoju

Dla każdego etapu mamy gotowe artefakty na dysku w `experiments/<kategoria>/<nazwa_eksperymentu>/`:
`history.png` (krzywe uczenia), `confusion_matrix.png` (macierz pomyłek), `metrics.json` (liczby),
`config.json` (ustawienia). Zbiorcze zestawienie: `experiments/results_summary.csv`.

---

## B. Słowniczek pojęć (do wykorzystania w tekście lub jako ramki „W skrócie")

Definicje są celowo przystępne — można je wpleść w narrację albo wstawić jako krótkie wyjaśnienia
przy pierwszym użyciu pojęcia.

**Sieć konwolucyjna (CNN, Convolutional Neural Network).** Rodzaj sieci neuronowej stworzony do pracy
z obrazami. Zamiast patrzeć na każdy piksel osobno, analizuje obraz małymi fragmentami i uczy się
rozpoznawać wzorce (najpierw krawędzie, potem kształty, w końcu całe obiekty).

**Warstwa konwolucyjna i filtr (kernel/jądro).** Filtr to małe „okienko" (u nas 3×3 piksele), które
przesuwa się po obrazie i wykrywa lokalny wzorzec (np. pionową krawędź). Jedna warstwa ma wiele filtrów —
każdy wyłapuje coś innego. Wynik to tzw. **mapa cech** (feature map): obraz pokazujący, gdzie dany wzorzec
wystąpił.

**Funkcja aktywacji ReLU.** Prosta operacja: ujemne wartości zamienia na zero, dodatnie zostawia bez zmian.
Wprowadza do sieci „nieliniowość", bez której sieć — choćby bardzo głęboka — potrafiłaby modelować jedynie
zależności liniowe. ReLU jest standardem, bo jest szybka i stabilna w treningu.

**Batch Normalization (normalizacja wsadowa).** Technika, która w trakcie treningu standaryzuje sygnał
płynący między warstwami (sprowadza go do podobnej skali — w przybliżeniu średnia 0, odchylenie 1).
Po co? W głębokiej sieci rozkład danych na wejściu kolejnych warstw ciągle „dryfuje" podczas uczenia, co
spowalnia i destabilizuje trening. Batch Normalization to stabilizuje: pozwala uczyć szybciej, używać
większego `learning rate` i działa lekko regularyzująco (utrudnia przeuczenie). Stąd w naszych sieciach po
każdej warstwie konwolucyjnej następuje Batch Normalization.

**Pooling (MaxPooling).** Zmniejszanie rozdzielczości mapy cech przez branie maksimum z małych obszarów
(u nas 2×2). Efekt: obraz „kurczy się" (32→16→8→4 piksele), sieć patrzy na coraz większy kontekst, a
liczba obliczeń maleje. MaxPooling zachowuje najsilniejsze sygnały (najwyraźniejsze cechy).

**Warstwa w pełni połączona (fully connected).** Klasyczna warstwa, w której każdy neuron łączy się ze
wszystkimi wejściami. U nas znajduje się na końcu sieci (część klasyfikująca) i na podstawie wykrytych cech
podejmuje ostateczną decyzję, do której z 10 klas należy obraz.

**Dropout.** Technika przeciw przeuczeniu: w trakcie treningu losowo „wyłącza" część neuronów (np. 25% lub
50%) w każdym kroku. Dzięki temu sieć nie może polegać na pojedynczych neuronach i uczy się bardziej
odpornych, ogólnych cech. Podczas testu dropout jest wyłączony.

**Funkcja straty (loss) i Cross-Entropy.** Liczba mówiąca, jak bardzo predykcje modelu odbiegają od
prawdy — im mniejsza, tym lepiej. Trening polega na jej minimalizowaniu. Dla klasyfikacji standardem jest
**Cross-Entropy** (entropia krzyżowa), która karze model tym mocniej, im pewniej wskazał błędną klasę.

**Label smoothing (wygładzanie etykiet).** Drobna modyfikacja celu uczenia: zamiast wymagać od modelu
absolutnej pewności (100% na poprawną klasę, 0% na resztę), „rozmywamy" cel (np. 90% / reszta rozłożona).
Zniechęca to model do nadmiernej pewności siebie i poprawia uogólnianie. Skutek uboczny: wartość `loss`
jest sztucznie wyższa, więc straty z różnymi ustawieniami `label smoothing` nie porównujemy wprost.

**Optymalizator.** Algorytm, który aktualizuje wagi sieci, by zmniejszać `loss`. Używaliśmy dwóch:
- **SGD** (Stochastic Gradient Descent) — „klasyk"; idzie w stronę najszybszego spadku błędu, z dodatkiem
  **momentum** (bezwładność — uśrednia kierunki z poprzednich kroków, dzięki czemu jedzie pewniej i mniej
  „skacze").
- **Adam** — optymalizator adaptacyjny; sam dobiera tempo uczenia osobno dla każdej wagi. Zwykle szybciej
  rusza z miejsca, ale bywa, że gorzej „dostraja się" na końcu treningu niż dobrze ustawiony SGD.

**Learning rate (współczynnik / tempo uczenia).** Rozmiar kroku przy aktualizacji wag. Za duży — model
„przeskakuje" dobre rozwiązania i jest niestabilny; za mały — uczy się boleśnie wolno. To jeden z
najważniejszych parametrów treningu.

**Harmonogram zmian `learning rate` i Cosine Annealing.** Zamiast trzymać `learning rate` stały, można go
stopniowo zmniejszać w trakcie treningu (taki plan zmian to po angielsku *scheduler*). **Cosine Annealing**
robi to płynnie wg krzywej kosinusa: na początku kroki są duże (szeroka eksploracja), pod koniec maleją
niemal do zera (precyzyjne „dostrojenie" w dobrym minimum). W naszych eksperymentach okazało się to
kluczowe (patrz Etap 2).

**Weight decay (regularyzacja L2).** Łagodne „ściąganie" wag w stronę zera podczas treningu. Zapobiega
nadmiernemu rozrostowi wag, co przekłada się na prostszy, lepiej uogólniający model (mniej przeuczenia).

**Epoka, batch (wsad), iteracja.** **Batch** to porcja obrazów przetwarzana naraz (np. 128). **Iteracja**
to jedna aktualizacja wag na podstawie jednego batcha. **Epoka** to jedno przejście przez cały zbiór
treningowy. Model trenuje się przez wiele epok.

**Early stopping (wczesne zatrzymanie).** Mechanizm, który przerywa trening, gdy wynik na zbiorze
walidacyjnym przestaje się poprawiać przez ustaloną liczbę epok (`patience`). Chroni przed przeuczeniem i
oszczędza czas. Uwaga: przy harmonogramie `cosine` celowo go wyłączamy, by harmonogram zdążył dobiec do
końca (inaczej `learning rate` nie zdążyłby opaść).

**Augmentacja danych.** Sztuczne powiększanie zróżnicowania zbioru treningowego przez drobne, losowe
przekształcenia obrazów w każdej epoce. Używamy dwóch: **RandomCrop** (losowe wycięcie po dodaniu marginesu —
uczy odporności na przesunięcie obiektu) oraz **HorizontalFlip** (losowe odbicie lustrzane w poziomie).
Model „widzi" za każdym razem lekko inny obraz, więc trudniej mu się przeuczyć.

**Przeuczenie (overfitting).** Sytuacja, gdy model świetnie radzi sobie na danych treningowych, ale słabo
na nowych (walidacja/test) — „nauczył się na pamięć" zamiast uogólniać. Objaw: duża różnica między
dokładnością treningową a walidacyjną.

**Podział train / validation / test.** Dane dzielimy na trzy części: **treningową** (model się uczy),
**walidacyjną** (wybór najlepszego modelu i strojenie — model jej nie „widzi" w trakcie uczenia wag) oraz
**testową** (jednorazowa, uczciwa ocena końcowa). Kluczowa zasada: zbioru testowego nie używamy do
podejmowania żadnych decyzji — dotykamy go dopiero na sam koniec, by wynik był wiarygodny.

**Grid search (przeszukiwanie siatki).** Systematyczne wypróbowanie wszystkich kombinacji wybranych
ustawień (hiperparametrów), by znaleźć najlepszą. U nas: 3 wartości `learning rate` × 3 `batch_size` ×
2 `dropout` × 2 (augmentacja tak/nie) = 36 kombinacji.

**Bloki rezydualne (residual blocks) i połączenia rezydualne (skip connections).** Pomysł z architektury
ResNet: obok zwykłej ścieżki przez warstwy dodaje się „skrót" (skip connection), który przepuszcza sygnał
dalej bez zmian i dodaje go do wyniku warstw. Ułatwia to uczenie bardzo głębokich sieci — gradient ma
krótszą drogę do wcześniejszych warstw, dzięki czemu nie zanika, a trening jest stabilniejszy.

**Metryki oceny.**
- **Accuracy (dokładność)** — odsetek poprawnie sklasyfikowanych obrazów. Prosta i czytelna; sensowna tu,
  bo zbiór jest idealnie zbalansowany.
- **F1-score (miara F1, wariant macro)** — średnia harmoniczna precyzji i czułości, liczona osobno dla
  każdej klasy i uśredniona. Pokazuje, czy model radzi sobie równo ze wszystkimi klasami, a nie tylko z
  łatwymi.
- **Macierz pomyłek (confusion matrix)** — tabela pokazująca, które klasy są ze sobą mylone (np. kot z psem).

**Liczba parametrów modelu.** Liczba wag, które sieć dostraja podczas treningu. Większa sieć = więcej
parametrów = większa pojemność, ale i większe ryzyko przeuczenia oraz dłuższy trening.

---

## C. Sekcja po sekcji — co, gdzie, dlaczego

### 4.1. Podejście badawcze (od prostego do złożonego)

**Co napisać:** Cały rozdział prowadzimy jako spójną „historię badawczą" — od najprostszego modelu do coraz
bardziej zaawansowanych. Każdy kolejny etap wynika z wniosków z poprzedniego. Kolejność:
1. zbudowanie prostego modelu bazowego jako punktu odniesienia,
2. ustalenie najlepszego sposobu treningu (optymalizator + harmonogram `learning rate`),
3. strojenie hiperparametrów na modelu bazowym,
4. przeniesienie najlepszych ustawień na dwie głębsze, nowoczesne architektury (VGG, ResNet).

**Dlaczego tak:** Pokazujemy proces myślowy i uzasadniamy każdą decyzję wynikami, zamiast od razu prezentować
gotowy najlepszy model. To jest istota dobrego raportu badawczego.

### 4.2. Metodyka eksperymentów

**Co napisać:**
- **Podział danych:** zbiór treningowy (50 000) dzielimy dodatkowo na właściwy treningowy (~45 000) i
  walidacyjny (~5 000, 10%), z zachowaniem proporcji klas (podział stratyfikowany). Zbiór testowy (10 000)
  zostaje nietknięty do oceny końcowej.
- **Reprodukowalność:** ustalone ziarno losowości (`seed`), by wyniki były powtarzalne.
- **Metryki:** główna to `accuracy`, uzupełniona o `F1-score (macro)` i macierz pomyłek. Wybór najlepszego
  modelu w każdym treningu odbywa się po najniższej stracie walidacyjnej (`val_loss`).
- **Sprzęt:** trening na karcie graficznej NVIDIA RTX 5070 (CUDA), framework PyTorch.

**Dlaczego tak:** Rozdzielenie walidacji od testu to fundament uczciwej oceny — bez tego wyniki byłyby
zawyżone (model pośrednio „podejrzałby" zbiór testowy podczas strojenia).

### 4.3. Etap 1 — model bazowy (Baseline CNN)

**Co napisać:** Prosta sieć konwolucyjna: 3 bloki (konwolucja 3×3 → Batch Normalization → ReLU → MaxPooling),
w których liczba filtrów rośnie 32 → 64 → 128, a rozmiar obrazu maleje 32 → 16 → 8 → 4. Na końcu część w pełni
połączona z `dropout`. Trening: optymalizator Adam, `learning rate` 0.001, bez augmentacji i regularyzacji
— celowo minimalna konfiguracja.

**Gdzie:** `experiments/baseline/baseline_bs128_lr0.001_drop0.5_aug0/` — wstawić `history.png` i
`confusion_matrix.png`. Liczba parametrów: ≈ 620 tys. (dokładnie 620 586).

**Dlaczego:** Potrzebujemy uczciwego punktu odniesienia, względem którego zmierzymy każde kolejne usprawnienie.

**Wynik:** test accuracy **`0.781`**, F1 macro **`0.782`** (najlepsza epoka 16, `early stopping` na 31).
**Obserwacja:** wyraźne przeuczenie — dokładność treningowa znacząco przewyższa walidacyjną, a `val_loss`
zaczyna rosnąć już po kilkunastu epokach. To celowo „goły" punkt odniesienia, który motywuje wszystkie
kolejne etapy (regularyzacja, augmentacja, lepszy harmonogram treningu, głębsze architektury).

### 4.4. Etap 2 — Adam vs SGD i rola harmonogramu uczenia

**Co napisać:** Kontrolowany eksperyment 2×2: dwa optymalizatory (Adam, SGD) × dwa tryby `learning rate`
(stały vs `cosine`). Pozostałe ustawienia identyczne. Celem jest odpowiedź: co naprawdę decyduje o jakości —
sam optymalizator czy sposób sterowania `learning rate`?

**Gdzie:** `experiments/opt_const/` (stały LR) oraz `experiments/opt_cosine/` (cosine). Wstawić wykres
porównawczy krzywych walidacyjnych z notebooka oraz tabelę 4 wariantów.

**Wynik (tabela do wstawienia):**

| Wariant | test accuracy | F1 macro | najlepsza epoka |
|---|---|---|---|
| SGD + cosine | `0.8018` | `0.8014` | 49 / 50 |
| Adam + cosine | `0.7913` | `0.7900` | 21 / 50 |
| Adam + stały LR | `0.7869` | `0.7867` | 23 / 38 |
| SGD + stały LR | `0.7660` | `0.7663` | 11 / 26 |

**Dlaczego i wnioski:**
- Przy **stałym** `learning rate` lepszy jest Adam — surowy SGD z wysokim, niegasnącym krokiem jest
  niestabilny i z czasem się pogarsza.
- Po włączeniu **cosine** sytuacja się odwraca: SGD wygrywa. Malejący `learning rate` pozwala mu pod koniec
  precyzyjnie „dostroić się" w dobrym minimum (najlepszy wynik wypada przy końcu treningu, gdy krok jest
  bliski zera).
- **Kluczowy wniosek:** decydujący okazał się **harmonogram uczenia**, nie sam wybór optymalizatora —
  `cosine` podniósł SGD aż o ~3,6 pp (z `0.766` do `0.8018`), a Adama jedynie o ~0,4 pp. Innymi słowy SGD
  nie był gorszym optymalizatorem; brakowało mu wyłącznie malejącego kroku. Dlatego do dalszych etapów
  przyjmujemy **SGD + cosine**.

### 4.5. Etap 3 — strojenie hiperparametrów (Grid Search)

**Co napisać:** Systematyczne przeszukanie 36 kombinacji: `learning rate` ∈ {0.01, 0.05, 0.1} × `batch_size`
∈ {64, 128, 256} × `dropout` ∈ {0.25, 0.5} × augmentacja ∈ {nie, tak}. Stałe: SGD + momentum, `cosine`,
`weight decay`, `label smoothing`. Wybór najlepszej kombinacji wyłącznie na zbiorze walidacyjnym.

**Gdzie:** wyniki w `experiments/baseline/` (po jednym folderze na kombinację); zbiorczo
`experiments/tuning/grid_results.csv` i `experiments/tuning/best_config.json`. Wstawić heatmapę
`learning rate × batch_size` oraz wykresy wpływu augmentacji i `dropout` (z notebooka).

**Wynik:** najlepsza konfiguracja (wg najniższego `val_loss`): `learning rate` **0.1**, `batch_size` **128**,
`dropout` **0.25**, augmentacja **włączona** — `val accuracy` **`0.872`**, `F1 macro` **`0.871`**
(zapis w `experiments/tuning/best_config.json`).

**Dlaczego i wnioski:**
- **Augmentacja okazała się zdecydowanie najważniejszym czynnikiem.** Konfiguracje z augmentacją osiągają
  ~`0.84`–`0.87` `val accuracy`, a bez niej ~`0.79`–`0.81` — różnica rzędu **+6–7 pp**, większa niż łączny
  wpływ pozostałych hiperparametrów. To bezpośrednio leczy przeuczenie obserwowane w Etapie 1.
- **`dropout`:** przy włączonej augmentacji lepszy jest niższy `dropout` 0.25 niż 0.5 — augmentacja już
  regularyzuje, więc mocny `dropout` dodatkowo „dusi" sieć i zaniża wynik.
- **`learning rate`:** wartości 0.05 i 0.1 wypadają najlepiej, a dzięki `cosine` wynik jest mało wrażliwy na
  dokładny wybór LR. **`batch_size`** ma wpływ drugorzędny (64 minimalnie lepszy, ale wybrano 128 dla
  szybszego treningu).
- Nastrojony baseline (~`0.87` `val`) to skok o **~9 pp** względem surowego baseline'u z Etapu 1 (`0.781`) —
  niemal w całości dzięki augmentacji.
- Najlepsze ustawienia (`learning rate`, `batch_size`, augmentacja) przenosimy na głębsze architektury.
  `dropout` dobieramy osobno dla każdej z nich. To świadomy kompromis: stroimy na tańszym obliczeniowo
  modelu bazowym jako reprezentancie, zamiast powtarzać kosztowny grid dla każdej architektury.

### 4.6. Etap 4 — architektura głęboka typu VGG

**Co napisać:** Głębsza sieć inspirowana VGG: grupy po kilka warstw konwolucyjnych 3×3 przed każdym
poolingiem (bloki [64,64] → [128,128] → [256,256,256]), bez połączeń rezydualnych. Od tego etapu włączamy
standaryzację per-kanał (normalizację wartości pikseli). Najlepsze ustawienia z gridu, pełny harmonogram
`cosine`.

**Gdzie:** `experiments/vgg_like/` — `history.png`, `confusion_matrix.png`. Liczba parametrów: ≈ 3,97 mln
(dokładnie 3 968 202).

**Wynik:** test accuracy **`0.938`**, F1 macro **`0.938`** (najlepsza epoka 198/200).

**Dlaczego i wnioski:** Sprawdzamy, czy większa głębokość (więcej warstw konwolucyjnych) przełoży się na
lepszą ekstrakcję cech niż prosty baseline. Tak — sama głębokość dała skok o **~6–7 pp** względem
nastrojonego baseline'u (~`0.87` → `0.938`). Model uczy się niemal do końca harmonogramu (best epoch 198/200),
co potwierdza, że głębsza sieć potrzebuje pełnych 200 epok i wygaszania `learning rate`. Ograniczenie:
dalsze pogłębianie czystej sieci VGG napotyka problem zanikającego gradientu — stąd ResNet w kolejnym etapie.

### 4.7. Etap 5 — architektura z połączeniami rezydualnymi (ResNet)

**Co napisać:** Sieć z blokami rezydualnymi (residual blocks ze skip connections), dostosowana do małych
obrazów 32×32. `dropout` ustawiony na 0 — w tej architekturze rolę regularyzatora przejmuje Batch
Normalization. Reszta ustawień jak w VGG.

**Gdzie:** `experiments/resnet_like/` — `history.png`, `confusion_matrix.png`. Liczba parametrów: ≈ 11,17 mln
(dokładnie 11 173 962).

**Wynik:** test accuracy **`0.952`**, F1 macro **`0.952`** (najlepsza epoka 197/200).

**Dlaczego i wnioski:** Połączenia rezydualne to jedna z najważniejszych nowoczesnych technik — umożliwiają
stabilny trening głębokich sieci. Dały przewagę nad zwykłą głęboką siecią: **+1,4 pp** względem VGG
(`0.938` → `0.952`). Zysk jest realny, ale malejący i okupiony kosztem — ResNet ma **~2,8× więcej parametrów**
niż VGG (11,17 mln vs 3,97 mln). Brak `dropout` nie zaszkodził: jego rolę regularyzatora w pełni przejmują
Batch Normalization i augmentacja.

### 5. Podsumowanie i wnioski

**5.1. Porównanie modeli** — tabela zbiorcza (baseline nastrojony z gridu vs VGG vs ResNet). Wstawić wykres
słupkowy z notebooka (Etap 6).

| Model | test accuracy | F1 macro | liczba parametrów |
|---|---|---|---|
| Baseline (nastrojony) | ~`0.87` (`val`; test z komórki ewaluacji w notebooku) | — | ≈ 0,62 mln |
| VGG-like | `0.938` | `0.938` | ≈ 3,97 mln |
| ResNet-like | **`0.952`** | **`0.952`** | ≈ 11,17 mln |

Najlepszy model to **ResNet-like (test accuracy `0.952`)**. Warto zwrócić uwagę na rosnący koszt jakości:
każdy kolejny skok wymaga istotnie większej sieci, a przyrosty maleją (baseline → VGG: +~6 pp przy ~6×
parametrów; VGG → ResNet: +1,4 pp przy ~2,8× parametrów).

**5.2. Analiza błędów** — z macierzy pomyłek najlepszego modelu (ResNet). Najsłabsze klasy to konsekwentnie
**kot** (F1 `0.889`) i **pies** (F1 `0.912`) — zwierzęta o podobnej sylwetce i teksturze, mylone głównie
między sobą oraz z pozostałymi zwierzętami (ptak, jeleń). Najłatwiejsze są pojazdy (samochód `0.976`,
ciężarówka, statek) oraz koń. Co istotne, **kot jest najtrudniejszą klasą na każdym etapie**, a jego F1
rośnie wraz z jakością modelu: `0.76` (baseline) → `0.86` (VGG) → `0.89` (ResNet). Powiązać z obserwacjami
z EDA (niska rozdzielczość 32×32, podobieństwo kształtów zwierząt).

**5.3. Wnioski końcowe** — synteza: dwa największe pojedyncze usprawnienia to **augmentacja danych**
(+~6–7 pp na etapie strojenia, a przy tym najtańszy „chwyt" w całym projekcie) oraz **głębsza architektura**
(VGG: +~6 pp względem baseline'u). Dla porównania, sam dobór harmonogramu `learning rate` (cosine) podniósł
SGD o ~3,6 pp w Etapie 2. Potwierdza to tezę, że **metodyka treningu (harmonogram, augmentacja) potrafi
ważyć tyle samo, co sama architektura** — a w dodatku wchodzi w grę wcześniej i mniejszym kosztem.
Połączenia rezydualne dołożyły kolejne +1,4 pp, lecz już przy wyraźnie gorszym stosunku zysku do kosztu.
Możliwe dalsze kroki: dłuższy trening, mocniejsza augmentacja (np. Cutout/Mixup), większy model lub
uśrednianie modeli (ensembling).

---

## D. Co podać koledze razem z tymi notatkami

Żeby mógł wstawić materiały wizualne i liczby, najwygodniej przekazać:
1. Ten plik.
2. Notebook `notebooks/CIFAR10_Modeling.ipynb` — komórki „Wnioski" przy każdym etapie zawierają gotowe,
   skrótowe podsumowania pokrywające się z punktami wyżej (dobre źródło zdań do wklejenia).
3. Wygenerowane obrazy z `experiments/.../history.png` i `.../confusion_matrix.png`.
4. Tabelę zbiorczą `experiments/results_summary.csv` (źródło liczb do tabel).

Wszystkie etapy są już policzone, a liczby w tym pliku uzupełnione realnymi wartościami z
`experiments/results_summary.csv` oraz plików `metrics.json`.
