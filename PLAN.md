# Optymalizacja treningu sieci do rozpoznawania twarzy

Dokument opisuje cel, zasady pracy i plan projektu. Stan: **planowanie, kod treningowy nie jest jeszcze zmieniany**.

## 1. Cel i priorytety

1. **Priorytet nadrzędny: optymalizacja bez utraty dokładności.** Każda zmiana musi mieć dowód, że jakość modelu nie spadła.
2. Priorytet drugi: zmiany, które przy minimalnej różnicy w wyniku dają duże przyspieszenie (sekcja 7).
3. Cel długoterminowy: wykorzystać możliwości komputera. Każda optymalizacja jest poparta profilerem.
4. Technologie: Python, a tam, gdzie profil to uzasadni, C/C++, CUDA, ewentualnie asembler.

**Metryka główna: time-to-accuracy**, czyli czas do osiągnięcia ustalonej dokładności (LFW, CFP-FP, AgeDB-30). Pomocniczo: obrazy na sekundę, szczytowe VRAM, czas oczekiwania GPU na dane.

**Kiedy uznamy komputer za wykorzystany:** GPU jest zajęte ponad 95% czasu, dane nigdy nie każą mu czekać, najdroższe kernele są blisko limitu teoretycznego (roofline w Nsight Compute), a każda pozostała strata jest wyjaśniona.

## 2. Środowisko

| Element | Wartość |
|---|---|
| CPU | Intel i7-11370H, 4 rdzenie / 8 wątków, AVX-512 z VNNI |
| RAM | 16 GB |
| GPU | NVIDIA RTX 3050 Ti Laptop, 4 GB, Ampere (sm_86), obsługa FP16/BF16/TF32, **brak FP8** |
| iGPU | Intel Iris Xe |
| Dyski | 2× NVMe |
| System docelowy | **Pop!_OS (natywnie)**: oficjalny Triton, pełne liczniki `perf`/VTune, brak narzutu WDDM |

Uwagi dla Pop!_OS:
- Użyj ISO z wbudowanym sterownikiem NVIDIA.
- Tryb grafiki `hybrid` (pulpit na Iris Xe) zostawia RTX całą pamięć i cały czas na trening.
- Pomiary tylko na zasilaczu, w trybie performance, z włączonym `nvidia-smi -pm 1`.
- Logować zegary, temperaturę i pobór mocy GPU (NVML) przy każdym pomiarze, bo laptop się przegrzewa i zwalnia.
- Repozytorium trzymać poza synchronizowanymi folderami (np. OneDrive), a dane i buildy poza gitem.

## 3. Punkt wyjścia

Brak własnego skryptu, więc bazą jest `arcface_torch` z insightface. Kopia referencyjna (**niezmieniana**) jest w [reference/arcface_torch/](reference/arcface_torch/). Użyte pliki: `train_v2.py`, `partial_fc_v2.py`, `losses.py`, `dataset.py`, `lr_scheduler.py`, `backbones/`, `utils/`, `eval/verification.py`, `configs/`.

Założenia: 1 GPU z 4 GB, zbiór **CASIA-WebFace** (10 572 osoby, ok. 490 tys. zdjęć), modele **MobileFaceNet (mbf)** i **IResNet-50 (r50)**, `fp16=True`, batch 128, SGD, ArcFace (margin 0.5, s=64). Licencje zbiorów dopuszczają tylko użycie badawcze.

## 4. Jak dowodzimy, że dokładność nie spada

Klasy zmian:

- **A: wynik identyczny co do bitu.** Test: 200 kroków w trybie deterministycznym (`use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`), ten sam seed i kolejność danych. Loss musi być identyczny w każdym kroku.
- **B: ta sama matematyka, inna kolejność zaokrągleń** (`channels_last`, `torch.compile`, fused optimizer). Test: te same 200 kroków z różnicami tylko na poziomie zaokrągleń, plus pełny trening.
- **C: zmiana przepisu treningu** (inny model, mniej epok, BF16). Rozstrzygają tylko pełne treningi.

Zasady:
1. **Szum seeda.** Baseline trenujemy 2–3 razy z różnymi seedami. Rozrzut wyników wyznacza próg, poniżej którego różnicy nie uznajemy za stratę.
2. **Ewaluacja jest przyrządem pomiarowym.** Po każdej jej zmianie oceniamy ponownie checkpoint baseline'u. Wynik musi być identyczny.
3. **Punktem odniesienia jest baseline na tym laptopie**, a nie liczby z publikacji (te pochodzą z 8 GPU).
4. Zmiany klas B i C sprawdzamy głównie na mbf. r50 potwierdzamy na końcu, bo trening jest długi.

Każda optymalizacja to jeden commit i jeden wpis w dzienniku: hipoteza → profil → zmiana → pomiar przed i po. Implementacja referencyjna zostaje na stałe jako test poprawności.

## 5. Co trzeba naprawić przed pierwszym pomiarem

- **mxnet** (dataset.py:7, verification.py:31) jest porzucony i wymaga starego numpy. Zastępujemy go własnym czytnikiem RecordIO (ok. 30 linii) i libjpeg-turbo. mxnet zostaje tylko w osobnym środowisku do jednorazowego porównania hashy zdekodowanych obrazów. Pułapka: `cv2.imdecode` zwraca BGR, mxnet RGB.
- **Ewaluacja zajmuje ok. 11,4 GB RAM** (verification.py:208-220): 38 tys. obrazów × 2 (z odbiciem) jako float32. W uint8 i bez odbić to ok. 1,4 GB. Klasa A.
- **VRAM.** Szacunek: r50 przy batchu 128 w FP16 potrzebuje ok. 4,5 GB na aktywacje. Do potwierdzenia pierwszym uruchomieniem. Obejścia mają pułapki:
  - *Activation checkpointing* (iresnet.py:6) daje identyczne gradienty, ale trening trwa ok. 30% dłużej. Przy ponownym liczeniu forwardu BatchNorm drugi raz aktualizuje statystyki, więc wymaga poprawki (momentum=0 podczas przeliczania).
  - *Gradient accumulation* ma w kodzie błąd: loss nie jest dzielony przez `gradient_acc` (train_v2.py:186, 194), więc efektywny learning rate rośnie. Statystyki BatchNorm liczą się z mniejszego batcha, więc to nie jest dokładny odpowiednik większego batcha.
  - Baseline r50 ustalamy w wariancie, który się mieści, i potem go nie zmieniamy. mbf mieści się bez problemu.
- **Zgodność z aktualnym PyTorchem:** `verbose` w schedulerze (lr_scheduler.py:7), `torch.cuda.amp.*`, `_LRScheduler`.

## 6. Priorytet 1: przyspieszenia bez utraty dokładności

| # | Zmiana | Gdzie | Klasa | Zysk |
|---|---|---|---|---|
| 1 | Usunięcie DDP i `fp16_compress_hook` przy jednym GPU | train_v2:98-105 | B (mniej błędu) | średni |
| 2 | Naprawa wyścigu w `DataLoaderX` (`record_stream`) | dataset:120-134 | poprawność | warunek dla #3 |
| 3 | Usunięcie synchronizacji CPU↔GPU w każdym kroku | niżej | A | mały dla r50, duży dla mbf |
| 4 | `SGD(fused=True)`, GradScaler bez synchronizacji | train_v2:121 | B | mały–średni |
| 5 | Uproszczenie PartialFC dla 1 GPU, `F.cross_entropy` zamiast `DistCrossEntropy` | partial_fc_v2:132-219 | A/B | mały–średni |
| 6 | `channels_last` dla modelu i wejścia | train_v2:95, dataset | B | duży |
| 7 | Surowe uint8 na GPU, normalizacja tam | dataset:140-145 | A | średni (mbf) |
| 8 | Bezpośrednie dekodowanie zamiast NDArray→numpy→PIL→Tensor; dobór `num_workers`, `persistent_workers`, `OMP_NUM_THREADS=1` | dataset:159-170, base:50 | A | duży, jeśli mbf czeka na dane |
| 9 | Alternatywa dla #8: dataset raz zdekodowany do surowych uint8 na NVMe (ok. 18 GB, memmap) | nowy plik | A | dekodowanie znika |
| 10 | `torch.compile` | train_v2 | B | średni–duży |
| 11 | CUDA Graphs dla całego kroku | train_v2 | B | duży (mbf) |
| 12 | Ewaluacja: batch 10→256, uint8, normalizacja na GPU, usunięcie martwego kodu, rzadsze uruchamianie | utils_callbacks:31-32, verification | A | duży (czas zegarowy) |
| 13 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | środowisko | A | lepsze wykorzystanie 4 GB |
| 14 | Tryb `hybrid`, profil performance, `nvidia-smi -pm 1` | system | A | stabilniejsze pomiary |

### Szczegóły

**#1.** `fp16_compress_hook` zamienia gradienty na fp16 i z powrotem także przy jednym GPU, bo DDP nadal grupuje je w kubełki. Efekty: zaokrąglenie gradientów do 11 bitów mantysy, więcej pominiętych kroków GradScalera, narzut `find_unused_parameters=True`, dodatkowa kopia gradientów (ok. 175 MB VRAM dla r50). Wynik po usunięciu nie będzie identyczny z baseline'em, bo znika źródło błędu numerycznego.

**#2.** `DataLoaderX` kopiuje batch w osobnym strumieniu bez `record_stream`. Alokator może oddać pamięć batcha *k* kolejnemu kopiowaniu, zanim backward kroku *k* skończy czytać wejście (conv1 potrzebuje go do gradientu wag). Dziś chroni przed tym tylko `loss.item()` w każdym kroku. Usunięcie synchronizacji bez tej poprawki po cichu psuje dane.

**#3.** Synchronizacje, które zatrzymują CPU do końca pracy GPU:
- `loss.item()` (train_v2:204, 210): sumujemy loss na GPU i odczytujemy co `frequent` kroków;
- `loss_gradient.item()` (partial_fc_v2:219): mnożymy przez tensor, wynik identyczny co do bitu;
- `torch.zeros(...).cuda()` ×2 (partial_fc_v2:133, 137): zbędne przy jednym GPU;
- `torch.where(labels != -1)` (losses:28, partial_fc_v2:190): dynamiczny rozmiar wyniku, a przy jednym GPU wszystkie etykiety są poprawne;
- `GradScaler.step` z `.item()` na `found_inf` (rozwiązanie w #4).

`torch.cuda.set_sync_debug_mode("warn")` wskaże synchronizacje spoza listy.

**#4.** Zostajemy przy FP16 z GradScalerem, bo to przepis referencyjny. Fused SGD dostaje `found_inf` i `grad_scale` na GPU i pomija krok bez `.item()`.

**#5.** ArcFace ma celowo uproszczony gradient: margines jest nakładany w `no_grad` (losses:42-49), więc gradient przechodzi przez cos(θ+m) jak przez funkcję tożsamościową. `F.cross_entropy`, `torch.compile` i własne kernele muszą zachować dokładnie to zachowanie. „Poprawny” matematycznie gradient zmieniłby przebieg treningu. Obcięcie do 1e-30 (partial_fc_v2:196) wpływa tylko na logowaną wartość lossu.

**#7.** Normalizacja (`/255`, `-0.5`, `/0.5`) na GPU jest identyczna co do bitu. Losowe odbicie zostaje na CPU, bo przeniesienie go zmienia strumień liczb losowych i uniemożliwia porównanie przebiegów bit po bicie. Robimy to osobną zmianą.

**#10–11.** Wymagają wcześniejszych #1, #3, #4, #5, inaczej graf się rozpadnie (graph breaks). CUDA Graphs wymagają stałych kształtów (`drop_last=True` już jest) i learning rate trzymanego jako tensor.

**#12.** Ewaluacja uruchamia się co `verbose=2000` kroków i przepuszcza przez sieć 76 tys. obrazów w batchach po 10, z normalizacją na CPU. Dodatkowo:
- `calculate_val` (verification:190-196) to pętla 4000 progów × 10 foldów, której wynik jest wyrzucany (verification:272), więc do usunięcia;
- pętla `xnorm` w Pythonie (verification:254-262) zastępowana jedną operacją numpy;
- progi w `calculate_roc` przez sortowanie i sumę skumulowaną, bo liczniki są całkowite i wynik wyjdzie identyczny;
- rzadsza ewaluacja nie zmienia finalnego modelu, bo skrypt zapisuje ostatni checkpoint, a nie najlepszy;
- po usunięciu DDP dane ewaluacyjne trzeba samemu przenosić na GPU (dziś robi to po cichu DDP, verification:242-243).

## 7. Priorytet 2: duże przyspieszenie przy minimalnej różnicy (klasa C)

| Zmiana | Zysk | Ryzyko |
|---|---|---|
| BF16 zamiast FP16 | znika GradScaler i pomijane kroki, łatwiejsze CUDA Graphs | 8 bitów mantysy zamiast 11 |
| r34 zamiast r50 | ok. 70% obliczeń (4,4 vs 6,3 GMAC) | spadek głównie na CFP-FP i AgeDB |
| r18 zamiast r50 | ok. 41% obliczeń (2,6 GMAC) | większy spadek |
| Mniej epok / krótszy harmonogram LR | proporcjonalny do skrócenia | decyduje time-to-accuracy |
| Większy batch ze skalowaniem LR (mbf) | lepsze wykorzystanie GPU | zmienia zachowanie BatchNorm i optymalizacji |
| Dekodowanie JPEG na GPU (DALI, nvJPEG) | odciąża 4 rdzenie CPU | piksele różnią się o ±1, ryzyko praktycznie zerowe |

## 8. Czego nie robić

- **FP8:** Ampere (sm_86) nie ma go w rdzeniach Tensor.
- **FP16/TF32 w głowicy i ostatniej warstwie `fc`:** autorzy celowo zostawili FP32 (train_v2:118, iresnet:160). To mniej niż 2% obliczeń, a margines przy s=64 jest wrażliwy na precyzję.
- **PartialFC z `sample_rate<1`:** opłaca się dopiero przy setkach tysięcy klas, a CASIA ma 10,5 tys.
- **Własny kernel ArcFace jako priorytet:** głowica to ok. 0,1% obliczeń r50 i 1–2% mbf. Dopiero gdy profil pokaże te kernele.
- **Asembler x86 w pętli treningowej:** obliczenia są na GPU. C++/asembler mają sens w loaderze danych, jeśli po #7–#9 mbf nadal czeka na dane.

## 9. Kolejność prac

1. **Nowe środowisko:** bez mxnet, ewaluacja w mniejszym RAM, zgodność z aktualnym PyTorchem. Baseline: mbf 2–3 razy z różnymi seedami, r50 raz lub dwa.
2. **Profilowanie baseline'u:** znaczniki NVTX (dane, forward, głowica, backward, optymalizator), `set_sync_debug_mode`, Nsight Systems.
3. **Zmiany klasy A** (#1, #2, #3, #5, #7, #8, #12, #13): jeden commit, jedna zmiana, jeden test równoważności.
4. **Zmiany klasy B** (#6, #4, #10, #11): test tolerancji po każdej, pełny trening mbf po całej serii.
5. **Ponowny profil:** na jego podstawie decyzja o loaderze w C++ i własnych kernelach.
6. **Eksperymenty z priorytetu 2:** każdy jako osobny pełny trening.

## 10. Zakres opcjonalny (po zakończeniu treningu)

- **Własny loader w C++:** plik z rekordami przez mmap, wątki poza GIL, libjpeg-turbo, augmentacja połączona z normalizacją i HWC→CHW w jednym przejściu, zapis do pinned memory. Łączenie z Pythonem: nanobind + scikit-build-core, tensory przez DLPack. Mikrokernele SIMD: porównanie auto-wektoryzacji, intrinsics AVX2/AVX-512 i asemblera (Google Benchmark, VTune).
- **Własne kernele GPU** (Triton, potem CUDA C++), rejestrowane przez `torch.library`, z testami zgodności wartości i gradientów z referencją.
- **Wyszukiwanie twarzy w dużej bazie** (np. 1 mln × 512): FP32 AVX-512 FMA, int8 z VNNI w asemblerze, BLAS, FAISS. Zadanie ograniczone przepustowością RAM, więc int8 daje ok. 4× mniej danych do czytania.
- **Iris Xe** do równoległej ewaluacji checkpointów (OpenVINO). CPU, iGPU i RTX dzielą budżet mocy i chłodzenia, więc trzeba zmierzyć wpływ na zegary RTX.

## 11. Planowana struktura repozytorium

```
reference/arcface_torch/   # kopia referencyjna insightface, niezmieniana
src/facetrain/             # Python: dane, modele, ArcFace, trening, ewaluacja
csrc/                      # C++ / CUDA / asm (jeśli profil to uzasadni)
bench/                     # harness i mikrobenchmarki
tests/                     # równoważność z referencją
results/                   # wyniki w JSON (duże trace'y poza gitem)
JOURNAL.md                 # hipoteza → profil → zmiana → pomiar
PLAN.md                    # ten dokument
```
