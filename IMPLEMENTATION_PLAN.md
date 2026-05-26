# Plan implementacji: RAFT MythBuster

## Spis treści
1. [Wymagania wstępne](#1-wymagania-wstępne)
2. [Konfiguracja środowiska](#2-konfiguracja-środowiska)
3. [Krok 1: Przygotowanie danych](#3-krok-1-przygotowanie-danych)
4. [Krok 2: Generowanie datasetu RAFT](#4-krok-2-generowanie-datasetu-raft)
5. [Krok 3: Fine-tuning modelu](#5-krok-3-fine-tuning-modelu)
6. [Krok 4: Ewaluacja RAFT vs RAG](#6-krok-4-ewaluacja-raft-vs-rag)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Wymagania wstępne

### Konta i klucze API
| Zasób | Link | Uwagi |
|:-------|:-----|:------|
| Google AI Studio (Gemini API) | https://aistudio.google.com/apikey | Darmowy tier: 15 req/min, 1M tokenów/dzień |
| Google Colab | https://colab.research.google.com | Darmowy T4 GPU (16GB VRAM) |
| HuggingFace (opcjonalnie) | https://huggingface.co/settings/tokens | Do zapisu/ładowania modelu |

### Sprzęt
- **Generowanie datasetu (Krok 1-2):** Dowolny komputer z Pythonem 3.10+ (CPU wystarczy)
- **Fine-tuning (Krok 3):** Google Colab z GPU T4 (darmowy) lub A100 (Pro)
- **Ewaluacja (Krok 4):** Google Colab z GPU T4

---

## 2. Konfiguracja środowiska

### 2.1 Klonowanie repozytorium

```bash
git clone https://github.com/<USER>/RAFT.git
cd RAFT
```

### 2.2 Instalacja lokalna (do kroków 1-2)

```bash
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 2.3 Konfiguracja klucza API

```bash
cp configs/.env.example configs/.env
```

Edytuj `configs/.env` i wpisz swój klucz Gemini:
```
GEMINI_API_KEY=AIzaSy...twój_klucz...
```

> ⚠️ **NIE commituj pliku `.env` do repozytorium.** Jest on w `.gitignore`.

### 2.4 Konfiguracja na Google Colab (do kroków 3-4)

Po otwarciu notebooka na Colab:
1. **Runtime → Change runtime type → T4 GPU**
2. Upload plików danych:
   ```python
   from google.colab import files
   files.upload()  # uploaduj data/raft_train.jsonl
   ```
   Lub zamontuj Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Sklonuj repo bezpośrednio na Colab:
   ```python
   !git clone https://github.com/<USER>/RAFT.git
   %cd RAFT
   ```

---

## 3. Krok 1: Przygotowanie danych

**Cel:** Zgromadzić korpus Golden Documents (wyroczni) i Dystraktorów (tekstów spiskowych).

**Gdzie:** Lokalnie lub na Colab (CPU wystarczy)

**Notebook:** `notebooks/01_data_preparation.ipynb`

### Co robi ten krok:
1. Ładuje wbudowane przykładowe dane (3 golden docs + 5 dystraktorów)
2. Pozwala dodać własne dokumenty ręcznie
3. Opcjonalnie scrapuje artykuły z Konkret24/Demagog
4. Zapisuje wynik do `data/golden_docs.jsonl` i `data/distractors.jsonl`

### Uruchomienie:

**Opcja A — Notebook (zalecane):**
```bash
cd notebooks/
jupyter notebook 01_data_preparation.ipynb
```
Uruchom komórki po kolei (Shift+Enter).

**Opcja B — Skrypt (quick-start z sample data):**
```bash
python3 src/data_collection.py
```

### Rezultat:
```
data/
├── golden_docs.jsonl   # 3-50 dokumentów fact-checkingowych
└── distractors.jsonl   # 5-150 tekstów spiskowych
```

### Rozszerzanie danych (opcjonalne, ale zalecane):
- Dodaj więcej golden docs ręcznie (patrz sekcja 2a w notebooku)
- Im więcej dokumentów, tym więcej pytań treningowych w Kroku 2
- Minimum: 5 golden docs → ~15 pytań
- Zalecane: 15-20 golden docs → ~50-60 pytań → ~200 przykładów po augmentacji

---

## 4. Krok 2: Generowanie datasetu RAFT

**Cel:** Wygenerować krotki treningowe {Pytanie, Kontekst, Odpowiedź CoT} za pomocą Gemini Pro.

**Gdzie:** Lokalnie (wymaga tylko CPU + internet)

**Notebook:** `notebooks/02_dataset_generation.ipynb`

**Wymaga:** Klucz API Gemini (`GEMINI_API_KEY`)

### Co robi ten krok:
1. Dzieli golden docs na chunki (300-500 tokenów)
2. Gemini Pro generuje 3 pytania per chunk
3. Gemini Pro generuje odpowiedzi Chain-of-Thought z cytatami
4. Buduje konteksty z regułą P=80% (80% z wyrocznią, 20% bez)
5. Opcjonalnie generuje syntetyczne dystraktory
6. Dzieli dane na train/test (80/20)

### Uruchomienie:

```bash
cd notebooks/
jupyter notebook 02_dataset_generation.ipynb
```

**WAŻNE:** Przed uruchomieniem ustaw klucz API:
```python
os.environ["GEMINI_API_KEY"] = "AIzaSy..."  # <- w komórce 2 notebooka
```

### Czas wykonania:
- ~200 przykładów: **20-40 minut** (darmowy Gemini tier: 15 RPM)
- Delay między requestami: 5 sekund (automatyczny w kodzie)

### Rate limiting:
Jeśli pojawi się błąd `429 Resource Exhausted`:
- Zwiększ parametr `delay` w `build_raft_dataset()` z 5.0 na 8.0
- Lub poczekaj 1 minutę i uruchom ponownie (ma wbudowaną obsługę błędów)

### Rezultat:
```
data/
├── raft_train.jsonl    # ~160 przykładów treningowych
└── raft_test.jsonl     # ~40 przykładów testowych
```

### Walidacja:
Po zakończeniu sprawdź w notebooku:
- Proporcja z/bez wyroczni (powinna być ~80%/20%)
- Jakość pytań (sekcja 6 notebooka — podgląd)
- Czy odpowiedzi zawierają `##begin_quote##` (cytaty)

---

## 5. Krok 3: Fine-tuning modelu

**Cel:** Wytrenować Mistral 7B z adapterami QLoRA na danych RAFT.

**Gdzie:** Google Colab (wymagany GPU T4 lub lepszy)

**Notebook:** `notebooks/03_finetuning.ipynb`

### Przygotowanie na Colab:

1. Otwórz notebook na Colab: `File → Upload notebook` lub sklonuj repo
2. Ustaw runtime: `Runtime → Change runtime type → T4 GPU`
3. Upload danych treningowych:
   ```python
   # Opcja 1: Upload pliku
   from google.colab import files
   uploaded = files.upload()  # wybierz data/raft_train.jsonl

   # Opcja 2: Z Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   !cp /content/drive/MyDrive/RAFT/data/raft_train.jsonl data/
   ```

### Co robi ten krok:
1. Instaluje Unsloth + zależności
2. Ładuje Mistral 7B (4-bit kwantyzacja) — ~5GB VRAM
3. Dodaje adaptery LoRA (r=16, alpha=32)
4. Trenuje 5 epok z loss masking
5. Zapisuje adaptery LoRA

### Parametry treningu:
| Parametr | Wartość | Uwagi |
|:---------|:--------|:------|
| Model | `unsloth/mistral-7b-v0.3-bnb-4bit` | Pre-kwantyzowany |
| LoRA rank | 16 | Balans jakość/pamięć |
| Learning rate | 2e-4 | Standardowe dla QLoRA |
| Epochs | 5 | Dla ~200 przykładów |
| Batch size | 2 (effective: 8) | Mieści się na T4 |
| Max seq length | 2048 | Wystarczające dla kontekstów |

### Czas treningu:
- **T4 (darmowy Colab):** ~30-60 minut dla 200 przykładów
- **A100 (Colab Pro):** ~10-15 minut

### Zapis modelu:
Model zapisuje się do `outputs/raft-mythbuster-mistral-7b-lora/`.

**Aby nie stracić wyniku po disconneccie Colab:**
```python
# Odkomentuj w komórce 7 notebooka:
from google.colab import drive
drive.mount('/content/drive')
model.save_pretrained('/content/drive/MyDrive/raft-mythbuster')
tokenizer.save_pretrained('/content/drive/MyDrive/raft-mythbuster')
```

### Weryfikacja sukcesu:
- Loss powinien spaść z ~2.5 do ~0.5-0.8
- Komórka 8 (test inferencji) powinna wygenerować odpowiedź z cytatami `##begin_quote##`

---

## 6. Krok 4: Ewaluacja RAFT vs RAG

**Cel:** Wykazać, że RAFT > RAG baseline w odrzucaniu dezinformacji.

**Gdzie:** Google Colab (GPU T4)

**Notebook:** `notebooks/04_evaluation.ipynb`

**Wymaga:** Wytrenowany model z Kroku 3 + klucz Gemini (LLM-as-a-Judge)

### Przygotowanie:

1. Załaduj wytrenowany model (z Drive lub outputs/)
2. Załaduj test set (`data/raft_test.jsonl`)
3. Ustaw `GEMINI_API_KEY` (do oceny LLM-as-a-Judge)

### Co robi ten krok:
1. Buduje RAG baseline: ChromaDB + vanilla Mistral 7B (bez fine-tuningu)
2. Indeksuje WSZYSTKIE dokumenty razem (golden + dystraktory) — symulacja brudnego indeksu
3. Generuje odpowiedzi obu modeli na test set
4. Oblicza metryki:
   - **CRR** (Conspiracy Rejection Rate)
   - **Golden Citation Rate**
   - **Quote Usage Rate**
5. Gemini Pro ocenia odpowiedzi jako sędzia (blind comparison)
6. Generuje wykresy porównawcze

### Uruchomienie:
Uruchom komórki po kolei. Sekcje 2-5 generują predykcje (~10-20 min), sekcje 6-8 obliczają metryki.

### Oczekiwane rezultaty:
| Metryka | RAFT (oczekiwane) | RAG Baseline |
|:--------|:------------------|:-------------|
| Conspiracy Rejection Rate | 0.7-0.9 | 0.3-0.5 |
| Golden Citation Rate | 0.6-0.8 | 0.1-0.2 |
| Quote Usage | 0.7-0.9 | 0.0-0.1 |
| LLM Judge Win Rate | 60-80% | 20-40% |

### Outputy:
```
outputs/
├── evaluation_results.json   # Pełne wyniki liczbowe
└── evaluation_plots.png      # Wykresy porównawcze
```

---

## 7. Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'requests'`
```bash
pip install requests beautifulsoup4
```

### Problem: Gemini 429 (Rate Limit Exceeded)
- Zwiększ `delay` w `build_raft_dataset()` do 8-10 sekund
- Darmowy tier: max 15 requestów/minutę
- Poczekaj 1 minutę i uruchom ponownie

### Problem: CUDA Out of Memory na Colab
- Zmniejsz `max_seq_length` z 2048 na 1024
- Zmniejsz `per_device_train_batch_size` z 2 na 1
- Upewnij się, że runtime to T4 (nie CPU)

### Problem: Colab disconnect w trakcie treningu
- Używaj checkpointów (co 50 kroków — domyślnie włączone)
- Zapisuj na Google Drive (odkomentuj kod w notebooku 03)
- Rozważ Colab Pro dla dłuższych sesji

### Problem: Model nie generuje cytatów `##begin_quote##`
- Sprawdź czy dataset ma poprawny format (notebook 02, sekcja 6)
- Zwiększ liczbę epok z 5 na 7-8
- Upewnij się, że loss spadł poniżej 1.0

### Problem: RAG Baseline zbyt dobry (mała delta)
- Dodaj więcej dystraktorów do ChromaDB (zwiększ szum)
- Użyj top-7 zamiast top-5 w retrieval (więcej szumu w kontekście)
- Upewnij się, że indeks zawiera zarówno golden docs JAK I dystraktory

---

## Kolejność wykonania (podsumowanie)

```
[LOKALNIE]                         [GOOGLE COLAB]
    │                                     │
    ▼                                     │
01_data_preparation.ipynb                 │
    │                                     │
    ▼                                     │
02_dataset_generation.ipynb               │
    │                                     │
    │── upload raft_train.jsonl ──────────▶│
    │   + raft_test.jsonl                  ▼
    │                            03_finetuning.ipynb
    │                                     │
    │                                     ▼
    │                            04_evaluation.ipynb
    │                                     │
    │◀── download results ────────────────│
    ▼
 Gotowe! (outputs/)
```

**Szacowany czas realizacji:** 4-6 godzin roboczych (bez czasu oczekiwania na Gemini API)
