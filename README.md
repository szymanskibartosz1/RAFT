# RAFT MythBuster: Uodparnianie LLM na teorie spiskowe

Implementacja architektury **RAFT (Retrieval-Augmented Fine-Tuning)** do uodpornienia modelu językowego na polskojęzyczne teorie spiskowe (COVID-19/szczepionki).

## Quickstart

### 1. Przygotowanie danych
```
src/data_collection.py
notebooks/01_data_preparation.ipynb 
```

### 2. Generowanie datasetu RAFT 
`notebooks/02_dataset_generation.ipynb`

### 3. Fine-tuning 
`notebooks/03_finetuning.ipynb`

### 4. Ewaluacja
`notebooks/04_evaluation.ipynb`

## Struktura projektu

```
RAFT/
├── README.md
├── requirements.txt
├── configs/
│   ├── training_config.yaml    # Hiperparametry treningu
│   └── .env.example            # Template dla API keys
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_dataset_generation.ipynb
│   ├── 03_finetuning.ipynb     # QLoRA na Colab
│   └── 04_evaluation.ipynb     # RAFT vs RAG porównanie
├── src/
│   ├── data_collection.py      # Scraping + ładowanie danych
│   ├── dataset_builder.py      # Generowanie krotek RAFT
│   └── evaluation.py           # Metryki + LLM-as-a-Judge
└── data/
    ├── golden_docs.jsonl        # Dokumenty wyroczni
    ├── distractors.jsonl        # Teksty spiskowe
    └── raft_train.jsonl         # Dataset treningowy
```

## Stack technologiczny

| Komponent | Narzędzie |
|:----------|:----------|
| Model bazowy | Mistral 7B v0.3 (4-bit) |
| Fine-tuning | Unsloth + QLoRA (r=16) |
| Teacher model | Gemini 2.5 Pro (darmowy tier) |
| Vector store | ChromaDB |
| Embeddingi | paraphrase-multilingual-MiniLM-L12-v2 |
| Ewaluacja | Custom CRR + LLM-as-a-Judge |

## Metodologia RAFT

Podział datasetu zgodnie z regułą **P = 80%**:
- **80% danych:** Pytanie + Golden Doc (D*) + 4 dystraktory → Odpowiedź CoT
- **20% danych:** Pytanie + 5 dystraktorów (bez D*) → Odpowiedź CoT

Model uczy się:
1. Identyfikować wiarygodne źródła w kontekście
2. Cytować dosłownie (`##begin_quote##`)
3. Odrzucać dezinformację z uzasadnieniem
4. Polegać na wiedzy wewnętrznej, gdy brakuje złotego dokumentu

## Metryki ewaluacji

| Metryka | Opis |
|:--------|:-----|
| **CRR** (Conspiracy Rejection Rate) | % poprawnych odrzuceń dezinformacji |
| **Golden Citation Rate** | % odpowiedzi cytujących wiarygodne źródło |
| **Faithfulness** | Zgodność twierdzeń z kontekstem (Ragas) |
| **LLM-as-a-Judge** | Blind comparison via Gemini Pro |
