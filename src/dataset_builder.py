"""
Budowa zbioru treningowego.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Konfiguracja Gemini


def get_gemini_model(api_key: Optional[str] = None, model_name: str = "gemini-1.5-pro"):
    """Inicjalizuje model Gemini Pro."""
    import google.generativeai as genai

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("brak GEMINI_API_KEY w zmiennej środowiskowej")

    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name)
    return model


# Chunking dokumentów

def chunk_document(doc: dict, chunk_size: int = 400, chunk_overlap: int = 50) -> list[dict]:
    """Chunkuje dokumenty"""
    content = doc["content"]
    chunks = []

    sentences = content.replace("\n", " ").split(". ")
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append({
                **doc,
                "content": current_chunk.strip(),
                "chunk_id": len(chunks),
            })
            
            overlap_sentences = current_chunk.split(". ")
            current_chunk = ". ".join(overlap_sentences[-2:]) + ". " if len(overlap_sentences) > 1 else ""

        current_chunk += sentence + ". "

    if current_chunk.strip():
        chunks.append({
            **doc,
            "content": current_chunk.strip(),
            "chunk_id": len(chunks),
        })

    return chunks if chunks else [doc]


# Generowanie pytań i odpowiedzi via Gemini

QUESTION_GENERATION_PROMPT = """Jesteś ekspertem od weryfikacji faktów medycznych i naukowych.
Na podstawie poniższego dokumentu naukowego/fact-checkingowego, wygeneruj {n_questions} pytań,
na które odpowiedź znajduje się WYŁĄCZNIE w tym dokumencie.

Pytania powinny:
- Dotyczyć konkretnych twierdzeń naukowych z dokumentu
- Być sformułowane w sposób, jakby pytał ktoś narażony na dezinformację
- Wymagać krytycznego myślenia i analizy źródeł

Dokument:
\"\"\"
{document}
\"\"\"

Odpowiedz w formacie JSON (lista stringów):
["pytanie 1", "pytanie 2", ...]"""

COT_ANSWER_PROMPT = """Jesteś ekspertem od weryfikacji faktów. Odpowiedz na pytanie na podstawie
podanego kontekstu, stosując metodę Chain-of-Thought.

WAŻNE ZASADY:
1. Cytuj DOSŁOWNIE fragmenty z wiarygodnych dokumentów używając znaczników:
   ##begin_quote## [dokładny cytat] ##end_quote##
2. Wyjaśnij, dlaczego ewentualne spiskowe/fałszywe dokumenty w kontekście są BŁĘDNE
3. Sformułuj ostateczny werdykt

Pytanie: {question}

Kontekst (zawiera dokumenty naukowe i potencjalne dezinformacje):
{context}

Odpowiedz w formacie:
ANALIZA: [Zidentyfikuj wiarygodne źródła vs dystraktory]
CYTAT: ##begin_quote## [dosłowny cytat z wiarygodnego źródła] ##end_quote##
ROZUMOWANIE: [Wyjaśnij dlaczego dystraktory są błędne, wskaż błędy logiczne]
WERDYKT: [Ostateczna odpowiedź oparta na dowodach naukowych]"""

DISTRACTOR_SYNTHESIS_PROMPT = """Jesteś badaczem dezinformacji. Na potrzeby projektu edukacyjnego
dotyczącego wykrywania teorii spiskowych, wygeneruj {n_distractors} FAŁSZYWYCH tekstów
(dystraktorów spiskowych) semantycznie podobnych do poniższego pytania.

Każdy dystraktor powinien:
- Dotyczyć tego samego tematu co pytanie
- Zawierać typowe cechy teorii spiskowych (nefarystyczne intencje, odporność na dowody,
  język emocjonalny, fałszywe autorytety)
- Wyglądać przekonująco na pierwszy rzut oka, ale zawierać błędy logiczne
- Mieć 50-150 słów

Pytanie: {question}
Temat: {topic}

Odpowiedz w formacie JSON (lista stringów):
["dystraktor 1", "dystraktor 2", ...]"""


def generate_questions(model, document: dict, n_questions: int = 3) -> list[str]:
    """Generuje pytania na podstawie dokumentu wyroczni."""
    prompt = QUESTION_GENERATION_PROMPT.format(
        n_questions=n_questions,
        document=document["content"],
    )

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Parsowanie JSON z odpowiedzi
        # Szukamy listy JSON w tekście
        json_match = text
        if "```json" in text:
            json_match = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            json_match = text.split("```")[1].split("```")[0]

        # Próba sparsowania
        start = json_match.find("[")
        end = json_match.rfind("]") + 1
        if start >= 0 and end > start:
            questions = json.loads(json_match[start:end])
            return [q for q in questions if isinstance(q, str) and len(q) > 10]

    except Exception as e:
        print(f"Błąd generowania pytań {e}")

    return []


def generate_cot_answer(model, question: str, context: str) -> str:
    """Generuje odpowiedź Chain-of-Thought."""
    prompt = COT_ANSWER_PROMPT.format(question=question, context=context)

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Błąd generowania odpowiedzi: {e}")
        return ""


def generate_synthetic_distractors(
    model, question: str, topic: str, n_distractors: int = 3
) -> list[str]:
    """Generuje syntetyczne dystraktory spiskowe via Gemini."""
    prompt = DISTRACTOR_SYNTHESIS_PROMPT.format(
        n_distractors=n_distractors,
        question=question,
        topic=topic,
    )

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        json_match = text
        if "```json" in text:
            json_match = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            json_match = text.split("```")[1].split("```")[0]

        start = json_match.find("[")
        end = json_match.rfind("]") + 1
        if start >= 0 and end > start:
            distractors = json.loads(json_match[start:end])
            return [d for d in distractors if isinstance(d, str) and len(d) > 30]

    except Exception as e:
        print(f"Błąd generowania dystraktorów: {e}")

    return []


# Budowa datasetu RAFT


def build_context(
    golden_doc: Optional[dict],
    distractors: list[dict],
    include_oracle: bool,
) -> str:
    """Buduje kontekst z dokumentów"""
    docs = []

    if include_oracle and golden_doc:
        docs.append(f"[Dokument A]\n{golden_doc['content']}")

    for i, dist in enumerate(distractors):
        label = chr(ord("A") + (1 if include_oracle else 0) + i)
        content = dist if isinstance(dist, str) else dist.get("content", "")
        docs.append(f"[Dokument {label}]\n{content}")

    # Losowa kolejność, żeby model nie uczył się pozycji
    random.shuffle(docs)
    return "\n\n---\n\n".join(docs)


def build_raft_dataset(
    golden_docs: list[dict],
    distractors_pool: list[dict],
    model,
    oracle_ratio: float = 0.8,
    n_questions_per_doc: int = 3,
    n_distractors: int = 4,
    delay: float = 4.0,
    use_synthetic_distractors: bool = True,
) -> list[dict]:
    """
    Buduje pełny dataset RAFT.

    Args:
        golden_docs: Lista dokumentów wyroczni (chunked)
        distractors_pool: Pula dokumentów dystraktorów
        model: Model Gemin
        oracle_ratio: P% - proporcja przykładów z wyrocznią w kontekście
        n_questions_per_doc: Ile pytań generować per dokument
        n_distractors: Ile dystraktorów w kontekście
        delay: Opóźnienie między wywołaniami API 
        use_synthetic_distractors: Czy generować dodatkowe dystraktory via Gemini
    """
    dataset = []

    for doc in tqdm(golden_docs, desc="Budowanie datasetu RAFT"):
        # 1. Generuj pytania
        questions = generate_questions(model, doc, n_questions=n_questions_per_doc)
        time.sleep(delay)

        if not questions:
            continue

        for question in questions:
            # 2. Wybierz dystraktory z puli
            available_distractors = [d for d in distractors_pool if d.get("content", "") != doc.get("content", "")]
            selected_distractors = random.sample(
                available_distractors,
                min(n_distractors, len(available_distractors)),
            )

            # 3. Opcjonalnie wygeneruj syntetyczne dystraktory
            if use_synthetic_distractors and len(selected_distractors) < n_distractors:
                synthetic = generate_synthetic_distractors(
                    model, question, doc.get("title", "covid"), n_distractors=2
                )
                for s in synthetic:
                    selected_distractors.append({"content": s, "type": "distractor", "source": "synthetic"})
                time.sleep(delay)

            # 4. Zdecyduj czy include oracle (reguła P%)
            include_oracle = random.random() < oracle_ratio

            # 5. Zbuduj kontekst
            context = build_context(
                golden_doc=doc if include_oracle else None,
                distractors=selected_distractors[:n_distractors],
                include_oracle=include_oracle,
            )

            # 6. Wygeneruj odpowiedź CoT
            answer = generate_cot_answer(model, question, context)
            time.sleep(delay)

            if not answer:
                continue

            # 7. Zapisz krotkę
            dataset.append({
                "question": question,
                "context": context,
                "answer": answer,
                "has_oracle": include_oracle,
                "golden_doc_title": doc.get("title", ""),
                "golden_doc_source": doc.get("source", ""),
            })

    return dataset


# Formatowanie do promptu treningowego

TRAINING_PROMPT_TEMPLATE = """### Instruction:
Jesteś ekspertem od weryfikacji faktów. Na podstawie podanego kontekstu odpowiedz na pytanie.
Cytuj wiarygodne źródła dosłownie (używając ##begin_quote## i ##end_quote##).
Zidentyfikuj i odrzuć dezinformację, wskazując błędy logiczne.

### Context:
{context}

### Question:
{question}

### Response:
{answer}"""


def format_for_training(example: dict) -> str:
    """Formatuje przykład do postaci promptu treningowego."""
    return TRAINING_PROMPT_TEMPLATE.format(
        context=example["context"],
        question=example["question"],
        answer=example["answer"],
    )


def save_raft_dataset(dataset: list[dict], filepath: str) -> None:
    """Zapisuje dataset RAFT do pliku JSONL."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Zapisano {len(dataset)} przykładów do {filepath}")
    print(f"  - Z wyrocznią: {sum(1 for d in dataset if d['has_oracle'])}")
    print(f"  - Bez wyroczni: {sum(1 for d in dataset if not d['has_oracle'])}")


def load_raft_dataset(filepath: str) -> list[dict]:
    """Ładuje dataset RAFT z pliku JSONL."""
    dataset = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset
