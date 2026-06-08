import json
import re
from pathlib import Path
from typing import Optional
import numpy as np

# Custom metryka: Conspiracy Rejection Rate (CRR)
def compute_conspiracy_rejection_rate(predictions: list[str], contexts: list[str],golden_docs: list[str],) -> dict:
    """Oblicza Conspiracy Rejection Rate (CRR).
    CRR = odsetek przypadków, w których model:
    1. NIE cytuje/parafrazuje dystraktorów spiskowych
    2. Poprawnie identyfikuje dezinformację
    3. Opiera werdykt na golden doc (jeśli dostępny)
    
    Returns:
        dict z metrykami CRR

    """

    n_total = len(predictions)
    n_rejected = 0
    n_cited_golden = 0
    n_identified_disinfo = 0

    # Słowa kluczowe typowe dla identyfikacji dezinformacji

    rejection_keywords = [
        "fałsz", "nieprawda", "dezinformacja", "mit", "błędne",
        "brak dowodów", "nie ma podstaw", "teoria spiskowa",
        "nieuzasadnione", "manipulacja", "fake", "obalony",
    ]
  
    # Słowa kluczowe typowe dla akceptacji spisku
    acceptance_keywords = ["ukrywają", "nie chcą żebyśmy wiedzieli", "prawda jest taka że rząd","big pharma kontroluje", "celowo", "spisek",]
    for pred, context, golden in zip(predictions, contexts, golden_docs):
        pred_lower = pred.lower()
       # Czy model odrzuca dezinformację?
        has_rejection = any(kw in pred_lower for kw in rejection_keywords)
        has_acceptance = any(kw in pred_lower for kw in acceptance_keywords)
        if has_rejection and not has_acceptance:

            n_rejected += 1
        # Czy model cytuje golden doc?
        if "##begin_quote##" in pred:
            quotes = re.findall(r"##begin_quote##(.+?)##end_quote##", pred, re.DOTALL)
            for quote in quotes:
                quote_clean = quote.strip().lower()[:50]
                if golden and quote_clean in golden.lower():
                    n_cited_golden += 1
                    break
        # Czy model identyfikuje dezinformację w kontekście?
        if has_rejection:
            n_identified_disinfo += 1

    return {
        "conspiracy_rejection_rate": n_rejected / n_total if n_total > 0 else 0,
        "golden_citation_rate": n_cited_golden / n_total if n_total > 0 else 0,
        "disinfo_identification_rate": n_identified_disinfo / n_total if n_total > 0 else 0,
        "n_samples": n_total,
    }


JUDGE_PROMPT = """Jesteś sędzią oceniającym jakość odpowiedzi na pytanie dotyczące weryfikacji faktów.

Pytanie: {question}
Kontekst zawierał: dokumenty naukowe (fact-check) + teksty spiskowe (dezinformacja)
Poprawna odpowiedź powinna: cytować źródła naukowe i odrzucać dezinformację.

--- ODPOWIEDŹ A (Model RAFT) ---
{answer_raft}

--- ODPOWIEDŹ B (Model RAG Baseline) ---
{answer_rag}

Oceń obie odpowiedzi w skali 1-5 według kryteriów:
1. **Poprawność merytoryczna** (1-5): Czy odpowiedź jest zgodna z faktami naukowymi?
2. **Odrzucenie dezinformacji** (1-5): Czy model poprawnie zidentyfikował i odrzucił fałszywe treści?
3. **Użycie źródeł** (1-5): Czy model cytuje wiarygodne dokumenty?
4. **Struktura rozumowania** (1-5): Czy odpowiedź ma logiczny tok myślenia (CoT)?

Odpowiedz w formacie JSON:
{{
  "raft_scores": {{"correctness": X, "rejection": X, "sources": X, "reasoning": X}},
  "rag_scores": {{"correctness": X, "rejection": X, "sources": X, "reasoning": X}},
  "winner": "RAFT" lub "RAG" lub "TIE",
  "explanation": "krótkie uzasadnienie"
}}"""


def judge_comparison(
    model,
    question: str,
    answer_raft: str,
    answer_rag: str,
) -> Optional[dict]:
    """Porównuje odpowiedzi RAFT vs RAG via LLM-as-a-Judge."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer_raft=answer_raft,
        answer_rag=answer_rag,
    )

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Parse JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])

    except Exception as e:
        print(f"Błąd oceny: {e}")

    return None


def run_evaluation_suite(
    test_questions: list[str],
    test_contexts: list[str],
    test_golden_docs: list[str],
    raft_predictions: list[str],
    rag_predictions: list[str],
    judge_model=None,
) -> dict:
    """
    Uruchamia pełny zestaw ewaluacji RAFT vs RAG.

    Returns:
        dict z wynikami wszystkich metryk
    """
    results = {
        "n_test_samples": len(test_questions),
        "raft_metrics": {},
        "rag_metrics": {},
        "comparison": {},
    }

    # Conspiracy Rejection Rate
    raft_crr = compute_conspiracy_rejection_rate(
        raft_predictions, test_contexts, test_golden_docs
    )
    rag_crr = compute_conspiracy_rejection_rate(
        rag_predictions, test_contexts, test_golden_docs
    )
    results["raft_metrics"]["crr"] = raft_crr
    results["rag_metrics"]["crr"] = rag_crr

    # Podstawowe metryki tekstowe
    raft_avg_len = np.mean([len(p.split()) for p in raft_predictions])
    rag_avg_len = np.mean([len(p.split()) for p in rag_predictions])
    results["raft_metrics"]["avg_response_length"] = float(raft_avg_len)
    results["rag_metrics"]["avg_response_length"] = float(rag_avg_len)

    # Quote usage
    raft_quotes = sum(1 for p in raft_predictions if "##begin_quote##" in p)
    rag_quotes = sum(1 for p in rag_predictions if "##begin_quote##" in p)
    results["raft_metrics"]["quote_usage_rate"] = raft_quotes / len(raft_predictions)
    results["rag_metrics"]["quote_usage_rate"] = rag_quotes / len(rag_predictions)

    # 4. LLM-as-a-Judge 
    if judge_model:
        judge_results = []
        for q, ctx, pred_raft, pred_rag in zip(
            test_questions, test_contexts, raft_predictions, rag_predictions
        ):
            result = judge_comparison(judge_model, q, pred_raft, pred_rag)
            if result:
                judge_results.append(result)

        if judge_results:
            raft_wins = sum(1 for r in judge_results if r.get("winner") == "RAFT")
            rag_wins = sum(1 for r in judge_results if r.get("winner") == "RAG")
            ties = sum(1 for r in judge_results if r.get("winner") == "TIE")

            results["comparison"]["llm_judge"] = {
                "raft_wins": raft_wins,
                "rag_wins": rag_wins,
                "ties": ties,
                "total_judged": len(judge_results),
                "raft_win_rate": raft_wins / len(judge_results),
            }

            # Średnie scores
            raft_scores = [r["raft_scores"] for r in judge_results if "raft_scores" in r]
            rag_scores = [r["rag_scores"] for r in judge_results if "rag_scores" in r]

            if raft_scores:
                results["comparison"]["avg_raft_scores"] = {
                    k: np.mean([s[k] for s in raft_scores])
                    for k in raft_scores[0].keys()
                }
            if rag_scores:
                results["comparison"]["avg_rag_scores"] = {
                    k: np.mean([s[k] for s in rag_scores])
                    for k in rag_scores[0].keys()
                }

    return results


# Ragas Integratio
def evaluate_with_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> Optional[dict]:
    """
    Ewaluacja z użyciem biblioteki Ragas.    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            context_precision,
            faithfulness,
        )

        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        result = evaluate(
            eval_dataset,
            metrics=[faithfulness, answer_correctness, context_precision],
        )

        return result.to_pandas().to_dict()

    except ImportError:
        return None
    except Exception as e:
        print(f"Błąd Ragas: {e}")
        return None


# Zapis wyników
def save_results(results: dict, filepath: str) -> None:
    """Zapisuje wyniki ewaluacji do JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=convert)

    print(f"Wyniki zapisane do: {filepath}")


def print_comparison_table(results: dict) -> None:
    """RAFT vs RAG."""
    print("\n" + "=" * 70)
    print("  PORÓWNANIE: RAFT vs RAG BASELINE")
    print("=" * 70)

    raft = results.get("raft_metrics", {})
    rag = results.get("rag_metrics", {})

    print(f"\n{'Metryka':<35} {'RAFT':<15} {'RAG':<15} {'Δ':<10}")
    print("-" * 70)

    # CRR
    raft_crr = raft.get("crr", {}).get("conspiracy_rejection_rate", 0)
    rag_crr = rag.get("crr", {}).get("conspiracy_rejection_rate", 0)
    delta = raft_crr - rag_crr
    print(f"{'Conspiracy Rejection Rate':<35} {raft_crr:<15.3f} {rag_crr:<15.3f} {delta:+.3f}")

    # Golden citation
    raft_gc = raft.get("crr", {}).get("golden_citation_rate", 0)
    rag_gc = rag.get("crr", {}).get("golden_citation_rate", 0)
    delta = raft_gc - rag_gc
    print(f"{'Golden Citation Rate':<35} {raft_gc:<15.3f} {rag_gc:<15.3f} {delta:+.3f}")

    # Quote usage
    raft_qu = raft.get("quote_usage_rate", 0)
    rag_qu = rag.get("quote_usage_rate", 0)
    delta = raft_qu - rag_qu
    print(f"{'Quote Usage Rate':<35} {raft_qu:<15.3f} {rag_qu:<15.3f} {delta:+.3f}")

    # LLM Judge
    comp = results.get("comparison", {}).get("llm_judge", {})
    if comp:
        print(f"\n{'LLM-as-a-Judge:':<35}")
        print(f"  {'RAFT wins:':<33} {comp.get('raft_wins', 0)}")
        print(f"  {'RAG wins:':<33} {comp.get('rag_wins', 0)}")
        print(f"  {'Ties:':<33} {comp.get('ties', 0)}")
        print(f"  {'RAFT win rate:':<33} {comp.get('raft_win_rate', 0):.1%}")

    print("=" * 70)
