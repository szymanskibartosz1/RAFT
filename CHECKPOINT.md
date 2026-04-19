## 1. Plan i metodologia

**Cel i pytania badawcze**
* **Cel:** Głównym celem projektu jest uodpornienie LLM-a na zjawisko "zatrucia kontekstu" (context poisoning) przez teorie spiskowe z wykorzystaniem fine-tuningu w architekturze RAFT (Retrieval-Augmented Fine Tuning).
* **Pytanie badawcze:** Czy model po fine-tuningu metodą RAFT (gdzie jako dystraktorów użyjemy teorii spiskowych) będzie lepiej ignorował szum informacyjny i trzymał się faktów naukowych w porównaniu do standardowego systemu RAG?

**Metryki oceny**
Do weryfikacji użyjemy podejścia LLM-as-a-judge oraz metryk:
    * *Faithfulness (Wierność):* Sprawdza, czy to, co wygenerował model, ma faktyczne potwierdzenie w poprawnej bazie wiedzy.
    * *Answer Relevancy (Trafność):* Ocenia, czy model odpowiada na temat
* **Odporność na szum:**
    * *NDR (No-Degradation Rate):* Procent przypadków, w których dodanie teorii spiskowych do promptu nie popsuło końcowej odpowiedzi.
    * *RSR (Retrieval Size Robustness):* Sprawdzenie, jak model sobie radzi, gdy dorzucimy mu dużo więcej błędnych dokumentów.
    * *ROR (Retrieval Order Robustness):* Czy model nie gubi się, gdy najbardziej toksyczny tekst wyląduje na samym początku wyników wyszukiwania.

**Baseline (Metoda odniesienia)**
* **Vanilla RAG:** Zwykły model spięty z bazą wektorową, bez żadnego dodatkowego fine-tuningu i bez wymuszania Chain-of-Thought. Taki RAG z założenia po prostu ufa temu, co wypluje mu wyszukiwarka, więc powinien szybko zacząć halucynować.

**Plan eksperymentu**
Przetestujemy 2 warianty:
1.  **Baseline (Vanilla RAG):** Ewaluacja, jak zwykły model radzi sobie (lub raczej nie radzi) z dokumentami, które używają retoryki spiskowej.
2.  **Model RAFT:** Model dostrojony metodą QLoRA. Uczymy go na krotkach zawierających poprawne źródła oraz szum. Kluczowe jest tu równanie RAFT – część danych (np. 20%) w ogóle nie będzie miała prawdziwego dokumentu w kontekście. Ma to wymusić na modelu asertywne odrzucanie bzdur bazując wyłącznie na jego własnej, wewnętrznej wiedzy.

**Dane i sposób ich pozyskania**
* **Źródła:** Zbiór treningowy wygenerujemy syntetycznie używając LLM'a.
    * *Złote dokumenty* weźmiemy ze sprawdzonych, polskich źródeł.
    * *Dystraktory* Wygeneruję z pomocą AI tak, aby imitowały typowe chwyty teorii spiskowych.
* **Format:** Model przygotuje poprawne odpowiedzi w formacie *Chain-of-Thought. Chodzi o to, żeby w zbiorze treningowym docelowy model widział, jak ignorować fałszywy szum krok po kroku.
