"""
Moduł do pozyskiwania danych referencyjnych (wyroczni) i dystraktorów.
Obsługuje scraping polskich serwisów fact-checkingowych oraz ładowanie
datasetów z HuggingFace.
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# Scraping polskich fact-checków (Konkret24, Demagog, AFP Fakty)
# ─────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research project; contact: student@university.edu)"
}


def scrape_konkret24_article(url: str) -> Optional[dict]:
    """Pobiera treść artykułu fact-checkingowego z Konkret24."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else ""

        # Konkret24 article body
        article_body = soup.find("div", class_="article-body")
        if not article_body:
            article_body = soup.find("article")

        if not article_body:
            return None

        paragraphs = article_body.find_all("p")
        content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        if len(content) < 100:
            return None

        return {
            "source": "konkret24",
            "url": url,
            "title": title_text,
            "content": content,
            "type": "golden_doc",
        }
    except (requests.RequestException, AttributeError):
        return None


def scrape_demagog_article(url: str) -> Optional[dict]:
    """Pobiera treść artykułu z Demagog.pl."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else ""

        content_div = soup.find("div", class_="analysis-content")
        if not content_div:
            content_div = soup.find("div", class_="entry-content")
        if not content_div:
            content_div = soup.find("article")

        if not content_div:
            return None

        paragraphs = content_div.find_all("p")
        content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        if len(content) < 100:
            return None

        return {
            "source": "demagog",
            "url": url,
            "title": title_text,
            "content": content,
            "type": "golden_doc",
        }
    except (requests.RequestException, AttributeError):
        return None


# ─────────────────────────────────────────────────────────────
# Ładowanie dystraktorów z plików / HuggingFace
# ─────────────────────────────────────────────────────────────


def load_distractors_from_jsonl(filepath: str) -> list[dict]:
    """Ładuje dystraktory z pliku JSONL."""
    docs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def load_covid_twitter_distractors(dataset_name: str = "laugustyniak/pl-covid19-twitter") -> list[dict]:
    """
    Ładuje polskie tweety COVID-19 z HuggingFace jako potencjalne dystraktory.
    Wymaga zainstalowanego pakietu `datasets`.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    distractors = []

    for item in ds:
        text = item.get("text", item.get("tweet", ""))
        if len(text) > 50:
            distractors.append({
                "source": "pl_covid_twitter",
                "content": text,
                "type": "distractor",
            })

    return distractors


# ─────────────────────────────────────────────────────────────
# Ręczne dodawanie dokumentów (dla małych zbiorów)
# ─────────────────────────────────────────────────────────────


def create_manual_golden_doc(title: str, content: str, source: str = "manual") -> dict:
    """Tworzy ręczny dokument wyroczni."""
    return {
        "source": source,
        "url": "",
        "title": title,
        "content": content,
        "type": "golden_doc",
    }


def create_manual_distractor(content: str, topic: str = "covid") -> dict:
    """Tworzy ręczny dystraktor spiskowy."""
    return {
        "source": "manual",
        "content": content,
        "topic": topic,
        "type": "distractor",
    }


# ─────────────────────────────────────────────────────────────
# Zapis danych
# ─────────────────────────────────────────────────────────────


def save_documents(docs: list[dict], filepath: str) -> None:
    """Zapisuje dokumenty do pliku JSONL."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Zapisano {len(docs)} dokumentów do {filepath}")


def batch_scrape(urls: list[str], scraper_fn, delay: float = 2.0) -> list[dict]:
    """
    Scraping wielu URL z opóźnieniem między requestami.
    """
    results = []
    for url in tqdm(urls, desc="Scraping"):
        doc = scraper_fn(url)
        if doc:
            results.append(doc)
        time.sleep(delay)
    return results


# ─────────────────────────────────────────────────────────────
# Przykładowe dane (do szybkiego startu bez scrapingu)
# ─────────────────────────────────────────────────────────────

SAMPLE_GOLDEN_DOCS = [
    {
        "source": "WHO",
        "title": "5G i COVID-19 - brak związku",
        "content": (
            "Wirusy nie mogą przemieszczać się za pomocą fal radiowych ani sieci komórkowych. "
            "COVID-19 rozprzestrzenia się w wielu krajach, w których nie ma sieci 5G. "
            "COVID-19 jest przenoszony drogą kropelkową, gdy zarażona osoba kaszle, kicha "
            "lub mówi. Można się również zarazić, dotykając skażonej powierzchni, a następnie "
            "oczu, ust lub nosa. Światowa Organizacja Zdrowia wielokrotnie potwierdziła, "
            "że technologia 5G nie rozprzestrzenia wirusa SARS-CoV-2."
        ),
        "type": "golden_doc",
    },
    {
        "source": "PZH/NIZP",
        "title": "Szczepionki mRNA - mechanizm działania",
        "content": (
            "Szczepionki mRNA (Pfizer-BioNTech, Moderna) zawierają instrukcję genetyczną "
            "do wytworzenia białka kolca (spike) wirusa SARS-CoV-2. Po wstrzyknięciu mRNA "
            "wnika do komórek i instruuje je do produkcji białka spike. Układ odpornościowy "
            "rozpoznaje to białko jako obce i wytwarza przeciwciała. mRNA nie wchodzi do "
            "jądra komórkowego i nie zmienia DNA człowieka. Jest degradowane przez enzymy "
            "komórkowe w ciągu kilku dni. Narodowy Instytut Zdrowia Publicznego potwierdza "
            "bezpieczeństwo i skuteczność szczepionek mRNA na podstawie badań klinicznych "
            "obejmujących dziesiątki tysięcy uczestników."
        ),
        "type": "golden_doc",
    },
    {
        "source": "EMA",
        "title": "Bezpieczeństwo szczepionek COVID-19",
        "content": (
            "Europejska Agencja Leków (EMA) zatwierdziła szczepionki przeciw COVID-19 po "
            "rygorystycznej ocenie danych z badań klinicznych fazy I, II i III. Proces "
            "przyspieszono dzięki równoległemu prowadzeniu faz badań, nie zaś skracaniu "
            "standardów bezpieczeństwa. Systemy nadzoru farmakologicznego (EudraVigilance) "
            "monitorują bezpieczeństwo na bieżąco. Rzadkie działania niepożądane, takie jak "
            "zakrzepica z trombocytopenią (VITT) przy szczepionce AstraZeneca, zostały "
            "zidentyfikowane i odpowiednio zakomunikowane. Stosunek korzyści do ryzyka "
            "pozostaje zdecydowanie pozytywny dla wszystkich zatwierdzonych preparatów."
        ),
        "type": "golden_doc",
    },
]

SAMPLE_DISTRACTORS = [
    {
        "source": "conspiracy_forum",
        "content": (
            "Szczepionki mRNA to tak naprawdę narzędzie kontroli populacji! Nikt nie wie, "
            "co dokładnie jest w tych ampułkach. Bill Gates od lat mówił o redukcji populacji "
            "i teraz nam to wstrzykują. mRNA ZMIENIA nasze DNA — tego nie da się cofnąć! "
            "Dlaczego mainstream media o tym milczą? Bo są opłacani przez Big Pharma. "
            "Obudźcie się ludzie!!!"
        ),
        "type": "distractor",
        "topic": "szczepionki_mRNA",
    },
    {
        "source": "conspiracy_forum",
        "content": (
            "Wieże 5G emitują promieniowanie, które osłabia nasz układ odpornościowy. "
            "Zastanawialiście się, dlaczego pandemia zaczęła się dokładnie wtedy, gdy "
            "uruchamiano sieci 5G? To nie przypadek! W Wuhan były jedne z pierwszych "
            "masowych instalacji 5G na świecie. Fale milimetrowe niszczą strukturę komórek "
            "i otwierają drogę wirusom. Naukowcy, którzy próbowali o tym mówić, byli uciszani."
        ),
        "type": "distractor",
        "topic": "5G_covid",
    },
    {
        "source": "conspiracy_telegram",
        "content": (
            "PILNE! Wyciekły dokumenty z Pfizera — skuteczność szczepionki to tylko 12%! "
            "Reszta to efekt placebo i manipulacja statystykami. Rządy na całym świecie "
            "zmuszają ludzi do szczepień, bo mają kontrakty z firmami farmaceutycznymi "
            "warte miliardy. Kto się nie szczepi, straci pracę, paszport, wolność. "
            "To nowy porządek świata w akcji — najpierw lockdowny, potem certyfikaty "
            "covidowe, a potem totalna kontrola."
        ),
        "type": "distractor",
        "topic": "big_pharma",
    },
    {
        "source": "conspiracy_forum",
        "content": (
            "Ivermektyna leczy COVID-19, ale Big Pharma blokuje tę informację, bo nie mogą "
            "na niej zarobić! W krajach, które stosowały ivermektynę (Indie, Japonia), "
            "pandemia skończyła się szybciej. WHO i FDA celowo odrzucają badania naukowe "
            "potwierdzające skuteczność, bo szczepionki generują większe zyski. Lekarze "
            "przepisujący ivermektynę tracą licencje — to cenzura medyczna!"
        ),
        "type": "distractor",
        "topic": "alternatywne_leczenie",
    },
    {
        "source": "conspiracy_forum",
        "content": (
            "COVID-19 został stworzony w laboratorium w Wuhan jako broń biologiczna. "
            "Dr Fauci finansował badania gain-of-function w tym laboratorium za pieniądze "
            "amerykańskich podatników. Wirus uciekł (albo został celowo wypuszczony), "
            "a potem wszyscy zaangażowani zaczęli zacierać ślady. Media społecznościowe "
            "cenzurują każdego, kto mówi o pochodzeniu laboratoryjnym. To największy "
            "cover-up w historii ludzkości."
        ),
        "type": "distractor",
        "topic": "lab_leak_conspiracy",
    },
]


if __name__ == "__main__":
    # Quick-start: zapisz przykładowe dane
    save_documents(SAMPLE_GOLDEN_DOCS, "data/golden_docs.jsonl")
    save_documents(SAMPLE_DISTRACTORS, "data/distractors.jsonl")
    print("Przykładowe dane zapisane. Rozszerz je o dodatkowe dokumenty.")
