import json
import re
import time
from pathlib import Path
from typing import Optional
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm



# Ładowanie dystraktorów z plików

def load_distractors_from_jsonl(filepath: str) -> list[dict]:
    """Ładuje dystraktory z pliku JSONL."""
    docs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


# Ręczne dodawanie dokumentów
def create_manual_golden_doc(title: str, content: str, source: str = "manual") -> dict:
    """Tworzy ręczny zloty dokument."""
    return {
        "source": source,
        "url": "",
        "title": title,
        "content": content,
        "type": "golden_doc",
    }


def create_manual_distractor(content: str, topic: str = "covid") -> dict:
    """Tworzy ręczny dokument z teoria spiskowa."""
    return {
        "source": "manual",
        "content": content,
        "topic": topic,
        "type": "distractor",
    }

# Zapis danych

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
