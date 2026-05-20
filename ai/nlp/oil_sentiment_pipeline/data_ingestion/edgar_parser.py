"""
edgar_parser.py
---------------
Collecte et extraction de texte depuis l'API SEC EDGAR.

Cible :
  - Filings 10-K, 10-Q, 8-K des grandes compagnies pétrolières
  - Extraction des sections textuelles pertinentes (risk factors, MD&A, etc.)

API utilisée : https://efts.sec.gov (full-text search) + EDGAR REST API
    https://data.sec.gov/submissions/{cik}.json
    https://efts.sec.gov/LATEST/search-index?q=...

Format de sortie standard :
    {"text": str, "date": str (ISO-8601), "source": str}
"""

import logging
import re
import time
import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional

from oil_sentiment_pipeline.data_ingestion.config import (
    REQUEST_DELAY_SEC,
    REQUEST_TIMEOUT,
    IngestionConfig,
    default_edgar_window,
    load_env,
)
from oil_sentiment_pipeline.data_ingestion.mock_data import build_mock_edgar

logger = logging.getLogger(__name__)

EDGAR_BASE_URL = "https://data.sec.gov"


def _sec_headers(host: str = "data.sec.gov") -> Dict[str, str]:
    cfg = IngestionConfig.from_env()
    ua = cfg.sec_user_agent()
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Host": host,
    }

# CIK des grandes compagnies pétrolières
OIL_COMPANY_CIKS = {
    "ExxonMobil":       "0000034088",
    "Chevron":          "0000093410",
    "BP":               "0000313807",
    "Shell":            "0001306965",
    "ConocoPhillips":   "0001163165",
    "HalliburtonCo":    "0000045012",
    "MarathonOil":      "0000101778",
    "OccidentalPetroleum": "0000797468",
}

# Sections 10-K/10-Q pertinentes pour le sentiment
RELEVANT_SECTIONS = [
    "item 1a",   # Risk Factors
    "item 7",    # MD&A
    "item 7a",   # Quantitative disclosures about market risk
    "item 8",    # Financial statements (partiel)
]

MAX_TEXT_CHARS = 3000  # tronque les sections longues

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_html(html_text: str) -> str:
    """Supprime les balises HTML et normalise les espaces."""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_section(text: str, section_name: str) -> Optional[str]:
    """
    Extrait le contenu d'une section à partir de son titre dans un texte brut.
    Utilise une heuristique de détection par marqueur.
    """
    pattern = re.compile(
        rf"(?i){re.escape(section_name)}[\.\s\-–:]+(.{{200,{MAX_TEXT_CHARS}}}?)(?=item\s+\d|$)",
        re.DOTALL,
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()[:MAX_TEXT_CHARS]
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Requêtes EDGAR
# ---------------------------------------------------------------------------

def _get_company_filings(cik: str, form_type: str = "10-K", max_filings: int = 5) -> List[Dict]:
    """
    Récupère la liste des filings d'une entreprise depuis l'API EDGAR.
    Retourne une liste de dicts {accession_number, filing_date, form_type, primary_document}.
    """
    url = f"{EDGAR_BASE_URL}/submissions/CIK{cik}.json"
    filings_list = []

    try:
        resp = requests.get(url, headers=_sec_headers(), timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        documents = recent.get("primaryDocument", [])

        count = 0
        for i, form in enumerate(forms):
            if form == form_type and count < max_filings:
                filings_list.append({
                    "accession_number": accessions[i].replace("-", ""),
                    "filing_date": dates[i],
                    "form_type": form,
                    "primary_document": documents[i] if i < len(documents) else "",
                })
                count += 1

        logger.info("CIK %s : %d filings %s trouvés.", cik, len(filings_list), form_type)

    except Exception as exc:
        logger.error("Erreur récupération filings CIK %s : %s", cik, exc)

    time.sleep(REQUEST_DELAY_SEC)
    return filings_list


def _fetch_filing_text(cik: str, accession_number: str, primary_document: str) -> Optional[str]:
    """
    Télécharge le texte brut d'un filing EDGAR.
    Tente d'abord le document primaire, puis le fichier .txt global.
    """
    base = f"{EDGAR_BASE_URL}/Archives/edgar/data/{int(cik)}/{accession_number}"
    urls_to_try = []

    if primary_document:
        urls_to_try.append(f"{base}/{primary_document}")

    # Fallback : fichier texte complet
    acc_dashed = f"{accession_number[:10]}-{accession_number[10:12]}-{accession_number[12:]}"
    urls_to_try.append(f"{base}/{acc_dashed}.txt")

    for url in urls_to_try:
        try:
            resp = requests.get(url, headers=_sec_headers(), timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                logger.debug("Fichier téléchargé : %s", url)
                return _clean_html(resp.text)
        except Exception as exc:
            logger.debug("Impossible de charger %s : %s", url, exc)
        time.sleep(REQUEST_DELAY_SEC)

    return None


def _extract_relevant_text(raw_text: str, company_name: str) -> str:
    """Extrait les sections pertinentes et les concatène."""
    extracted = []
    for section in RELEVANT_SECTIONS:
        section_text = _extract_section(raw_text, section)
        if section_text:
            extracted.append(f"[{section.upper()}] {section_text}")

    if extracted:
        return " | ".join(extracted)[:MAX_TEXT_CHARS * 2]

    # Si aucune section trouvée, retourne les premiers MAX_TEXT_CHARS caractères
    return raw_text[:MAX_TEXT_CHARS]


# ---------------------------------------------------------------------------
# EDGAR Full-Text Search (recherche globale "crude oil")
# ---------------------------------------------------------------------------

def fetch_edgar_full_text_search(
    keyword: str = "crude oil",
    form_types: List[str] = None,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    max_results: int = 20,
) -> List[Dict]:
    """
    Utilise l'API EDGAR EFTS pour rechercher des filings contenant un mot-clé.
    """
    if form_types is None:
        form_types = ["10-K", "10-Q", "8-K"]

    results = []

    for form_type in form_types:
        url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q=%22{keyword.replace(' ', '+')}%22"
            f"&dateRange=custom&startdt={start_date}&enddt={end_date}"
            f"&forms={form_type}"
        )
        try:
            resp = requests.get(
                url, headers=_sec_headers(host="efts.sec.gov"), timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            for hit in hits[:max_results]:
                source_data = hit.get("_source", {})
                file_date = source_data.get("file_date", "")
                entity_name = source_data.get("entity_name", "EDGAR")
                display_names = source_data.get("display_names", [])
                name = display_names[0] if display_names else entity_name

                # Texte extrait du résumé de recherche
                highlights = hit.get("highlight", {})
                text_fragments = highlights.get("file_date", []) or highlights.get("period_of_report", [])
                description = source_data.get("file_date", "")

                # Construction texte depuis les highlights
                text_parts = []
                for key, vals in highlights.items():
                    if isinstance(vals, list):
                        text_parts.extend(vals)
                text = _clean_html(" ".join(text_parts)) if text_parts else f"{name} {form_type} filing mentioning {keyword}"

                if file_date:
                    try:
                        date_str = datetime.strptime(file_date, "%Y-%m-%d").strftime("%Y-%m-%dT00:00:00Z")
                    except Exception:
                        date_str = _now_iso()
                else:
                    date_str = _now_iso()

                results.append({
                    "text": text[:MAX_TEXT_CHARS],
                    "date": date_str,
                    "source": f"edgar_{form_type.lower()}_{name[:20]}",
                })

            logger.info("EDGAR EFTS [%s] : %d résultats.", form_type, len(hits))

        except Exception as exc:
            logger.error("Erreur EDGAR EFTS [%s] : %s", form_type, exc)

        time.sleep(REQUEST_DELAY_SEC)

    return results


# ---------------------------------------------------------------------------
# Collecte par CIK (profondeur filing)
# ---------------------------------------------------------------------------

def fetch_company_filings_text(
    company_name: str,
    cik: str,
    form_type: str = "10-K",
    max_filings: int = 3,
) -> List[Dict]:
    """
    Pour une entreprise donnée, télécharge et extrait le texte des N derniers filings.
    """
    results = []
    filings = _get_company_filings(cik, form_type=form_type, max_filings=max_filings)

    for filing in filings:
        raw_text = _fetch_filing_text(cik, filing["accession_number"], filing["primary_document"])
        if not raw_text:
            logger.warning("Texte vide pour %s %s %s", company_name, form_type, filing["filing_date"])
            continue

        text = _extract_relevant_text(raw_text, company_name)
        date_str = f"{filing['filing_date']}T00:00:00Z"

        results.append({
            "text": text,
            "date": date_str,
            "source": f"edgar_{form_type.lower()}_{company_name[:20]}",
        })

        logger.info("Extrait %s %s du %s (%d chars)", company_name, form_type, filing["filing_date"], len(text))

    return results


# ---------------------------------------------------------------------------
# Mode mock
# ---------------------------------------------------------------------------

def fetch_mock_edgar(n: int = 5) -> List[Dict]:
    """Retourne des données EDGAR simulées avec dates récentes."""
    logger.warning("Mode MOCK EDGAR activé — %d documents simulés utilisés.", n)
    return build_mock_edgar(n)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def fetch_all_edgar(
    use_full_text_search: bool = True,
    use_company_filings: bool = False,
    start_date: str = None,
    end_date: str = None,
    max_results: int = 20,
    allow_mock: bool = True,
) -> List[Dict]:
    """
    Collecte les données EDGAR.
    - use_full_text_search : recherche EFTS (plus rapide, moins profond)
    - use_company_filings  : téléchargement complet par CIK (lent mais riche)
    - allow_mock           : si erreur réseau, retourne des données mock
    """
    if start_date is None or end_date is None:
        start_date, end_date = default_edgar_window()

    all_docs: List[Dict] = []

    try:
        if use_full_text_search:
            docs = fetch_edgar_full_text_search(
                keyword="crude oil",
                start_date=start_date,
                end_date=end_date,
                max_results=max_results,
            )
            all_docs.extend(docs)

        if use_company_filings:
            for company, cik in OIL_COMPANY_CIKS.items():
                docs = fetch_company_filings_text(company, cik, form_type="10-K", max_filings=1)
                all_docs.extend(docs)

        if not all_docs and allow_mock:
            logger.warning("Aucun résultat EDGAR — basculement mode mock.")
            all_docs = fetch_mock_edgar()

    except Exception as exc:
        logger.error("Erreur EDGAR : %s", exc)
        if allow_mock:
            logger.warning("Basculement mode mock EDGAR.")
            all_docs = fetch_mock_edgar()
        else:
            raise

    logger.info("Total documents EDGAR : %d", len(all_docs))
    return all_docs


# ---------------------------------------------------------------------------
# CLI rapide pour test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import csv, os
    from datetime import datetime, timezone
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    docs = fetch_all_edgar(
        use_full_text_search=True,
        use_company_filings=False,
        max_results=10,
    )
    for d in docs[:3]:
        print(f"[{d['date']}] [{d['source']}]")
        print(f"  {d['text'][:200]}...")
        print()
    print(f"Total : {len(docs)} documents EDGAR collectés.")

    from oil_sentiment_pipeline.paths import RAW_DIR

    os.makedirs(RAW_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = str(RAW_DIR / f"edgar_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "date", "source"])
        writer.writeheader()
        writer.writerows(docs)
    print(f"Sauvegardé : {path}")
