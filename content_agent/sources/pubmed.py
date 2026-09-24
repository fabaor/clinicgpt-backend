"""Busca de artigos científicos no PubMed via NCBI E-utilities (gratuito)."""
import os
import time
from typing import Dict, List
from xml.etree import ElementTree

import requests

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def buscar_pubmed(termo: str, max_resultados: int = 8) -> List[Dict]:
    """Retorna artigos recentes do PubMed para o termo informado."""
    params_busca = {
        "db": "pubmed",
        "term": termo,
        "retmax": max_resultados,
        "sort": "date",
        "retmode": "json",
    }
    if NCBI_API_KEY:
        params_busca["api_key"] = NCBI_API_KEY

    resp = requests.get(ESEARCH_URL, params=params_busca, timeout=15)
    resp.raise_for_status()
    pmids = resp.json().get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    time.sleep(0.1 if NCBI_API_KEY else 0.34)  # respeita o limite de requisições do NCBI

    params_fetch = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if NCBI_API_KEY:
        params_fetch["api_key"] = NCBI_API_KEY

    resp = requests.get(EFETCH_URL, params=params_fetch, timeout=20)
    resp.raise_for_status()

    root = ElementTree.fromstring(resp.content)
    artigos = []
    for artigo in root.findall(".//PubmedArticle"):
        pmid = artigo.findtext(".//PMID", default="")
        titulo = (artigo.findtext(".//ArticleTitle", default="") or "").strip()
        resumo = " ".join(el.text or "" for el in artigo.findall(".//AbstractText")).strip()
        ano = artigo.findtext(".//PubDate/Year") or artigo.findtext(".//PubDate/MedlineDate") or ""
        autores = []
        for autor in artigo.findall(".//Author"):
            sobrenome = autor.findtext("LastName")
            iniciais = autor.findtext("Initials")
            if sobrenome:
                autores.append(f"{sobrenome} {iniciais or ''}".strip())
        artigos.append(
            {
                "fonte": "pubmed",
                "pmid": pmid,
                "titulo": titulo,
                "resumo": resumo,
                "ano": ano,
                "autores": autores[:5],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return artigos
