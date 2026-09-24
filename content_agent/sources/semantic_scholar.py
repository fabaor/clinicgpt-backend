"""Busca de evidências científicas via Semantic Scholar.

A Consensus (app de busca de evidências) não oferece hoje uma API pública para
integração de terceiros. O Semantic Scholar cobre a mesma necessidade — busca
semântica de papers com resumo e contagem de citações — com API gratuita
(uma SEMANTIC_SCHOLAR_API_KEY é opcional e apenas aumenta o limite de taxa).
"""
import os
from typing import Dict, List

import requests

API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def buscar_evidencias(termo: str, max_resultados: int = 8) -> List[Dict]:
    params = {
        "query": termo,
        "limit": max_resultados,
        "fields": "title,abstract,year,authors,url,citationCount,venue",
    }
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    resp = requests.get(SEARCH_URL, params=params, headers=headers, timeout=15)
    resp.raise_for_status()

    artigos = []
    for item in resp.json().get("data", []):
        autores = [a.get("name") for a in item.get("authors", [])][:5]
        artigos.append(
            {
                "fonte": "evidencias",
                "titulo": item.get("title", ""),
                "resumo": item.get("abstract") or "",
                "ano": item.get("year"),
                "autores": autores,
                "citacoes": item.get("citationCount"),
                "revista": item.get("venue"),
                "url": item.get("url", ""),
            }
        )
    return artigos
