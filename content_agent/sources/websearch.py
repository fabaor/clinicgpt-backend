"""Busca na web geral via Tavily (requer TAVILY_API_KEY).

Fonte de menor confiabilidade que PubMed/ClinicalTrials — usar apenas como
apoio de contexto/atualidade, nunca como base isolada para conteúdo clínico.
Se a chave não estiver configurada, a busca é simplesmente ignorada.
"""
import os
from typing import Dict, List

import requests

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"


def buscar_web(termo: str, max_resultados: int = 5) -> List[Dict]:
    if not TAVILY_API_KEY:
        return []

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": termo,
        "search_depth": "advanced",
        "max_results": max_resultados,
        "include_answer": False,
    }
    resp = requests.post(TAVILY_URL, json=payload, timeout=20)
    resp.raise_for_status()

    resultados = []
    for item in resp.json().get("results", []):
        resultados.append(
            {
                "fonte": "web",
                "titulo": item.get("title", ""),
                "resumo": item.get("content", ""),
                "url": item.get("url", ""),
            }
        )
    return resultados
