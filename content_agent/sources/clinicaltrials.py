"""Busca de estudos clínicos via ClinicalTrials.gov API v2 (pública, sem chave)."""
from typing import Dict, List

import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def buscar_clinical_trials(termo: str, max_resultados: int = 8) -> List[Dict]:
    params = {
        "query.term": termo,
        "pageSize": max_resultados,
        "sort": "LastUpdatePostDate:desc",
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    dados = resp.json()

    estudos = []
    for estudo in dados.get("studies", []):
        protocolo = estudo.get("protocolSection", {})
        identificacao = protocolo.get("identificationModule", {})
        status_modulo = protocolo.get("statusModule", {})
        descricao = protocolo.get("descriptionModule", {})
        nct_id = identificacao.get("nctId", "")
        estudos.append(
            {
                "fonte": "clinicaltrials",
                "nct_id": nct_id,
                "titulo": identificacao.get("briefTitle", ""),
                "resumo": descricao.get("briefSummary", ""),
                "status": status_modulo.get("overallStatus", ""),
                "atualizado_em": status_modulo.get("lastUpdatePostDateStruct", {}).get("date", ""),
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
            }
        )
    return estudos
