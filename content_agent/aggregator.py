"""Orquestra a busca de um termo em múltiplas fontes."""
from typing import Dict, List, Optional, Tuple

from .sources import clinicaltrials, pubmed, semantic_scholar, websearch

FONTES_DISPONIVEIS = {
    "pubmed": pubmed.buscar_pubmed,
    "clinicaltrials": clinicaltrials.buscar_clinical_trials,
    "web": websearch.buscar_web,
    "evidencias": semantic_scholar.buscar_evidencias,
}


def buscar_em_fontes(
    termo: str, fontes: Optional[List[str]] = None
) -> Tuple[Dict[str, List[Dict]], Dict[str, str]]:
    """Busca o termo em cada fonte solicitada.

    Retorna (resultados_por_fonte, erros_por_fonte). Uma fonte que falha não
    interrompe as demais — o erro fica registrado em `erros`.
    """
    nomes = fontes or list(FONTES_DISPONIVEIS.keys())
    resultados: Dict[str, List[Dict]] = {}
    erros: Dict[str, str] = {}

    for nome in nomes:
        buscar = FONTES_DISPONIVEIS.get(nome)
        if not buscar:
            erros[nome] = "Fonte desconhecida"
            continue
        try:
            resultados[nome] = buscar(termo)
        except Exception as e:
            resultados[nome] = []
            erros[nome] = str(e)

    return resultados, erros
