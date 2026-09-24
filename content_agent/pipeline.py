"""Pipeline principal: busca em fontes -> síntese com IA -> persistência."""
from typing import Dict, List, Optional

from . import store
from .aggregator import buscar_em_fontes
from .synthesizer import sintetizar
from .topics import AREAS, termo_padrao


def executar_pesquisa(
    area: str,
    subtema: Optional[str] = None,
    publico: str = "ambos",
    fontes: Optional[List[str]] = None,
) -> Dict:
    if area not in AREAS:
        raise ValueError(f"Área desconhecida: {area}. Áreas válidas: {list(AREAS.keys())}")

    termo = subtema or termo_padrao(area)
    resultados, erros = buscar_em_fontes(termo, fontes)
    fontes_usadas = list(resultados.keys())
    label = AREAS[area]["label"]

    sintese = sintetizar(label, resultados, publico)

    return store.salvar_conteudo(
        area=area,
        termo_busca=termo,
        publico=publico,
        fontes_usadas=fontes_usadas,
        erros=erros,
        achados=resultados,
        resumo_clinico=sintese["resumo_clinico"],
        conteudo_paciente=sintese["conteudo_paciente"],
    )
