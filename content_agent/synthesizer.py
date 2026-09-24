"""Sintetiza os achados brutos em conteúdo clínico e/ou para pacientes, via IA."""
import os
from typing import Dict, List, Optional

import openai

MODELO = os.getenv("CONTENT_AGENT_MODEL", "gpt-3.5-turbo")


def _formatar_achados(resultados: Dict[str, List[Dict]]) -> str:
    linhas = []
    for fonte, itens in resultados.items():
        for item in itens:
            titulo = item.get("titulo") or "(sem título)"
            resumo = (item.get("resumo") or "")[:600]
            url = item.get("url", "")
            linhas.append(f"- [{fonte}] {titulo}\n  Resumo: {resumo}\n  Fonte: {url}")
    return "\n".join(linhas) if linhas else "Nenhum achado recuperado nas fontes selecionadas."


def _chamar_ia(prompt: str, max_tokens: int = 700) -> str:
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY não configurada.")
    res = openai.ChatCompletion.create(
        model=MODELO,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return res.choices[0].message["content"].strip()


def gerar_resumo_clinico(area_label: str, achados_formatados: str) -> str:
    prompt = f"""Você é um assistente de pesquisa médica especializado em {area_label}.
Com base nos achados abaixo (artigos científicos, ensaios clínicos e evidências), produza um
resumo técnico para um médico, destacando:
1. Principais achados recentes.
2. Nível de evidência, quando identificável.
3. Implicações práticas para a conduta clínica.
4. Lacunas ou controvérsias.
Cite a fonte de cada achado mencionado (ex: [pubmed], [clinicaltrials]). Não invente dados que
não estejam nos achados abaixo.

Achados:
{achados_formatados}
"""
    return _chamar_ia(prompt, max_tokens=800)


def gerar_conteudo_paciente(area_label: str, achados_formatados: str) -> str:
    prompt = f"""Você é um redator de conteúdo de saúde para pacientes, especializado em {area_label}.
Com base nos achados científicos abaixo, escreva um texto curto (formato blog/post de rede social),
em linguagem clara e acolhedora, sem jargão técnico, sem prometer curas ou resultados garantidos.
Termine com um aviso de que o conteúdo é informativo e não substitui consulta médica.

Achados:
{achados_formatados}
"""
    return _chamar_ia(prompt, max_tokens=600)


def sintetizar(area_label: str, resultados: Dict[str, List[Dict]], publico: str) -> Dict[str, Optional[str]]:
    achados_formatados = _formatar_achados(resultados)
    saida: Dict[str, Optional[str]] = {"resumo_clinico": None, "conteudo_paciente": None}
    if publico in ("clinico", "ambos"):
        saida["resumo_clinico"] = gerar_resumo_clinico(area_label, achados_formatados)
    if publico in ("paciente", "ambos"):
        saida["conteudo_paciente"] = gerar_conteudo_paciente(area_label, achados_formatados)
    return saida
