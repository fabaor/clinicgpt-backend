"""Persistência em SQLite para os conteúdos gerados pelo agente."""
import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional

DB_PATH = os.getenv("CONTENT_AGENT_DB", "content_agent.db")


def _conectar():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def inicializar_db():
    with _conectar() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conteudos (
                id TEXT PRIMARY KEY,
                area TEXT NOT NULL,
                termo_busca TEXT,
                publico TEXT,
                criado_em TEXT,
                fontes_usadas TEXT,
                erros_busca TEXT,
                achados_brutos TEXT,
                resumo_clinico TEXT,
                conteudo_paciente TEXT
            )
            """
        )


def salvar_conteudo(
    area: str,
    termo_busca: str,
    publico: str,
    fontes_usadas: List[str],
    erros: Dict[str, str],
    achados: Dict[str, List[Dict]],
    resumo_clinico: Optional[str],
    conteudo_paciente: Optional[str],
) -> Dict:
    item_id = str(uuid.uuid4())
    criado_em = datetime.utcnow().isoformat()
    with _conectar() as conn:
        conn.execute(
            """INSERT INTO conteudos
               (id, area, termo_busca, publico, criado_em, fontes_usadas, erros_busca,
                achados_brutos, resumo_clinico, conteudo_paciente)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item_id,
                area,
                termo_busca,
                publico,
                criado_em,
                json.dumps(fontes_usadas),
                json.dumps(erros),
                json.dumps(achados),
                resumo_clinico,
                conteudo_paciente,
            ),
        )
    return {
        "id": item_id,
        "area": area,
        "termo_busca": termo_busca,
        "publico": publico,
        "criado_em": criado_em,
        "fontes_usadas": fontes_usadas,
        "erros_busca": erros,
        "achados_brutos": achados,
        "resumo_clinico": resumo_clinico,
        "conteudo_paciente": conteudo_paciente,
    }


def listar_conteudos(
    area: Optional[str] = None, publico: Optional[str] = None, limite: int = 20
) -> List[Dict]:
    query = (
        "SELECT id, area, termo_busca, publico, criado_em, resumo_clinico, conteudo_paciente "
        "FROM conteudos"
    )
    condicoes, params = [], []
    if area:
        condicoes.append("area = ?")
        params.append(area)
    if publico:
        condicoes.append("publico = ?")
        params.append(publico)
    if condicoes:
        query += " WHERE " + " AND ".join(condicoes)
    query += " ORDER BY criado_em DESC LIMIT ?"
    params.append(limite)

    with _conectar() as conn:
        linhas = conn.execute(query, params).fetchall()
    return [dict(linha) for linha in linhas]


def obter_conteudo(item_id: str) -> Optional[Dict]:
    with _conectar() as conn:
        linha = conn.execute("SELECT * FROM conteudos WHERE id = ?", (item_id,)).fetchone()
    if not linha:
        return None
    dados = dict(linha)
    dados["fontes_usadas"] = json.loads(dados["fontes_usadas"] or "[]")
    dados["erros_busca"] = json.loads(dados["erros_busca"] or "{}")
    dados["achados_brutos"] = json.loads(dados["achados_brutos"] or "{}")
    return dados
