# db.py
"""Persistência em SQLite das entidades principais do ClinicGPT (usuários, pacientes, atendimentos)."""
import os
import sqlite3
from contextlib import contextmanager
from typing import Dict, List, Optional

DB_PATH = os.getenv("CLINICGPT_DB", "clinicgpt.db")


@contextmanager
def _conectar():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def inicializar_db():
    with _conectar() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usuarios (
                email TEXT PRIMARY KEY,
                nome TEXT NOT NULL,
                hashed_password TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pacientes (
                id TEXT PRIMARY KEY,
                nome TEXT NOT NULL,
                data_nascimento TEXT,
                cpf TEXT,
                sexo TEXT,
                telefone TEXT,
                email TEXT,
                endereco TEXT,
                observacoes TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS atendimentos (
                id TEXT PRIMARY KEY,
                paciente_id TEXT NOT NULL,
                queixas TEXT,
                doencas_med TEXT,
                historico TEXT,
                habitos TEXT,
                sono TEXT,
                atividade_fisica TEXT,
                antecedentes TEXT,
                exames_texto TEXT,
                resumo_ia TEXT,
                diagnosticos_ia TEXT
            )
            """
        )


# --- usuários ---

def obter_usuario(email: str) -> Optional[Dict]:
    with _conectar() as conn:
        linha = conn.execute("SELECT * FROM usuarios WHERE email = ?", (email,)).fetchone()
    return dict(linha) if linha else None


def criar_usuario(email: str, nome: str, hashed_password: str) -> Dict:
    with _conectar() as conn:
        conn.execute(
            "INSERT INTO usuarios (email, nome, hashed_password) VALUES (?, ?, ?)",
            (email, nome, hashed_password),
        )
    return {"email": email, "nome": nome, "hashed_password": hashed_password}


# --- pacientes ---

def criar_paciente(paciente: Dict) -> Dict:
    with _conectar() as conn:
        conn.execute(
            """INSERT INTO pacientes
               (id, nome, data_nascimento, cpf, sexo, telefone, email, endereco, observacoes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                paciente["id"],
                paciente["nome"],
                paciente["data_nascimento"],
                paciente["cpf"],
                paciente["sexo"],
                paciente["telefone"],
                paciente["email"],
                paciente["endereco"],
                paciente.get("observacoes"),
            ),
        )
    return paciente


def obter_paciente(paciente_id: str) -> Optional[Dict]:
    with _conectar() as conn:
        linha = conn.execute("SELECT * FROM pacientes WHERE id = ?", (paciente_id,)).fetchone()
    return dict(linha) if linha else None


# --- atendimentos ---

def criar_atendimento(atendimento_id: str, atendimento: Dict) -> Dict:
    with _conectar() as conn:
        conn.execute(
            """INSERT INTO atendimentos
               (id, paciente_id, queixas, doencas_med, historico, habitos, sono,
                atividade_fisica, antecedentes, exames_texto, resumo_ia, diagnosticos_ia)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                atendimento_id,
                atendimento["paciente_id"],
                atendimento["queixas"],
                atendimento["doencas_med"],
                atendimento["historico"],
                atendimento["habitos"],
                atendimento["sono"],
                atendimento["atividade_fisica"],
                atendimento["antecedentes"],
                atendimento.get("exames_texto", ""),
                atendimento.get("resumo_ia", ""),
                atendimento.get("diagnosticos_ia", ""),
            ),
        )
    return {"id": atendimento_id, **atendimento}
