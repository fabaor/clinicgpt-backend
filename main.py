# main.py
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
import uuid
import openai
import os
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

import db
from auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user,
)
from content_agent.router import router as content_router
from content_agent.store import inicializar_db as inicializar_db_conteudo
from content_agent.scheduler import iniciar_scheduler, parar_scheduler

# Configurações OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Modelos
class Usuario(BaseModel):
    email: str
    nome: str
    senha: str

class Token(BaseModel):
    access_token: str
    token_type: str

class Paciente(BaseModel):
    id: Optional[str] = None
    nome: str
    data_nascimento: str
    cpf: str
    sexo: str
    telefone: str
    email: str
    endereco: str
    observacoes: Optional[str] = None

class Atendimento(BaseModel):
    paciente_id: str
    queixas: str
    doencas_med: str
    historico: str
    habitos: str
    sono: str
    atividade_fisica: str
    antecedentes: str
    exames_texto: Optional[str] = ""
    resumo_ia: Optional[str] = ""
    diagnosticos_ia: Optional[str] = ""

# App
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(content_router)


@app.on_event("startup")
def iniciar_agente_de_conteudo():
    db.inicializar_db()
    inicializar_db_conteudo()
    if os.getenv("CONTENT_AGENT_SCHEDULER", "true").lower() == "true":
        iniciar_scheduler()


@app.on_event("shutdown")
def parar_agente_de_conteudo():
    parar_scheduler()


@app.post("/auth/signup")
def signup(usuario: Usuario):
    if db.obter_usuario(usuario.email):
        raise HTTPException(status_code=400, detail="Usuário já existe")
    hashed = get_password_hash(usuario.senha)
    db.criar_usuario(usuario.email, usuario.nome, hashed)
    return {"message": "Usuário criado com sucesso"}

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = db.obter_usuario(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")
    token = create_access_token(data={"sub": user["email"]})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/patients")
def criar_paciente(paciente: Paciente, current_user: dict = Depends(get_current_user)):
    paciente.id = str(uuid.uuid4())
    db.criar_paciente(paciente.dict())
    return paciente

@app.get("/patients/{paciente_id}")
def get_paciente(paciente_id: str, current_user: dict = Depends(get_current_user)):
    return db.obter_paciente(paciente_id)

@app.post("/appointments")
def registrar_atendimento(atendimento: Atendimento, current_user: dict = Depends(get_current_user)):
    atendimento_id = str(uuid.uuid4())
    try:
        atendimento.resumo_ia = gerar_resumo_clinico(atendimento)
    except Exception as e:
        atendimento.resumo_ia = f"Erro: {e}"
    db.criar_atendimento(atendimento_id, atendimento.dict())
    return {"id": atendimento_id, **atendimento.dict()}

@app.post("/exams/upload")
def upload_exame(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    filename = f"uploaded_{file.filename}"
    with open(filename, "wb") as buffer:
        buffer.write(file.file.read())
    texto = extrair_texto_pdf(filename)
    interpretacao = interpretar_exame(texto)
    return {"message": "Interpretado", "interpretacao_ia": interpretacao, "conteudo_extraido": texto}

# IA functions
def gerar_resumo_clinico(at: Atendimento) -> str:
    prompt = f"""Você é um assistente médico. Gere um resumo clínico com base nos seguintes dados:
    Queixas: {at.queixas}
    Doenças: {at.doencas_med}
    Histórico: {at.historico}
    Hábitos: {at.habitos}
    Sono: {at.sono}
    Atividade física: {at.atividade_fisica}
    Antecedentes: {at.antecedentes}
    Exames: {at.exames_texto}"""
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )
    return res.choices[0].message['content'].strip()

def extrair_texto_pdf(path: str) -> str:
    texto = ""
    doc = fitz.open(path)
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        texto += pytesseract.image_to_string(img, lang='por') + "\n"
    return texto

def interpretar_exame(texto: str) -> str:
    prompt = f"Analise o exame abaixo e destaque achados relevantes:\n\n{texto}"
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )
    return res.choices[0].message['content'].strip()
