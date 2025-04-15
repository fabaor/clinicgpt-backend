# main.py
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
import uuid
import openai
import os
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

# Configurações OpenAI e segurança
openai.api_key = os.getenv("OPENAI_API_KEY")
SECRET_KEY = "clinicgptsecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
usuarios = {}

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None or email not in usuarios:
            raise HTTPException(status_code=401, detail="Token inválido")
        return usuarios[email]
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido")

# Modelos
class Usuario(BaseModel):
    email: str
    nome: str
    senha: str

class Token(BaseModel):
    access_token: str
    token_type: str

class Paciente(BaseModel):
    id: Optional[str]
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

pacientes = {}
atendimentos = {}

@app.post("/auth/signup")
def signup(usuario: Usuario):
    if usuario.email in usuarios:
        raise HTTPException(status_code=400, detail="Usuário já existe")
    hashed = get_password_hash(usuario.senha)
    usuarios[usuario.email] = {"email": usuario.email, "nome": usuario.nome, "hashed_password": hashed}
    return {"message": "Usuário criado com sucesso"}

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = usuarios.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")
    token = create_access_token(data={"sub": user["email"]})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/patients")
def criar_paciente(paciente: Paciente, current_user: dict = Depends(get_current_user)):
    paciente.id = str(uuid.uuid4())
    pacientes[paciente.id] = paciente
    return paciente

@app.get("/patients/{paciente_id}")
def get_paciente(paciente_id: str, current_user: dict = Depends(get_current_user)):
    return pacientes.get(paciente_id)

@app.post("/appointments")
def registrar_atendimento(atendimento: Atendimento, current_user: dict = Depends(get_current_user)):
    atendimento_id = str(uuid.uuid4())
    try:
        atendimento.resumo_ia = gerar_resumo_clinico(atendimento)
    except Exception as e:
        atendimento.resumo_ia = f"Erro: {e}"
    atendimentos[atendimento_id] = atendimento
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
