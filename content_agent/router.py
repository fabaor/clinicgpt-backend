"""Endpoints HTTP do agente de conteúdo: pesquisa sob demanda e biblioteca."""
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import get_current_user

from . import store
from .pipeline import executar_pesquisa
from .topics import AREAS

router = APIRouter(prefix="/content", tags=["conteudo"])


class PesquisaRequest(BaseModel):
    area: str
    subtema: Optional[str] = None
    publico: Literal["clinico", "paciente", "ambos"] = "ambos"
    fontes: Optional[List[str]] = None


@router.get("/areas")
def listar_areas(current_user: dict = Depends(get_current_user)):
    """Lista as áreas de foco disponíveis para pesquisa."""
    return {chave: dados["label"] for chave, dados in AREAS.items()}


@router.post("/research")
def pesquisar(req: PesquisaRequest, current_user: dict = Depends(get_current_user)):
    """Busca conteúdo sob demanda (PubMed, ClinicalTrials.gov, web, evidências) e sintetiza com IA."""
    try:
        return executar_pesquisa(
            area=req.area, subtema=req.subtema, publico=req.publico, fontes=req.fontes
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/library")
def listar_biblioteca(
    area: Optional[str] = None,
    publico: Optional[str] = None,
    limite: int = 20,
    current_user: dict = Depends(get_current_user),
):
    """Lista os conteúdos já gerados (pela rotina agendada ou por pesquisas manuais)."""
    return store.listar_conteudos(area=area, publico=publico, limite=limite)


@router.get("/library/{item_id}")
def obter_item(item_id: str, current_user: dict = Depends(get_current_user)):
    """Retorna um item da biblioteca, incluindo os achados brutos usados na síntese."""
    item = store.obter_conteudo(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Conteúdo não encontrado")
    return item
