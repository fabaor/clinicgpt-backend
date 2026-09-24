"""Rotina agendada: atualiza a biblioteca de conteúdo para todas as áreas de foco."""
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from .pipeline import executar_pesquisa
from .topics import AREAS

logger = logging.getLogger("content_agent.scheduler")
_scheduler = BackgroundScheduler(timezone="America/Sao_Paulo")


def atualizar_todas_as_areas():
    for area in AREAS:
        try:
            executar_pesquisa(area=area, publico="ambos")
            logger.info("Conteúdo atualizado para a área: %s", area)
        except Exception as e:
            logger.error("Falha ao atualizar a área %s: %s", area, e)


def iniciar_scheduler():
    if not _scheduler.running:
        _scheduler.add_job(
            atualizar_todas_as_areas, "cron", hour=6, minute=0, id="atualizar_conteudo_diario"
        )
        _scheduler.start()


def parar_scheduler():
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
