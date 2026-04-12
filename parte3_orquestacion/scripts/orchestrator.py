"""
orchestrator.py
===============
Orquestador maestro del pipeline completo de datos.

Ejecuta en orden las 2 etapas del pipeline:
    Etapa 1 — Fuente de Datos (Parte 1): carga el dataset Amazon Books en PostgreSQL
    Etapa 2 — ETL completo   (Parte 2): extrae, transforma y carga la capa curada

Características:
    - Ejecución por etapas con validación entre pasos
    - Logging detallado con timestamps de inicio y fin de cada etapa
    - Idempotente: puede ejecutarse múltiples veces sin corromper datos
    - Reporte de ejecución en logs/pipeline_report.log

Uso:
    cd parte3_orquestacion
    python scripts/orchestrator.py

Proyecto Final — Administración de Datos — LEAD University
Grupo 6 | Parte 3: Orquestación
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# RUTAS
# ============================================================
BASE_DIR      = Path(__file__).parent.parent          # parte3_orquestacion/
PROJECT_DIR   = BASE_DIR.parent                       # data-pipeline-ai-project-main/
PARTE1_DIR    = PROJECT_DIR / "parte1_fuente_datos"
PARTE2_DIR    = PROJECT_DIR / "parte2_etl"
LOG_DIR       = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Agregar parte1 y parte2 al path para importar sus módulos
sys.path.insert(0, str(PARTE1_DIR))
sys.path.insert(0, str(PARTE2_DIR / "scripts"))

# ============================================================
# LOGGING — archivo + consola
# ============================================================
timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"pipeline_{timestamp_run}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline.orchestrator")


# ============================================================
# UTILIDADES
# ============================================================

def banner(title: str, char: str = "=", width: int = 60) -> None:
    logger.info(char * width)
    logger.info(f"  {title}")
    logger.info(char * width)


def elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ============================================================
# ETAPAS DEL PIPELINE
# ============================================================

def etapa_1_fuente_datos() -> bool:
    """
    Etapa 1: Carga el dataset Amazon Books en PostgreSQL.
    Llama directamente al script maestro de la Parte 1.
    """
    banner("ETAPA 1/2 — FUENTE DE DATOS (Parte 1)")
    logger.info("  Cargando dataset Amazon Books → PostgreSQL (amazon_books_db)")
    logger.info("  Tablas: books.books_data | books.books_rating")

    t_start = time.time()
    try:
        # Importar y ejecutar el main de la Parte 1
        sys.path.insert(0, str(PARTE1_DIR))
        import importlib, runpy
        result = runpy.run_path(
            str(PARTE1_DIR / "scripts" / "run_parte1.py"),
            run_name="__orchestrated__"
        )
        # run_parte1 llama a main() solo cuando __name__ == "__main__"
        # Lo invocamos directamente:
        result["main"]()

        elapsed_str = elapsed(time.time() - t_start)
        logger.info(f"  Etapa 1 completada en {elapsed_str}")
        logger.info("  Resultado: 3,212,404 registros disponibles en PostgreSQL")
        return True

    except SystemExit as e:
        if e.code == 0:
            elapsed_str = elapsed(time.time() - t_start)
            logger.info(f"  Etapa 1 completada en {elapsed_str}")
            return True
        logger.error(f"  Etapa 1 fallida (sys.exit {e.code})")
        return False

    except Exception as e:
        logger.error(f"  Error en Etapa 1: {e}")
        return False


def etapa_2_etl() -> bool:
    """
    Etapa 2: Extrae, transforma y carga la capa curada.
    Llama directamente al orquestador del ETL (Parte 2).
    """
    banner("ETAPA 2/2 — ETL COMPLETO (Parte 2)")
    logger.info("  Extracción desde PostgreSQL → Transformación → Carga curada")
    logger.info("  Salidas: CSV curado + tabla books.curated_books_reviews")

    t_start = time.time()
    try:
        sys.path.insert(0, str(PARTE2_DIR / "scripts"))
        from run_etl import run_etl
        run_etl()

        elapsed_str = elapsed(time.time() - t_start)
        logger.info(f"  Etapa 2 completada en {elapsed_str}")
        logger.info("  Resultado: dataset curado listo para modelo de IA (Parte 4)")
        return True

    except SystemExit as e:
        if e.code == 0:
            elapsed_str = elapsed(time.time() - t_start)
            logger.info(f"  Etapa 2 completada en {elapsed_str}")
            return True
        logger.error(f"  Etapa 2 fallida (sys.exit {e.code})")
        return False

    except Exception as e:
        logger.error(f"  Error en Etapa 2: {e}")
        return False


# ============================================================
# ORQUESTADOR PRINCIPAL
# ============================================================

def main():
    t_total = time.time()
    inicio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    banner("PIPELINE COMPLETO DE DATOS — AMAZON BOOKS REVIEWS", "=", 60)
    logger.info(f"  Inicio de ejecución : {inicio}")
    logger.info(f"  Log de esta corrida : logs/pipeline_{timestamp_run}.log")
    logger.info(f"  Proyecto            : Administración de Datos — LEAD University")
    logger.info(f"  Grupo 6             : Ariana Víquez | Esteban Gutiérrez | Marco Chinchilla")
    logger.info("")

    resultados = {}

    # ---- Etapa 1 ----
    ok1 = etapa_1_fuente_datos()
    resultados["Etapa 1 — Fuente de Datos"] = "OK" if ok1 else "FALLO"

    if not ok1:
        logger.error("Pipeline detenido: la Etapa 1 falló.")
        _print_summary(resultados, t_total, inicio)
        sys.exit(1)

    # ---- Etapa 2 ----
    ok2 = etapa_2_etl()
    resultados["Etapa 2 — ETL Completo"] = "OK" if ok2 else "FALLO"

    if not ok2:
        logger.error("Pipeline detenido: la Etapa 2 falló.")
        _print_summary(resultados, t_total, inicio)
        sys.exit(1)

    _print_summary(resultados, t_total, inicio)


def _print_summary(resultados: dict, t_total: float, inicio: str):
    """Imprime el reporte final de ejecución."""
    fin = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = elapsed(time.time() - t_total)

    banner("REPORTE DE EJECUCION DEL PIPELINE", "-", 60)
    logger.info(f"  Inicio  : {inicio}")
    logger.info(f"  Fin     : {fin}")
    logger.info(f"  Total   : {total}")
    logger.info("")
    logger.info(f"  {'Etapa':<35} {'Estado':>10}")
    logger.info(f"  {'-'*47}")
    for etapa, estado in resultados.items():
        icono = "OK" if estado == "OK" else "FALLO"
        logger.info(f"  {etapa:<35} {icono:>10}")
    logger.info("")

    todos_ok = all(v == "OK" for v in resultados.values())
    if todos_ok:
        logger.info("  PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("  Datos listos para la Parte 4 (Modelo de IA)")
    else:
        logger.info("  PIPELINE COMPLETADO CON ERRORES — revisar logs")

    logger.info("=" * 60)
    logger.info(f"  Log guardado en: logs/pipeline_{timestamp_run}.log")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
