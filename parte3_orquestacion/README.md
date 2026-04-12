# Parte 3: Orquestación del Pipeline
## Pipeline Completo de Datos con IA | Grupo 6 | LEAD University

**Responsable:** Esteban Gutiérrez Saborío
**Curso:** Administración de Datos
**Profesor:** Alejandro Zamora

---

## Descripción

Esta parte implementa la **orquestación del pipeline completo**. Un script maestro ejecuta en orden las etapas de Parte 1 y Parte 2, valida que cada etapa sea exitosa antes de continuar, y genera evidencia de ejecución mediante logs con timestamps.

El pipeline es **idempotente**: puede ejecutarse múltiples veces sin corromper los datos.

---

## Flujo de Ejecución

```
[ORQUESTADOR]
      |
      ├── Etapa 1: Fuente de Datos (Parte 1)
      │     ├── Conexión a PostgreSQL
      │     ├── Verificación de archivos CSV
      │     ├── Creación del schema books (tablas + índices)
      │     └── Carga: 3,212,404 registros → amazon_books_db
      │
      ├── [validación: si Etapa 1 falla → pipeline se detiene]
      │
      └── Etapa 2: ETL Completo (Parte 2)
            ├── Extracción desde PostgreSQL
            ├── Transformación (join, limpieza, feature engineering)
            └── Carga: CSV curado + tabla books.curated_books_reviews
```

---

## Estructura de Archivos

```
parte3_orquestacion/
├── scripts/
│   └── orchestrator.py    ← Script maestro del pipeline
├── logs/                  ← Logs de cada ejecución (auto-generados)
└── README.md
```

---

## Requisitos Previos

1. PostgreSQL corriendo (Postgres.app en macOS)
2. Base de datos `amazon_books_db` creada (`createdb amazon_books_db`)
3. Archivo `.env` configurado en `parte1_fuente_datos/config/.env`
4. CSVs del dataset en `parte1_fuente_datos/data/raw/`
5. Dependencias instaladas:

```bash
pip install -r ../parte1_fuente_datos/requirements.txt
pip install -r ../parte2_etl/requirements.txt
```

---

## Ejecución

```bash
cd parte3_orquestacion
python scripts/orchestrator.py
```

### Salida esperada

```
============================================================
  PIPELINE COMPLETO DE DATOS — AMAZON BOOKS REVIEWS
============================================================
  Inicio de ejecución : 2026-04-07 13:15:00

============================================================
  ETAPA 1/2 — FUENTE DE DATOS (Parte 1)
============================================================
  ...
  Etapa 1 completada en 1m 35s

============================================================
  ETAPA 2/2 — ETL COMPLETO (Parte 2)
============================================================
  ...
  Etapa 2 completada en 4s

------------------------------------------------------------
  REPORTE DE EJECUCION DEL PIPELINE
------------------------------------------------------------
  Etapa                               Estado
  -----------------------------------------------
  Etapa 1 — Fuente de Datos              OK
  Etapa 2 — ETL Completo                 OK

  PIPELINE COMPLETADO EXITOSAMENTE
  Datos listos para la Parte 4 (Modelo de IA)
============================================================
```

---

## Evidencia de Orquestación

Cada ejecución genera un log con timestamp en `logs/pipeline_YYYYMMDD_HHMMSS.log` con:
- Hora exacta de inicio y fin de cada etapa
- Estado de cada etapa (OK / FALLO)
- Tiempo total de ejecución
- Validación entre etapas (si Etapa 1 falla, la Etapa 2 no se ejecuta)

---

## Outputs del Pipeline

| Salida                              | Ubicación                                        |
|-------------------------------------|--------------------------------------------------|
| Tabla raw de libros                 | `amazon_books_db → books.books_data`             |
| Tabla raw de reseñas                | `amazon_books_db → books.books_rating`           |
| Tabla curada para IA                | `amazon_books_db → books.curated_books_reviews`  |
| CSV curado                          | `parte2_etl/data/curated/amazon_books_curated.csv` |
| Log de ejecución                    | `parte3_orquestacion/logs/pipeline_*.log`        |
