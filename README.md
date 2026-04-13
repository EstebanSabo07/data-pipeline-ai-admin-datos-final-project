# Pipeline Completo de Datos con App de IA
## Proyecto Final Grupal — Administración de Datos
### Bachillerato en Ingeniería en Ciencia de Datos — LEAD University

---

**Grupo 6 - Estudiantes**
- Ariana Víquez Solano
- Esteban Gutiérrez Saborío
- Marco Chinchilla Barrantes

**Profesor:** Alejandro Zamora  
**Entrega:** 14 de abril de 2026

---

## Descripción General

Este proyecto implementa un pipeline completo de datos end-to-end, desde la ingesta de datos crudos hasta una aplicación web con Inteligencia Artificial, usando el dataset público **Amazon Books Reviews** de Kaggle.

El pipeline consta de 4 etapas:

1. **Parte 1 — Fuente de Datos:** Ingesta de CSVs a PostgreSQL
2. **Parte 2 — ETL:** Extracción, Transformación y Carga de datos curados
3. **Parte 3 — Orquestación:** Ejecución automatizada del pipeline completo
4. **Parte 4 — App IA:** Entrenamiento de modelo y app web interactiva

---

## Dataset

**Amazon Books Reviews** (Kaggle)

| Tabla | Descripción | Registros |
|-------|-------------|-----------|
| `books_data` | Catálogo de libros (título, autor, categoría, descripción) | ~212,000 |
| `books_rating` | Reseñas de usuarios (puntaje, texto, resumen) | ~3,000,000 |

- **Base de datos:** PostgreSQL 18 — esquema `books`, base de datos `amazon_books_db`
- **Descarga manual desde:** https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews
- Archivos requeridos: `books_data.csv` y `Books_rating.csv` → colocar en `parte1_fuente_datos/data/raw/`

---

## Tecnologías

| Componente | Tecnología |
|------------|------------|
| Base de Datos | PostgreSQL 18 (Postgres.app) |
| Lenguaje | Python 3.13 (Miniconda) |
| ORM / Conexión | SQLAlchemy >=2.0.36, psycopg2-binary 2.9.10 |
| Manipulación de datos | pandas >=2.2.2, numpy >=1.26.4 |
| Machine Learning | scikit-learn >=1.4.0 |
| App Web | Streamlit >=1.32.0 |
| Visualizaciones | Plotly >=5.20.0 |
| Variables de entorno | python-dotenv |

---

## Estructura del Proyecto

```
data-pipeline-ai-project-main/
├── parte1_fuente_datos/        ← Ingesta de datos a PostgreSQL
│   ├── config/
│   │   └── .env.example        ← Plantilla de credenciales
│   ├── data/
│   │   └── raw/                ← CSVs crudos (no incluidos en repo)
│   ├── scripts/
│   │   ├── db_connection.py
│   │   ├── load_books_data.py
│   │   ├── load_books_rating.py
│   │   ├── run_parte1.py       ← Orquestador Parte 1
│   │   └── verify_load.py
│   ├── sql/
│   │   └── 05_create_books_tables.sql
│   └── requirements.txt
│
├── parte2_etl/                 ← Proceso ETL
│   ├── data/
│   │   └── curated/            ← CSV curado (no incluido en repo)
│   ├── scripts/
│   │   ├── extract.py
│   │   ├── transform.py
│   │   ├── load.py
│   │   └── run_etl.py          ← Orquestador ETL
│   └── requirements.txt
│
├── parte3_orquestacion/        ← Pipeline orquestado
│   ├── scripts/
│   │   └── orchestrator.py     ← Ejecuta Parte 1 + Parte 2 en secuencia
│   └── logs/                   ← Logs de ejecución (auto-generados)
│
├── parte4_app_ia/              ← Aplicativo con IA
│   ├── scripts/
│   │   ├── train_model.py      ← Entrena y guarda el modelo
│   │   └── app.py              ← Aplicación Streamlit
│   ├── model/                  ← Modelo entrenado (incluido en repo)
│   │   ├── modelo_sentimiento.pkl
│   │   ├── vectorizer.pkl
│   │   └── metricas.pkl
│   └── requirements.txt
│
└── README.md
```

---

## Instrucciones de Ejecución

### Prerequisitos

- PostgreSQL 18 corriendo localmente (Postgres.app en macOS)
- Python 3.13 con Miniconda
- Base de datos `amazon_books_db` creada en PostgreSQL

```bash
createdb amazon_books_db
```

### Paso 1: Configurar credenciales

```bash
cp parte1_fuente_datos/config/.env.example parte1_fuente_datos/config/.env
# Editar .env con tus credenciales de PostgreSQL
```

### Paso 2: Descargar el dataset

Descargar manualmente desde Kaggle y colocar en `parte1_fuente_datos/data/raw/`:
- `books_data.csv`
- `Books_rating.csv`

### Paso 3: Instalar dependencias

```bash
# Parte 1
pip install -r parte1_fuente_datos/requirements.txt

# Parte 2
pip install -r parte2_etl/requirements.txt

# Parte 4
pip install -r parte4_app_ia/requirements.txt
```

### Paso 4: Ejecutar el pipeline completo (Parte 3 — Orquestación)

```bash
python parte3_orquestacion/scripts/orchestrator.py
```

Esto ejecuta automáticamente la ingesta (Parte 1) y el ETL (Parte 2) en secuencia.

O bien, ejecutar por separado:

```bash
# Solo Parte 1
python parte1_fuente_datos/scripts/run_parte1.py

# Solo Parte 2
python parte2_etl/scripts/run_etl.py
```

### Paso 5: Entrenar el modelo de IA

```bash
python parte4_app_ia/scripts/train_model.py
```

> Nota: El modelo ya entrenado está incluido en el repo (`parte4_app_ia/model/`). Este paso solo es necesario si se desea reentrenar desde cero.

### Paso 6: Lanzar la aplicación web

```bash
streamlit run parte4_app_ia/scripts/app.py
```

La app abre en `http://localhost:8501`

---

## Modelo de IA

| Componente | Detalle |
|------------|---------|
| Tipo | Clasificación binaria |
| Algoritmo | Regresión Logística |
| Features | TF-IDF sobre `review_text` + `review_summary` (8,000 términos, bigramas) |
| Variable target | `review_score` ≥ 4 → Positiva / `review_score` < 4 → Negativa/Neutral |
| Split | 80% entrenamiento / 20% prueba |
| Accuracy | 89.49% |
| F1-Score | 93.59% |

---

## Parte 1: Fuente de Datos

Ingesta de los CSVs crudos a PostgreSQL. Crea el esquema `books` con dos tablas:

- `books_data`: Catálogo de libros
- `books_rating`: Reseñas de usuarios

Ver instrucciones detalladas en [`parte1_fuente_datos/README.md`](parte1_fuente_datos/README.md)

---

## Parte 2: ETL

Pipeline de Extracción, Transformación y Carga:

1. **Extracción:** Consulta las tablas desde PostgreSQL
2. **Transformación:** Limpieza de nulos, normalización de fechas, join de tablas, filtrado de columnas relevantes
3. **Carga:** Guarda el dataset curado en tabla `books.curated_books_reviews` y en `data/curated/amazon_books_curated.csv`

Ver instrucciones en [`parte2_etl/README.md`](parte2_etl/README.md)

---

## Parte 3: Orquestación

Script maestro que ejecuta el pipeline completo (Parte 1 → Parte 2) en secuencia, con logging detallado y manejo de errores.

Ver instrucciones en [`parte3_orquestacion/README.md`](parte3_orquestacion/README.md)

---

## Parte 4: Aplicativo con IA

Aplicación web interactiva (Streamlit) que permite:

- **Predictor:** Escribe una reseña y obtiene predicción de sentimiento con porcentaje de confianza
- **Métricas del Modelo:** Accuracy, Precision, Recall, F1-Score, Matriz de Confusión, palabras más influyentes
- **Sobre el Proyecto:** Descripción del pipeline e información del equipo

Ver instrucciones en [`parte4_app_ia/README.md`](parte4_app_ia/README.md)
