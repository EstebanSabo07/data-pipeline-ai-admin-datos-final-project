# Parte 4: Aplicativo con IA — Predictor de Sentimiento
## Pipeline Completo de Datos con IA | Grupo 6 | LEAD University

**Autores:** Ariana Víquez Solano | Esteban Gutiérrez Saborío | Marco Chinchilla Barrantes
**Profesor:** Alejandro Zamora
**Curso:** Administración de Datos — Bachillerato en Ingeniería en Ciencia de Datos
**Entrega:** 14 de abril de 2026

---

## Descripción

Aplicación web interactiva que utiliza un modelo de **Inteligencia Artificial** para predecir el sentimiento de reseñas de libros. El usuario escribe el texto de una reseña y el modelo predice instantáneamente si es **Positiva (4-5 ⭐)** o **Negativa/Neutral (1-3 ⭐)**.

---

## Modelo de IA

| Componente       | Detalle                                                  |
|------------------|----------------------------------------------------------|
| Tipo             | Clasificación binaria (Positiva vs Negativa/Neutral)     |
| Algoritmo        | Regresión Logística                                      |
| Features         | TF-IDF sobre review_text + review_summary (8,000 términos, bigramas) |
| Variable target  | review_score ≥ 4 → Positiva / review_score < 4 → Negativa |
| Split            | 80% entrenamiento / 20% prueba                          |
| Dataset fuente   | amazon_books_curated.csv (Parte 2 ETL)                  |

---

## Estructura de Archivos

```
parte4_app_ia/
├── scripts/
│   ├── train_model.py   ← Entrenamiento y guardado del modelo
│   └── app.py           ← Aplicación Streamlit
├── model/               ← Archivos del modelo entrenado (auto-generados)
│   ├── modelo_sentimiento.pkl
│   ├── vectorizer.pkl
│   └── metricas.pkl
├── requirements.txt
└── README.md
```

---

## Instrucciones de Ejecución

### Paso 1: Instalar dependencias

```bash
cd parte4_app_ia
pip install -r requirements.txt
```

### Paso 2: Entrenar el modelo

```bash
python scripts/train_model.py
```

Esto lee el dataset curado de `parte2_etl/data/curated/amazon_books_curated.csv`, entrena el modelo y guarda los archivos en `model/`.

### Paso 3: Lanzar la aplicación

```bash
streamlit run scripts/app.py
```

La app abre automáticamente en el navegador en `http://localhost:8501`

---

## Funcionalidades de la App

- **🔮 Predictor:** El usuario escribe una reseña y obtiene la predicción con porcentaje de confianza
- **📊 Métricas del Modelo:** Accuracy, Precision, Recall, F1-Score, Matriz de Confusión y palabras más influyentes
- **ℹ️ Sobre el Proyecto:** Descripción completa del pipeline e información del equipo
