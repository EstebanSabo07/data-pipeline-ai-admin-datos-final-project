"""
app.py
======
Aplicación Streamlit — Predictor de Sentimiento de Reseñas de Libros

El usuario escribe el texto de una reseña y el modelo de IA predice
si es una reseña Positiva (4-5 ⭐) o Negativa/Neutral (1-3 ⭐).

Ejecución:
    cd parte4_app_ia
    streamlit run scripts/app.py

Proyecto Final — Administración de Datos — LEAD University
Grupo 6 | Parte 4: Aplicativo con IA
Autores: Ariana Víquez Solano | Esteban Gutiérrez Saborío | Marco Chinchilla Barrantes
Profesor: Alejandro Zamora
Entrega: 14 de abril de 2026
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Predictor de Reseñas — Grupo 6",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# RUTAS
# ============================================================
BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model"

# ============================================================
# ESTILOS CSS
# ============================================================
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #0f1117; }

    /* Header del proyecto */
    .project-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #e94560;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
    }
    .project-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .project-subtitle {
        font-size: 1.1rem;
        color: #a0aec0;
        margin-top: 0.3rem;
    }
    .badge {
        display: inline-block;
        background: #e94560;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-top: 0.5rem;
    }
    .badge-blue {
        background: #0f3460;
        border: 1px solid #4a90d9;
    }

    /* Tarjeta de métrica */
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e94560;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 0.2rem;
    }

    /* Resultado de predicción */
    .result-positive {
        background: linear-gradient(135deg, #1a3a2a, #0d2b1e);
        border: 2px solid #38a169;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #3a1a1a, #2b0d0d);
        border: 2px solid #e53e3e;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .result-subtitle {
        font-size: 1rem;
        color: #a0aec0;
    }

    /* Sección de info */
    .info-box {
        background: #1a1f2e;
        border-left: 4px solid #e94560;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }

    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CARGA DEL MODELO
# ============================================================
@st.cache_resource
def load_model():
    modelo     = joblib.load(MODEL_DIR / "modelo_sentimiento.pkl")
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
    metricas   = joblib.load(MODEL_DIR / "metricas.pkl")
    return modelo, vectorizer, metricas

try:
    modelo, vectorizer, metricas = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📚 Predictor de Reseñas")
    st.markdown("---")
    st.markdown("**Proyecto Final**")
    st.markdown("Administración de Datos")
    st.markdown("**Universidad:** LEAD University")
    st.markdown("**Profesor:** Alejandro Zamora")
    st.markdown("**Entrega:** 14 de abril de 2026")
    st.markdown("---")
    st.markdown("**Grupo 6**")
    st.markdown("- Ariana Víquez Solano")
    st.markdown("- Esteban Gutiérrez Saborío")
    st.markdown("- Marco Chinchilla Barrantes")
    st.markdown("---")
    st.markdown("**Modelo**")
    st.markdown("TF-IDF + Regresión Logística")
    if model_loaded:
        st.markdown(f"**Accuracy:** `{metricas['accuracy']*100:.2f}%`")
        st.markdown(f"**F1-Score:** `{metricas['f1']*100:.2f}%`")
        st.success("Modelo cargado ✓")
    else:
        st.error("Modelo no encontrado")
        st.code("python scripts/train_model.py")


# ============================================================
# HEADER PRINCIPAL
# ============================================================
st.markdown("""
<div class="project-header">
    <p class="project-title"> Predictor de Sentimiento de Reseñas</p>
    <p class="project-subtitle">Amazon Books Reviews — Pipeline Completo de Datos con IA</p>
    <span class="badge">LEAD University</span>
    <span class="badge badge-blue">Administración de Datos</span>
    <span class="badge badge-blue">Grupo 6</span>
    <span class="badge badge-blue">Entrega: 14 abril 2026</span>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ El modelo no está entrenado. Ejecutá primero: `python scripts/train_model.py`")
    st.stop()


# ============================================================
# TABS
# ============================================================
tab_predictor, tab_metricas, tab_info = st.tabs([
    "📚 Predictor",
    "📊 Métricas del Modelo",
    "ℹ️ Sobre el Proyecto"
])


# ============================================================
# TAB 1: PREDICTOR
# ============================================================
with tab_predictor:
    st.markdown("### ¿Qué opinas del libro?")
    st.markdown("Escribí el texto de tu reseña y el modelo de IA predecirá si es una experiencia **positiva** o **negativa/neutral**.")

    col_input, col_result = st.columns([1.2, 1], gap="large")

    with col_input:
        # Ejemplos rápidos
        st.markdown("** Ejemplos rápidos:**")
        col_e1, col_e2, col_e3 = st.columns(3)
        ejemplos = {
            "⭐⭐⭐⭐⭐ Positivo": "This book was absolutely incredible! The story captivated me from the very first page. The characters are beautifully developed and the writing style is elegant. I couldn't put it down. One of the best books I've ever read. Highly recommend to everyone!",
            "⭐ Negativo": "Terrible book. Complete waste of money and time. The plot makes no sense, the characters are flat and uninteresting. I forced myself to finish it hoping it would get better but it never did. Very disappointing.",
            "⭐⭐⭐ Neutral": "The book was okay. Nothing extraordinary but not bad either. Some parts were interesting but overall it felt average. The ending was predictable. Might be enjoyable for fans of the genre but not for everyone."
        }
        ejemplo_seleccionado = None
        if col_e1.button("⭐⭐⭐⭐⭐", use_container_width=True):
            ejemplo_seleccionado = list(ejemplos.values())[0]
        if col_e2.button("⭐ Negativo", use_container_width=True):
            ejemplo_seleccionado = list(ejemplos.values())[1]
        if col_e3.button("⭐⭐⭐ Neutral", use_container_width=True):
            ejemplo_seleccionado = list(ejemplos.values())[2]

        texto_default = ejemplo_seleccionado or ""
        texto = st.text_area(
            "Texto de la reseña:",
            value=texto_default,
            height=200,
            placeholder="Escribí aquí tu reseña en inglés o español...\n\nEjemplo: This book was amazing! The story was captivating and the characters felt real...",
            label_visibility="collapsed"
        )

        predecir = st.button(" Predecir Sentimiento", type="primary", use_container_width=True)

    with col_result:
        st.markdown("**Resultado:**")

        if predecir and texto.strip():
            texto_vec = vectorizer.transform([texto])
            prediccion = modelo.predict(texto_vec)[0]
            probabilidades = modelo.predict_proba(texto_vec)[0]

            prob_negativa = probabilidades[0]
            prob_positiva = probabilidades[1]

            if prediccion == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <div style="font-size:3.5rem">⭐⭐⭐⭐⭐</div>
                    <div class="result-title" style="color:#68d391">Reseña Positiva</div>
                    <div class="result-subtitle">El modelo predice 4-5 estrellas</div>
                    <div style="margin-top:1rem;font-size:1.8rem;font-weight:700;color:#68d391">
                        {prob_positiva*100:.1f}% confianza
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <div style="font-size:3.5rem">⭐</div>
                    <div class="result-title" style="color:#fc8181">Reseña Negativa / Neutral</div>
                    <div class="result-subtitle">El modelo predice 1-3 estrellas</div>
                    <div style="margin-top:1rem;font-size:1.8rem;font-weight:700;color:#fc8181">
                        {prob_negativa*100:.1f}% confianza
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Barra de probabilidades
            fig_prob = go.Figure(go.Bar(
                x=[prob_negativa * 100, prob_positiva * 100],
                y=["Negativa/Neutral (1-3 ⭐)", "Positiva (4-5 ⭐)"],
                orientation='h',
                marker=dict(
                    color=[
                        f"rgba(229, 62, 62, {0.4 + prob_negativa * 0.6})",
                        f"rgba(72, 187, 120, {0.4 + prob_positiva * 0.6})"
                    ]
                ),
                text=[f"{prob_negativa*100:.1f}%", f"{prob_positiva*100:.1f}%"],
                textposition='outside',
                textfont=dict(color='white', size=14),
            ))
            fig_prob.update_layout(
                title="Probabilidad por categoría",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(range=[0, 110], showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False),
                height=200,
                margin=dict(l=10, r=30, t=40, b=10),
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        elif predecir and not texto.strip():
            st.warning("Por favor, escribí un texto antes de predecir.")
        else:
            st.markdown("""
            <div style="background:#1a1f2e;border:1px dashed #4a5568;border-radius:12px;
                        padding:3rem;text-align:center;color:#718096;">
                <div style="font-size:3rem">🤖</div>
                <div style="margin-top:1rem">El resultado aparecerá aquí</div>
                <div style="font-size:0.85rem;margin-top:0.5rem">
                    Escribí una reseña y presioná "Predecir"
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# TAB 2: MÉTRICAS
# ============================================================
with tab_metricas:
    st.markdown("### Evaluación del Modelo")
    st.markdown("Métricas obtenidas sobre el conjunto de prueba (20% del dataset curado).")

    # Tarjetas de métricas
    col1, col2, col3, col4 = st.columns(4)
    metricas_display = [
        (col1, "Accuracy",  metricas['accuracy'],  "Registros clasificados correctamente"),
        (col2, "Precision", metricas['precision'], "De los positivos predichos, cuántos son realmente positivos"),
        (col3, "Recall",    metricas['recall'],    "De los positivos reales, cuántos detectó el modelo"),
        (col4, "F1-Score",  metricas['f1'],        "Balance entre Precision y Recall"),
    ]
    for col, nombre, valor, desc in metricas_display:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{valor*100:.1f}%</div>
            <div class="metric-label" style="font-weight:700;color:white">{nombre}</div>
            <div class="metric-label">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_words = st.columns(2, gap="large")

    # Matriz de confusión
    with col_cm:
        st.markdown("#### Matriz de Confusión")
        cm = np.array(metricas['confusion_matrix'])
        etiquetas = ["Negativa/Neutral", "Positiva"]

        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicción", y="Real", color="Cantidad"),
            x=etiquetas,
            y=etiquetas,
            color_continuous_scale=[[0, "#1a1f2e"], [1, "#e94560"]],
            text_auto=True,
        )
        fig_cm.update_traces(textfont=dict(size=20, color="white"))
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        n_test = metricas.get('n_test', '?')
        st.caption(f"Evaluado sobre {n_test:,} reseñas del conjunto de prueba")

    # Palabras más influyentes
    with col_words:
        st.markdown("#### Palabras más influyentes")

        top_pos = metricas['top_palabras_positivas'][:10]
        top_neg = metricas['top_palabras_negativas'][:10]

        tab_pos, tab_neg = st.tabs(["✅ Para Positivas", "❌ Para Negativas"])

        with tab_pos:
            fig_pos = go.Figure(go.Bar(
                x=list(range(len(top_pos), 0, -1)),
                y=top_pos,
                orientation='h',
                marker_color='#38a169',
            ))
            fig_pos.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_pos, use_container_width=True)
            st.caption("Términos que el modelo asocia más con reseñas positivas (4-5 ⭐)")

        with tab_neg:
            fig_neg = go.Figure(go.Bar(
                x=list(range(len(top_neg), 0, -1)),
                y=top_neg,
                orientation='h',
                marker_color='#e53e3e',
            ))
            fig_neg.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_neg, use_container_width=True)
            st.caption("Términos que el modelo asocia más con reseñas negativas (1-3 ⭐)")

    # Detalles técnicos del modelo
    st.markdown("---")
    st.markdown("#### Detalles Técnicos")
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.markdown(f"""
    <div class="info-box">
        <b>Tipo de modelo</b><br>
        Regresión Logística<br>
        <small style="color:#a0aec0">solver=lbfgs, C=1.0, max_iter=1000</small>
    </div>""", unsafe_allow_html=True)
    col_t2.markdown(f"""
    <div class="info-box">
        <b>Vectorización</b><br>
        TF-IDF (8,000 features)<br>
        <small style="color:#a0aec0">bigramas, min_df=3, sublinear_tf=True</small>
    </div>""", unsafe_allow_html=True)
    col_t3.markdown(f"""
    <div class="info-box">
        <b>Dataset</b><br>
        Amazon Books Reviews<br>
        <small style="color:#a0aec0">Train: {metricas['n_train']:,} | Test: {metricas['n_test']:,}</small>
    </div>""", unsafe_allow_html=True)


# ============================================================
# TAB 3: SOBRE EL PROYECTO
# ============================================================
with tab_info:
    st.markdown("### Sobre el Proyecto")

    col_desc, col_team = st.columns([1.5, 1], gap="large")

    with col_desc:
        st.markdown("""
        <div class="info-box">
            <b> Objetivo</b><br>
            Construir un pipeline completo de ingeniería de datos desde la conexión a una base de datos
            real hasta una aplicación funcional con un modelo de inteligencia artificial para predecir
            el sentimiento de reseñas de libros.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("####  Arquitectura del Pipeline")
        pipeline_steps = [
            (" Parte 1 — Fuente de Datos",
             "PostgreSQL · amazon_books_db · 3,212,404 registros",
             "Dataset Amazon Books Reviews cargado en 2 tablas relacionales (books_data y books_rating)"),
            (" Parte 2 — ETL Completo",
             "Extracción → Transformación → Carga Curada",
             "Join entre libros y reseñas, limpieza de nulos, filtrado de fechas inválidas y exportación a CSV curado"),
            (" Parte 3 — Orquestación",
             "Script maestro con logging por etapas",
             "Pipeline automatizado e idempotente con evidencia de ejecución y reporte de estado"),
            (" Parte 4 — Aplicativo con IA",
             "TF-IDF + Regresión Logística · Streamlit",
             "App funcional que predice el sentimiento de cualquier reseña en tiempo real"),
        ]
        for titulo, subtitulo, desc in pipeline_steps:
            st.markdown(f"""
            <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:10px;
                        padding:1rem 1.2rem;margin:0.5rem 0;">
                <b style="color:white">{titulo}</b><br>
                <span style="color:#e94560;font-size:0.85rem">{subtitulo}</span><br>
                <span style="color:#a0aec0;font-size:0.85rem">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_team:
        st.markdown("""
        <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:12px;padding:1.5rem;">
            <div style="font-size:1.1rem;font-weight:700;color:white;margin-bottom:1rem">
                 Equipo — Grupo 6
            </div>
            <div style="color:#a0aec0;font-size:0.9rem;line-height:2">
                - Ariana Víquez Solano<br>
                - Esteban Gutiérrez Saborío<br>
                - Marco Chinchilla Barrantes
            </div>
            <hr style="border-color:#2d3748;margin:1rem 0">
            <div style="color:#a0aec0;font-size:0.85rem;line-height:1.8">
                 <b style="color:white">Universidad</b><br>
                LEAD University<br><br>
                 <b style="color:white">Curso</b><br>
                Administración de Datos<br>
                Bachillerato en Ingeniería en Ciencia de Datos<br><br>
                 <b style="color:white">Profesor</b><br>
                Alejandro Zamora<br><br>
              <b style="color:white">Fecha de Entrega</b><br>
                14 de abril de 2026
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:12px;padding:1.5rem;">
            <div style="font-size:1rem;font-weight:700;color:white;margin-bottom:1rem">
                 Dataset
            </div>
            <div style="color:#a0aec0;font-size:0.85rem;line-height:1.8">
                <b style="color:white">Amazon Books Reviews</b><br>
                Kaggle — mohamedbakheet<br><br>
                📊 3,212,404 registros totales<br>
                📖 212,404 libros únicos<br>
                ⭐ 3,000,000 reseñas<br>
                🎯 Target: review_score (1-5 ⭐)
            </div>
        </div>
        """, unsafe_allow_html=True)
