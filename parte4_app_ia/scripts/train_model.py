"""
train_model.py
==============
Entrenamiento del modelo de IA para predicción de sentimiento en reseñas de libros.

Enfoque:
    - Features : TF-IDF sobre review_text + review_summary (hasta 8,000 términos, bigramas)
    - Target   : Clasificación binaria
                   1 → Positiva  (review_score 4-5 estrellas)
                   0 → Negativa/Neutral (review_score 1-3 estrellas)
    - Modelo   : Regresión Logística (rápida, interpretable, alta precisión en texto)

Salidas (guardadas en model/):
    - modelo_sentimiento.pkl  → modelo entrenado
    - vectorizer.pkl          → TF-IDF vectorizer ajustado
    - metricas.pkl            → métricas de evaluación + matriz de confusión

Proyecto Final — Administración de Datos — LEAD University
Grupo 6 | Parte 4: Aplicativo con IA
Autores: Ariana Víquez Solano | Esteban Gutiérrez Saborío | Marco Chinchilla Barrantes
Profesor: Alejandro Zamora
Entrega: 14 de abril de 2026
"""

import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ============================================================
# RUTAS
# ============================================================
BASE_DIR    = Path(__file__).parent.parent
MODEL_DIR   = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_DIR = BASE_DIR.parent
CURATED_CSV = PROJECT_DIR / "parte2_etl" / "data" / "curated" / "amazon_books_curated.csv"


def load_data() -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  CARGANDO DATASET CURADO")
    print(f"{'='*55}")
    print(f"  Fuente: {CURATED_CSV}")

    if not CURATED_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset curado en:\n  {CURATED_CSV}\n"
            "Asegurate de haber ejecutado la Parte 2 (ETL) primero."
        )

    df = pd.read_csv(CURATED_CSV)
    print(f"  Registros cargados: {len(df):,}")
    print(f"  Columnas: {list(df.columns)}")
    return df


def prepare_features(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print("  PREPARACIÓN DE FEATURES")
    print(f"{'='*55}")

    # Combinar texto de reseña y resumen
    df['texto_completo'] = (
        df['review_summary'].fillna('') + ' ' + df['review_text'].fillna('')
    ).str.strip()

    # Target binario: 1 = Positiva (4-5 ⭐), 0 = Negativa/Neutral (1-3 ⭐)
    df['sentimiento'] = (df['review_score'] >= 4).astype(int)

    # Eliminar filas sin texto
    df = df[df['texto_completo'].str.len() > 5].copy()

    positivas  = df['sentimiento'].sum()
    negativas  = len(df) - positivas
    print(f"  Reseñas positivas (4-5 ⭐) : {positivas:,} ({positivas/len(df)*100:.1f}%)")
    print(f"  Reseñas negativas (1-3 ⭐) : {negativas:,} ({negativas/len(df)*100:.1f}%)")
    print(f"  Total para entrenamiento  : {len(df):,}")

    return df['texto_completo'], df['sentimiento']


def train(X, y):
    print(f"\n{'='*55}")
    print("  ENTRENAMIENTO DEL MODELO")
    print(f"{'='*55}")

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # TF-IDF vectorizer
    print("  Vectorizando texto (TF-IDF, hasta 8,000 términos, bigramas)...")
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # Modelo
    print("  Entrenando Regresión Logística...")
    t0 = time.time()
    modelo = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=42,
    )
    modelo.fit(X_train_vec, y_train)
    elapsed = time.time() - t0
    print(f"  Entrenamiento completado en {elapsed:.1f}s")

    # Evaluación
    y_pred = modelo.predict(X_test_vec)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*55}")
    print("  MÉTRICAS DE EVALUACIÓN")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Matriz de Confusión:")
    print(f"  {cm}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Negativa/Neutral','Positiva'])}")

    # Top palabras más influyentes
    feature_names = vectorizer.get_feature_names_out()
    coeficientes  = modelo.coef_[0]
    top_positivas = [feature_names[i] for i in np.argsort(coeficientes)[-20:]][::-1]
    top_negativas = [feature_names[i] for i in np.argsort(coeficientes)[:20]]

    metricas = {
        "accuracy"      : round(acc, 4),
        "precision"     : round(prec, 4),
        "recall"        : round(rec, 4),
        "f1"            : round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "top_palabras_positivas": top_positivas,
        "top_palabras_negativas": top_negativas,
        "n_train"       : len(X_train),
        "n_test"        : len(X_test),
    }

    return modelo, vectorizer, metricas


def save_artifacts(modelo, vectorizer, metricas):
    print(f"\n{'='*55}")
    print("  GUARDANDO MODELO")
    print(f"{'='*55}")

    joblib.dump(modelo,     MODEL_DIR / "modelo_sentimiento.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    joblib.dump(metricas,   MODEL_DIR / "metricas.pkl")

    print(f"  modelo_sentimiento.pkl → {MODEL_DIR}")
    print(f"  vectorizer.pkl         → {MODEL_DIR}")
    print(f"  metricas.pkl           → {MODEL_DIR}")
    print(f"\n  ✅ Modelo listo para la aplicación Streamlit")


def main():
    print("\n" + "="*55)
    print("  ENTRENAMIENTO — MODELO DE SENTIMIENTO DE RESEÑAS")
    print("  Amazon Books Reviews | Grupo 6 | LEAD University")
    print("="*55)

    df = load_data()
    X, y = prepare_features(df)
    modelo, vectorizer, metricas = train(X, y)
    save_artifacts(modelo, vectorizer, metricas)

    print("\n" + "="*55)
    print(f"  Accuracy final: {metricas['accuracy']*100:.2f}%")
    print("  Próximo paso: streamlit run scripts/app.py")
    print("="*55)


if __name__ == "__main__":
    main()
