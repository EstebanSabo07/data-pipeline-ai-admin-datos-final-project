-- ============================================================
-- 05_create_books_tables.sql
-- Esquema de referencia para la base de datos Amazon Books
-- Base de datos: amazon_books_db | Schema: books
--
-- NOTA: Las tablas son creadas directamente por run_parte1.py
-- con autocommit para mayor robustez. Este archivo documenta
-- el esquema y puede usarse para referencia o recreación manual.
-- ============================================================

CREATE SCHEMA IF NOT EXISTS books;

-- Tabla de metadatos de libros
CREATE TABLE IF NOT EXISTS books.books_data (
    title           TEXT,
    description     TEXT,
    authors         TEXT,
    image           TEXT,
    preview_link    TEXT,
    publisher       TEXT,
    published_date  TEXT,
    info_link       TEXT,
    categories      TEXT,
    ratings_count   NUMERIC
);

-- Tabla de reseñas (variable target: review_score)
CREATE TABLE IF NOT EXISTS books.books_rating (
    id              TEXT,
    title           TEXT,
    price           NUMERIC,
    user_id         TEXT,
    profile_name    TEXT,
    helpfulness     TEXT,
    review_score    NUMERIC,
    review_time     BIGINT,
    review_summary  TEXT,
    review_text     TEXT
);

-- Índices para optimización de queries del ETL y modelo de IA
CREATE INDEX IF NOT EXISTS idx_rating_title ON books.books_rating(title);
CREATE INDEX IF NOT EXISTS idx_rating_score ON books.books_rating(review_score);
CREATE INDEX IF NOT EXISTS idx_rating_user  ON books.books_rating(user_id);
