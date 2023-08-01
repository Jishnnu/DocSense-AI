import os
from chromadb.config import Settings

# Defining ChromaDB settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl = 'duckdb+parquet',
    persist_directory = 'Database',
    anonymized_telemetry = False
)