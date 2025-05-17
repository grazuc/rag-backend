"""
ingest.py
----------
Ingiere documentos en múltiples formatos desde ./docs, los trocea,
genera embeddings y los guarda en la base de datos vectorial.
Soporta procesamiento paralelo, procesamiento incremental y chunking dinámico.
"""

import os
import time
import hashlib
import argparse
import logging
import json
import re
import sys
from typing import List, Dict, Optional, Set, Tuple, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from langchain_community.embeddings import HuggingFaceEmbeddings
import multiprocessing
import shutil
import pickle
from datetime import datetime
import statistics

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from langdetect import detect, LangDetectException

from langchain_community.document_loaders import (
    PyPDFLoader, 
    UnstructuredFileLoader,
    DirectoryLoader,
    TextLoader, 
    Docx2txtLoader
)
from langchain_community.document_loaders.pdf import PyPDFium2Loader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from threading import Lock
import psutil
import traceback

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ingest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración por defecto
@dataclass
class IngestConfig:
    docs_dir: Path = Path("docs")
    cache_dir: Path = Path(".cache")
    embed_model: str = "intfloat/multilingual-e5-base" 
    collection_name: str = "manual_e5_multi"
    chunk_size: int = 384  # Chunk size base
    chunk_overlap: int = 64  # Chunk overlap base
    batch_size: int = 128  # Batch size más grande para mejor rendimiento
    max_workers: int = None  # Permitir autodetección
    max_embed_workers: int = 1  # Workers para embeddings
    allowed_languages: Set[str] = field(default_factory=lambda: {"es", "en"})
    file_extensions: List[str] = field(default_factory=lambda: ["pdf", "txt", "docx", "md"])
    deduplication: bool = True
    similarity_threshold: float = 0.95  # Para deduplicación
    incremental: bool = True  # Procesamiento incremental de archivos
    semantic_chunking: bool = True  # Chunking semántico
    dynamic_chunking: bool = True  # Chunking dinámico adaptativo
    reset_vector_collection: bool = True # Nuevo campo


# 1) Entorno
def load_config() -> Tuple[IngestConfig, str]:
    """Carga la configuración desde argumentos y env vars"""
    load_dotenv()
    
    # Verificar variable de entorno obligatoria
    pg_conn = os.environ.get("PG_CONN")
    if not pg_conn:
        raise EnvironmentError("Variable de entorno PG_CONN no definida")
    
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Ingesta de documentos para RAG")
    parser.add_argument("--docs-dir", type=str, default="docs", 
                        help="Directorio con documentos a ingerir")
    parser.add_argument("--cache-dir", type=str, default=".cache",
                        help="Directorio para caché de procesamiento")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-base",
                        help="Modelo de embeddings a utilizar")
    parser.add_argument("--collection", type=str, default="manual_e5_multi",
                        help="Nombre de la colección en PGVector")
    parser.add_argument("--chunk-size", type=int, default=384,
                        help="Tamaño base de los chunks de texto")
    parser.add_argument("--chunk-overlap", type=int, default=64,
                        help="Superposición base entre chunks")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Tamaño de lote para inserción en BD")
    parser.add_argument("--workers", type=int, default=None,
                        help="Número de workers para procesamiento paralelo (None=auto)")
    parser.add_argument("--embed-workers", type=int, default=1,
                        help="Número de workers para embeddings paralelos")
    parser.add_argument("--langs", type=str, default="es,en",
                        help="Idiomas permitidos (códigos ISO separados por comas)")
    parser.add_argument("--extensions", type=str, default="pdf,txt,docx,md",
                        help="Extensiones de archivo a procesar (separadas por comas)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Desactivar deduplicación de chunks")
    parser.add_argument("--no-incremental", action="store_true",
                        help="Desactivar procesamiento incremental")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Desactivar chunking semántico")
    parser.add_argument("--no-dynamic", action="store_true",
                        help="Desactivar chunking dinámico adaptativo")
    parser.add_argument("--reset-cache", action="store_true",
                        help="Eliminar caché y reprocesar todos los archivos")
    parser.add_argument("--reset-vector-collection", action="store_true",
    help="Eliminar y recrear la colección en PGVector antes de la ingesta.")
    
    args = parser.parse_args()
    
    # Auto-detección del número de workers si no se especifica
    max_workers = args.workers if args.workers is not None else max(1, multiprocessing.cpu_count() - 1)
    
    # Crear configuración
    config = IngestConfig(
        docs_dir=Path(args.docs_dir),
        cache_dir=Path(args.cache_dir),
        embed_model=args.model,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        max_workers=max_workers,
        max_embed_workers=args.embed_workers,
        allowed_languages=set(args.langs.split(",")),
        file_extensions=args.extensions.split(","),
        deduplication=not args.no_dedup,
        incremental=not args.no_incremental,
        semantic_chunking=not args.no_semantic,
        dynamic_chunking=not args.no_dynamic,
        reset_vector_collection=args.reset_vector_collection
    )
    
    # Resetear caché si se solicita
    if args.reset_cache and config.cache_dir.exists():
        shutil.rmtree(config.cache_dir)
    
    # Asegurar que existe el directorio de caché
    config.cache_dir.mkdir(exist_ok=True, parents=True)
    
    return config, pg_conn


class DocumentProcessor:
    """Clase para manejo de procesamiento de documentos con caché"""
    
    def __init__(self, config: IngestConfig):
        self.config = config
        self.cache_file = config.cache_dir / "processed_files.json"
        self.processed_files = self._load_processed_files()
        
    def _load_processed_files(self) -> Dict[str, Dict[str, Any]]:
        """Carga el registro de archivos procesados desde caché"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error al cargar caché: {e}")
                return {}
        return {}
    
    def _save_processed_files(self):
        """Guarda el registro de archivos procesados en caché"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_files, f)
    
    def get_files_to_process(self) -> List[Tuple[Path, bool]]:
        """
        Devuelve lista de archivos a procesar, indicando si necesitan procesamiento
        Retorna: Lista de tuplas (archivo, necesita_procesamiento)
        """
        files_to_process = []
        
        for ext in self.config.file_extensions:
            for file_path in self.config.docs_dir.rglob(f"*.{ext}"):
                # Obtener información del archivo
                file_key = str(file_path.relative_to(self.config.docs_dir))
                file_stat = file_path.stat()
                file_size = file_stat.st_size
                file_mtime = file_stat.st_mtime
                
                # Verificar si el archivo necesita procesamiento
                needs_processing = True
                if self.config.incremental and file_key in self.processed_files:
                    cached_info = self.processed_files[file_key]
                    # Si tamaño y fecha de modificación coinciden, no procesar
                    if cached_info.get("size") == file_size and cached_info.get("mtime") == file_mtime:
                        needs_processing = False
                
                files_to_process.append((file_path, needs_processing))
        
        return files_to_process
    
    def mark_file_processed(self, file_path: Path):
        """Marca un archivo como procesado en el caché"""
        file_stat = file_path.stat()
        self.processed_files[file_path.name] = {
            "size": file_stat.st_size,
            "mtime": file_stat.st_mtime,
            "last_processed": time.time()
        }
        self._save_processed_files()
    
    def get_chunks_cache_path(self, file_path: Path) -> Path:
        """Devuelve la ruta para caché de chunks de un archivo"""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()
        return self.config.cache_dir / f"chunks_{file_hash}.pkl"
    
    def get_cached_chunks(self, file_path: Path) -> Optional[List[Document]]:
        """Intenta obtener chunks cacheados para un archivo"""
        if not self.config.incremental:
            return None
            
        cache_path = self.get_chunks_cache_path(file_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error al cargar chunks desde caché: {e}")
        return None
    
    def save_chunks_to_cache(self, file_path: Path, chunks: List[Document]):
        """Guarda chunks en caché para un archivo"""
        if not self.config.incremental:
            return
            
        cache_path = self.get_chunks_cache_path(file_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(chunks, f)
        except Exception as e:
            logger.warning(f"Error al guardar chunks en caché: {e}")


def get_file_loader(file_path: Path, allowed_extensions):
    """Selecciona el loader más adecuado según el tipo de archivo con mejor manejo de errores"""
    extension = file_path.suffix.lower().lstrip(".")
    
    if extension not in allowed_extensions:
        raise ValueError(f"Extensión no soportada: {extension}")
    
    try:
        # Verificar que el archivo existe y tiene tamaño
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo {file_path} no existe")
        
        if file_path.stat().st_size == 0:
            raise ValueError(f"El archivo {file_path} está vacío")
        
        if extension == "pdf":
            # Intentar primero con PyPDFium2Loader, y si falla, usar PyPDFLoader como respaldo
            try:
                return PyPDFium2Loader(str(file_path))
            except Exception as e:
                logger.warning(f"Error con PyPDFium2Loader: {e}, intentando con PyPDFLoader")
                return PyPDFLoader(str(file_path))
        elif extension == "docx":
            return Docx2txtLoader(str(file_path))
        elif extension == "md":
            return TextLoader(str(file_path), encoding='utf-8', autodetect_encoding=True)
        elif extension == "txt":
            return TextLoader(str(file_path), encoding='utf-8', autodetect_encoding=True)
        else:
            return UnstructuredFileLoader(str(file_path))
    except Exception as e:
        logger.error(f"Error al crear loader para {file_path}: {e}")
        raise


# En ingest.py

def detect_language(text: str, min_length_for_detection: int = 20, fallback_lang="es") -> Optional[str]:
    """
    Detecta el idioma del texto con mejor manejo de textos cortos y casos límite.
    Incluye fallback para manejar casos donde la detección no es confiable.
    """
    stripped_text = text.strip()
    # Usar un umbral más bajo para mejorar detección en chunks pequeños
    if not stripped_text:
        return fallback_lang  # Retornar idioma por defecto si está vacío
        
    if len(stripped_text) < min_length_for_detection:
        # Para textos muy cortos, intentar detectar pero con precaución
        logger.debug(f"Texto corto para detección confiable (longitud: {len(stripped_text)})")
        # Si contiene palabras clave en español, podemos asumir español
        if re.search(r'\b(el|la|los|las|de|en|por|para|con|que)\b', stripped_text.lower()):
            return "es"
        # Si contiene palabras clave en inglés
        elif re.search(r'\b(the|of|in|to|and|that|for|with)\b', stripped_text.lower()):
            return "en"
        return fallback_lang
    
    try:
        # Usar langdetect con mayor robustez
        lang = detect(stripped_text)
        return lang
    except LangDetectException as e:
        logger.warning(f"Fallo en detección de idioma: '{stripped_text[:50]}...'. Error: {e}")
        return fallback_lang


def sanitize_pg_identifier(name: str) -> str:
    """Convierte un nombre arbitrario a un identificador SQL seguro para PostgreSQL"""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def compute_text_hash(text: str) -> str:
    """Calcula un hash del texto para deduplicación"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def analyze_document_structure(text: str, file_extension: str) -> Dict[str, Any]:
    """
    Analiza la estructura del documento para determinar características relevantes
    para el chunking dinámico, con mejor identificación de tipos de documento
    y análisis más profundo de la estructura.
    """
    # Inicializar análisis con valores por defecto mejorados
    analysis = {
        "extension": file_extension,
        "doc_type": "general",
        "avg_sentence_length": 0,
        "avg_paragraph_length": 0,
        "semantic_density": 0.0,
        "has_structures": False,
        "has_tables": False,
        "has_code": False,
        "has_lists": False,
        "title": None,
        "language": None,
        "recommended_chunk_size": 450,  # Valor por defecto mejorado
        "recommended_chunk_overlap": 80  # Valor por defecto mejorado
    }
    
    # Intento de detección de idioma para ajustar análisis
    try:
        analysis["language"] = detect_language(text[:3000])  # Usar una muestra más grande para mejor detección
    except:
        pass  # Si falla, seguimos sin idioma detectado
    
    # Detección más sofisticada del tipo de documento
    if file_extension == "pdf":
        analysis["doc_type"] = "pdf"
        # Detectar si es un PDF técnico, académico o general
        if re.search(r"(?i)(abstract|keywords|references|bibliography|doi|isbn)", text[:3000]):
            analysis["doc_type"] = "scientific_pdf"
        elif re.search(r"(?i)(contrato|acuerdo|cláusula|artículo|ley|decreto|resolución|expediente)", text[:3000]):
            analysis["doc_type"] = "legal_pdf"
    elif file_extension == "docx":
        analysis["doc_type"] = "docx"
    elif file_extension == "md":
        analysis["doc_type"] = "markdown"
        # Detectar si es documentación técnica
        if re.search(r"```(python|javascript|java|c\+\+|bash|sql)", text):
            analysis["doc_type"] = "technical_markdown"
            analysis["has_code"] = True
    elif file_extension == "txt":
        # Análisis más profundo del tipo de texto
        if re.search(r"(?i)(artículo|cláusula|contrato|acuerdo|legal|legislación)", text[:2000]):
            analysis["doc_type"] = "legal"
        elif re.search(r"(?i)(from:|to:|subject:|sent:|cc:|bcc:)", text[:1000]):
            analysis["doc_type"] = "email"
        elif re.search(r"(?i)(abstract|introducción|metodología|conclusión|references)", text[:3000]):
            analysis["doc_type"] = "scientific"
        elif re.search(r"(?i)(def |class |import |function |var |const |let |```python|```javascript)", text):
            analysis["doc_type"] = "code"
            analysis["has_code"] = True
    
    # Extraer título si es posible
    title_match = re.search(r"(?i)^(?:\s*#\s*|\s*título\s*:?\s*|\s*title\s*:?\s*)(.*?)$", text[:1000], re.MULTILINE)
    if title_match:
        analysis["title"] = title_match.group(1).strip()
    
    # Análisis de estructura mejorado
    paragraphs = re.split(r'\n\s*\n', text)
    if len(paragraphs) > 1:
        paragraph_lengths = [len(p.split()) for p in paragraphs if p.strip()]
        if paragraph_lengths:
            analysis["avg_paragraph_length"] = sum(paragraph_lengths) / len(paragraph_lengths)
            analysis["paragraph_count"] = len(paragraph_lengths)
            # Analizar variabilidad en longitud de párrafos
            if len(paragraph_lengths) > 3:
                analysis["paragraph_length_std"] = statistics.stdev(paragraph_lengths)
            else:
                analysis["paragraph_length_std"] = 0
    
    # Análisis de oraciones más preciso
    sentence_pattern = r'[.!?][\s"\')\]]+'  # Mejor regex para detectar finales de oración
    sentences = re.split(sentence_pattern, text)
    if sentences:
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            analysis["avg_sentence_length"] = sum(sentence_lengths) / len(sentence_lengths)
            analysis["sentence_count"] = len(sentence_lengths)
            
            # Medir complejidad por desviación estándar de longitud de oraciones
            if len(sentence_lengths) > 5:
                analysis["sentence_length_std"] = statistics.stdev(sentence_lengths)
                # Complejidad: alta desviación estándar indica texto con estructuras variadas
                analysis["complexity"] = min(1.0, analysis["sentence_length_std"] / 10)
            else:
                analysis["sentence_length_std"] = 0
                analysis["complexity"] = 0.2  # Valor predeterminado bajo para textos cortos
    
    # Análisis de densidad semántica mejorado (aproximación por unique words / total words)
    words = re.findall(r'\b\w+\b', text.lower())
    if words:
        analysis["word_count"] = len(words)
        unique_words = set(words)
        analysis["unique_word_count"] = len(unique_words)
        # Calcular densidad semántica con penalización para textos cortos
        min_words_for_full_density = 100  # Umbral para considerar muestra suficiente
        discount_factor = min(1.0, len(words) / min_words_for_full_density)
        analysis["semantic_density"] = (len(unique_words) / max(1, len(words))) * discount_factor
    
    # Detectar estructuras especiales
    # Tablas
    if re.search(r'[|+][-+]+[|+]', text) or re.search(r'\b\d+\s*&\s*\d+\b', text):
        analysis["has_tables"] = True
        analysis["has_structures"] = True
    
    # Listas
    if re.search(r'^\s*[*\-+•○◦]\s+\w+.*$', text, re.MULTILINE) or \
       re.search(r'^\s*\d+\.\s+\w+.*$', text, re.MULTILINE):
        analysis["has_lists"] = True
        analysis["has_structures"] = True
    
    # Bloques de código
    if re.search(r'```.*?```', text, re.DOTALL) or re.search(r'<code>.*?</code>', text, re.DOTALL):
        analysis["has_code"] = True
        analysis["has_structures"] = True
    
    # URLs y referencias
    urls_count = len(re.findall(r'https?://\S+', text))
    if urls_count > 0:
        analysis["urls_count"] = urls_count
        analysis["has_references"] = True
    
    # Determinar chunk_size y chunk_overlap basados en el análisis
    determine_chunking_parameters(analysis)
    
    return analysis


def determine_chunking_parameters(analysis: Dict[str, Any]):
    """
    Determina los parámetros óptimos de chunking basados en el análisis del documento,
    con mejor adaptación según el tipo de contenido y posibilidad de recuperación contextual.
    """
    doc_type = analysis["doc_type"]
    
    # Chunking más inteligente por tipo de documento
    if doc_type == "legal":
        # Documentos legales requieren chunks más grandes y mayor overlap para mantener contexto legal
        base_chunk_size = 600
        base_chunk_overlap = 150  # Mayor overlap para mejor contexto de referencias legales
    elif doc_type == "scientific":
        # Documentos científicos con chunks medianos pero mayor overlap
        base_chunk_size = 500
        base_chunk_overlap = 120  # Mayor overlap para mantener coherencia en ecuaciones y referencias
    elif doc_type == "email":
        # Emails típicamente son más cortos y autocontenidos
        base_chunk_size = 350
        base_chunk_overlap = 60
    elif doc_type == "markdown":
        # Markdown ya tiene estructura que ayuda a la segmentación
        base_chunk_size = 450
        base_chunk_overlap = 90
    else:
        # Valores base para otros tipos, ligeramente más grandes que los anteriores
        base_chunk_size = 450
        base_chunk_overlap = 80
    
    # Ajustes por densidad semántica con mayor sensibilidad
    density_factor = 1.0
    if analysis["semantic_density"] > 0.75:  # Alta diversidad léxica = texto técnico/especializado
        # Chunks más pequeños para texto denso, pero mayor overlap
        density_factor = 0.85
        # Aumentar overlap proporcionalmente
        base_chunk_overlap = int(base_chunk_overlap * 1.2)
    elif analysis["semantic_density"] < 0.35:  # Baja diversidad léxica = texto simple/repetitivo
        # Chunks más grandes para texto simple
        density_factor = 1.3
    
    # Ajustes por longitud de oraciones con mejor calibración
    sentence_factor = 1.0
    avg_sentence_length = analysis.get("avg_sentence_length", 0)
    if avg_sentence_length > 30:  # Oraciones muy largas (ej. texto académico o legal)
        # Reducir tamaño para evitar cortar en medio de ideas complejas
        sentence_factor = 0.75
        # Mayor overlap para texto con oraciones complejas
        base_chunk_overlap = int(base_chunk_overlap * 1.3)
    elif avg_sentence_length > 20:  # Oraciones largas
        sentence_factor = 0.85
        base_chunk_overlap = int(base_chunk_overlap * 1.15)
    elif avg_sentence_length < 8:  # Oraciones muy cortas (diálogos, listas, etc.)
        # Aumentar tamaño para capturar suficiente contexto
        sentence_factor = 1.2
    
    # Ajuste para documentos estructurados
    structure_factor = 0.85 if analysis["has_structures"] else 1.0
    
    # Ajuste por complejidad 
    complexity = analysis.get("complexity", 0)
    # Mayor complejidad = más variabilidad en las oraciones = necesita chunks más adaptativos
    complexity_factor = max(0.8, min(1.2, 1.0 - (complexity * 0.2)))
    
    # Calcular tamaños finales ajustados con todos los factores
    chunk_size = int(base_chunk_size * density_factor * sentence_factor * structure_factor * complexity_factor)
    
    # Ajustar overlap basado en todos los factores, pero especialmente en complejidad
    # Cuando el texto es más complejo, necesitamos más overlap para preservar el contexto
    overlap_factor = 1.0 + (complexity * 0.6) 
    # También añadir factor por densidad semántica
    if analysis["semantic_density"] > 0.65:
        overlap_factor *= 1.15
    
    chunk_overlap = int(base_chunk_overlap * overlap_factor)
    
    # Restricciones para mantener valores razonables
    chunk_size = max(300, min(chunk_size, 1200))  # Entre 300 y 1200
    
    # El overlap debe ser proporcional al chunk_size pero no excesivo
    min_overlap = max(50, chunk_size // 8)  # Al menos 50 o 1/8 del chunk
    max_overlap = chunk_size // 3  # Máximo 1/3 del chunk para evitar duplicación excesiva
    chunk_overlap = max(min_overlap, min(chunk_overlap, max_overlap))
    
    # Actualizar el análisis con los valores recomendados
    analysis["recommended_chunk_size"] = chunk_size
    analysis["recommended_chunk_overlap"] = chunk_overlap



def get_smart_text_splitter(
    doc_extension: str, 
    config: IngestConfig, 
    # Parámetros que ahora se pasan directamente
    chunk_size: int, 
    chunk_overlap: int,
    # doc_content es opcional y solo relevante si se quiere que esta función aún pueda determinar
    # los parámetros si no se pasaron, pero la idea es que process_file ya lo haga.
    # Para este refactor, asumiremos que chunk_size y chunk_overlap ya vienen determinados.
):
    """
    Devuelve el splitter más adecuado según el tipo de documento.
    Chunk_size y chunk_overlap son determinados externamente y pasados como parámetros.
    """
    logger.debug(f"Usando get_smart_text_splitter con chunk_size={chunk_size}, chunk_overlap={chunk_overlap} para extensión {doc_extension}")

    # Si no se usa chunking semántico, usar RecursiveCharacterTextSplitter simple
    if not config.semantic_chunking:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len, # Es buena práctica definirla explícitamente
        )
    
    # Personalizar el splitter según el tipo de documento
    if doc_extension == "md":
        # Para archivos Markdown, usar splitter basado en encabezados
        headers_to_split_on = [
            ("#", "Heading1"),
            ("##", "Heading2"),
            ("###", "Heading3"),
            ("####", "Heading4"),
        ]
    
        # Para mantenerlo simple y alineado con la estructura de un solo splitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # Párrafos
                "\n",    # Líneas
                "```",   # Bloques de código
                "##",    # Encabezados
                "#",     # Encabezados
                ". ", 
                " ", 
                ""
            ],
            length_function=len,
        )
    
    elif doc_extension == "pdf" or doc_extension == "docx":
        # Para PDF y DOCX, usar separadores que respeten estructura de documento
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n",  # Separación entre secciones principales
                "\n\n",    # Separación entre párrafos
                "\n",      # Separación entre líneas
                ".\n",     # Puntos finales de párrafo
                ". ",      # Separación entre frases
                "; ",      # Separación dentro de frases complejas
                ", ",      # Separación dentro de frases
                " ",       # Separación entre palabras
                ""         # Último recurso: caracteres individuales
            ],
            length_function=len,
        )
    else: # txt y otros
        # Para otros tipos, usar configuración estándar
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )



def process_file(file_path: Path, config: IngestConfig, doc_processor: DocumentProcessor) -> List[Document]:
    """Procesa un archivo y devuelve los chunks resultantes"""
    try:
        logger.info(f"Procesando {file_path.name}...")
        
        cached_chunks = doc_processor.get_cached_chunks(file_path) # Movido al inicio para chequear caché primero
        
        if cached_chunks is not None and config.incremental: # Asegurarse que incremental esté activo
            logger.info(f"Usando chunks cacheados para {file_path.name}")
            # Actualizar algunos metadatos que pueden cambiar si es necesario
            for chunk in cached_chunks:
                chunk.metadata["ingest_time"] = time.time() # Ejemplo de metadato a actualizar
            return cached_chunks

        # Si no hay caché o no es incremental, procesar
        loader = get_file_loader(file_path, config.file_extensions)
        docs = loader.load()
        
        if not docs:
            logger.warning(f"El loader no devolvió documentos para {file_path.name}")
            return []

        file_extension = file_path.suffix.lower().lstrip(".")
        
        full_content = "\n\n".join([doc.page_content for doc in docs if doc.page_content])
        
        if not full_content.strip():
            logger.warning(f"El contenido extraído de {file_path.name} está vacío.")
            return []

        current_chunk_size = config.chunk_size
        current_chunk_overlap = config.chunk_overlap
        doc_analysis_results = None # Para almacenar los resultados del análisis

        if config.dynamic_chunking:
            doc_analysis_results = analyze_document_structure(full_content, file_extension)
            current_chunk_size = doc_analysis_results["recommended_chunk_size"]
            current_chunk_overlap = doc_analysis_results["recommended_chunk_overlap"]
            logger.info(
                f"Chunking dinámico para {file_path.name}: "
                f"size={current_chunk_size}, overlap={current_chunk_overlap} "
                f"(doc tipo: {doc_analysis_results.get('doc_type', 'general')}, "
                f"densidad: {doc_analysis_results.get('semantic_density', 0):.2f})"
            )
        
        splitter = get_smart_text_splitter(
            file_extension, 
            config, 
            current_chunk_size, # Pasar tamaño determinado
            current_chunk_overlap # Pasar overlap determinado
        )
        
        # Dividir en chunks
        # split_documents espera una lista de Documentos. Si docs ya es una lista de Documentos, está bien.
        chunks = splitter.split_documents(docs)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Actualizar metadatos del chunk
            # Detectar título, sección o keywords (puede basarse en heurísticas simples)
            first_line = chunk.page_content.strip().split('\n')[0]
            if len(first_line) < 80:
                chunk.metadata["title"] = first_line

            # Extraer sección por regex básica (opcional)
            match_section = re.search(r"(?i)(sección|capítulo|título|parte)\s+\d+[:\-]?\s*(.+)", chunk.page_content[:300])
            if match_section:
                 chunk.metadata["section"] = match_section.group(2).strip()

            # Opcional: keywords (si tenés algún algoritmo o regex)
            keywords = re.findall(r'\b[A-Z]{3,}\b', chunk.page_content)
            if keywords:
                chunk.metadata["keywords"] = list(set(keywords))


            logger.info(f"🧪 DOCUMENT_ID={os.environ.get('DOCUMENT_ID')}")
            logger.info(f"🧪 file_path.parent.name={file_path.parent.name}")
            
            document_id = file_path.parent.name or file_path.stem
            chunk.metadata["document_id"] = document_id  # ← usa el nombre del directorio que es el documentId
            chunk.metadata["source"] = file_path.name
            chunk.metadata["extension"] = file_extension
            chunk.metadata["ingest_time"] = time.time()
            chunk.metadata["chunk_id"] = f"{file_path.name}-{i}" # ID de chunk más único
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(chunks)
            chunk.metadata["file_size"] = file_path.stat().st_size
            # doc_type ya debería estar en doc_analysis_results si dynamic_chunking está activo
            chunk.metadata["doc_type"] = doc_analysis_results.get('doc_type', file_extension) if doc_analysis_results else file_extension
            chunk.metadata["process_date"] = datetime.now().isoformat()
            logger.info(f"🧪 document_id final seteado: {document_id}")


            if i > 0:
                chunk.metadata["prev_chunk_id"] = f"{file_path.name}-{i-1}"
            if i < len(chunks) - 1:
                chunk.metadata["next_chunk_id"] = f"{file_path.name}-{i+1}"
                
            if config.dynamic_chunking and doc_analysis_results:
                chunk.metadata["chunk_size_used"] = current_chunk_size
                chunk.metadata["chunk_overlap_used"] = current_chunk_overlap
                chunk.metadata["doc_semantic_density"] = doc_analysis_results.get("semantic_density")
                # Guardar una copia del análisis sin los parámetros ya usados para no ser redundante
                analysis_to_store = doc_analysis_results.copy()
                analysis_to_store.pop("recommended_chunk_size", None)
                analysis_to_store.pop("recommended_chunk_overlap", None)
                chunk.metadata["doc_analysis_details"] = analysis_to_store
            
            # Filtrar por idioma
            if config.allowed_languages:
                lang = detect_language(chunk.page_content) # Asumiendo que detect_language está definida
                if lang and lang in config.allowed_languages:
                    chunk.metadata["language"] = lang
                else:
                    logger.debug(f"Chunk de {file_path.name} (ID: {chunk.metadata['chunk_id']}) descartado por idioma: {lang if lang else 'no detectado'}")
                    continue # Saltar este chunk
            
            # Añadir hash para deduplicación
            chunk.metadata["content_hash"] = compute_text_hash(chunk.page_content) # Asumiendo que compute_text_hash está definida
            processed_chunks.append(chunk)
        
        if not processed_chunks:
            logger.info(f"No se generaron chunks válidos (post-filtrado) para {file_path.name}")
            # Aún así, marcar el archivo como procesado para no reintentar indefinidamente si el contenido es problemático
            doc_processor.mark_file_processed(file_path)
            return []

        doc_processor.save_chunks_to_cache(file_path, processed_chunks)
        doc_processor.mark_file_processed(file_path)
        
        return processed_chunks
            
    except Exception as e:
        logger.error(f"Error procesando {file_path.name}: {e}", exc_info=True) # Añadir exc_info para traceback
        return []


# Lock global para deduplicación
dedup_lock = Lock()

def deduplicate_chunks(chunks: List[Document], similarity_threshold=0.95) -> List[Document]:
    """Elimina chunks duplicados o muy similares con sincronización y mejor detección de similitudes"""
    with dedup_lock:
        seen_hashes = set()
        seen_ngrams = {}  # Para detectar similitud parcial
        unique_chunks = []
        
        for chunk in chunks:
            chunk_hash = chunk.metadata.get("content_hash", "")
            content = chunk.page_content.strip()
            
            # Si el hash ya existe, es un duplicado exacto
            if chunk_hash and chunk_hash in seen_hashes:
                continue
            
            # Verificar similitud por n-gramas para duplicados no exactos
            # Esto es más rápido que comparar coseno de embeddings
            content_words = content.lower().split()
            if len(content_words) > 10:  # Solo para chunks de tamaño razonable
                # Crear firma por 5-gramas
                content_ngrams = set()
                for i in range(len(content_words) - 5):
                    ngram = " ".join(content_words[i:i+5])
                    content_ngrams.add(ngram)
                
                # Verificar similitud con chunks previos
                is_similar = False
                for prev_id, prev_ngrams in seen_ngrams.items():
                    if len(content_ngrams) > 0 and len(prev_ngrams) > 0:
                        # Calcular coeficiente de Jaccard para medir similitud
                        overlap = len(content_ngrams.intersection(prev_ngrams))
                        similarity = overlap / len(content_ngrams.union(prev_ngrams))
                        
                        if similarity > similarity_threshold:
                            is_similar = True
                            break
                
                if is_similar:
                    continue
                
                # Guardar n-gramas para comparaciones futuras
                seen_ngrams[chunk.metadata.get("chunk_id", str(len(seen_ngrams)))] = content_ngrams
            
            # Marcamos como visto
            if chunk_hash:
                seen_hashes.add(chunk_hash)
                
            # Añadimos a chunks únicos
            unique_chunks.append(chunk)
    
    return unique_chunks


def batch_process_embeddings(chunks: List[Document], embeddings, batch_size: int) -> List[Tuple[Document, List[float]]]:
    """
    Procesa embeddings en batch para mejor rendimiento con manejo mejorado de errores y monitoreo
    Retorna lista de tuplas (documento, embedding)
    """
    if not chunks:
        return []
    
    # Dividir en lotes para embedding eficiente
    chunks_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    results = []
    failed_chunks = []
    
    for i, batch in enumerate(tqdm(chunks_batches, desc="Procesando embeddings en batch", unit="batch")):
        # Extraer textos
        texts = [doc.page_content for doc in batch]
        
        try:
            # Calcular embeddings para todo el lote
            batch_embeddings = embeddings.embed_documents(texts)
            
            # Validar dimensiones
            if batch_embeddings and not all(len(emb) == len(batch_embeddings[0]) for emb in batch_embeddings):
                logger.warning(f"Batch {i+1}: Dimensiones inconsistentes en embeddings, intentando recuperar...")
                # Intentar procesar individualmente los documentos problemáticos
                for j, doc in enumerate(batch):
                    try:
                        single_emb = embeddings.embed_documents([doc.page_content])[0]
                        results.append((doc, single_emb))
                    except Exception as e_single:
                        logger.error(f"Error en embedding individual: {e_single}")
                        failed_chunks.append(doc)
                continue  # Pasar al siguiente batch
            
            # Asociar cada documento con su embedding
            for j, doc in enumerate(batch):
                if j < len(batch_embeddings):  # Verificar índices
                    results.append((doc, batch_embeddings[j]))
                else:
                    logger.error(f"Índice fuera de rango: {j} >= {len(batch_embeddings)}")
                    failed_chunks.append(doc)
                    
        except Exception as e:
            logger.error(f"Error en batch {i+1}: {e}")
            # Intentar procesamiento uno por uno para recuperar los que se puedan
            for doc in batch:
                try:
                    single_emb = embeddings.embed_documents([doc.page_content])[0]
                    results.append((doc, single_emb))
                except Exception as e_single:
                    logger.error(f"Error en embedding individual: {e_single}")
                    failed_chunks.append(doc)
    
    # Informar sobre fallos
    if failed_chunks:
        logger.warning(f"No se pudieron procesar {len(failed_chunks)} chunks")
    
    return results


def parallel_embed_chunks(chunks: List[Document], embeddings, batch_size: int, workers: int) -> List[Tuple[Document, List[float]]]:
    """
    Calcula embeddings en paralelo usando múltiples workers con mejor manejo de errores
    y balanceo de carga dinámico para mayor eficiencia.
    """
    if not chunks:
        return []
    
    # Si hay pocos chunks, procesarlos directamente sin paralelismo
    if len(chunks) <= batch_size or workers <= 1:
        return batch_process_embeddings(chunks, embeddings, batch_size)
    
    # Estrategia de distribución mejorada: dividir en más bloques que workers
    # para mejor balanceo dinámico de carga
    chunks_per_worker = max(batch_size, len(chunks) // (workers * 3))
    chunk_blocks = [chunks[i:i + chunks_per_worker] for i in range(0, len(chunks), chunks_per_worker)]
    
    logger.info(f"Procesando embeddings en paralelo: {len(chunks)} chunks, {workers} workers, {len(chunk_blocks)} bloques")
    
    results = []
    failed_blocks = []
    
    # Usar ThreadPoolExecutor en lugar de ProcessPoolExecutor para compartir modelo
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(batch_process_embeddings, block, embeddings, batch_size): i
            for i, block in enumerate(chunk_blocks)
        }
        
        # Usar tqdm para monitoreo en tiempo real
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculando embeddings en paralelo"):
            block_idx = futures[future]
            try:
                batch_results = future.result()
                results.extend(batch_results)
                
                # Monitoreo periódico
                if len(results) % (batch_size * 5) == 0:
                    logger.info(f"Progreso: {len(results)}/{len(chunks)} chunks procesados ({len(results)/len(chunks)*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"Error en bloque {block_idx}: {e}")
                failed_blocks.append(block_idx)
    
    # Intentar recuperar bloques fallidos con batch_size más pequeño
    if failed_blocks:
        logger.warning(f"Reintentando {len(failed_blocks)} bloques fallidos con batch_size reducido")
        for block_idx in failed_blocks:
            if block_idx < len(chunk_blocks):  # Verificar índice
                block = chunk_blocks[block_idx]
                try:
                    # Reintentar con batch_size más pequeño
                    recovery_results = batch_process_embeddings(block, embeddings, max(1, batch_size // 4))
                    results.extend(recovery_results)
                except Exception as e:
                    logger.error(f"Error en recuperación de bloque {block_idx}: {e}")
    
    return results# En ingest.py

def insert_into_pgvector(
    doc_embed_pairs: List[Tuple[Document, List[float]]], 
    pg_conn: str, 
    collection_name: str, 
    embeddings_model_instance,
    config: IngestConfig
):
    """Inserta documentos y sus embeddings en la base de datos PGVector con mejor rendimiento y validación"""
    if not doc_embed_pairs:
        logger.info("No hay documentos para insertar en PGVector.")
        return

    should_pre_delete = getattr(config, 'reset_vector_collection', False)

    try:
        logger.info(f"Conectando a PGVector. Colección: {collection_name}. Reset colección: {should_pre_delete}")
        
        # Validar conexión antes de proceder
        import psycopg2
        try:
            # Verificar conexión a la base de datos
            conn = psycopg2.connect(pg_conn)
            conn.close()
        except Exception as e:
            logger.error(f"Error al conectar con la base de datos: {e}")
            raise
            
        vector_store = PGVector(
            embedding_function=embeddings_model_instance,
            collection_name=collection_name,
            connection_string=pg_conn,
            pre_delete_collection=should_pre_delete 
        )

        total_inserted = 0
        db_batch_size = min(config.batch_size, 100)  # Limitar tamaño de batch para evitar problemas de memoria
        num_batches = (len(doc_embed_pairs) + db_batch_size - 1) // db_batch_size

        logger.info(f"Iniciando inserción de {len(doc_embed_pairs)} embeddings en {num_batches} lotes de tamaño {db_batch_size}.")

        # Usar tqdm con unidad específica para mejor monitoreo
        for i in tqdm(range(num_batches), desc="Insertando en PGVector", unit="batch"):
            batch_pairs = doc_embed_pairs[i * db_batch_size : (i + 1) * db_batch_size]
            if not batch_pairs:
                continue

            texts_batch = [doc.page_content for doc, _ in batch_pairs]
            embeddings_batch = [emb for _, emb in batch_pairs]
            metadatas_batch = [doc.metadata for doc, _ in batch_pairs]
            
            # Validar que todos los embeddings tengan la misma dimensionalidad
            emb_lens = set(len(emb) for emb in embeddings_batch)
            if len(emb_lens) > 1:
                logger.warning(f"Dimensiones inconsistentes en embeddings: {emb_lens}")
                # Filtrar solo los que tienen la dimensión correcta (la más común)
                from collections import Counter
                most_common_dim = Counter(len(emb) for emb in embeddings_batch).most_common(1)[0][0]
                
                valid_indices = [i for i, emb in enumerate(embeddings_batch) if len(emb) == most_common_dim]
                texts_batch = [texts_batch[i] for i in valid_indices]
                embeddings_batch = [embeddings_batch[i] for i in valid_indices]
                metadatas_batch = [metadatas_batch[i] for i in valid_indices]
                
                logger.info(f"Filtrados {len(batch_pairs) - len(valid_indices)} embeddings con dimensiones incorrectas")

            try:
                # Usar transacción única para el lote entero para mejor rendimiento
                vector_store.add_embeddings(
                    texts=texts_batch,
                    embeddings=embeddings_batch,
                    metadatas=metadatas_batch
                )
                total_inserted += len(texts_batch)
            except Exception as e_batch:
                logger.error(f"Error al insertar lote {i+1}/{num_batches}: {e_batch}")
                # Intentar insertar uno por uno para salvar los que se puedan
                for j in range(len(texts_batch)):
                    try:
                        vector_store.add_embeddings(
                            texts=[texts_batch[j]],
                            embeddings=[embeddings_batch[j]],
                            metadatas=[metadatas_batch[j]]
                        )
                        total_inserted += 1
                    except Exception:
                        # Simplemente seguimos adelante
                        pass
        
        logger.info(f"Insertados {total_inserted} documentos en la colección '{collection_name}'.")
        
    except Exception as e_main:
        logger.error(f"Error general durante la inserción en PGVector: {e_main}", exc_info=True)
        raise

def improve_chunk_quality(chunks: List[Document]) -> List[Document]:
    """
    Mejora la calidad de los chunks para RAG mediante:
    - Enriquecimiento de metadatos
    - Limpieza y normalización
    - Filtrado de chunks de baja calidad
    - Mejor manejo de referencias cruzadas
    """
    if not chunks:
        return []
    
    improved_chunks = []
    
    # Agrupar chunks por fuente para facilitar referencias cruzadas
    chunks_by_source = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        if source not in chunks_by_source:
            chunks_by_source[source] = []
        chunks_by_source[source].append(chunk)
    
    for i, chunk in enumerate(chunks):
        # Filtrar chunks extremadamente cortos
        if len(chunk.page_content.strip()) < 50:
            continue
        
        # Normalización de contenido
        content = chunk.page_content
        
        # Eliminar múltiples saltos de línea y espacios extra
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        content = content.strip()
        
        # Eliminar títulos duplicados al inicio del chunk
        lines = content.split('\n')
        if len(lines) > 2:
            if lines[0].strip() and (lines[0].strip() == lines[1].strip() or 
                                    lines[0].strip() in lines[1] or lines[1].strip() in lines[0]):
                content = '\n'.join(lines[1:])
        
        # Análisis de riqueza de contenido
        words = content.split()
        unique_words = set(w.lower() for w in words if len(w) > 3)
        
        # Filtrar chunks con muy poco contenido único
        if len(unique_words) < 10:
            continue
        
        # Obtener metadatos básicos
        source = chunk.metadata.get("source", "unknown")
        chunk_index = chunk.metadata.get("chunk_index", i)
        chunk_total = chunk.metadata.get("chunk_total", len(chunks))
        
        # Extraer entidades potenciales (palabras que comienzan con mayúscula después de puntuación)
        entities = set()
        words_with_context = re.findall(r'[.!?]\s+([A-Z][a-zA-Z]*)', content)
        words_with_context.extend(re.findall(r'\b([A-Z][a-zA-Z]*\s+[A-Z][a-zA-Z]*)\b', content))
        if words_with_context:
            entities = set(words_with_context)
        
        # Análisis de frecuencia de palabras para keywords
        word_freq = {}
        for word in words:
            word = word.lower()
            if len(word) > 3 and word.isalnum():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Stopwords en español e inglés
        stopwords = {'este', 'esta', 'estos', 'estas', 'aquellos', 'aquellas', 'para', 'como', 'desde', 'cuando', 'donde', 'porque', 'como', 'tambien', 'pero', 'the', 'and', 'that', 'this', 'with', 'from', 'have', 'for', 'not', 'was', 'were', 'are', 'they', 'their', 'them'}
        keywords = [w for w, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True) 
                  if w not in stopwords][:5]
        
        # Detectar secciones importantes basadas en patrones
        section_match = re.search(r"(?i)(sección|capítulo|título|parte)\s+[\dIVXLC]+[:\-]?\s*(.+?)(?=\n|$)", content[:300])
        section_name = section_match.group(2).strip() if section_match else None
        
        # Detectar fechas en el texto
        dates = re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?\d{2,4}\b', content)
        
        # Enriquecer metadatos
        chunk.metadata.update({
            "content_quality": len(unique_words) / max(1, len(words)),
            "processed_date": datetime.now().isoformat(),
            "word_count": len(words),
            "unique_word_count": len(unique_words),
            "entities": list(entities) if entities else None,
            "keywords": keywords,
            "position_percent": chunk_index / max(1, chunk_total - 1) if chunk_total > 1 else 0.5,
            "is_first": chunk_index == 0,
            "is_last": chunk_index == chunk_total - 1,
            "section": section_name,
            "dates": dates if dates else None
        })
        
        # Crear referencias a chunks vecinos para mejor navegación y contexto
        source_chunks = chunks_by_source.get(source, [])
        if len(source_chunks) > 1:
            if chunk_index > 0:
                chunk.metadata["prev_chunk_ref"] = f"{source}:{chunk_index-1}"
            if chunk_index < chunk_total - 1:
                chunk.metadata["next_chunk_ref"] = f"{source}:{chunk_index+1}"
        
        # Analizar contexto semántico
        if len(content) > 100:
            first_paragraph = content.split('\n\n')[0] if '\n\n' in content else content[:min(300, len(content))]
            chunk.metadata["summary"] = first_paragraph.strip()
        
        chunk.page_content = content
        improved_chunks.append(chunk)
    
    return improved_chunks


def optimize_vector_search_config(pg_conn: str, collection_name: str):
    """
    Optimiza la configuración de búsqueda vectorial en PGVector
    para mejorar el rendimiento y la relevancia.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(pg_conn)
        cursor = conn.cursor()
        
        # Verificar existencia de índices
        safe_name = sanitize_pg_identifier(collection_name)
        cursor.execute(f"""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = '{safe_name}'
        """)
        existing_indexes = cursor.fetchall()
        index_names = [idx[0] for idx in existing_indexes]
        
        # Crear índice GIN para búsquedas rápidas en metadatos
        metadata_index_name = f"{safe_name}_metadata_gin_idx"
        if metadata_index_name not in index_names:
            logger.info(f"Creando índice GIN para metadatos en {collection_name}")
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {metadata_index_name}
                ON public.{safe_name} USING GIN (metadata);
            """)
        
        # Optimizar índice vectorial existente para mejorar eficiencia
        cursor.execute(f"""
            ALTER INDEX public.{safe_name}_langchain_vector_idx
            SET (lists_growth_with_size = yes, intermediate_compression_threshold = 1024);
        """)
        
        # Crear vista materializada para consultas frecuentes
        view_name = f"{safe_name}_search_view"
        cursor.execute(f"""
            SELECT relname FROM pg_class 
            WHERE relkind = 'm' AND relname = '{view_name}'
        """)
        if not cursor.fetchone():
            logger.info(f"Creando vista materializada para búsqueda en {collection_name}")
            cursor.execute(f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS public.{view_name} AS
                SELECT uuid, document, embedding, metadata, 
                       metadata->>'source' as source,
                       metadata->>'language' as language,
                       metadata->>'doc_type' as doc_type,
                       metadata->>'chunk_id' as chunk_id
                FROM public.{safe_name};
                
                CREATE INDEX IF NOT EXISTS {view_name}_source_idx
                ON public.{view_name} (source);
                
                CREATE INDEX IF NOT EXISTS {view_name}_language_idx
                ON public.{view_name} (language);
                
                CREATE INDEX IF NOT EXISTS {view_name}_doc_type_idx
                ON public.{view_name} (doc_type);
            """)
        
        # Configurar parámetros para optimización de búsqueda vectorial
        cursor.execute(f"""
            ALTER TABLE public.{safe_name}
            SET (autovacuum_vacuum_scale_factor = 0.05, autovacuum_analyze_scale_factor = 0.02);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Optimización de búsqueda vectorial completada para {collection_name}")
        
    except Exception as e:
        logger.error(f"Error al optimizar búsqueda vectorial: {e}")


def monitor_vector_db_health(pg_conn: str, collection_name: str):
    """
    Monitora la salud de la base de datos vectorial y reporta estadísticas.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(pg_conn)
        cursor = conn.cursor()
        
        # Contar total de documentos
        cursor.execute(f"""
            SELECT COUNT(*) FROM public.{safe_name};
        """)
        total_docs = cursor.fetchone()[0]
        
        # Estadísticas de metadatos
        cursor.execute(f"""
            SELECT 
                COUNT(DISTINCT metadata->>'source') as unique_sources,
                COUNT(DISTINCT metadata->>'language') as unique_languages,
                COUNT(DISTINCT metadata->>'doc_type') as unique_doc_types
            FROM public.{safe_name};
        """)
        metadata_stats = cursor.fetchone()
        
        # Estadísticas de embeddings
        cursor.execute(f"""
            SELECT AVG(array_length(embedding, 1)) as avg_dimensions,
                   MIN(array_length(embedding, 1)) as min_dimensions,
                   MAX(array_length(embedding, 1)) as max_dimensions
            FROM public.{safe_name};
        """)
        dimension_stats = cursor.fetchone()
        
        # Estadísticas de tamaño y calidad
        cursor.execute(f"""
            SELECT 
                AVG((metadata->>'word_count')::float) as avg_words,
                AVG((metadata->>'content_quality')::float) as avg_quality,
                MIN((metadata->>'content_quality')::float) as min_quality,
                MAX((metadata->>'content_quality')::float) as max_quality
            FROM public.{safe_name}
            WHERE metadata->>'word_count' IS NOT NULL;
        """)
        content_stats = cursor.fetchone()
        
        # Información de fragmentación
        cursor.execute(f"""
            SELECT 
                pg_size_pretty(pg_total_relation_size('public.{safe_name}')) as total_size,
                pg_size_pretty(pg_table_size('public.{safe_name}')) as table_size,
                pg_size_pretty(pg_indexes_size('public.{safe_name}')) as index_size
            FROM pg_catalog.pg_class c
            LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = '{safe_name}' AND n.nspname = 'public';
        """)
        size_stats = cursor.fetchone()
        
        # Reportar estadísticas
        logger.info(f"Estadísticas de la base de datos vectorial ({collection_name}):")
        logger.info(f"  - Total de documentos: {total_docs}")
        logger.info(f"  - Fuentes únicas: {metadata_stats[0]}")
        logger.info(f"  - Idiomas: {metadata_stats[1]}")
        logger.info(f"  - Tipos de documentos: {metadata_stats[2]}")
        logger.info(f"  - Dimensiones de embeddings: avg={dimension_stats[0]}, min={dimension_stats[1]}, max={dimension_stats[2]}")
        
        if content_stats[0]:
            logger.info(f"  - Promedio de palabras por chunk: {content_stats[0]:.2f}")
            logger.info(f"  - Calidad promedio de contenido: {content_stats[1]:.2f} (min={content_stats[2]:.2f}, max={content_stats[3]:.2f})")
        
        if size_stats:
            logger.info(f"  - Tamaño total: {size_stats[0]}")
            logger.info(f"  - Tamaño tabla: {size_stats[1]}")
            logger.info(f"  - Tamaño índices: {size_stats[2]}")
        
        # Alertas
        if dimension_stats[1] != dimension_stats[2]:
            logger.warning(f"¡ALERTA! Dimensiones inconsistentes en los embeddings: min={dimension_stats[1]}, max={dimension_stats[2]}")
        
        if content_stats and content_stats[2] < 0.2:
            logger.warning(f"¡ALERTA! Chunks de baja calidad detectados (mínimo: {content_stats[2]:.2f})")
        
        # Recomendaciones
        cursor.execute(f"""
            SELECT reltuples::bigint, relpages::bigint
            FROM pg_class
            WHERE relname = '{safe_name}';
        """)
        rel_stats = cursor.fetchone()
        
        if rel_stats and rel_stats[0] > 0 and rel_stats[0] / max(1, rel_stats[1]) < 10:
            logger.info("Recomendación: Considerar realizar VACUUM FULL para optimizar espacio")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error al monitorear la base de datos vectorial: {e}")


def search_quality_validation(pg_conn: str, collection_name: str, embeddings, test_queries: List[str] = None):
    """
    Valida la calidad de búsqueda con consultas de prueba para ajustar parámetros.
    """
    if not test_queries:
        test_queries = [
            "¿Cuáles son los principales conceptos del documento?",
            "Resumen del contenido",
            "Aspectos más importantes",
            "Información técnica específica",
            "Detalles legales relevantes"
        ]
    
    try:
        vector_store = PGVector(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_string=pg_conn
        )
        
        logger.info("Realizando validación de calidad de búsqueda...")
        
        results = []
        for query in test_queries:
            logger.info(f"Consultando: '{query}'")
            
            # Probar diferentes configuraciones de búsqueda
            similarity_search_results = vector_store.similarity_search(query, k=3)
            mmr_results = vector_store.max_marginal_relevance_search(query, k=3, fetch_k=10)
            
            # Evaluar diversidad
            similarity_content = [doc.page_content[:100] for doc in similarity_search_results]
            mmr_content = [doc.page_content[:100] for doc in mmr_results]
            
            unique_sources_similarity = len(set([doc.metadata.get('source', '') for doc in similarity_search_results]))
            unique_sources_mmr = len(set([doc.metadata.get('source', '') for doc in mmr_results]))
            
            results.append({
                'query': query,
                'similarity_sources': unique_sources_similarity,
                'mmr_sources': unique_sources_mmr,
                'similarity_vs_mmr_overlap': len(set(similarity_content) & set(mmr_content))
            })
        
        # Analizar resultados
        avg_similarity_sources = sum(r['similarity_sources'] for r in results) / len(results)
        avg_mmr_sources = sum(r['mmr_sources'] for r in results) / len(results)
        avg_overlap = sum(r['similarity_vs_mmr_overlap'] for r in results) / len(results)
        
        logger.info(f"Resultados de validación:")
        logger.info(f"  - Promedio de fuentes por consulta (similarity): {avg_similarity_sources:.2f}")
        logger.info(f"  - Promedio de fuentes por consulta (MMR): {avg_mmr_sources:.2f}")
        logger.info(f"  - Superposición media entre métodos: {avg_overlap:.2f}")
        
        # Recomendaciones basadas en resultados
        if avg_overlap < 1.5:
            logger.info("Recomendación: La diversidad entre métodos es alta. MMR proporciona resultados significativamente diferentes.")
        else:
            logger.info("Recomendación: Considerar aumentar la diversidad con MMR ajustando el parámetro lambda.")
        
        if avg_mmr_sources > avg_similarity_sources * 1.2:
            logger.info("Recomendación: MMR muestra mayor diversidad de fuentes. Recomendado para consultas exploratorias.")
        else:
            logger.info("Recomendación: La búsqueda por similitud y MMR muestran diversidad comparable de fuentes.")
        
        return results
        
    except Exception as e:
        logger.error(f"Error en validación de calidad: {e}")
        return []


def get_hybrid_retriever(pg_conn: str, collection_name: str, embeddings, use_compression: bool = False):
    """
    Crea un retriever híbrido que combina búsqueda vectorial y por keywords
    para mejorar la calidad y relevancia de resultados.
    """
    try:
        # Crear almacén vectorial
        vector_store = PGVector(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_string=pg_conn
        )
        
        # Vector retriever base
        vector_retriever = vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
        )
        
        # Si se requiere compresión para chunks extensos
        if use_compression:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import LLMChainExtractor
            
            # Nota: Esto requiere configurar un LLM para la compresión
            # llm = ... (configurar OpenAI, Anthropic, etc.)
            # compressor = LLMChainExtractor.from_llm(llm)
            # compressed_retriever = ContextualCompressionRetriever(
            #     base_compressor=compressor,
            #     base_retriever=vector_retriever
            # )
            # return compressed_retriever
        
        # Crear un retriever de búsqueda por keywords usando metadatos
        # Esto es una simplificación, para implementación real se necesitaría desarrollar esto
        
        # Combinar ambos retrievers en una función
        def hybrid_retrieve(query: str, **kwargs):
            k = kwargs.get("k", 5)
            
            # Vector search
            vector_docs = vector_retriever.get_relevant_documents(query)
            
            # Extraer keywords usando alguna biblioteca como KeyBERT o métodos simples
            keywords = set(re.findall(r'\b\w{4,}\b', query.lower()))
            
            # Aumentar resultados con una re-clasificación basada en coincidencia de keywords
            for doc in vector_docs:
                doc_keywords = set(re.findall(r'\b\w{4,}\b', doc.page_content.lower()))
                keyword_match_score = len(keywords.intersection(doc_keywords)) / max(1, len(keywords))
                doc.metadata["keyword_score"] = keyword_match_score
            
            # Ordenar por score combinado (ya ordenados por vector + boost por keywords)
            reranked_docs = sorted(
                vector_docs, 
                key=lambda d: d.metadata.get("keyword_score", 0), 
                reverse=True
            )
            
            return reranked_docs[:k]
        
        # Crear un objeto RetrieverLike que encapsule nuestra función híbrida
        class HybridRetriever:
            def __init__(self, retriever_func):
                self.retrieve = retriever_func
                
            def get_relevant_documents(self, query: str, **kwargs):
                return self.retrieve(query, **kwargs)
                
        return HybridRetriever(hybrid_retrieve)
        
    except Exception as e:
        logger.error(f"Error al crear retriever híbrido: {e}")
        # Fallback al retriever básico
        return vector_store.as_retriever()


def main():
    """Función principal del proceso de ingesta optimizada para rendimiento y calidad"""
    try:
        # Cargar configuración
        config, pg_conn = load_config()
        logger.info(f"ARGUMENTOS RECIBIDOS:" ,sys.argv)
        print("🧪 ARGUMENTOS RECIBIDOS:", sys.argv)
        logger.info(f"Configuración de ingesta cargada: {vars(config)}")

        # Verificar directorio de documentos
        if not config.docs_dir.exists():
            raise FileNotFoundError(f"El directorio de documentos '{config.docs_dir}' no existe")
        if not config.docs_dir.is_dir():
            raise NotADirectoryError(f"La ruta de documentos '{config.docs_dir}' no es un directorio")

        # Iniciar medición de rendimiento
        overall_start_time = time.time()
        stage_times = {}
        memory_usage = []

        # Monitorear memoria inicial
        memory_usage.append(psutil.Process().memory_info().rss / 1024**2)
        logger.info(f"Memoria inicial: {memory_usage[-1]:.2f} MB")

        try:
            # 1. Cargar modelo de embeddings
            model_load_start = time.time()
            logger.info(f"Cargando modelo de embeddings: {config.embed_model}...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs = {
                "device": device
            }
            encode_kwargs = {
                "normalize_embeddings": True,
                "batch_size": config.batch_size
            }
            
            embeddings = HuggingFaceEmbeddings(
                model_name=config.embed_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            stage_times["load_model"] = time.time() - model_load_start
            logger.info(f"Modelo de embeddings cargado en {stage_times['load_model']:.2f}s")
            logger.info(f"Dimensión de embeddings: {embeddings.client.get_sentence_embedding_dimension()}")
            
            # Monitorear memoria después de cargar modelo
            memory_usage.append(psutil.Process().memory_info().rss / 1024**2)
            logger.info(f"Memoria después de cargar modelo: {memory_usage[-1]:.2f} MB")

            # 2. Inicializar DocumentProcessor y obtener archivos
            doc_processor = DocumentProcessor(config)
            file_scan_start = time.time()
            files_to_process_info = doc_processor.get_files_to_process()
            stage_times["file_scan"] = time.time() - file_scan_start

            total_files_found = len(files_to_process_info)
            files_needing_processing = [(fp, needs_proc) for fp, needs_proc in files_to_process_info if needs_proc]
            num_files_to_process = len(files_needing_processing)
            
            total_size_mb = sum(f.stat().st_size for f, _ in files_to_process_info) / (1024 * 1024)
            processing_size_mb = sum(f.stat().st_size for f, _ in files_needing_processing) / (1024 * 1024)

            logger.info(f"Escaneo de archivos completado en {stage_times['file_scan']:.2f}s")
            logger.info(f"Archivos encontrados: {total_files_found} (Total: {total_size_mb:.2f} MB)")
            logger.info(f"Archivos a procesar: {num_files_to_process} (Tamaño: {processing_size_mb:.2f} MB)")

            if num_files_to_process == 0:
                logger.info("No hay archivos nuevos para procesar")
                return

            # 3. Procesar archivos en paralelo
            processing_start = time.time()
            all_chunks = []
            
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                futures = []
                for file_path, needs_processing in files_needing_processing:
                    if needs_processing:
                        futures.append(
                            executor.submit(process_file, file_path, config, doc_processor)
                        )
                
                # Procesar resultados con barra de progreso
                for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando archivos"):
                    try:
                        chunks = future.result()
                        if chunks:
                            all_chunks.extend(chunks)
                    except Exception as e:
                        logger.error(f"Error procesando archivo: {e}")
                        continue

            stage_times["file_processing"] = time.time() - processing_start
            logger.info(f"Procesamiento de archivos completado en {stage_times['file_processing']:.2f}s")
            logger.info(f"Chunks generados: {len(all_chunks)}")

            # Monitorear memoria después de procesar archivos
            memory_usage.append(psutil.Process().memory_info().rss / 1024**2)
            logger.info(f"Memoria después de procesar archivos: {memory_usage[-1]:.2f} MB")

            if not all_chunks:
                logger.warning("No se generaron chunks. Verificar archivos y configuración.")
                return

            # 4. Mejorar calidad de chunks
            quality_start = time.time()
            improved_chunks = improve_chunk_quality(all_chunks)
            stage_times["quality_improvement"] = time.time() - quality_start
            logger.info(f"Mejora de calidad completada en {stage_times['quality_improvement']:.2f}s")
            logger.info(f"Chunks después de mejora: {len(improved_chunks)}")

            # 5. Deduplicar chunks si está habilitado
            if config.deduplication:
                dedup_start = time.time()
                final_chunks = deduplicate_chunks(improved_chunks, config.similarity_threshold)
                stage_times["deduplication"] = time.time() - dedup_start
                logger.info(f"Deduplicación completada en {stage_times['deduplication']:.2f}s")
                logger.info(f"Chunks después de deduplicación: {len(final_chunks)}")
            else:
                final_chunks = improved_chunks

            # 6. Generar embeddings
            embedding_start_time = time.time()
            
            # Usar parallel_embed_chunks o batch_process_embeddings según configuración
            if len(final_chunks) > encode_kwargs["batch_size"] * 2 and config.max_embed_workers > 1:
                doc_embed_pairs = parallel_embed_chunks(
                    final_chunks,
                    embeddings,
                    encode_kwargs["batch_size"],
                    config.max_embed_workers
                )
            else:
                logger.info("Procesando embeddings en un solo hilo")
                doc_embed_pairs = batch_process_embeddings(
                    final_chunks,
                    embeddings,
                    encode_kwargs["batch_size"]
                )
            
            stage_times["embedding_generation"] = time.time() - embedding_start_time
            logger.info(f"Generación de embeddings completada en {stage_times['embedding_generation']:.2f}s")
            logger.info(f"Pares Doc-Embedding generados: {len(doc_embed_pairs)}")

            # Monitorear memoria después de generar embeddings
            memory_usage.append(psutil.Process().memory_info().rss / 1024**2)
            logger.info(f"Memoria después de generar embeddings: {memory_usage[-1]:.2f} MB")

            if not doc_embed_pairs:
                logger.error("No se generaron embeddings. Verificar modelo y datos.")
                return

            # 7. Inserción en PGVector
            db_insert_start_time = time.time()
            logger.info(f"Insertando {len(doc_embed_pairs)} embeddings en PGVector...")
            print(f"🧪 INSERTANDO en colección: {config.collection_name}")
            print(f"🧪 RESET COLLECTION: {config.reset_vector_collection}")
            insert_into_pgvector(doc_embed_pairs, pg_conn, config.collection_name, embeddings, config)
            stage_times["db_insertion"] = time.time() - db_insert_start_time
            logger.info(f"Inserción en base de datos completada en {stage_times['db_insertion']:.2f}s")

            # 8. Optimización y validación
            optimization_start = time.time()
            
            # Optimizar configuración de búsqueda
            optimize_vector_search_config(pg_conn, config.collection_name)
            
            # Validar calidad de búsqueda
            search_quality_validation(pg_conn, config.collection_name, embeddings)
            
            stage_times["optimization"] = time.time() - optimization_start
            logger.info(f"Optimización y validación completadas en {stage_times['optimization']:.2f}s")

            # 9. Monitoreo final
            total_time = time.time() - overall_start_time
            memory_usage.append(psutil.Process().memory_info().rss / 1024**2)
            
            # Generar reporte de rendimiento
            performance_report = {
                "total_time": total_time,
                "stage_times": stage_times,
                "memory_usage": {
                    "initial": memory_usage[0],
                    "after_model": memory_usage[1],
                    "after_processing": memory_usage[2],
                    "after_embeddings": memory_usage[3],
                    "final": memory_usage[4]
                },
                "files_processed": num_files_to_process,
                "total_chunks": len(all_chunks),
                "final_chunks": len(final_chunks),
                "embeddings_generated": len(doc_embed_pairs)
            }
            
            logger.info("Reporte de rendimiento:")
            logger.info(json.dumps(performance_report, indent=2))

            # 10. Limpieza
            if config.reset_vector_collection:
                logger.info("Limpieza de recursos...")
                # Aquí se podrían agregar tareas de limpieza adicionales

        except Exception as e:
            logger.error(f"Error durante el procesamiento: {e}")
            logger.error(traceback.format_exc())
            raise

    except Exception as e:
        logger.critical(f"Error crítico en el proceso de ingesta: {e}")
        logger.critical(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
