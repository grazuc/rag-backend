# api/main.py
import os
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from functools import lru_cache
import json
import asyncio
from contextlib import asynccontextmanager
import nest_asyncio
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import sys
import multiprocessing
import psutil
import torch
import traceback
import psycopg2
import warnings
from langdetect import detect, LangDetectException
from asyncio import TimeoutError
import async_timeout

from fastapi import FastAPI, HTTPException, Depends, Request, status, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE
from pydantic import BaseModel, Field, ConfigDict, validator, root_validator, model_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores.pgvector import PGVector
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from .translation import translate_text

from transformers import pipeline
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

import shutil
import subprocess
from supabase import create_client, Client  # pip install supabase

# MODIFICADO: Configuración de logging mejorada
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MODIFICADO: Configuración mejorada con validación
class Settings(BaseSettings):
    pg_conn: str
    deepseek_api_key: str
    hf_token: Optional[str] = None
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_model: str = "intfloat/multilingual-e5-base"
    llm_model: str = "deepseek-chat"
    collection_name: str = "manual_e5_multi"
    initial_retrieval_k: int = Field(default=20, ge=1, le=100)
    final_docs_count: int = Field(default=4, ge=1, le=20)
    api_key: Optional[str] = None
    cache_ttl_hours: int = Field(default=6, ge=1, le=24)
    environment: str = Field(default="production", pattern="^(production|development|testing)$")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    query_rewrite_strategy: str = Field(default="original_only", pattern="^(original_only|rewrite_only|concatenate)$")
    max_retries: int = Field(default=3, ge=1, le=5)
    timeout_seconds: int = Field(default=30, ge=5, le=120)
    batch_size: int = Field(default=16, ge=1, le=64)
    max_concurrent_requests: int = Field(default=100, ge=10, le=1000)
    query_timeout: int = Field(default=60, ge=30, le=300)  # Main query timeout in seconds
    translation_timeout: int = Field(default=10, ge=5, le=30)  # Translation timeout in seconds
    retrieval_timeout: int = Field(default=20, ge=10, le=60)  # Document retrieval timeout in seconds

    @model_validator(mode='after')
    def validate_settings(self):
        if self.environment == "production" and self.cors_origins == ["*"]:
            logger.warning("Production environment with wildcard CORS origins is not recommended")
        return self

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

# MODIFICADO: Funciones cacheadas para componentes
@lru_cache(maxsize=1)
def get_llm():
    """Inicializa y retorna el modelo LLM"""
    try:
        if settings.llm_model == "deepseek-chat":
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            return ChatDeepSeek(
                api_key=settings.deepseek_api_key,
                model_name="deepseek-chat",
                temperature=0.3,
                max_tokens=1000,
                callback_manager=callback_manager
            )
        else:
            raise ValueError(f"Modelo LLM no soportado: {settings.llm_model}")
    except Exception as e:
        logger.error(f"Error al inicializar LLM: {e}")
        raise RuntimeError(f"No se pudo inicializar el modelo LLM: {e}")

@lru_cache(maxsize=1)
def get_qa_prompt():
    """Retorna el prompt para el QA chain"""
    return PromptTemplate(
        template="""Eres un asistente experto que ayuda a responder preguntas basándose en la documentación proporcionada.
        
Utiliza SOLO la información del siguiente contexto para responder la pregunta. Si la información no es suficiente para responder,
indica claramente que no puedes responder basándote solo en el contexto proporcionado.

Contexto:
{context}

Pregunta:
{question}

Respuesta:""",
        input_variables=["context", "question"]
    )

@lru_cache(maxsize=1)
def get_qa_chain():
    """Inicializa y retorna el QA chain"""
    try:
        return LLMChain(
            llm=get_llm(),
            prompt=get_qa_prompt(),
            verbose=True
        )
    except Exception as e:
        logger.error(f"Error al inicializar QA chain: {e}")
        raise RuntimeError(f"No se pudo inicializar el QA chain: {e}")

@lru_cache(maxsize=1)
async def get_deepseek_query_rewriter_llm():
    """Inicializa y retorna el LLM para reescritura de consultas"""
    try:
        return ChatDeepSeek(
            api_key=settings.deepseek_api_key,
            model_name="deepseek-chat",
            temperature=0.2,
            max_tokens=200
        )
    except Exception as e:
        logger.warning(f"Error al inicializar LLM para reescritura: {e}")
        return None

# MODIFICADO: Función para logging de métricas
async def log_query_metrics(query: str, num_sources: int, execution_time: float, language: Optional[str] = None):
    """Registra métricas de la consulta para análisis"""
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "query_length": len(query),
            "num_sources": num_sources,
            "execution_time": execution_time,
            "language": language or "unknown"
        }
        logger.info(f"Métricas de consulta: {json.dumps(metrics)}")
    except Exception as e:
        logger.error(f"Error al registrar métricas: {e}")

# MODIFICADO: Función para fusionar información de documentos
def merge_document_info(docs: List[Document]) -> List[Document]:
    """Fusiona documentos consecutivos del mismo origen para mejor contexto"""
    if not docs:
        return []
        
    merged = []
    current_doc = None
    
    for doc in docs:
        if not current_doc:
            current_doc = doc
            continue
            
        # Si los documentos son consecutivos y del mismo origen, fusionarlos
        if (current_doc.metadata.get("source") == doc.metadata.get("source") and
            abs(current_doc.metadata.get("chunk_index", 0) - doc.metadata.get("chunk_index", 0)) == 1):
            
            current_doc.page_content += "\n\n" + doc.page_content
            # Actualizar metadatos
            if "page_range" in current_doc.metadata:
                current_doc.metadata["page_range"] = f"{current_doc.metadata['page_range']}-{doc.metadata.get('page', '')}"
            else:
                current_doc.metadata["page_range"] = f"{current_doc.metadata.get('page', '')}-{doc.metadata.get('page', '')}"
        else:
            merged.append(current_doc)
            current_doc = doc
    
    if current_doc:
        merged.append(current_doc)
    
    return merged

# MODIFICADO: Gestión de lifespan mejorada
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando aplicación API RAG")
    load_dotenv()
    
    # Inicializar recursos
    try:
        # Verificar conexión a la base de datos
        conn = psycopg2.connect(settings.pg_conn)
        conn.close()
        logger.info("Conexión a base de datos verificada")
        
        # Verificar GPU si está disponible
        if torch.cuda.is_available():
            logger.info(f"GPU disponible: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            logger.info("No se detectó GPU, usando CPU")
            
        # Verificar memoria del sistema
        memory = psutil.virtual_memory()
        logger.info(f"Memoria total: {memory.total / 1024**3:.2f} GB")
        logger.info(f"Memoria disponible: {memory.available / 1024**3:.2f} GB")
        
    except Exception as e:
        logger.error(f"Error durante la inicialización: {e}")
        raise
    
    yield
    
    logger.info("Cerrando recursos y conexiones")
    # Limpiar cachés
    get_embeddings.cache_clear()
    get_vectorstore.cache_clear()
    get_retriever.cache_clear()
    get_qa_chain.cache_clear()
    get_llm.cache_clear()

# MODIFICADO: Middleware mejorado con métricas y seguridad
class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0
        self.start_time = time.time()
        
    async def dispatch(self, request: Request, call_next):
        self.request_count += 1
        start_time = time.time()
        
        try:
            # Validar headers de seguridad
            if request.headers.get("X-Forwarded-For"):
                logger.warning(f"Request con X-Forwarded-For: {request.headers['X-Forwarded-For']}")
            
            # Procesar request
            response = await call_next(request)
            
            # Calcular métricas
            duration = time.time() - start_time
            self.total_response_time += duration
            
            # Logging mejorado
            logger.info(
                f"Request: {request.method} {request.url.path} | "
                f"Status: {response.status_code} | "
                f"Duration: {duration:.4f}s | "
                f"Client: {request.client.host if request.client else 'Unknown'}"
            )
            
            return response
            
        except Exception as e:
            self.error_count += 1
            duration = time.time() - start_time
            
            logger.error(
                f"Error en request: {request.method} {request.url.path} | "
                f"Error: {str(e)} | "
                f"Duration: {duration:.4f}s | "
                f"Client: {request.client.host if request.client else 'Unknown'}"
            )
            
            if isinstance(e, HTTPException):
                raise e
                
            return JSONResponse(
                status_code=500,
                content={"detail": "Error interno del servidor"}
            )

# MODIFICADO: Rate limiter mejorado
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = asyncio.Lock()
        
    async def check_rate_limit(self, request: Request):
        async with self.lock:
            client_id = request.headers.get("X-API-Key") or request.client.host
            now = time.time()
            
            # Limpiar requests antiguos
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < 60
            ]
            
            # Verificar límite
            if len(self.requests[client_id]) >= self.requests_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail="Demasiadas peticiones. Por favor, intente más tarde."
                )
            
            # Registrar nueva request
            self.requests[client_id].append(now)
            return True

# MODIFICADO: Inicialización de la aplicación
try:
    settings = Settings()
except Exception as e:
    logger.critical(f"Error al cargar configuración: {e}")
    raise

app = FastAPI(
    title="RAG API",
    description="API para consultas RAG sobre documentación técnica",
    version="1.0.0",
    lifespan=lifespan
)

# MODIFICADO: Middleware y seguridad
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(MetricsMiddleware)

# MODIFICADO: Inicialización de componentes con mejor manejo de errores
@lru_cache(maxsize=1)
def get_embeddings():
    """Inicializa y retorna el modelo de embeddings con mejor manejo de errores"""
    logger.info("Inicializando modelo de embeddings")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {
            "device": device
        }
        encode_kwargs = {
            "normalize_embeddings": True,
            "batch_size": settings.batch_size
        }
        
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    except Exception as e:
        logger.error(f"Error al inicializar embeddings: {e}")
        raise RuntimeError(f"No se pudo inicializar el modelo de embeddings: {e}")

@lru_cache(maxsize=1)
def get_vectorstore():
    """Inicializa el vectorstore con reintentos y validación"""
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _init_vectorstore():
        try:
            # Ignorar warnings de deprecación por ahora ya que seguimos usando la versión community
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
                
                return PGVector(
                    embedding_function=get_embeddings(),
                    collection_name=settings.collection_name,
                    connection_string=settings.pg_conn
                )
        except Exception as e:
            logger.error(f"Error al inicializar vectorstore: {e}")
            raise
            
    return _init_vectorstore()

@lru_cache(maxsize=1)
def get_retriever():
    """Inicializa el retriever optimizado con procesamiento por lotes"""
    logger.info("Inicializando retriever con optimización de lotes")
    try:
        vectorstore = get_vectorstore()
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": settings.initial_retrieval_k,
                "filter": {},
                "batch_size": settings.batch_size
            }
        )
        
        # Optimización del reranker
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cross_enc = HuggingFaceCrossEncoder(
            model_name=settings.reranker_model,
            model_kwargs={"device": device}
        )
        reranker = CrossEncoderReranker(
            model=cross_enc, 
            top_n=settings.final_docs_count
        )
        
        return ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )
    except Exception as e:
        logger.error(f"Error al inicializar retriever: {e}")
        raise RuntimeError(f"No se pudo inicializar el retriever: {e}")

# MODIFICADO: Modelos Pydantic para la API
class SourceInfo(BaseModel):
    source: str
    title: Optional[str] = None
    page: Optional[int] = None
    page_range: Optional[str] = None
    relevance_score: Optional[float] = None
    content_preview: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = None
    detailed_feedback: bool = False
    userId: Optional[str] = None
    documentId: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("La consulta no puede estar vacía")
        if len(v) > 1000:
            raise ValueError("La consulta es demasiado larga (máximo 1000 caracteres)")
        return v.strip()

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    execution_time: float
    feedback_url: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

# MODIFICADO: Funciones de autenticación y rate limiting
def verify_api_key_if_configured(request: Request) -> bool:
    """Verifica la API key si está configurada en settings"""
    if not settings.api_key:
        return True
        
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key requerida"
        )
        
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida"
        )
        
    return True

# MODIFICADO: Inicialización del rate limiter
rate_limiter = RateLimiter(requests_per_minute=settings.max_concurrent_requests)

# MODIFICADO: Implementación de caché para consultas
class AsyncQueryCache:
    def __init__(self, ttl_hours: int = 6):
        self.cache = {}
        self.ttl = timedelta(hours=ttl_hours)
        self._lock = asyncio.Lock()

    async def get(self, query: str) -> Optional[QueryResponse]:
        """Obtiene una respuesta cacheada si existe y no ha expirado"""
        async with self._lock:
            # Normalizar la consulta para mejor matching
            normalized_query = query.lower().strip()
            
            # Limpiar entradas expiradas
            now = datetime.now()
            self.cache = {
                k: v for k, v in self.cache.items()
                if now - v["timestamp"] < self.ttl
            }
            
            # Buscar coincidencia exacta o similar
            for cached_query, cache_data in self.cache.items():
                if (cached_query == normalized_query or 
                    (len(normalized_query) > 10 and 
                     self._similarity_score(cached_query, normalized_query) > 0.9)):
                    return cache_data["response"]
            
            return None

    async def set(self, query: str, response: QueryResponse) -> None:
        """Almacena una respuesta en caché"""
        async with self._lock:
            normalized_query = query.lower().strip()
            self.cache[normalized_query] = {
                "response": response,
                "timestamp": datetime.now()
            }
            
            # Mantener el tamaño del caché bajo control
            if len(self.cache) > 1000:  # Límite arbitrario
                # Eliminar las entradas más antiguas
                sorted_items = sorted(
                    self.cache.items(),
                    key=lambda x: x[1]["timestamp"]
                )
                self.cache = dict(sorted_items[-1000:])

    def _similarity_score(self, query1: str, query2: str) -> float:
        """Calcula un score de similitud simple entre dos consultas"""
        # Convertir a conjuntos de palabras
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        # Calcular coeficiente de Jaccard
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

# MODIFICADO: Inicialización del caché
query_cache = AsyncQueryCache(ttl_hours=settings.cache_ttl_hours)

# MODIFICADO: Función sync para reescribir consultas
def rewrite_query(query: str, llm) -> Optional[str]:
    """Reescribe la consulta de forma síncrona"""
    try:
        rewrite_prompt = PromptTemplate(
            template=(
                "Eres un experto en reformular preguntas para recuperación de información. "
                "Tu objetivo es generar una versión mejorada de la consulta del usuario que maximice "
                "la posibilidad de recuperar información relevante de documentos técnicos.\n\n"
                "Aplica estas técnicas:\n"
                "1. Usa terminología técnica y sinónimos relevantes\n"
                "2. Expande acrónimos y abreviaturas\n"
                "3. Incorpora términos relacionados temáticamente\n"
                "4. Elimina palabras ambiguas o innecesarias\n"
                "5. Mantén la intención original del usuario\n\n"
                "PREGUNTA ORIGINAL: \"{user_query}\"\n\n"
                "PREGUNTA REFORMULADA:"
            ),
            input_variables=["user_query"]
        )
        rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)
        
        rewrite_response = rewrite_chain.invoke({"user_query": query})
        temp_rewritten_query = rewrite_response.get("text", "").strip()
        
        if "PREGUNTA REFORMULADA:" in temp_rewritten_query:
            temp_rewritten_query = temp_rewritten_query.split("PREGUNTA REFORMULADA:")[-1].strip()
            
        if temp_rewritten_query and temp_rewritten_query.lower() != query.lower():
            return temp_rewritten_query.replace('"', '')
        return None
        
    except Exception as e:
        logger.warning(f"Error en reescritura de consulta: {e}")
        return None

# MODIFICADO: Variable global para el traductor
_translator = None

def get_translator():
    """Inicializa y retorna el modelo de traducción"""
    global _translator
    
    try:
        if _translator is None:
            # Usar MarianMT que usa el tokenizer estándar
            _translator = pipeline(
                task="translation",
                model="Helsinki-NLP/opus-mt-es-en",
                tokenizer="Helsinki-NLP/opus-mt-es-en",
                framework="pt"
            )
        return _translator
            
    except Exception as e:
        logger.error(f"Error al inicializar traductor: {e}")
        return None

def translate_query(query: str, source_lang: str, target_lang: str) -> str:
    """Translate the query if needed"""
    if source_lang == target_lang:
        return query
        
    translated = translate_text(query, source_lang, target_lang)
    if translated is None:
        logger.warning(f"Translation failed, using original query")
        return query
        
    logger.info(f"Query translated from {source_lang} to {target_lang}")
    return translated

def merge_document_results(original_docs: List[Document], translated_docs: List[Document]) -> List[Document]:
    """Merge and deduplicate documents from both queries, prioritizing higher relevance scores"""
    if not original_docs:
        return translated_docs
    if not translated_docs:
        return original_docs
        
    # Create a dictionary to store unique documents by content
    unique_docs = {}
    
    # Process original documents
    for doc in original_docs:
        content_hash = hash(doc.page_content)
        if content_hash not in unique_docs:
            unique_docs[content_hash] = doc
        else:
            # If document exists, keep the one with higher relevance score
            existing_score = unique_docs[content_hash].metadata.get("relevance_score", 0)
            new_score = doc.metadata.get("relevance_score", 0)
            if new_score > existing_score:
                unique_docs[content_hash] = doc
    
    # Process translated documents
    for doc in translated_docs:
        content_hash = hash(doc.page_content)
        if content_hash not in unique_docs:
            unique_docs[content_hash] = doc
        else:
            # If document exists, keep the one with higher relevance score
            existing_score = unique_docs[content_hash].metadata.get("relevance_score", 0)
            new_score = doc.metadata.get("relevance_score", 0)
            if new_score > existing_score:
                unique_docs[content_hash] = doc
    
    # Convert back to list and sort by relevance score
    merged_docs = list(unique_docs.values())
    merged_docs.sort(key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
    
    return merged_docs

async def perform_retrieval_with_timeout(retriever, query: str, timeout: int) -> List[Document]:
    """Perform document retrieval with timeout"""
    try:
        async with async_timeout.timeout(timeout):
            return await asyncio.to_thread(lambda: retriever.invoke(query))
    except asyncio.TimeoutError:
        logger.warning(f"Retrieval timeout for query: {query[:100]}...")
        return []

# Initialize Supabase client (do this at the top-level, not per-request)
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_vectorstore_for_collection(collection_name: str):
    """Return a vectorstore for a specific collection (no cache)."""
    return PGVector(
        embedding_function=get_embeddings(),
        collection_name=collection_name,
        connection_string=settings.pg_conn
    )

def get_retriever_for_collection(collection_name: str):
    """Return a retriever for a specific collection (no cache)."""
    vectorstore = get_vectorstore_for_collection(collection_name)
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": settings.initial_retrieval_k,
            "filter": {},
            "batch_size": settings.batch_size
        }
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cross_enc = HuggingFaceCrossEncoder(
        model_name=settings.reranker_model,
        model_kwargs={"device": device}
    )
    reranker = CrossEncoderReranker(
        model=cross_enc, 
        top_n=settings.final_docs_count
    )
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request_data: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    api_auth: bool = Depends(verify_api_key_if_configured),
    rate_limit: bool = Depends(rate_limiter.check_rate_limit)
):
    start_time = time.time()
    
    try:
        userId = request_data.userId
        documentId = request_data.documentId

        # Dynamic collection name
        collection_name = settings.collection_name
        if userId and documentId:
            collection_name = f"{userId}_{documentId}"

        # Use custom retriever for this collection
        retriever = get_retriever_for_collection(collection_name)

        # --- Supabase logic example (fetch user metadata) ---
        # user_meta = supabase.table("users").select("*").eq("id", userId).single().execute()
        # logger.info(f"User metadata: {user_meta.data}")

        # Validación de entrada
        if not request_data.query.strip():
            raise HTTPException(
                status_code=400,
                detail="La consulta no puede estar vacía"
            )
            
        # Generar ID único para la consulta
        query_id = f"q-{int(time.time())}-{hashlib.md5(request_data.query.encode()).hexdigest()[:6]}"
        
        # Registrar la consulta
        query_log = {
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(),
            "query_length": len(request_data.query),
            "detected_language": request_data.language,
            "client_ip": request.client.host if request.client else "unknown",
            "detailed_feedback_requested": request_data.detailed_feedback
        }
        logger.info(f"Nueva consulta recibida: {json.dumps(query_log)}")
        
        # Verificar caché antes de procesar
        cached_response = await query_cache.get(request_data.query)
        if cached_response:
            logger.info(f"Respuesta obtenida de caché para query similar a: '{request_data.query}'")
            cached_response.execution_time = round(time.time() - start_time, 4)
            return cached_response
            
        # Procesar consulta con timeout
        async with async_timeout.timeout(settings.query_timeout):
            # Reescritura de consulta si está habilitada
            original_query = request_data.query
            rewritten_query = None

            if settings.query_rewrite_strategy != "original_only":
                llm = get_llm()
                rewritten_query = rewrite_query(original_query, llm)
                if rewritten_query:
                    logger.info(f"Consulta reescrita: '{rewritten_query}'")

            # Determinar query a usar según estrategia
            if settings.query_rewrite_strategy == "original_only":
                query_to_use = original_query
            elif settings.query_rewrite_strategy == "rewrite_only":
                query_to_use = rewritten_query if rewritten_query else original_query
            else:  # concatenate
                query_to_use = f"{original_query} | {rewritten_query}" if rewritten_query else original_query
                
            # Recuperar documentos usando el nuevo método invoke
            raw_source_docs = await perform_retrieval_with_timeout(
                retriever, 
                query_to_use,
                settings.retrieval_timeout
            )
            
            if not raw_source_docs:
                logger.warning("No se encontraron documentos relevantes")
                return QueryResponse(
                    answer="No se encontró información relevante en los documentos para responder a su pregunta.",
                    sources=[],
                    execution_time=round(time.time() - start_time, 4)
                )
                
            # Detect document language and handle translation
            original_docs = []
            translated_docs = []
            
            if raw_source_docs:
                try:
                    doc_language = detect_document_language(raw_source_docs)
                    query_language = request_data.language or detect(query_to_use)
                    
                    if doc_language != query_language:
                        logger.info(f"Document in {doc_language}, query in {query_language}")
                        
                        # Get results for original query with timeout
                        original_docs = await perform_retrieval_with_timeout(
                            retriever, 
                            query_to_use,
                            settings.retrieval_timeout
                        )
                        
                        if doc_language == "en" and query_language == "es":
                            # Translate with timeout
                            try:
                                async with async_timeout.timeout(settings.translation_timeout):
                                    translated_query = translate_query(query_to_use, "es", "en")
                            except asyncio.TimeoutError:
                                logger.warning("Translation timeout, using original query")
                                translated_query = query_to_use
                                
                            if translated_query and translated_query != query_to_use:
                                # Get results for translated query with timeout
                                translated_docs = await perform_retrieval_with_timeout(
                                    retriever,
                                    translated_query,
                                    settings.retrieval_timeout
                                )
                                logger.info(f"Retrieved {len(translated_docs)} documents from translated query")
                        
                        # Merge results from both queries
                        raw_source_docs = merge_document_results(original_docs, translated_docs)
                        logger.info(f"After merging: {len(raw_source_docs)} unique documents")
                        
                except Exception as e:
                    logger.warning(f"Error in translation process: {e}, using original query results")
                    raw_source_docs = original_docs if original_docs else raw_source_docs

            # Fusionar documentos consecutivos
            merged_docs = merge_document_info(raw_source_docs)
            
            # Generar respuesta de forma síncrona
            context = "\n\n".join([doc.page_content for doc in merged_docs])
            llm = get_llm()
            qa_chain = get_qa_chain()
            
            try:
                async with async_timeout.timeout(settings.query_timeout - (time.time() - start_time)):
                    result = await asyncio.to_thread(
                        qa_chain.invoke,
                        {"context": context, "question": original_query}
                    )
            except asyncio.TimeoutError:
                logger.warning("QA chain timeout, returning partial results")
                result = {"text": "The query took too long to process. Please try rephrasing or breaking it into smaller parts."}

            # Preparar fuentes
            sources_info = []
            for doc in merged_docs:
                metadata = doc.metadata or {}
                content_preview = None
                if request_data.detailed_feedback:
                    content_preview = (
                        doc.page_content[:150] + "..." 
                        if len(doc.page_content) > 150 
                        else doc.page_content
                    )
                    
                sources_info.append(SourceInfo(
                    source=metadata.get("source", "unknown"),
                    title=metadata.get("title"),
                    page=metadata.get("page"),
                    page_range=metadata.get("page_range"),
                    relevance_score=metadata.get("relevance_score"),
                    content_preview=content_preview
                ))
                
            # Generar URL de feedback si está configurada
            feedback_url = None
            if os.environ.get("FEEDBACK_URL"):
                feedback_id = hashlib.md5(
                    f"{original_query}-{int(time.time())}".encode()
                ).hexdigest()[:10]
                feedback_url = f"{os.environ['FEEDBACK_URL']}?id={feedback_id}&query={original_query[:50]}"
                
            # Registrar métricas en background
            background_tasks.add_task(
                log_query_metrics,
                query=original_query,
                num_sources=len(sources_info),
                execution_time=time.time() - start_time,
                language=request_data.language
            )
            
            # Crear respuesta
            response = QueryResponse(
                answer=result.get("text", "No se pudo generar una respuesta."),
                sources=sources_info,
                execution_time=round(time.time() - start_time, 4),
                feedback_url=feedback_url
            )
            
            # Almacenar en caché
            await query_cache.set(original_query, response)
            
            return response
            
    except asyncio.TimeoutError:
        logger.error(f"Overall query timeout: {request_data.query}")
        raise HTTPException(
            status_code=504,
            detail="The query exceeded the maximum processing time. Please try rephrasing or breaking it into smaller parts."
        )
    except Exception as e:
        logger.error(f"Error in query: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    userId: str = Form(None),
    documentId: str = Form(None)
):
    allowed_extensions = {"pdf", "docx", "txt"}
    filename = file.filename
    ext = filename.split(".")[-1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Tipo de archivo no soportado. Solo PDF, DOCX y TXT.")

    # Create a user-specific directory if userId is provided
    docs_dir = os.path.join(os.path.dirname(__file__), "../docs")
    if userId:
        docs_dir = os.path.join(docs_dir, userId)
        if documentId:
            docs_dir = os.path.join(docs_dir, documentId)
    
    os.makedirs(docs_dir, exist_ok=True)
    file_path = os.path.join(docs_dir, filename)

    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error guardando archivo: {e}")
        raise HTTPException(status_code=500, detail="Error al guardar el archivo.")

    # Use the Python of the current environment
    python_exec = sys.executable
    logger.info(f"Usando Python para ingest: {python_exec}")

    # Run ingest.py as a subprocess to index ONLY this file
    try:
        ingest_path = os.path.join(os.path.dirname(__file__), "../ingest.py")
        
        # Add collection name based on user and document if provided
        collection_name = settings.collection_name
        if userId and documentId:
            collection_name = f"{userId}_{documentId}"
        
        result = subprocess.run([
            python_exec, ingest_path,
            "--docs-dir", docs_dir,
            "--extensions", ext,
            "--collection", collection_name,
            "--reset-cache",
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"Error en ingest.py: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Error al procesar el documento: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("Ingest timeout")
        raise HTTPException(status_code=500, detail="El procesamiento del documento excedió el tiempo límite.")
    except Exception as e:
        logger.error(f"Error ejecutando ingest.py: {e}")
        raise HTTPException(status_code=500, detail="Error al indexar el documento.")

    return {"detail": "Documento procesado correctamente"}

# MODIFICADO: Punto de entrada mejorado
if __name__ == "__main__":
    import uvicorn
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    print(f"🚀 Iniciando API RAG en puerto {port}")
    print(f"📄 Colección: {settings.collection_name}")
    print(f"🧠 Modelo de embeddings: {settings.embedding_model}")
    print(f"🔄 Modelo de reranking: {settings.reranker_model}")
    print(f"🤖 LLM: {settings.llm_model}")
    print(f"🌍 Entorno: {settings.environment}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=settings.environment.lower() != "production",
        workers=min(multiprocessing.cpu_count(), 4)  # Limitar workers para evitar sobrecarga
    )

def detect_document_language(docs: List[Document]) -> str:
    """Detecta el idioma predominante en los documentos"""
    try:
        # Tomar una muestra de texto de cada documento
        text_samples = []
        for doc in docs[:5]:  # Limitar a 5 documentos para eficiencia
            content = doc.page_content
            # Tomar los primeros 1000 caracteres de cada documento
            text_samples.append(content[:1000])
        
        # Detectar idioma de cada muestra
        languages = []
        for sample in text_samples:
            try:
                lang = detect(sample)
                languages.append(lang)
            except LangDetectException:
                continue
        
        # Retornar el idioma más común
        if languages:
            from collections import Counter
            return Counter(languages).most_common(1)[0][0]
        return "en"  # Default a inglés si no se puede detectar
    except Exception as e:
        logger.warning(f"Error detectando idioma de documentos: {e}")
        return "en"