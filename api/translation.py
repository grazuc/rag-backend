"""
Simple translation module for the RAG service
"""
import logging
from typing import Optional
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)

def get_translator(source_lang: str = "es", target_lang: str = "en"):
    """
    Get a translation pipeline for the specified language pair.
    Returns None if initialization fails.
    """
    try:
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Initializing translation model {model_name} on {'GPU' if device == 0 else 'CPU'}")
        
        translator = pipeline(
            "translation",
            model=model_name,
            device=device
        )
        return translator
    except Exception as e:
        logger.error(f"Failed to initialize translation model: {e}")
        return None

def translate_text(text: str, source_lang: str = "es", target_lang: str = "en") -> Optional[str]:
    """
    Translate text from source_lang to target_lang.
    Returns None if translation fails.
    """
    if not text or source_lang == target_lang:
        return text
        
    try:
        translator = get_translator(source_lang, target_lang)
        if translator is None:
            logger.warning("Translation model not available")
            return None
            
        result = translator(text, max_length=512)
        if isinstance(result, list) and len(result) > 0:
            translated = result[0]['translation_text']
            logger.info(f"Successfully translated text from {source_lang} to {target_lang}")
            return translated
            
        logger.warning("Translation returned empty result")
        return None
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return None 