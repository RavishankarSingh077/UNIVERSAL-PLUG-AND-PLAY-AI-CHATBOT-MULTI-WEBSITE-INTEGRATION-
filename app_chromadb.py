#!/usr/bin/env python3
"""
ChromaDB RAG Chatbot
Complete implementation with multi-URL crawling and vector database.
"""

import os
import logging
import hashlib
import uuid
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Core libraries
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, root_validator
import uvicorn
import requests
import httpx
from bs4 import BeautifulSoup
import re

# Task scheduling
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Rate limiting and security
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Vector database and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Intent classification
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, skipping Hugging Face intent classification")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Load from environment variables
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MAX_URLS = int(os.getenv("MAX_URLS", "20"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "1.0"))  # Maximum distance for relevant results (lower is better, stricter matching)

def get_adaptive_threshold(query_intent: str) -> float:
    """
    Get relevance threshold based on query intent (universal approach, no hardcoded keywords).
    Different query types need different thresholds for optimal results.
    """
    if query_intent == "COMPANY_INFO_QUERY":
        return 1.8  # More lenient for company info queries (alumni, team size, etc.)
    elif query_intent == "SERVICE_QUERY":
        return 1.0  # Strict for service queries (need exact matches)
    elif query_intent == "GENERAL_QUESTION":
        return 1.3  # Moderate for general questions
    else:
        return 1.2  # Default threshold for other query types

# Company Configuration - Load from environment variables
# WEBSITE_URL is required - everything else will be extracted automatically
WEBSITE_URL = os.getenv("WEBSITE_URL", "https://example.com")


def _parse_csv_env(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def extract_basic_info_from_domain(website_url: str) -> Dict[str, Optional[str]]:
    """
    Extract basic company information from website domain as fallback.
    This ensures we always have non-None values even if extraction fails.
    """
    try:
        parsed = urlparse(website_url)
        domain = parsed.netloc.replace("www.", "").strip()
        
        if not domain:
            return {
                "company_name": "Company",
                "contact_email": "info@example.com",
                "contact_phone": None
            }
        
        # Extract domain name (e.g., "websjyoti" from "websjyoti.com")
        domain_parts = domain.split(".")
        domain_name = domain_parts[0] if domain_parts else "company"
        
        # Generate company name from domain (capitalize first letter of each word)
        # Handle camelCase or hyphenated domains
        company_name = domain_name.replace("-", " ").replace("_", " ")
        # Split by capital letters if camelCase
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', company_name) or [company_name]
        company_name = " ".join(word.capitalize() for word in words if word)
        
        # Generate contact email
        contact_email = f"info@{domain}"
        
        return {
            "company_name": company_name,
            "contact_email": contact_email,
            "contact_phone": None  # Cannot extract from domain
        }
    except Exception as e:
        logger.warning(f"Failed to extract info from domain: {e}")
        return {
            "company_name": "Company",
            "contact_email": "info@example.com",
            "contact_phone": None
        }


# Extract basic info from domain as fallback (ensures non-None values)
_domain_info = extract_basic_info_from_domain(WEBSITE_URL)

# Initialize with domain-based values (will be updated by extraction at startup)
COMPANY_NAME = os.getenv("COMPANY_NAME") or _domain_info["company_name"]
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL") or _domain_info["contact_email"]
CONTACT_PHONE = os.getenv("CONTACT_PHONE") or _domain_info["contact_phone"]
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL") or CONTACT_EMAIL

BASE_WEBSITE_URL = WEBSITE_URL.rstrip("/") or WEBSITE_URL
WEBSITE_DOMAIN = urlparse(WEBSITE_URL).netloc.replace("www.", "")
# Contact page URL - optional, defaults to /contact if not specified
_contact_page_env = os.getenv("CONTACT_PAGE_URL", "").strip()
if _contact_page_env:
    CONTACT_PAGE_URL = _contact_page_env
else:
    # Try common contact page paths
    CONTACT_PAGE_URL = urljoin(f"{BASE_WEBSITE_URL}/", "contact")
company = COMPANY_NAME
company_lower = COMPANY_NAME.lower()


COMPANY_TAGLINE = os.getenv("COMPANY_TAGLINE")
COMPANY_INDUSTRY = os.getenv("COMPANY_INDUSTRY")
PRIMARY_OFFERINGS = _parse_csv_env(os.getenv("PRIMARY_OFFERINGS"))
FLAGSHIP_CLIENTS = _parse_csv_env(os.getenv("FLAGSHIP_CLIENTS"))
BRAND_TONE = os.getenv("BRAND_TONE", "friendly")
BRAND_VOICE = os.getenv("BRAND_VOICE", "professional")
BRAND_KEYWORDS = _parse_csv_env(os.getenv("BRAND_KEYWORDS"))


def _clean_term(value: str) -> str:
    return re.sub(r'\s+', ' ', value).strip()


def human_join(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


@dataclass
class DomainProfile:
    company_name: str
    website_url: str
    contact_email: str
    contact_phone: str
    tagline: Optional[str] = None
    industry: Optional[str] = None
    primary_offerings: List[str] = field(default_factory=list)
    flagship_clients: List[str] = field(default_factory=list)
    tone: str = "friendly"
    voice: str = "professional"
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.primary_offerings = self._normalize_terms(self.primary_offerings)
        self.flagship_clients = self._normalize_terms(self.flagship_clients)
        self.keywords = self._normalize_terms(self.keywords)

    def _normalize_terms(self, terms: List[str]) -> List[str]:
        normalized: List[str] = []
        seen: set = set()
        for term in terms:
            cleaned = _clean_term(term)
            lowered = cleaned.lower()
            if cleaned and lowered not in seen:
                normalized.append(cleaned)
                seen.add(lowered)
        return normalized

    def describe_company_category(self) -> str:
        if self.industry:
            # CRITICAL: Ensure industry is a string, not a coroutine
            industry_str = self.industry
            if hasattr(industry_str, '__await__'):
                logger.error(f"[describe_company_category] CRITICAL: industry is coroutine!")
                return "organization"
            if not isinstance(industry_str, str):
                industry_str = str(industry_str) if industry_str else ""
            if industry_str:
                try:
                    return f"{industry_str.strip()} organization"
                except:
                    return "organization"
        return "organization"

    def describe_offerings_label(self) -> str:
        if self.primary_offerings:
            return human_join([off.split(",")[0] for off in self.primary_offerings[:2]])
        if self.industry:
            # CRITICAL: Ensure industry is a string, not a coroutine
            industry_str = self.industry
            if hasattr(industry_str, '__await__'):
                logger.error(f"[describe_offerings_label] CRITICAL: industry is coroutine!")
                return "solutions"
            if not isinstance(industry_str, str):
                industry_str = str(industry_str) if industry_str else ""
            if industry_str:
                try:
                    return f"{industry_str.strip()} services"
                except:
                    return "solutions"
        return "solutions"

    def summary_of_offerings(self) -> str:
        if self.primary_offerings:
            return human_join(self.primary_offerings[:4])
        if self.keywords:
            return human_join(self.keywords[:4])
        return "specialized services"

    def summary_sentence(self) -> str:
        summary = self.summary_of_offerings()
        return f"{self.company_name} provides {summary}."

    def register_offerings(self, offerings: List[str]):
        merged = self.primary_offerings[:]
        seen = {term.lower() for term in merged}
        for offering in offerings:
            cleaned = _clean_term(offering)
            if not cleaned or len(cleaned.split()) > 15:
                continue
            lowered = cleaned.lower()
            if lowered not in seen:
                merged.append(cleaned)
                seen.add(lowered)
            if len(merged) >= 8:
                break
        self.primary_offerings = merged

    def ingest_content_chunks(self, content_chunks: List[Dict[str, Any]]):
        harvested: List[str] = []
        for chunk in content_chunks:
            text = _clean_term(chunk.get('text', ''))
            if not text:
                continue
            metadata = chunk.get('metadata') or {}
            chunk_type = metadata.get('type', '')
            if chunk_type == 'heading':
                if 2 <= len(text.split()) <= 8:
                    harvested.append(text)
            else:
                lowered = text.lower()
                if any(keyword in lowered for keyword in ['service', 'solution', 'program', 'package', 'treatment', 'class', 'menu', 'offering']):
                    fragments = re.split(r'[.;:\n]', text)
                    for fragment in fragments:
                        fragment_clean = _clean_term(fragment)
                        if fragment_clean and len(fragment_clean.split()) <= 10:
                            harvested.append(fragment_clean)
        if harvested:
            self.register_offerings(harvested)

    def flagship_clients_summary(self) -> Optional[str]:
        if self.flagship_clients:
            return human_join(self.flagship_clients[:3])
        return None

    def brand_voice(self) -> str:
        voice_parts = []
        if self.tone:
            voice_parts.append(self.tone.strip())
        if self.voice and self.voice != self.tone:
            voice_parts.append(self.voice.strip())
        return human_join(voice_parts) or "professional"


DOMAIN_PROFILE = DomainProfile(
    company_name=COMPANY_NAME,
    website_url=WEBSITE_URL,
    contact_email=CONTACT_EMAIL,
    contact_phone=CONTACT_PHONE,
    tagline=COMPANY_TAGLINE,
    industry=COMPANY_INDUSTRY,
    primary_offerings=PRIMARY_OFFERINGS,
    flagship_clients=FLAGSHIP_CLIENTS,
    tone=BRAND_TONE,
    voice=BRAND_VOICE,
    keywords=BRAND_KEYWORDS,
)


def get_domain_profile() -> DomainProfile:
    return DOMAIN_PROFILE


def update_domain_profile_from_chunks(chunks: List[Dict[str, Any]]):
    if not chunks:
        return
    try:
        DOMAIN_PROFILE.ingest_content_chunks(chunks)
    except Exception as ingest_exc:
        logger.debug(f"Failed to update domain profile from chunks: {ingest_exc}")


def get_safe_fallback_reply() -> str:
    statement = DOMAIN_PROFILE.summary_sentence()
    return apply_company_placeholders(statement)


def get_general_service_fallback(additional_context: Optional[str] = None) -> str:
    summary = DOMAIN_PROFILE.summary_sentence()
    if additional_context:
        summary = f"{summary} {additional_context.strip()}"
    return apply_company_placeholders(summary)


# Utility helpers for dynamic company placeholders
def apply_company_placeholders(text: Optional[str]) -> Optional[str]:
    """Replace legacy hardcoded company references with configured values."""
    if not text:
        return text
    profile = get_domain_profile()
    replacements = {
        "{company} Ventures Private Limited": COMPANY_NAME,
        "{company} Ventures Pvt Ltd": COMPANY_NAME,
        "{company}": COMPANY_NAME,
        "{company_descriptor}": profile.describe_company_category(),
        "{company_offerings_label}": profile.describe_offerings_label(),
        "{primary_offerings_summary}": profile.summary_of_offerings(),
        "{company_voice}": profile.brand_voice(),
        "{company_lower}.com/contact": CONTACT_PAGE_URL,
        "https://{company_lower}.com/contact": CONTACT_PAGE_URL,
        "http://{company_lower}.com/contact": CONTACT_PAGE_URL,
        "https://{company_lower}.com": BASE_WEBSITE_URL,
        "http://{company_lower}.com": BASE_WEBSITE_URL,
        "{company_lower}.com": WEBSITE_DOMAIN or BASE_WEBSITE_URL,
        "info@{company_lower}.com": CONTACT_EMAIL,
        "support@{company_lower}.com": SUPPORT_EMAIL,
    }
    
    for legacy, updated in replacements.items():
        if updated is not None:  # Only replace if updated is not None
            text = text.replace(legacy, str(updated))
    
    placeholder_replacements = {
        "{company}": COMPANY_NAME,
        "{company_lower}": COMPANY_NAME.lower() if COMPANY_NAME else "",
        "{website}": BASE_WEBSITE_URL or "",
        "{website_domain}": WEBSITE_DOMAIN or BASE_WEBSITE_URL or "",
        "{contact_email}": CONTACT_EMAIL or "",
        "{support_email}": SUPPORT_EMAIL or "",
        "{contact_phone}": CONTACT_PHONE or "",
        "{contact_page}": CONTACT_PAGE_URL or "",
        "{company_tagline}": (COMPANY_TAGLINE or profile.summary_sentence()) if profile else "",
    }
    for placeholder, value in placeholder_replacements.items():
        if value is not None:  # Only replace if value is not None
            text = text.replace(placeholder, str(value))

    # Removed hardcoded IT-specific phrase replacements
    # All placeholders now use dynamic profile methods which adapt to any company type
    # No assumptions about IT services, technology, or specific industries
    
    return text

def _build_company_tokens() -> List[str]:
    tokens = []
    if COMPANY_NAME:
        tokens.append(COMPANY_NAME.lower())
        tokens.append(COMPANY_NAME.lower().replace(" ", ""))
    if WEBSITE_DOMAIN:
        domain_lower = WEBSITE_DOMAIN.lower()
        tokens.append(domain_lower)
        if "." in domain_lower:
            tokens.append(domain_lower.split(".")[0])
    return list(dict.fromkeys(filter(None, tokens)))

COMPANY_TOKENS = _build_company_tokens()

# Generate collection name from website URL if not specified
_collection_name_env = os.getenv("COLLECTION_NAME")
if _collection_name_env:
    COLLECTION_NAME = _collection_name_env
else:
    # Extract domain from website URL and create collection name
    parsed_url = urlparse(WEBSITE_URL)
    domain = parsed_url.netloc.replace("www.", "").replace(".", "_")
    COLLECTION_NAME = f"{domain}_content"
    logger.info(f"Auto-generated collection name: {COLLECTION_NAME} from {WEBSITE_URL}")

# Priority queries that must be pre-ingested to avoid on-demand scraping latency
# These are generic queries - will work for any company website
PRIORITY_PRELOAD_QUERIES = [
    # AI & Automation
    "automation services",
    "ai solutions",
    "ai implementation services",
    "intelligent automation",
    # ERP & CRM
    "erp implementation",
    "crm services",
    "enterprise resource planning",
    "customer relationship management",
    # Cloud & IoT
    "cloud transformation services",
    "cloud hosting",
    "iot solutions",
    # Pricing & Contact
    "pricing",
    "cost estimate",
    "how to contact",
    "support email",
    # Portfolio & Clients
    "projects portfolio",
    "client success",
    "case studies",
    # General service info
    "what services does the company offer",
    "digital transformation services"
]

AI_AUTOMATION_QUERY_KEYWORDS = {
    "ai company",
    "ai service",
    "ai services",
    "ai solution",
    "ai solutions",
    "ai implementation",
    "ai implementations",
    "artificial intelligence",
    "machine learning",
    "ml",
    "deep learning",
    "neural network",
    "neural networks",
    "computer vision",
    "machine vision",
    "image processing",
    "object detection",
    "pattern recognition",
    "natural language processing",
    "nlp",
    "conversational ai",
    "virtual agent",
    "virtual assistant",
    "chatbot",
    "intelligent automation",
    "ai automation",
    "automation",
    "hyperautomation",
    "smart automation",
    "cognitive automation",
    "generative ai",
    "foundation model",
    "foundation models",
    "large language model",
    "large language models",
    "llm",
    "llms",
    "robotic process automation",
    "rpa",
    "autonomous systems",
    "edge ai",
}

# API Configuration - Groq
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Secure API keys from environment variables (supports rotation)
_raw_groq_keys = os.getenv("GROQ_API_KEYS", "")
GROQ_API_KEYS = [key.strip() for key in _raw_groq_keys.split(",") if key.strip()]

if not GROQ_API_KEYS:
    single_key = os.getenv("GROQ_API_KEY")
    if single_key and single_key.strip():
        GROQ_API_KEYS = [single_key.strip()]

# Groq is optional if Mistral is provided
GROQ_API_KEYS = list(dict.fromkeys(GROQ_API_KEYS))
_current_groq_key_index = 0
_groq_key_lock = asyncio.Lock()

# API Configuration - Mistral
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = os.getenv("MISTRAL_API_URL", "https://api.mistral.ai/v1")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")

if not GROQ_API_KEYS and not MISTRAL_API_KEY:
    raise ValueError("No LLM API keys configured. Set GROQ_API_KEY or MISTRAL_API_KEY before running.")

# Classification cache to reduce LLM calls (universal, no hardcoded keywords)

# Classification cache to reduce LLM calls (universal, no hardcoded keywords)
_classification_cache: Dict[str, str] = {}
_cache_lock = asyncio.Lock()
MAX_CACHE_SIZE = 1000  # Limit cache size to prevent memory issues


def _mask_api_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"


async def _get_active_groq_key() -> str:
    async with _groq_key_lock:
        return GROQ_API_KEYS[_current_groq_key_index]


async def _advance_groq_key() -> str:
    global _current_groq_key_index
    async with _groq_key_lock:
        _current_groq_key_index = (_current_groq_key_index + 1) % len(GROQ_API_KEYS)
        return GROQ_API_KEYS[_current_groq_key_index]


async def _call_groq_with_rotation(
    messages: List[Dict[str, str]],
    max_tokens: int = 100,
    temperature: float = 0.7,
    model: str = "llama-3.1-8b-instant"
) -> Optional[str]:
    """
    Centralized function for Groq API calls with automatic key rotation and exponential backoff.
    Handles all error cases (401, 403, 429, network errors) and tries all available keys.
    Implements exponential backoff for rate limiting (429 errors).
    
    Args:
        messages: List of message dictionaries for the API
        max_tokens: Maximum tokens for response (default: 100)
        temperature: Temperature for response generation (default: 0.7)
        model: Model to use (default: llama-3.1-8b-instant)
    
    Returns:
        Cleaned AI response string or None if all keys fail
    """
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    attempts = 0
    response: Optional[httpx.Response] = None
    base_delay = 1.0  # Base delay in seconds for exponential backoff
    max_delay = 3.0  # Maximum delay in seconds (optimized for faster responses)

    async with httpx.AsyncClient(timeout=10.0) as client:
        while attempts < len(GROQ_API_KEYS):
            active_key = await _get_active_groq_key()
            headers = {
                "Authorization": f"Bearer {active_key}",
                "Content-Type": "application/json",
            }

            try:
                response = await client.post(GROQ_API_URL, headers=headers, json=data)
            except httpx.RequestError as exc:
                logger.error(f"Groq API request error with key {_mask_api_key(active_key)}: {exc}")
                attempts += 1
                await _advance_groq_key()
                # Small delay for network errors before retrying
                await asyncio.sleep(0.5)
                continue

            if response.status_code == 200:
                try:
                    result = response.json()
                    ai_reply = result["choices"][0]["message"]["content"]
                    await _advance_groq_key()  # Advance key for next request (continuous rotation)
                    # Fix 2: Clean LLM response (remove HTML/metadata/boilerplate)
                    cleaned_reply = strip_markdown(ai_reply.strip())
                    cleaned_reply = clean_context_for_llm(cleaned_reply)
                    return cleaned_reply
                except (KeyError, IndexError, ValueError) as exc:
                    logger.error(f"Unexpected Groq response structure: {exc}")
                    return None

            if response.status_code in (401, 403):
                logger.warning(
                    f"Groq API key {_mask_api_key(active_key)} returned status {response.status_code}. Rotating to next key."
                )
                attempts += 1
                await _advance_groq_key()
                continue

            if response.status_code == 429:
                # Rate limiting: Use exponential backoff
                delay = min(base_delay * (2 ** attempts), max_delay)
                logger.warning(
                    f"Groq API key {_mask_api_key(active_key)} rate limited (429). "
                    f"Waiting {delay:.1f}s before trying next key (attempt {attempts + 1}/{len(GROQ_API_KEYS)})"
                )
                await asyncio.sleep(delay)
                attempts += 1
                await _advance_groq_key()
                continue

            logger.error(
                f"Groq API error {response.status_code} with key {_mask_api_key(active_key)}: {response.text}"
            )
            attempts += 1
            await _advance_groq_key()
            if attempts < len(GROQ_API_KEYS):
                continue
            break

    return None


async def _call_mistral(
    messages: List[Dict[str, str]],
    max_tokens: int = 100,
    temperature: float = 0.7,
    model: str = None
) -> Optional[str]:
    """
    Call Mistral AI API with provided configuration.
    """
    if not MISTRAL_API_KEY:
        return None

    model = model or MISTRAL_MODEL
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(f"{MISTRAL_API_URL}/chat/completions", headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                ai_reply = result["choices"][0]["message"]["content"]
                return strip_markdown(ai_reply.strip())
            else:
                logger.error(f"Mistral API error {response.status_code}: {response.text}")
        except Exception as exc:
            logger.error(f"Mistral API request error: {exc}")
    return None


async def _get_llm_response(
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
    model_override: str = None
) -> Optional[str]:
    """
    Generic LLM caller that prioritizes Mistral if configured, otherwise uses Groq.
    """
    # Try Mistral first if configured
    if MISTRAL_API_KEY:
        response = await _call_mistral(messages, max_tokens, temperature, model_override)
        if response:
            return response

    # Fallback to Groq
    if GROQ_API_KEYS:
        return await _call_groq_with_rotation(messages, max_tokens, temperature, model_override or "llama-3.1-8b-instant")

    return None


async def classify_query_intent_with_llm(query: str) -> str:
    """
    Use LLM to intelligently classify query intent with enhanced categories.
    Uses cache to reduce LLM calls for similar queries (universal, no hardcoded keywords).
    Returns one of: SERVICE_QUERY, COMPANY_INFO_QUERY, SERVICE_INFO_QUERY, 
    CONTACT_QUERY, PRICING_QUERY, GENERAL_QUESTION
    """
    try:
        # Normalize query for cache key (lowercase, strip whitespace)
        cache_key = query.lower().strip()
        
        # Check cache first
        async with _cache_lock:
            if cache_key in _classification_cache:
                logger.info(f"Using cached classification for query: {query[:50]}...")
                return _classification_cache[cache_key]
        
        classification_prompt = f"""Analyze this user query and classify it into ONE category.

Query: "{query}"

Categories:
- SERVICE_QUERY: User is asking if company provides a specific service (e.g., "do you provide car servicing", "can you do X")
- COMPANY_INFO_QUERY: User is asking about company details (founder, history, team, alumni, students, graduates, established, started, who created, team size, employees, location, office, address, certifications, awards, experience, years in business, portfolio, clients, case studies, company info, about company, company details, company statistics, how many projects, how many services, what makes you different, competitors, comparison, industries served, personnel, executives, sales team, noted people, key people, key personnel, management team, leadership team, who are the executives, who are the key people)
- SERVICE_INFO_QUERY: User is asking about services company provides (e.g., "tell me about your software solutions", "what services do you offer", "what are your services")
- CONTACT_QUERY: User is asking for contact information (email, phone, location, address, how to reach, where are you)
- PRICING_QUERY: User is asking about pricing, cost, packages, fees, rates, charges, how much
- GENERAL_QUESTION: General questions about the company, services, or anything else

Return ONLY the category name (e.g., "COMPANY_INFO_QUERY"). No explanation, just the category."""

        messages = [
            {"role": "system", "content": "You are a query classification assistant. Return only the category name."},
            {"role": "user", "content": classification_prompt}
        ]
        
        classification = await _get_llm_response(
            messages=messages,
            max_tokens=20,  # Very short response needed
            temperature=0.3  # Low temperature for consistent classification
        )
        
        if classification:
            classification = classification.strip().upper()
            # Validate classification
            valid_categories = [
                "SERVICE_QUERY", "COMPANY_INFO_QUERY", "SERVICE_INFO_QUERY",
                "CONTACT_QUERY", "PRICING_QUERY", "GENERAL_QUESTION"
            ]
            if classification in valid_categories:
                logger.info(f"LLM classified query as: {classification}")
                # Store in cache
                async with _cache_lock:
                    # Limit cache size to prevent memory issues
                    if len(_classification_cache) >= MAX_CACHE_SIZE:
                        # Remove oldest entry (simple FIFO)
                        oldest_key = next(iter(_classification_cache))
                        del _classification_cache[oldest_key]
                    _classification_cache[cache_key] = classification
                return classification
        
        # Fallback to GENERAL_QUESTION if classification fails
        logger.warning(f"LLM classification failed or invalid, defaulting to GENERAL_QUESTION")
        fallback = "GENERAL_QUESTION"
        # Cache the fallback too
        async with _cache_lock:
            if len(_classification_cache) >= MAX_CACHE_SIZE:
                oldest_key = next(iter(_classification_cache))
                del _classification_cache[oldest_key]
            _classification_cache[cache_key] = fallback
        return fallback
        
    except Exception as e:
        logger.error(f"Error in LLM query classification: {e}")
        return "GENERAL_QUESTION"  # Safe fallback


def generate_context_based_fallback(
    query: str,
    query_intent: str,
    context: str,
    search_results: List[Dict[str, Any]],
    company_name: str = None
) -> str:
    """
    Generate a context-based fallback response when LLM fails (universal, no hardcoded keywords).
    Uses query-aware sentence extraction with semantic matching, distance-based prioritization,
    and query type detection for accurate, complete responses.
    Improved: Better cleaning, sentence filtering, and company description handling.
    
    Args:
        query: User's query
        query_intent: Classified intent (COMPANY_INFO_QUERY, SERVICE_QUERY, etc.)
        context: Combined context from RAG search results
        search_results: List of search results with metadata and distance scores
        company_name: Company name for personalization
    
    Returns:
        Natural language response based on context (500-600 characters)
    """
    try:
        # Fix 3: Clean context at the start of fallback function
        if context:
            context = clean_context_for_llm(context)
        
        # Clean search results content before processing
        if search_results:
            for result in search_results:
                if 'content' in result:
                    result['content'] = clean_context_for_llm(result.get('content', ''))
        
        if not context or not context.strip():
            # No context available
            if query_intent == "COMPANY_INFO_QUERY":
                return f"Based on available information, {company_name or 'we'} can provide details about our company. For specific information, please visit our website or contact us."
            elif query_intent == "SERVICE_QUERY":
                return f"{company_name or 'We'} offer a range of services. For detailed information about specific services, please visit our website or contact us."
            elif query_intent == "CONTACT_QUERY":
                return f"You can reach {company_name or 'us'} at {CONTACT_PHONE or 'our contact number'} or visit our website for contact information."
            else:
                return f"Thank you for your query. For more information, please visit our website or contact us."
        
        # Extract query keywords (remove common stop words)
        query_lower = query.lower()
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                     'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 
                     'about', 'with', 'for', 'from', 'to', 'in', 'on', 'at', 'by', 'of', 'and', 'or', 'but', 
                     'if', 'then', 'else', 'when', 'where', 'how', 'what', 'which', 'who', 'why', 'you', 'your', 
                     'yours', 'we', 'our', 'ours', 'they', 'their', 'them', 'it', 'its', 'this', 'that', 'these', 
                     'those', 'i', 'me', 'my', 'mine', 'he', 'she', 'his', 'her', 'hers', 'tell', 'me', 'about'}
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        query_keywords = [w for w in query_words if w not in stop_words and len(w) > 2]
        
        # Detect query type (universal approach, no hardcoded keywords)
        query_type = None
        is_company_description_query = False
        if query_lower.startswith('who'):
            query_type = 'who'  # Extract names, people, entities
        elif query_lower.startswith('what'):
            query_type = 'what'  # Extract definitions, descriptions
            # Check if it's a company description query ("what is [company]?")
            if 'what is' in query_lower and (company_name and company_name.lower() in query_lower):
                is_company_description_query = True
        elif 'how many' in query_lower:
            query_type = 'how_many'  # Extract numbers, statistics, counts
        elif query_lower.startswith('when'):
            query_type = 'when'  # Extract dates, time periods
        elif query_lower.startswith('where'):
            query_type = 'where'  # Extract locations, places
        elif query_lower.startswith('how'):
            query_type = 'how'  # Extract processes, methods
        
        # Map sentences to source chunks with distance scores
        # Create a mapping: sentence -> distance score from search_results
        sentence_to_distance = {}
        if search_results:
            # Split context to match with search_results content
            for result in search_results:
                result_content = result.get('content', '')
                result_distance = result.get('distance', 999.0)
                # Find sentences that appear in this result
                if result_content:
                    result_sentences = re.split(r'(?<=[.!?])\s+', result_content)
                    for sent in result_sentences:
                        sent_clean = sent.strip()
                        if sent_clean and len(sent_clean) > 15:
                            # Use first 50 chars as key for matching
                            sent_key = sent_clean[:50].lower()
                            if sent_key not in sentence_to_distance or result_distance < sentence_to_distance[sent_key]:
                                sentence_to_distance[sent_key] = result_distance
        
        # Split context into sentences (better splitting to preserve complete sentences)
        sentences = re.split(r'(?<=[.!?])\s+', context)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
        
        # Fix 5 & 6: Filter out boilerplate/legal/irrelevant sentences (enhanced)
        boilerplate_keywords = [
            'i agree', 'terms and conditions', 'disclaimer', 'show more', 'view all',
            'read more', 'click here', 'learn more', 'explore more', 'view more',
            'accept cookies', 'reject cookies', 'manage preferences', 'subscribe',
            'follow us', 'posted on', 'updated on', 'published on', 'created on'
        ]
        
        # Fix 5: Additional patterns for registration numbers, copyright, UI elements
        metadata_patterns = [
            r'U\d{5}[A-Z]{2}\d{4}[A-Z]{2}\d{6}',  # Company registration numbers
            r'Powered\s+By',  # "Powered By" text
            r'All\s+Pictures.*?illustration\s+purpose',  # "All Pictures...illustration purpose"
            r'While\s+we\s+strive\s+for\s+accuracy',  # "While we strive for accuracy"
            r'do\s+not\s+guarantee',  # "do not guarantee"
            r'Copyright\s+©',  # Copyright symbols
            r'All\s+rights\s+reserved',  # "All rights reserved"
            r'chatInquire\s+Now',  # UI elements
            r'apartmentBook\s+Plot',  # UI elements
            r'Gallery\s+Images',  # UI elements
            r'keyboard_arrow_down',  # UI elements
            r'Actual\s+product\s+may\s+vary',  # "Actual product may vary"
            r'Investors\s+are\s+required\s+to\s+verify',  # Legal disclaimers
        ]
        
        filtered_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Skip sentences that are mostly boilerplate
            if any(keyword in sentence_lower for keyword in boilerplate_keywords):
                # Only skip if the sentence is mostly boilerplate (short or contains multiple boilerplate keywords)
                if len(sentence) < 50 or sum(1 for kw in boilerplate_keywords if kw in sentence_lower) >= 2:
                    continue
            
            # Fix 5: Skip sentences containing registration numbers, copyright, UI elements
            contains_metadata = False
            for pattern in metadata_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    contains_metadata = True
                    break
            
            if contains_metadata:
                # Only skip if sentence is short or mostly metadata
                if len(sentence) < 60:
                    continue
                # If sentence is longer, check if it's mostly metadata
                metadata_chars = sum(len(m.group()) for m in re.finditer('|'.join(metadata_patterns), sentence, re.IGNORECASE))
                if metadata_chars > len(sentence) * 0.3:  # More than 30% metadata
                    continue
            
            filtered_sentences.append(sentence)
        
        sentences = filtered_sentences if filtered_sentences else sentences
        
        if not sentences:
            # Fallback if no sentences found
            if query_intent == "COMPANY_INFO_QUERY":
                return f"Based on our knowledge base, {company_name or 'we'} have relevant information about this topic. For specific details, please visit our website."
            else:
                return f"Thank you for your query. We have information about this topic. For more details, please visit our website or contact us."
        
        # Score sentences based on query keyword matching, distance, and query type
        scored_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = 0
            
            # 1. Keyword matching: count how many query keywords appear in sentence
            keyword_matches = sum(1 for kw in query_keywords if kw in sentence_lower)
            score += keyword_matches * 2  # Weight: 2x
            
            # 2. Distance-based boost (higher priority for more relevant chunks)
            sent_key = sentence[:50].lower()
            if sent_key in sentence_to_distance:
                distance = sentence_to_distance[sent_key]
                if distance < 0.5:
                    score += 3  # Highly relevant chunk
                elif distance < 1.0:
                    score += 2  # Moderately relevant chunk
                elif distance < 1.5:
                    score += 1  # Somewhat relevant chunk
            
            # 3. Query type relevance (universal approach)
            if query_type == 'who':
                # Boost sentences with names, people, entities (detect capitalized words, titles)
                if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', sentence):  # Name pattern
                    score += 2
                if any(word in sentence_lower for word in ['founder', 'director', 'manager', 'president', 'ceo', 'owner']):
                    score += 2
            elif query_type == 'what':
                # Boost sentences with definitions, descriptions
                if any(word in sentence_lower for word in ['is', 'are', 'means', 'refers', 'includes', 'consists']):
                    score += 2
                # Fix 2: Special handling for company description queries
                if is_company_description_query:
                    # Prioritize descriptive sentences about company purpose/services
                    descriptive_keywords = ['is a', 'provides', 'offers', 'specializes', 'focuses', 'delivers', 
                                          'develops', 'creates', 'builds', 'designs', 'solutions', 'services',
                                          'company', 'organization', 'firm', 'business', 'enterprise']
                    if any(keyword in sentence_lower for keyword in descriptive_keywords):
                        score += 4  # High boost for descriptive sentences
                    # Penalize sentences that are just the company name
                    if sentence.strip() == company_name or len(sentence.strip()) < 30:
                        score -= 5  # Heavy penalty for just company name or very short sentences
            elif query_type == 'how_many':
                # Boost sentences with numbers, statistics, counts
                if re.search(r'\d+', sentence):  # Contains numbers
                    score += 3
                if any(word in sentence_lower for word in ['number', 'count', 'total', 'many', 'several', 'multiple']):
                    score += 2
            elif query_type == 'when':
                # Boost sentences with dates, time periods
                if re.search(r'\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', sentence):  # Date pattern
                    score += 3
                if any(word in sentence_lower for word in ['year', 'month', 'day', 'date', 'established', 'founded', 'since']):
                    score += 2
            elif query_type == 'where':
                # Boost sentences with locations, places
                if any(word in sentence_lower for word in ['location', 'address', 'place', 'city', 'country', 'located']):
                    score += 2
            
            # 4. Prefer longer, more informative sentences
            if len(sentence) > 50:
                score += 1
            
            scored_sentences.append((score, sentence))
        
        # Sort by score (highest first) and select top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Select top 4-5 most relevant sentences (increased from 2-3)
        top_sentences = [s for _, s in scored_sentences[:5] if scored_sentences[0][0] > 0]  # Only if we have matches
        
        # If no keyword matches, use top sentences by default
        if not top_sentences:
            top_sentences = [s for _, s in scored_sentences[:4]]
        
        if not top_sentences:
            # Ultimate fallback
            if query_intent == "COMPANY_INFO_QUERY":
                return f"Based on our knowledge base, {company_name or 'we'} have relevant information about this topic. For specific details, please visit our website."
            else:
                return f"Thank you for your query. We have information about this topic. For more details, please visit our website or contact us."
        
        # Build response from top sentences (use top 4 sentences, increased from 2)
        # Fix 2: For company description queries, ensure we have enough sentences (at least 2-3)
        if is_company_description_query:
            # Use more sentences for company descriptions to ensure completeness
            num_sentences = min(5, len(top_sentences))
            response = " ".join(top_sentences[:num_sentences])
        else:
            response = " ".join(top_sentences[:4])
        
        # Ensure response is complete (ends with punctuation)
        if not response.endswith(('.', '!', '?')):
            response += "."
        
        # Fix 2: Ensure minimum response length for company descriptions (at least 150-200 chars)
        min_length = 150 if is_company_description_query else 100
        if len(response) < min_length:
            # Add more context if response is too short
            if len(top_sentences) > len(top_sentences[:4] if not is_company_description_query else top_sentences[:5]):
                if is_company_description_query:
                    response = " ".join(top_sentences[:6])
                else:
                    response = " ".join(top_sentences[:5])
                if not response.endswith(('.', '!', '?')):
                    response += "."
        
        # Limit to 600 chars for better completeness (increased from 300)
        if len(response) > 600:
            # Truncate at last complete sentence before 600 chars
            truncated = response[:600]
            last_period = truncated.rfind('.')
            last_exclamation = truncated.rfind('!')
            last_question = truncated.rfind('?')
            last_punct = max(last_period, last_exclamation, last_question)
            min_content = 200 if is_company_description_query else 150
            if last_punct > min_content:  # Only if we have enough content
                response = response[:last_punct + 1]
            else:
                response = response[:600] + "..."
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating context-based fallback: {e}")
        # Ultimate fallback
        return f"Thank you for your query. For more information, please visit our website or contact us at {CONTACT_PHONE or 'our contact number'}."


async def generate_query_variations(query: str) -> List[str]:
    """
    Generate multiple query variations for better RAG search coverage.
    Uses LLM to create semantic variations of the original query.
    Includes fallback mechanism if LLM fails.
    """
    try:
        variation_prompt = f"""Generate 2-3 different ways to ask the same question. 
Keep the meaning the same but use different words or phrasing.

Original query: "{query}"

Return ONLY the variations, one per line, no numbering, no explanation.
Example:
what are your services
what services do you provide
tell me about your offerings"""

        messages = [
            {"role": "system", "content": "You are a query variation generator. Return only the variations, one per line."},
            {"role": "user", "content": variation_prompt}
        ]
        
        variations_response = await _get_llm_response(
            messages=messages,
            max_tokens=50,
            temperature=0.7
        )
        
        if variations_response:
            # Parse variations (one per line)
            variations = [v.strip() for v in variations_response.strip().split('\n') if v.strip()]
            # Limit to 3 variations max
            variations = variations[:3]
            # Add original query at the beginning
            all_queries = [query] + variations
            logger.info(f"Generated {len(variations)} query variations for: {query}")
            return all_queries
        
        # Fallback: Try again with a different prompt if first attempt failed
        logger.warning(f"First LLM attempt failed for query variations, trying fallback prompt")
        fallback_prompt = f"""Create alternative phrasings for this question: "{query}"

Generate 2-3 reworded versions that mean the same thing.
Return one variation per line, no numbers or explanations."""

        fallback_messages = [
            {"role": "system", "content": "Generate query variations. One per line."},
            {"role": "user", "content": fallback_prompt}
        ]
        
        fallback_response = await _get_llm_response(
            messages=fallback_messages,
            max_tokens=50,
            temperature=0.8  # Slightly higher temperature for more diversity
        )
        
        if fallback_response:
            variations = [v.strip() for v in fallback_response.strip().split('\n') if v.strip()]
            variations = variations[:3]
            all_queries = [query] + variations
            logger.info(f"Generated {len(variations)} query variations using fallback for: {query}")
            return all_queries
        
        # Final fallback: return original query only
        logger.warning(f"Both LLM attempts failed for query variations, returning original query only")
        return [query]
        
    except Exception as e:
        logger.error(f"Error generating query variations: {e}")
        return [query]  # Fallback to original query


def multi_query_search_chroma(queries: List[str], collection_name: str = COLLECTION_NAME, n_results: int = 3, query_intent: str = None) -> List[Dict[str, Any]]:
    """
    Search ChromaDB with multiple query variations and combine results.
    Returns deduplicated and ranked results.
    Uses adaptive threshold based on query_intent (universal approach).
    """
    all_results = []
    seen_content = set()  # To deduplicate based on content
    
    for query in queries:
        try:
            results = search_chroma(query, collection_name, n_results=n_results, query_intent=query_intent)
            for result in results:
                # Deduplicate by content (first 100 chars)
                content_key = result['content'][:100] if result.get('content') else ""
                if content_key and content_key not in seen_content:
                    seen_content.add(content_key)
                    all_results.append(result)
        except Exception as e:
            logger.error(f"Error in multi-query search for query '{query}': {e}")
            continue
    
    # Sort by distance (lower is better) and return top results
    all_results.sort(key=lambda x: x.get('distance', 999.0))
    
    # Return top n_results
    final_results = all_results[:n_results]
    logger.info(f"Multi-query search: {len(queries)} queries → {len(final_results)} unique results")
    return final_results


async def _call_groq_with_messages(
    messages: List[Dict[str, str]],
    temperature: float = 0.5,
    max_tokens: int = 100
) -> Optional[str]:
    """Utility to call Groq with rotation support and return the cleaned content."""
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    attempts = 0
    response: Optional[httpx.Response] = None

    async with httpx.AsyncClient(timeout=10.0) as client:
        while attempts < len(GROQ_API_KEYS):
            active_key = await _get_active_groq_key()
            headers = {
                "Authorization": f"Bearer {active_key}",
                "Content-Type": "application/json",
            }

            try:
                response = await client.post(GROQ_API_URL, headers=headers, json=data)
            except httpx.RequestError as exc:
                logger.error(f"Groq API request error with key {_mask_api_key(active_key)}: {exc}")
                attempts += 1
                await _advance_groq_key()
                continue

            if response.status_code == 200:
                try:
                    result = response.json()
                    ai_reply = result["choices"][0]["message"]["content"]
                    await _advance_groq_key()  # Advance key for next request (continuous rotation)
                    # Fix 2: Clean LLM response (remove HTML/metadata/boilerplate)
                    cleaned_reply = strip_markdown(ai_reply.strip())
                    cleaned_reply = clean_context_for_llm(cleaned_reply)
                    return cleaned_reply
                except (KeyError, IndexError, ValueError) as exc:
                    logger.error(f"Unexpected Groq response structure: {exc}")
                    return None

            if response.status_code in (401, 403, 429):
                logger.warning(
                    f"Groq API key {_mask_api_key(active_key)} returned status {response.status_code}. Rotating to next key."
                )
                attempts += 1
                await _advance_groq_key()
                continue

            logger.error(
                f"Groq API error {response.status_code} with key {_mask_api_key(active_key)}: {response.text}"
            )
            break

    return None

SOFT_NEGATIVE_PHRASES = [
    "you are mad",
    "you're mad",
    "you are irritating",
    "you're irritating",
    "you are annoying",
    "you're annoying",
    "you are rude",
    "you're rude",
    "i am not your client",
    "i'm not your client",
    "i am not your customer",
    "i'm not your customer",
    "i don't need your help",
    "i dont need your help",
    "i do not need your help",
]

# --------------------------------------------------------------------------------------------------
# Response sanitization utilities
# --------------------------------------------------------------------------------------------------
def sanitize_response_text(reply: Optional[str]) -> str:
    """Keep friendly statements while stripping questions and forbidden phrases."""
    if not reply:
        return get_safe_fallback_reply()

    reply = reply.strip()
    if not reply:
        return get_safe_fallback_reply()

    original_reply = reply
    sentence_split_pattern = re.compile(r'(?<=[.!?])\s+')
    question_phrase_patterns = [
        r'would you like to',
        r'do you want to',
        r'can i help you',
        r'what would you like',
        r'how can i assist',
        r'is there anything',
        r'are you exploring',
        r'what challenges',
        r"what's on your mind",
        r'what would you like to explore',
        r'how can we support',
        r'want to hear about',
        r'let me ask',
        r"since you're here",
        r'is there anything about',
        r'would you like to know',
        r'would you like to learn',
        r'would you like to discuss',
        r'would you like to explore',
        r'would you like more',
        r'would you like additional',
        r'would you like to hear',
        r'would you like to find out',
        r'can we support',
        r'how can we',
        r'what do you need',
        r'what do you want',
    ]

    sentences = sentence_split_pattern.split(reply)
    kept_sentences: List[str] = []

    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean:
            continue

        sentence_lower = sentence_clean.lower()
        if '?' in sentence_lower:
            continue
        if any(re.search(pattern, sentence_lower) for pattern in question_phrase_patterns):
            continue

        kept_sentences.append(sentence_clean.rstrip('.!?'))
        if len(kept_sentences) >= 2:
            break

    if not kept_sentences:
        logger.warning(f"Reply removed during sanitization. Original: {original_reply[:120]}")
        sanitized = get_safe_fallback_reply()
    else:
        sanitized = '. '.join(kept_sentences).strip()
        if not sanitized.endswith('.'):
            sanitized = f"{sanitized}."

    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Fix 3: Clean HTML/metadata/boilerplate from final response
    sanitized = clean_context_for_llm(sanitized)

    correct_number = CONTACT_PHONE
    wrong_number_patterns = [
        r'\+91\s*11\s*4567\s*8900',
        r'\+91\s*11-4567-8900',
        r'\+91\s*11\s*4567-8900',
        r'\+91\s*11-4567\s*8900',
        r'11\s*4567\s*8900',
        r'11-4567-8900'
    ]
    for pattern in wrong_number_patterns:
        sanitized = re.sub(pattern, correct_number, sanitized, flags=re.IGNORECASE)

    phone_pattern = r'\+91[\s-]?\d{2}[\s-]?\d{4}[\s-]?\d{4}'
    found_numbers = re.findall(phone_pattern, sanitized)
    for found_num in found_numbers:
        normalized_found = re.sub(r'[\s-]', '', found_num)
        normalized_correct = re.sub(r'[\s-]', '', correct_number)
        if normalized_found != normalized_correct:
            sanitized = sanitized.replace(found_num, correct_number)

    if '?' in sanitized:
        sanitized = sanitized.replace('?', '.').strip()

    if not sanitized:
        sanitized = get_safe_fallback_reply()

    sanitized = apply_company_placeholders(sanitized)
    
    # Fix 4: Improve response completeness validation
    sanitized = _validate_and_complete_response(sanitized)

    return sanitized


def _validate_and_complete_response(response: str) -> str:
    """
    Validate and complete incomplete responses (e.g., "920 sq" → "920 sq km").
    Universal approach - works for any incomplete patterns.
    """
    if not response:
        return response
    
    import re
    
    # Fix incomplete area measurements (e.g., "920 sq" → "920 sq km")
    # Pattern: number followed by "sq" but missing unit
    incomplete_area_pattern = r'(\d+)\s+sq\s*([^km]|$)'
    matches = list(re.finditer(incomplete_area_pattern, response, re.IGNORECASE))
    if matches:
        # Check if "km" or "m" appears later in the response
        response_lower = response.lower()
        if 'km' in response_lower or 'kilometer' in response_lower:
            # Likely should be "sq km"
            response = re.sub(r'(\d+)\s+sq\s*([^km]|$)', r'\1 sq km\2', response, flags=re.IGNORECASE)
        elif 'm' in response_lower and 'meter' in response_lower:
            # Likely should be "sq m"
            response = re.sub(r'(\d+)\s+sq\s*([^m]|$)', r'\1 sq m\2', response, flags=re.IGNORECASE)
        else:
            # Default to "sq km" for area measurements
            response = re.sub(r'(\d+)\s+sq\s*([^km]|$)', r'\1 sq km\2', response, flags=re.IGNORECASE)
    
    # Fix incomplete sentences that end abruptly (e.g., "The total project area encompasses 920 sq.")
    # Pattern: sentence ending with incomplete measurement or number
    incomplete_sentence_pattern = r'(\d+)\s+(sq|square)\s*\.\s*$'
    if re.search(incomplete_sentence_pattern, response, re.IGNORECASE):
        response = re.sub(r'(\d+)\s+(sq|square)\s*\.\s*$', r'\1 \2 km.', response, flags=re.IGNORECASE)
    
    # Fix truncated sentences (e.g., "What is the Dholera project area? The total project area encompasses 920 sq")
    # Pattern: sentence ending with incomplete measurement without punctuation
    truncated_pattern = r'(\d+)\s+sq\s*$'
    if re.search(truncated_pattern, response, re.IGNORECASE):
        response = re.sub(truncated_pattern, r'\1 sq km.', response, flags=re.IGNORECASE)
    
    # Ensure response ends with proper punctuation
    if response and not response.rstrip().endswith(('.', '!', '?')):
        response = response.rstrip() + '.'
    
    return response


def consolidate_rag_results(search_results: List[Dict[str, Any]], query_intent: str = None) -> List[Dict[str, Any]]:
    """
    Consolidate RAG results to ensure consistent answers.
    - Groups similar information
    - Prioritizes: company-wide > project-specific
    - Uses lower distance (higher confidence) when conflicting
    - Removes duplicates
    Universal approach - no hardcoded keywords.
    """
    if not search_results:
        return []
    
    # Sort by distance (lower = better/more relevant)
    sorted_results = sorted(search_results, key=lambda x: x.get('distance', 999.0))
    
    # Deduplicate by content (first 150 chars to catch similar chunks)
    seen_content = set()
    consolidated = []
    
    for result in sorted_results:
        content = result.get('content', '')
        if not content:
            continue
        
        # Create content key for deduplication
        content_key = content[:150].lower().strip()
        
        # Skip if we've seen very similar content
        if content_key in seen_content:
            continue
        
        seen_content.add(content_key)
        consolidated.append(result)
    
    # For COMPANY_INFO_QUERY, prioritize company-wide stats over project-specific
    if query_intent == "COMPANY_INFO_QUERY":
        # Try to identify and prioritize company-wide information
        # Look for patterns that suggest company-wide stats (e.g., "total", "overall", "company-wide")
        company_wide_keywords = ['total', 'overall', 'company-wide', 'all', 'entire', 'complete']
        project_specific_keywords = ['project', 'specific', 'particular', 'individual', 'this project']
        
        def prioritize_score(result):
            content_lower = result.get('content', '').lower()
            score = 0
            
            # Boost company-wide indicators
            for keyword in company_wide_keywords:
                if keyword in content_lower:
                    score += 10
                    break
            
            # Reduce project-specific indicators
            for keyword in project_specific_keywords:
                if keyword in content_lower:
                    score -= 5
                    break
            
            # Lower distance = higher priority
            distance = result.get('distance', 999.0)
            score += (100.0 - distance * 10)  # Convert distance to priority score
            
            return score
        
        # Re-sort with prioritization
        consolidated = sorted(consolidated, key=prioritize_score, reverse=True)
    
    logger.info(f"Consolidated {len(search_results)} results to {len(consolidated)} unique results")
    return consolidated


def extract_numerical_stats(context: str, query: str) -> Dict[str, Any]:
    """
    Extract numerical statistics from context and identify company-wide vs project-specific.
    Universal approach - uses regex patterns and heuristics, no hardcoded values.
    """
    import re
    
    # Generic patterns for numerical stats (enhanced for better detection)
    patterns = {
        'numbers_with_plus': r'(\d{1,3}(?:,\d{3})*)\+',  # 5,000+, 350+
        'numbers_with_units': r'(\d{1,3}(?:,\d{3})*)\s+(residential|commercial|plots?|townships?|projects?|employees?|clients?|years?|villas?|shops?|properties?)',  # 5,000 residential plots
        'numbers_with_dash': r'(\d{1,3}(?:,\d{3})*)\s*-\s*(\d{1,3}(?:,\d{3})*)\s+(residential|commercial|plots?|townships?)',  # 5,000-10,000 plots
        'plain_numbers': r'\b(\d{1,3}(?:,\d{3})*)\b',  # Standalone numbers
    }
    
    stats = {
        'company_wide': [],
        'project_specific': [],
        'all_numbers': []
    }
    
    # Extract all numerical patterns
    for pattern_name, pattern in patterns.items():
        matches = re.finditer(pattern, context, re.IGNORECASE)
        for match in matches:
            number = match.group(1).replace(',', '')
            unit = match.group(2) if len(match.groups()) > 1 and pattern_name != 'numbers_with_dash' else None
            if pattern_name == 'numbers_with_dash' and len(match.groups()) >= 3:
                unit = match.group(3)
            full_match = match.group(0)
            
            stats['all_numbers'].append({
                'value': number,
                'unit': unit,
                'text': full_match,
                'context': context[max(0, match.start()-100):match.end()+100]  # Increased context window
            })
    
    # Use enhanced heuristics to identify company-wide vs project-specific
    # (Universal approach - no hardcoded keywords)
    context_lower = context.lower()
    query_lower = query.lower()
    
    for stat in stats['all_numbers']:
        context_snippet = stat['context'].lower()
        number_value = int(stat['value'])
        
        # Enhanced indicators of company-wide stats (universal patterns)
        company_wide_indicators = [
            'total', 'overall', 'company-wide', 'all', 'entire', 'complete', 'combined', 'sum',
            'across all', 'company offers', 'we offer', 'we have', 'company has',
            'total of', 'overall', 'entire portfolio', 'all projects', 'combined total'
        ]
        # Enhanced indicators of project-specific stats (universal patterns)
        project_indicators = [
            'this project', 'specific project', 'particular project', 'individual project',
            'one project', 'project offers', 'project has', 'in this project',
            'project includes', 'project features', 'project provides'
        ]
        
        # Check for company-wide indicators
        is_company_wide = any(indicator in context_snippet for indicator in company_wide_indicators)
        # Check for project-specific indicators
        is_project_specific = any(indicator in context_snippet for indicator in project_indicators)
        
        # Enhanced logic: prioritize company-wide detection
        if is_company_wide and not is_project_specific:
            stats['company_wide'].append(stat)
        elif is_project_specific:
            stats['project_specific'].append(stat)
        # If unclear, use size-based heuristics (universal approach)
        # Larger numbers are more likely to be company-wide totals
        elif number_value > 1000:  # Threshold for likely company-wide
            stats['company_wide'].append(stat)
        elif number_value > 100:
            # Medium numbers: check if query asks for totals
            if any(word in query_lower for word in ['total', 'overall', 'all', 'how many', 'total number']):
                stats['company_wide'].append(stat)
            else:
                stats['project_specific'].append(stat)
        else:
            stats['project_specific'].append(stat)
    
    # Sort company-wide by value (descending) to prioritize largest totals
    stats['company_wide'].sort(key=lambda x: int(x['value']), reverse=True)
    # Sort project-specific by value (ascending) for reference
    stats['project_specific'].sort(key=lambda x: int(x['value']))
    
    return stats


def clean_context_for_llm(context: str) -> str:
    """
    Clean context before sending to LLM: remove HTML tags, markers, and metadata.
    Universal approach - works for any website content.
    Improved: Better HTML detection, error handling, and boilerplate removal.
    """
    import re
    from bs4 import BeautifulSoup
    import warnings
    
    if not context:
        return context
    
    # Check if input is HTML-like (contains HTML tags)
    is_html_like = bool(re.search(r'<[^>]+>', context))
    
    # Remove HTML tags using BeautifulSoup (universal approach)
    cleaned = context
    if is_html_like:
        try:
            # Suppress BeautifulSoup warnings for non-HTML input
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                soup = BeautifulSoup(context, 'html.parser')
                # Get text content, preserving structure
                cleaned = soup.get_text(separator=' ', strip=True)
        except Exception as e:
            # Fallback: use regex to remove HTML tags
            cleaned = re.sub(r'<[^>]+>', '', context)
    else:
        # Not HTML, just clean text - use as is but still clean markers
        cleaned = context
    
    # Remove common content markers and boilerplate (universal patterns)
    markers_to_remove = [
        r'---\s*Content from.*?---',
        r'Disclaimer.*?At\s+\w+',
        r'Show more',
        r'Posted on.*?ago',
        r'View All',
        r'Read more',
        r'Click here',
        r'Learn more',
        r'Explore More',
        r'View More',
        r'I agree to abide by.*?terms and conditions',
        r'Terms and conditions',
        r'Privacy policy',
        r'Cookie policy',
        r'Accept.*?cookies',
        r'Subscribe.*?newsletter',
        r'Follow us on',
    ]
    
    for marker_pattern in markers_to_remove:
        cleaned = re.sub(marker_pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove boilerplate/legal text patterns (universal approach)
    boilerplate_patterns = [
        r'I agree.*?terms',
        r'By continuing.*?agree',
        r'Accept.*?terms',
        r'Read.*?terms',
        r'Accept all cookies',
        r'Reject all cookies',
        r'Manage preferences',
    ]
    
    for pattern in boilerplate_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Fix 1: Remove company registration numbers (e.g., "U70200GJ2012PTC100931")
    # Pattern: U followed by digits, letters, digits (common Indian company registration format)
    cleaned = re.sub(r'\bU\d{5}[A-Z]{2}\d{4}[A-Z]{2}\d{6}\b', '', cleaned, flags=re.IGNORECASE)
    # Also handle other registration number formats
    cleaned = re.sub(r'\b[A-Z]{1,3}\d{5,15}[A-Z]{0,5}\d{0,10}\b', '', cleaned)
    
    # Fix 1: Remove "Powered By" patterns
    cleaned = re.sub(r'Powered\s+By\s*:?\s*\w+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Powered\s+by\s+\w+', '', cleaned, flags=re.IGNORECASE)
    
    # Fix 1: Remove "All Pictures/Images shown" patterns
    cleaned = re.sub(r'All\s+Pictures[/\s]*Images\s+shown\s+on\s+this\s+website\s+are\s+for\s+illustration\s+purpose\s+only', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'All\s+Images\s+shown.*?illustration\s+purpose', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Fix 1: Remove "While we strive for accuracy" patterns
    cleaned = re.sub(r'While\s+we\s+strive\s+for\s+accuracy.*?do\s+not\s+guarantee', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'do\s+not\s+guarantee\s+the\s+completeness.*?availability', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Fix 1: Remove copyright patterns
    cleaned = re.sub(r'Copyright\s+©\s*\d{4}\s+All\s+rights\s+reserved', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'©\s*\d{4}\s+All\s+rights\s+reserved', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Copyright\s+©\s*\d{4}', '', cleaned, flags=re.IGNORECASE)
    
    # Fix 1: Remove UI elements (universal patterns)
    ui_elements = [
        r'chatInquire\s+Now',
        r'apartmentBook\s+Plot',
        r'Sq\.yard',
        r'SCO\s+Plots',
        r'Shop-cum-office',
        r'Commercial\s+Plots',
        r'Common\s+Plots\s+for\s+Parking',
        r'Gallery\s+Images',
        r'Images\s+Images',
        r'AmenitiesGated\s+Community',
        r'Recreational\s+Spaces',
        r'SecurityWide\s+Internal\s+Roads',
        r'Car\s+Parking',
        r'24\s+Hrs\s+Backup\s+Electricity',
        r'Cctv\s+Camera',
        r'Landscape\s+Garden',
        r'LocationdistanceView\s+On\s+Map',
        r'Inquire\s+Now',
        r'Select\s+Projects',
        r'VillaPlotLooking\s+for',
        r'keyboard_arrow_down',
        r'First\s+Name',
        r'Last\s+Name',
        r'Phone\s+Number',
        r'Email\s+Address',
        r'CommentsSubmit',
        r'OngoingDholera',
        r'Dholera\s+Expres',
    ]
    
    for ui_pattern in ui_elements:
        cleaned = re.sub(ui_pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Fix 1: Remove "Actual product may vary" patterns
    cleaned = re.sub(r'Actual\s+product\s+may\s+vary\s+due\s+to\s+product\s+enhancement', '', cleaned, flags=re.IGNORECASE)
    
    # Fix 1: Remove "Investors are required to verify" patterns
    cleaned = re.sub(r'Investors\s+are\s+required\s+to\s+verify\s+all\s+the\s+details.*?independently', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove excessive whitespace and newlines
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n+', '\n', cleaned)
    
    # Remove metadata-like patterns (universal approach)
    # Remove patterns like "Posted on", "Updated on", etc.
    cleaned = re.sub(r'(Posted|Updated|Published|Created)\s+on.*?(\n|$)', '', cleaned, flags=re.IGNORECASE)
    
    # Remove email patterns that might be metadata (keep actual contact emails in context)
    # Only remove if they appear in isolation
    cleaned = re.sub(r'\b\w+@\w+\.\w+\b(?=\s*$)', '', cleaned)
    
    # Remove URLs that appear as standalone text (keep URLs in sentences)
    cleaned = re.sub(r'\bhttps?://\S+\b(?=\s*[.!?]?\s*$)', '', cleaned)
    
    # Clean up any remaining artifacts
    cleaned = cleaned.strip()
    
    # Remove empty lines
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    cleaned = ' '.join(lines)
    
    return cleaned


def extract_context_snippet(search_results: Optional[List[Dict[str, Any]]], max_words: int = 25) -> Optional[str]:
    """Extract a short informative snippet from search results to enrich terse replies."""
    if not search_results:
        return None
    for result in search_results:
        if not isinstance(result, dict):
            continue
        text = result.get('content', '')
        if not text:
            continue
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            cleaned = re.sub(r'\s+', ' ', sentence).strip()
            if not cleaned:
                continue
            if len(cleaned.split()) < 6:
                continue
            if len(cleaned.split()) > max_words:
                cleaned = ' '.join(cleaned.split()[:max_words]) + '...'
            cleaned = cleaned.replace('?', '.')
            return cleaned
    return None

# FastAPI app
app = FastAPI(title="ChromaDB RAG Chatbot", version="2.0.0")

# Global exception handler to catch ALL errors and log full traceback
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all exceptions and log full traceback for debugging"""
    import traceback
    full_traceback = traceback.format_exc()
    error_type = type(exc).__name__
    error_msg = str(exc)
    
    logger.error(f"[GLOBAL EXCEPTION HANDLER] Error Type: {error_type}")
    logger.error(f"[GLOBAL EXCEPTION HANDLER] Error Message: {error_msg}")
    logger.error(f"[GLOBAL EXCEPTION HANDLER] Full Traceback:\n{full_traceback}")
    logger.error(f"[GLOBAL EXCEPTION HANDLER] Request Path: {request.url.path}")
    logger.error(f"[GLOBAL EXCEPTION HANDLER] Request Method: {request.method}")
    
    # Try to get request body for debugging
    try:
        body = await request.body()
        logger.error(f"[GLOBAL EXCEPTION HANDLER] Request Body: {body.decode('utf-8')[:500]}")
    except:
        pass
    
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={
            "detail": error_msg,
            "error_type": error_type,
            "traceback": full_traceback
        }
    )

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration - Allow all origins for universal website integration
allowed_origins = ["*"]

# Remove duplicates while preserving order
allowed_origins = list(dict.fromkeys(allowed_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Note: Middleware approach removed - handling in validator and __init__ is sufficient

# Global variables for caching
chroma_client = None
embedding_model = None
intent_classifier = None
# Embedding cache to avoid regenerating same query embeddings
embedding_cache = {}

# Global scheduler for periodic tasks
scheduler = None

# In-memory conversation storage
# Structure: { 
#   "session_id": {
#     "conversations": [{"role": "user", "content": "message"}, {"role": "assistant", "content": "reply"}],
#     "language": "english" or "hindi"
#   }
# }
conversation_sessions = {}

# Language detection function
def detect_language(message: str) -> str:
    """Detect if message is in Hindi or English based on comprehensive word list"""
    # DEBUG: Log what we receive
    logger.info(f"[DEBUG detect_language] Received type: {type(message)}, has __await__: {hasattr(message, '__await__') if message else False}")
    
    # Defensive check: ensure message is a string, not a coroutine
    if hasattr(message, '__await__'):
        # If it's a coroutine, this shouldn't happen but handle it
        logger.error(f"[DEBUG detect_language] CRITICAL: Received coroutine! Type: {type(message)}")
        return 'english'
    
    if not message:
        return 'english'  # Default for empty messages
    
    # Ensure message is a string before calling strip()
    if not isinstance(message, str):
        logger.warning(f"[DEBUG detect_language] message is not string, type: {type(message)}, converting...")
        message = str(message)
    
    try:
        if not message.strip():
            return 'english'  # Default for empty messages
        message = message.strip()
    except Exception as e:
        logger.error(f"[DEBUG detect_language] Error calling strip(): {e}, type: {type(message)}")
        return 'english'
    message_lower = message.lower()
    
    # Devanagari script range (Hindi characters)
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    
    # Check if message contains Devanagari script - if yes, it's Hindi
    if bool(devanagari_pattern.search(message)):
        return 'hindi'
    
    # Check for common English greetings first (before Hindi word detection)
    english_greetings = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'hiya', 'howdy', 'greetings'
    ]
    
    # Check for exact matches or short greeting phrases
    if message_lower in english_greetings:
        return 'english'
    
    # Check if message starts with an English greeting and has 3 or fewer words
    for greeting in english_greetings:
        if message_lower.startswith(greeting) and len(message_lower.split()) <= 3:
            return 'english'
    # Comprehensive list of common Hindi words in Roman script
    hindi_words = {
        # Common verbs and conjugations
        'ho', 'hai', 'hain', 'hun', 'hoon', 'hoa', 'hoga', 'hogi', 'honge',
        'karta', 'karti', 'karte', 'kiya', 'kiye', 'kiyi', 'karega', 'karegi', 'karenge',
        'kar', 'karo', 'kariye', 'karein', 'kare',
        'bolo', 'boliye', 'bolte', 'bola', 'bol', 'kaho',
        'dekh', 'dekho', 'dekhna', 'dekhte', 'dekha', 'dekhega',
        'banaya', 'banai', 'banate', 'banega', 'banegi',
        'liya', 'li', 'le', 'lete', 'leta', 'leti', 'letiye', 'lena', 'lene',
        'gaya', 'gayi', 'gaye', 'jao', 'ja', 'jaate', 'jaata', 'jaati', 'jana', 'jane',
        'aayega', 'aayegi', 'aayenge', 'aao', 'aa', 'aate', 'aata', 'aati', 'aana', 'aane',
        'diya', 'diye', 'dete', 'deta', 'deti', 'dene', 'de', 'do', 'dena', 'dene',
        
        # Pronouns
        'tum', 'tumhare', 'tumhari', 'tumhara', 'tumhe', 'tumko', 'tumse',
        'aap', 'aapka', 'aapki', 'aapke', 'aapko', 'aapse',
        'main', 'mera', 'meri', 'mere', 'mujhe', 'mujhse', 'mujhko',
        'woh', 'uska', 'uski', 'uske', 'use', 'usse',
        'yeh', 'ye', 'iske', 'iski', 'iska', 'ise', 'isse', 'isne',
        'hum', 'hamara', 'hamari', 'hamare', 'humhe', 'humko', 'hamse',
        'kaun', 'kaunse', 'kaunsi', 'kaunka', 'kiska', 'kiski', 'kiske',
        
        # Question words
        'kya', 'kyun', 'kahan', 'kab', 'kaise', 'kitna', 'kitni', 'kitne', 'kisko', 'kisne',
        'kyon', 'kaun', 'kaise', 'kabhi', 'kab',
        
        # Common prepositions/particles
        'ko', 'se', 'par', 'mein', 'ne', 'toh', 'bhi', 'na', 'nahi', 'nahin',
        'tak', 'ke', 'ki', 'ka', 'kar', 'ke', 'hie',
        
        # Common adjectives/adverbs
        'acha', 'achcha', 'accha', 'badhiya', 'theek', 'sahi', 'galat', 'bura', 'bura',
        'bahut', 'bohot', 'bhot', 'zyada', 'kam', 'khub', 'kaafi',
        'abhi', 'ab', 'phir', 'fir', 'tab', 'toh',
        
        # Common nouns
        'ghar', 'ghar', 'kaam', 'kaam', 'log', 'insaan', 'admi', 'aurat',
        'dost', 'dosti', 'pyar', 'mohabbat', 'khushi', 'dukh', 'gham',
        
        # Time words
        'aaj', 'kal', 'parso', 'roz', 'hamesha', 'kabhi', 'abhi', 'phir', 'fir',
        'subah', 'shaam', 'raat', 'din', 'mahina', 'saal', 'samay', 'waqt',
        
        # Greetings and common phrases
        'namaste', 'namaskar', 'dhanyawad', 'shukriya', 'kripya', 'maaf',
        'hain', 'sab', 'hota', 'hoti', 'hote',
        
        # Common words
        'bilkul', 'zaroor', 'pakka', 'sach', 'sacchi', 'sach',
        'acha', 'theek', 'sahi', 'galat', 'bura', 'sahi',
        'bhi', 'sirf', 'bas', 'abhi', 'phir',
        'mai', 'tumhari', 'mera', 'tera', 'hamara',
        'nahi', 'han', 'haan', 'na', 'ji', 'ji',
        
        # Action words
        'karo', 'karein', 'karna', 'karne', 'kar', 'karne',
        'bolo', 'boliye', 'bolna', 'bolne', 'bol',
        'dekh', 'dekho', 'dekhna', 'dekhne',
        
        # More common words
        'usne', 'unhone', 'unke', 'unka', 'unki',
        'raha', 'rahi', 'rahe', 'rahon', 'rahun',
        'chahiye', 'chahiye', 'chahta', 'chahti', 'chahte',
        'milna', 'milne', 'repository', 'project', 'repository', 'budget',
    }
    
    # Extract words from message
    words = re.findall(r'\b\w+\b', message_lower)
    
    if not words:
        return 'english'  # No words found
    
    # Count Hindi words
    hindi_count = sum(1 for word in words if word in hindi_words)
    total_words = len(words)
    
    # If 30% or more words are Hindi, it's Hindi
    if hindi_count > 0 and (hindi_count / total_words) >= 0.3:
        return 'hindi'
    elif hindi_count > total_words - hindi_count:
        return 'hindi'
    else:
        return 'english'

def is_ai_automation_query(message: str) -> bool:
    """Quick heuristic to detect AI / automation themed queries."""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in AI_AUTOMATION_QUERY_KEYWORDS)


def is_soft_negative_message(message: str) -> bool:
    """Identify negative/frustrated phrasing that should still be answered by the LLM."""
    # CRITICAL: Ensure message is a string before any operations
    if hasattr(message, '__await__'):
        logger.error(f"[is_soft_negative_message] CRITICAL: message is coroutine!")
        return False
    if not isinstance(message, str):
        message = str(message) if message else ""
    if not message:
        return False
    try:
        message_lower = message.lower()
    except:
        return False
    return any(phrase in message_lower for phrase in SOFT_NEGATIVE_PHRASES)

# Hugging Face Intent Classifier Functions
def get_intent_classifier():
    """Load and return the Hugging Face intent classifier model"""
    global intent_classifier
    
    if intent_classifier is not None:
        return intent_classifier
    
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers library not available, skipping intent classifier")
        return None
    
    try:
        logger.info("Loading Hugging Face intent classifier model (DistilBERT)...")
        # Using DistilBERT for faster zero-shot classification (3-4x faster, 6x smaller)
        intent_classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",  # Fast and lightweight intent classification
            device=-1  # Use CPU
        )
        logger.info("Intent classifier model loaded successfully")
        return intent_classifier
    except Exception as e:
        logger.error(f"Error loading intent classifier: {e}")
        intent_classifier = None
        return None
def detect_intent_with_hf(message: str) -> dict:
    """
    Detect intent using Hugging Face model with zero-shot classification
    
    Returns:
        dict with 'intent' and 'confidence' keys, or None if model unavailable
    """
    try:
        classifier = get_intent_classifier()
        if not classifier:
            return None
        
        # Define intent candidates with descriptive phrases based on pattern checks
        # These descriptions include keywords from pattern-based functions for better accuracy
        intent_candidates = [
            # Greeting - from is_greeting() patterns
            "user is greeting or saying hello with words like hi, hello, hey, good morning, good afternoon, good evening, namaste, namaskar, salam, hiya, howdy, greetings, kaise ho, kaise hain",
            
            # Goodbye - from is_goodbye() patterns
            "user is saying goodbye or ending conversation with words like bye, goodbye, see you, later, end chat, exit, quit, alvida, phir milte hain",
            
            # Project inquiry - from is_project_intent() patterns
            "user wants to start a new project, work with company, need project help, want to work with you, planning a project, need help with project",
            
            # Service inquiry - generic (works for any company type)
            "user is asking about services offered, what services do you offer, tell me about your services, what do you provide",
            
            # Business info - from is_business_info_query() patterns
            "user is asking about business information like success rate, customer satisfaction, client satisfaction, team size, how many employees, experience, years of experience, portfolio, case studies, clients, customer base",
            
            # Existing customer - from is_existing_customer_query() patterns
            "user is an existing customer asking about their project, I am a client, I have a project with you, my project name is",
            
            # Personal introduction - from is_personal_introduction() patterns
            "user is introducing themselves by sharing their name, my name is, I am, call me",
            
            # Bot identity - from is_bot_identity_question() patterns
            "user is asking about bot's identity or name like who are you, what is your name, tell me your name, apka naam kya hai",
            
            # Contact inquiry - from is_contact_query() patterns
            "user is asking for contact information like location, address, where are you, office, email, phone, how to contact, get in touch, talk to someone",
            
            # Capability question - from is_capability_question() patterns (GENERIC ONLY, not service-specific)
            "user is asking GENERICALLY about chatbot capabilities like what can you do, what are your capabilities, what can you assist with, BUT NOT asking about specific services",
            
            # Complaint - from is_complaint() patterns
            "user has a complaint or is dissatisfied like not happy, unhappy, disappointed, poor service, bad service, complaint, problem with service",
            
            # Help request - GENERIC help only (not service-specific)
            "user needs GENERIC help or has questions like help, need help, can you help, I need assistance, how can you help me, BUT NOT asking about specific services",
            
            # Pricing inquiry - from is_pricing_query() patterns (with negative exclusions)
            "user is asking about pricing information like price, pricing, cost, how much, expensive, cheap, fees, charge, payment, quote, estimate BUT NOT asking about success rate, conversion rate, performance metrics, or business statistics",
            
            # Off topic - from is_off_topic() patterns
            "user is asking something completely unrelated to company services like movies, weather, recipes, other companies like Google or Flipkart, unrelated topics"
        ]
        
        # Create mapping from descriptive labels to simple intent names
        intent_mapping = {
            "user is greeting or saying hello": "greeting",
            "user is saying goodbye or ending": "goodbye",
            "user wants to start a new project": "project_inquiry",
            "user is asking about services offered": "service_inquiry",
            "user is asking about business information": "business_info",
            "user is an existing customer": "existing_customer",
            "user is introducing themselves": "personal_introduction",
            "user is asking about bot's identity": "bot_identity",
            "user is asking for contact information": "contact_inquiry",
            "user is asking GENERICALLY about chatbot capabilities": "capability_question",
            "user has a complaint": "complaint",
            "user needs GENERIC help": "help_request",
            "user is asking about pricing information": "pricing_inquiry",
            "user is asking something completely unrelated": "off_topic"
        }
        
        # Dynamic confidence thresholds based on intent type (LOWERED for better coverage)
        intent_thresholds = {
            "greeting": 0.55,
            "goodbye": 0.55,
            "help_request": 0.55,
            "contact_inquiry": 0.55,
            "service_inquiry": 0.50,
            "pricing_inquiry": 0.55,
            "business_info": 0.55,
            "project_inquiry": 0.55,
            "bot_identity": 0.55,
            "capability_question": 0.55,
            "existing_customer": 0.55,
            "personal_introduction": 0.55,
            "complaint": 0.60,
            "off_topic": 0.65
        }
        
        # Classify the message
        result = classifier(message, intent_candidates)
        
        if result and len(result['labels']) > 0:
            top_intent_label = result['labels'][0]
            confidence = result['scores'][0]
            
            # Map descriptive label back to simple intent name
            detected_intent = None
            # Use generic mapping (no service-specific intents)
            for label_key, intent_name in intent_mapping.items():
                if label_key in top_intent_label.lower():
                    detected_intent = intent_name
                    break
            
            # If no mapping found, try to extract from label
            if not detected_intent:
                # Extract first few words as fallback
                if "greeting" in top_intent_label.lower():
                    detected_intent = "greeting"
                elif "goodbye" in top_intent_label.lower():
                    detected_intent = "goodbye"
                elif "erp" in top_intent_label.lower():
                    detected_intent = "erp_inquiry"
                elif "crm" in top_intent_label.lower():
                    detected_intent = "crm_inquiry"
                elif "cloud" in top_intent_label.lower() or "hosting" in top_intent_label.lower():
                    detected_intent = "cloud_inquiry"
                elif "iot" in top_intent_label.lower():
                    detected_intent = "iot_inquiry"
                elif "ai" in top_intent_label.lower() or "artificial intelligence" in top_intent_label.lower():
                    detected_intent = "ai_inquiry"
                elif "business information" in top_intent_label.lower():
                    detected_intent = "business_info"
                elif "pricing" in top_intent_label.lower():
                    detected_intent = "pricing_inquiry"
                elif "service" in top_intent_label.lower():
                    detected_intent = "service_inquiry"
                elif "project" in top_intent_label.lower():
                    detected_intent = "project_inquiry"
                else:
                    detected_intent = "off_topic"  # Default fallback
            
            # Get threshold for this intent
            threshold = intent_thresholds.get(detected_intent, 0.75)
            
            logger.info(f"HF Intent detected: {detected_intent} with confidence: {confidence:.2f} (threshold: {threshold:.2f})")
            
            return {
                'intent': detected_intent,
                'confidence': confidence,
                'threshold': threshold,
                'all_intents': list(zip(result['labels'], result['scores']))
            }
        
        return None
    
    except Exception as e:
        logger.error(f"Error in Hugging Face intent detection: {e}")
        return None
# Query classification functions
def get_off_topic_category(message: str) -> str:
    """Categorize off-topic queries for appropriate responses"""
    message_lower = message.lower().strip()
    
    if is_soft_negative_message(message):
        return None
    
    # Check if it's a Hindi service query (should be allowed)
    hindi_service_keywords = [
        'services', 'solutions', '{company_lower}',
        'aapke', 'ke bare', 'batao', 'kya hain', 'provide', 'karte hain'
    ]
    
    # If Hindi message contains service-related keywords, don't mark as off-topic
    if any(keyword in message_lower for keyword in hindi_service_keywords):
        return None
    
    # Universal: Exclude personnel/executives queries from off-topic detection
    # These should be handled as COMPANY_INFO_QUERY (universal approach - no hardcoded keywords)
    personnel_keywords = [
        'personnel', 'executives', 'executive', 'sales team', 'sales executive',
        'noted people', 'noted personnel', 'key people', 'key personnel',
        'management team', 'leadership team', 'who are the executives',
        'who are the key people', 'team members', 'staff', 'employees',
        'founder', 'founders', 'co-founder', 'co-founders', 'ceo', 'director',
        'leadership', 'management', 'who is', 'who are'
    ]
    
    # If query contains personnel-related keywords, don't mark as off-topic
    # Let it be classified as COMPANY_INFO_QUERY by the LLM classifier
    if any(keyword in message_lower for keyword in personnel_keywords):
        return None
    
    # Abusive or inappropriate language patterns
    abusive_patterns = [
        'pagal', 'idiot', 'stupid', 'harami', 'fool', 'nonsense', 'shut up',
        'fuck', 'shit', 'damn', 'hate', 'useless', 'garbage', 'trash',
        'chup', 'chup sale', 'sale', 'kutta', 'kutte', 'bevakoof', 'ullu'
    ]
    
    # Other company keywords
    other_company_keywords = [
        'google', 'microsoft', 'amazon', 'facebook', 'meta', 'apple', 
        'netflix', 'tesla', 'ibm', 'oracle', 'salesforce'
    ]
    
    # Job-related keywords for other companies
    job_other_company = [
        'google job', 'job in google', 'job at google', 'microsoft job', 'amazon job',
        'facebook job', 'apple job', 'work at google', 'work at microsoft'
    ]
    
    # Unrelated topics
    unrelated_keywords = [
        'bhojpuri', 'recipe', 'cooking', 'movie', 'song', 'game', 'sport',
        'weather', 'news', 'politics', 'celebrity', 'fashion', 'shopping',
        'love', 'dating', 'relationship', 'marriage'
    ]
    
    # Generic unrelated
    generic_unrelated = ['close it', 'closeit', 'stop', 'don\'t show', 'hide']
    
    # Check for abusive language
    for pattern in abusive_patterns:
        if pattern in message_lower:
            return 'abusive'
    
    # Check for job queries at other companies
    for keyword in job_other_company:
        if keyword in message_lower:
            return 'job_other_company'
    
    # Check for other company mentions
    for keyword in other_company_keywords:
        if keyword in message_lower and 'job' not in message_lower:
            return 'other_company'
    
    # Check for unrelated topics
    for keyword in unrelated_keywords:
        if keyword in message_lower:
            return 'unrelated'
    
    # Check for generic unrelated
    if message_lower in generic_unrelated:
        return 'unrelated'
    
    return None

def is_how_are_you_question(message: str) -> bool:
    """Check if message is asking 'how are you' type questions"""
    message_lower = message.lower().strip()
    how_are_you_patterns = [
        'how are you', 'how are you doing', 'how do you do', 'how\'s it going',
        'how\'s your day', 'how\'s everything', 'how\'s life', 'what\'s up',
        'how\'s work', 'how\'s things', 'how are things', 'how\'s your day going',
        'kaise ho', 'kaise hain', 'aap kaise hain', 'tum kaise ho',
        'aap kaise hain', 'kaise chal raha hai', 'sab theek hai'
    ]
    
    for pattern in how_are_you_patterns:
        if pattern in message_lower:
            return True
    
    return False

def is_emotional_expression(message: str) -> bool:
    """Check if user is expressing emotions or feelings"""
    message_lower = message.lower().strip()
    
    emotional_patterns = [
        'i am happy', 'i am sad', 'i am frustrated', 'i am angry', 'i am upset',
        'i am excited', 'i am worried', 'i am disappointed', 'i am pleased',
        'i am annoyed', 'i am good', 'i am bad', 'i am fine', 'i am alright',
        'i am perfect', 'i am excellent', 'i am wonderful', 'i am great',
        'i am terrible', 'i am awful', 'i am amazing', 'i am fantastic',
        'i am perform well', 'i am facing difficulties', 'i am upset with',
        'i am frustrated with', 'i am happy with', 'i am sad about',
        'i am excited about', 'i am worried about', 'i am disappointed with',
        'i am pleased with', 'i am angry about', 'i am annoyed with',
        'upset with', 'frustrated with', 'happy with', 'sad about',
        'excited about', 'worried about', 'disappointed with', 'pleased with',
        'angry about', 'annoyed with', 'excited for', 'worried for',
        'disappointed in', 'pleased about'
    ]
    
    return any(pattern in message_lower for pattern in emotional_patterns)

def is_user_doubt(message: str) -> bool:
    """Check if user is expressing doubt about chatbot's ability to help"""
    message_lower = message.lower().strip()
    
    doubt_patterns = [
        'i don\'t think you could help me', 'i don\'t think you can help',
        'you probably can\'t help', 'i doubt you can help', 'not sure you can help',
        'i don\'t think you could', 'i don\'t think you can', 'you can\'t help',
        'you probably can\'t', 'i doubt you', 'not sure you', 'you won\'t be able',
        'you might not be able', 'i\'m not sure you', 'i don\'t believe you',
        'you probably won\'t', 'you likely can\'t', 'you may not be able',
        'i don\'t think so', 'probably not', 'doubt it', 'not confident',
        'you\'re not helpful', 'you\'re not useful', 'you can\'t do it',
        'you won\'t understand', 'you don\'t know', 'you\'re not smart enough'
    ]
    
    return any(pattern in message_lower for pattern in doubt_patterns)

def is_help_request(message: str) -> bool:
    """Check if user is asking for help or facing difficulties - simplified to core keywords only"""
    message_lower = message.lower().strip()
    
    # FIRST: Check for minimal negative patterns - if found, return False (let RAG handle)
    negative_patterns = [
        "don't need", "dont need", "do not need",
        "don't want", "dont want", "do not want",
        "i don't need", "i dont need", "i do not need",
        "i am not your client", "i am not your customer", "i'm not your client"
    ]
    
    if any(pattern in message_lower for pattern in negative_patterns):
        return False  # Negative case - let RAG handle with human-like responses
    
    # SECOND: Check for core help keywords only (5-10 keywords)
    # Service-specific queries will be handled by project_manager or RAG flow
    help_keywords = [
        'help', 'assistance', 'support', 'guidance', 'trouble',
        'stuck', 'confused', 'problems', 'issues', 'difficulties'
    ]
    
    return any(keyword in message_lower for keyword in help_keywords)

def is_new_user_indication(message: str) -> bool:
    """Check if user is indicating they are new/first time"""
    message_lower = message.lower().strip()
    
    new_user_patterns = [
        'first time', 'new here', 'just started', 'new user', 'new customer',
        'i am coming', 'coming on', 'first visit', 'never been', 'never used',
        'don\'t know', 'don\'t have', 'don\'t think', 'could help',
        'how you found', 'found my project', 'i am coming on {company_lower} first time',
        'coming on {company_lower} first time', 'first time here', 'never used this',
        'never been here', 'just started using', 'new to this'
    ]
    
    for token in COMPANY_TOKENS:
        new_user_patterns.extend([
            f'new to {token}',
            f'first visit to {token}',
            f'never used {token}',
            f'never been on {token}'
        ])
    
    return any(pattern in message_lower for pattern in new_user_patterns)
def clear_project_context(session_id: str):
    """Clear project context for a session"""
    if session_id in conversation_sessions:
        if 'project_context' in conversation_sessions[session_id]:
            conversation_sessions[session_id]['project_context'] = {}
            logger.info(f"Cleared project context for session: {session_id}")

def is_personal_introduction(message: str) -> bool:
    """Check if message contains personal introduction/name sharing"""
    message_lower = message.lower().strip()
    
    # First check if it's dissatisfaction - if so, don't treat as name introduction
    if is_dissatisfaction(message):
        return False
    
    # COMPREHENSIVE CONTEXT ANALYSIS - Check this FIRST before any pattern matching
    def has_context_words(text):
        """Check if text contains context words that indicate it's not a name introduction"""
        context_indicators = [
            # Action words and verbs
            'looking', 'searching', 'finding', 'seeking', 'asking', 'telling',
            'wanting', 'needing', 'trying', 'doing', 'going', 'coming',
            'working', 'playing', 'running', 'walking', 'sitting', 'standing',
            'eating', 'drinking', 'sleeping', 'waking', 'buying', 'selling',
            'helping', 'using', 'opening', 'closing', 'starting', 'stopping',
            'beginning', 'ending', 'finishing', 'getting', 'making', 'taking',
            'giving', 'seeing', 'knowing', 'thinking', 'having', 'being',
            # Purpose and intention words
            'for', 'about', 'regarding', 'concerning', 'to', 'with', 'by',
            'want', 'need', 'require', 'interested', 'curious', 'wondering',
            # Technology and service words
            'website', 'chatbot', 'bot', 'system', 'service', 'company', 'business',
            'project', 'work', 'job', 'task', 'problem', 'issue', 'question', 'answer',
            'solution', 'help', 'assistance', 'support', 'information', 'data',
            'crm', 'erp', 'ai', 'cloud', 'computing', 'development', 'software',
            # Comparison words
            'just like', 'similar to', 'for my', 'for your', 'for the', 'like you',
            'same as', 'like this', 'like that', 'as you', 'as me',
            # Common nouns that are not names
            'time', 'day', 'night', 'morning', 'evening', 'year', 'month', 'week',
            'place', 'location', 'area', 'city', 'country', 'world', 'earth',
            'thing', 'stuff', 'item', 'object', 'product', 'tool',
            'book', 'movie', 'music', 'food', 'water', 'money', 'price', 'cost'
        ]
        words = text.lower().split()
        return any(indicator in words for indicator in context_indicators)
    
    # If context words are detected, reject immediately
    if has_context_words(message_lower):
        return False
    
    # Check for emotional words that should not be treated as names
    # Include common misspellings
    emotional_words = ['frustrated', 'frustated', 'frustate', 'angry', 'sad', 'happy', 'excited', 'worried', 'disappointed', 'pleased', 'upset', 'annoyed', 'irritated']
    if any(word in message_lower for word in emotional_words):
        return False
    
    # Check for phone number patterns - these are not introductions
    import re
    phone_patterns = [
        r'\b\d{10}\b',  # 10 digits
        r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
        r'\b\+?\d{1,3}\s?\d{10}\b',  # +91 1234567890
        r'\b\d{5}\s?\d{5}\b'  # 12345 67890
    ]
    for pattern in phone_patterns:
        if re.search(pattern, message):
            return False
    
    # More specific name introduction patterns - require explicit introduction phrases
    intro_patterns = [
        'my name is', 'i\'m called', 'people call me', 'you can call me',
        'main hun', 'mera naam', 'mujhe kehte hain'
    ]
    
    # Check for name patterns - but be very strict
    for pattern in intro_patterns:
        if pattern in message_lower:
            # Additional validation: make sure it's not a complaint or question
            words = message_lower.split()
            # Exclude if contains negative words, questions, or action verbs
            excluded_words = ['not', 'bad', 'wrong', 'satisfied', 'happy', 'good', 'helpful', 
                            'asking', 'wondering', 'curious', 'want', 'need', 'looking', 
                            'interested', 'about', 'regarding', 'concerning', 'frustrated', 'frustated',
                            'angry', 'sad', 'excited', 'worried', 'disappointed', 'pleased', 'upset',
                            'on', 'at', 'in', 'to', 'for', 'with', 'by', 'from']
            if not any(word in excluded_words for word in words):
                return True
    
    # ENHANCED "i am" pattern handling - much stricter but supports full names
    if ('i am' in message_lower or 'i\'m' in message_lower):
        parts = message_lower.split()
        for i, part in enumerate(parts):
            if part in ['am', 'i\'m'] and i + 1 < len(parts):
                # Check for full names (up to 3 words after "i am")
                potential_name_parts = []
                for j in range(i + 1, min(i + 4, len(parts))):  # Check up to 3 words
                    potential_name_parts.append(parts[j])
                
                # Check if any of the potential name parts are context words
                potential_name_str = ' '.join(potential_name_parts)
                if has_context_words(potential_name_str):
                    continue
                
                # Check if all parts look like valid name components
                valid_name_parts = []
                for part in potential_name_parts:
                    # Skip common words that shouldn't be in names
                    if part in ['frustrated', 'frustated', 'frustate', 'angry', 'sad', 'happy', 'excited', 'worried', 
                               'disappointed', 'pleased', 'good', 'bad', 'right', 'wrong', 'upset', 'annoyed',
                               'looking', 'searching', 'finding', 'seeking', 'asking', 'telling', 'wanting', 'needing',
                               'website', 'chatbot', 'bot', 'system', 'service', 'company', 'business', 'project',
                               'work', 'job', 'task', 'problem', 'issue', 'question', 'answer', 'solution',
                               'help', 'assistance', 'support', 'information', 'for', 'about', 'regarding',
                               'just', 'like', 'similar', 'to', 'with', 'by', 'from', 'on', 'at', 'in',
                               'the', 'a', 'an', 'and', 'or', 'but', 'so', 'yet', 'nor', 'for', 'of',
                               'difficulties', 'perform', 'well', 'system', 'aren\'t', 'working', 'can\'t', 'find',
                               'facing', 'trying', 'going', 'wanting', 'needing', 'looking', 'searching',
                               'first', 'time', 'new', 'here', 'just', 'started', 'coming', 'found']:
                        break
                    
                    # Check if it looks like a valid name part (alphabetic, reasonable length)
                    if part.isalpha() and 2 <= len(part) <= 20:
                        valid_name_parts.append(part)
                    else:
                        break
                
                # If we have valid name parts, check if there are more words after
                if valid_name_parts:
                    remaining_words = parts[i + 1 + len(valid_name_parts):]
                    # If there are more words after the name, they should not be context words
                    if remaining_words and has_context_words(' '.join(remaining_words)):
                        continue
                    
                    # Final validation: ensure no context words in the entire sentence
                    if not has_context_words(message_lower):
                        return True
    
    return False

def is_name_recall_question(message: str) -> bool:
    """Check if user is asking about their own name"""
    message_lower = message.lower().strip()
    
    # More specific name recall patterns - avoid false positives
    name_recall_patterns = [
        'what is my name', 'what\'s my name', 'do you know my name', 
        'do you remember my name', 'what did i tell you my name', 
        'can you tell me my name', 'tell me my name',
        'mera naam kya hai', 'naam yaad hai', 'kya naam hai'
    ]
    
    for pattern in name_recall_patterns:
        if pattern in message_lower:
            return True
    
    # Special case for "my name" - only if it's a question or request
    if 'my name' in message_lower:
        # Check if it's part of a question or request
        if any(word in message_lower for word in ['what', 'tell', 'remember', 'know', 'recall']):
            return True
    
    return False

def is_personality_question(message: str) -> bool:
    """Check if message is asking about personality/friendliness"""
    message_lower = message.lower().strip()
    personality_patterns = [
        'are you friendly', 'are you nice', 'are you helpful', 'are you good',
        'would you like to be my friend', 'can we be friends', 'be my friend',
        'are you a friend', 'do you like me', 'do you care', 'are you kind',
        'are you warm', 'are you personable', 'are you approachable',
        'are you welcoming', 'are you supportive', 'are you understanding',
        'are you patient', 'are you gentle', 'are you caring',
        'tum dost ban sakte ho', 'dost bano', 'aap dost hain', 'dost banoge',
        'aap achhe hain', 'aap friendly hain', 'aap helpful hain'
    ]
    
    for pattern in personality_patterns:
        if pattern in message_lower:
            return True
    
    return False
def get_how_are_you_response(user_language: str = 'english') -> str:
    """Generate natural response for 'how are you' questions"""
    import random
    
    if user_language == 'hindi':
        responses = [
            f"Main bilkul theek hun! Dhanyawad puchhne ke liye. Main yahan hun aapko {COMPANY_NAME} ke services ke baare mein batane ke liye. Aap kya janna chahenge?",
            f"Main achha hun! Main aapka AI assistant hun aur {COMPANY_NAME} ki services ke baare mein madad karne ke liye ready hun.",
            f"Main theek hun! Bataiye, aapko {COMPANY_NAME} ke baare mein kya jankari chahiye?"
        ]
    else:
        responses = [
            f"I'm doing great, thank you! I'm here to help you explore {COMPANY_NAME}'s services.",
            f"I'm excellent and ready to share details about {COMPANY_NAME}'s offerings.",
            f"I'm here and happy to help you learn more about what {COMPANY_NAME} can do."
        ]
    
    return random.choice(responses)
def extract_name_from_message(message: str) -> Optional[str]:
    """Extract user's name from introduction message with enhanced validation"""
    import re
    message_lower = message.lower().strip()
    name = None
    
    # Skip if it's dissatisfaction
    if is_dissatisfaction(message):
        return None
    
    # Comprehensive blacklists for name extraction
    excluded_words = {
        # Prepositions
        'on', 'at', 'in', 'to', 'for', 'with', 'by', 'from', 'about', 'into', 
        'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 
        'off', 'over', 'under', 'again', 'further', 'then', 'once',
        # Articles and common words
        'the', 'a', 'an', 'and', 'or', 'but', 'so', 'yet', 'for', 'nor',
        # Negative words
        'not', 'no', 'never', 'neither', 'none', 'nothing', 'nobody',
        # Satisfaction/emotion words - expanded list with common misspellings
        'satisfied', 'happy', 'good', 'bad', 'wrong', 'right', 'sad', 'angry',
        'frustrated', 'frustated', 'frustate', 'disappointed', 'pleased', 'excited', 'worried',
        'mad', 'upset', 'annoyed', 'irritated', 'confused', 'lost', 'tired',
        # Common verbs (expanded)
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'can', 'shall', 'go', 'come', 'get', 'make', 'take',
        'give', 'see', 'know', 'think', 'want', 'need', 'like', 'love', 'hate',
        'find', 'look', 'search', 'seek', 'ask', 'tell', 'say', 'speak', 'talk',
        'listen', 'hear', 'read', 'write', 'work', 'play', 'run', 'walk', 'sit',
        'stand', 'eat', 'drink', 'sleep', 'wake', 'buy', 'sell', 'help', 'try',
        'use', 'open', 'close', 'start', 'stop', 'begin', 'end', 'finish',
        'looking', 'searching', 'finding', 'seeking', 'asking', 'telling',
        'wanting', 'needing', 'trying', 'doing', 'going', 'coming', 'being',
        'having', 'getting', 'making', 'taking', 'giving', 'seeing', 'knowing',
        'thinking', 'working', 'playing', 'running', 'walking', 'sitting',
        'standing', 'eating', 'drinking', 'sleeping', 'waking', 'buying',
        'selling', 'helping', 'using', 'opening', 'closing', 'starting',
        'stopping', 'beginning', 'ending', 'finishing',
        # Pronouns
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
        # Numbers (common ones)
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'zero', 'first', 'second', 'third', 'last',
        # Other common words
        'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'why', 'how',
        'what', 'who', 'which', 'whom', 'whose',
        # Common nouns that are not names
        'website', 'chatbot', 'bot', 'system', 'service', 'company', 'business',
        'project', 'work', 'job', 'task', 'problem', 'issue', 'question', 'answer',
        'solution', 'help', 'assistance', 'support', 'information', 'data',
        'time', 'day', 'night', 'morning', 'evening', 'year', 'month', 'week',
        'place', 'location', 'area', 'city', 'country', 'world', 'earth',
        'thing', 'stuff', 'item', 'object', 'product', 'service', 'tool',
        'book', 'movie', 'music', 'food', 'water', 'money', 'price', 'cost'
    }
    
    def is_phone_number(text):
        """Check if text is a phone number"""
        # Remove common separators and spaces
        clean_text = re.sub(r'[\s\-\(\)\+]', '', text)
        # Check if it's all digits and reasonable length for phone number
        if clean_text.isdigit() and 7 <= len(clean_text) <= 15:
            return True
        # Check for specific patterns
        phone_patterns = [
            r'\b\d{10}\b',  # 10 digits
            r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
            r'\b\+?\d{1,3}\s?\d{10}\b',  # +91 1234567890
            r'\b\d{5}\s?\d{5}\b'  # 12345 67890
        ]
        for pattern in phone_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def contains_phone_number(text):
        """Check if text contains phone number pattern"""
        return is_phone_number(text)
    
    def has_context_words(text):
        """Check if text contains context words that indicate it's not a name introduction"""
        context_indicators = [
            'looking', 'searching', 'finding', 'seeking', 'asking', 'telling',
            'wanting', 'needing', 'trying', 'doing', 'going', 'coming',
            'for', 'about', 'regarding', 'concerning', 'website', 'chatbot',
            'bot', 'system', 'service', 'company', 'business', 'project',
            'work', 'job', 'task', 'problem', 'issue', 'question', 'answer',
            'solution', 'help', 'assistance', 'support', 'information'
        ]
        words = text.lower().split()
        return any(indicator in words for indicator in context_indicators)
    
    def is_valid_name(word):
        """Enhanced name validation with multiple layers"""
        if not word or len(word) < 2:
            return False
        
        # Remove punctuation
        word = word.strip('.,!?;:')
        
        # Must be alphabetic only
        if not word.isalpha():
            return False
        
        # Length check: 2-20 characters
        if len(word) < 2 or len(word) > 20:
            return False
        
        # Must not be in excluded words blacklist
        if word.lower() in excluded_words:
            return False
        
        # Must not be a phone number
        if is_phone_number(word):
            return False
        
        # Must not start with common non-name prefixes
        if word.lower().startswith(('mr', 'mrs', 'ms', 'dr', 'prof')):
            return False
    
        # Capitalization check: proper names should be properly capitalized
        # Allow: "John", "Ravi", "Sarah"
        # Reject: "JOHN" (all caps), "john" (all lowercase if > 3 chars)
        if word.isupper() and len(word) > 1:
            return False
        if word.islower() and len(word) > 10:  # Very long lowercase words are likely not names
            return False
        
        # Additional check: reject if it looks like a verb/noun
        if len(word) > 6 and word.lower() in ['looking', 'searching', 'finding', 'seeking', 'asking', 'telling']:
            return False
        
        return True
    
    # Context analysis - reject if sentence contains context words
    if has_context_words(message_lower):
        return None
    
    # Try to extract name from very specific patterns only
    if 'my name is' in message_lower:
        parts = message_lower.split('my name is')
        if len(parts) > 1:
            remaining = parts[1].strip()
            # Get first word only and validate
            words = remaining.split()
            if words:
                potential_name = words[0]
                if is_valid_name(potential_name):
                    name = potential_name.capitalize()
    
    elif 'i am' in message_lower and not contains_phone_number(message):
        # Only extract if "i am" is followed by a single word and it's the end of meaningful content
        parts = message_lower.split()
        for i, part in enumerate(parts):
            if part == 'am' and i + 1 < len(parts):
                potential_name = parts[i + 1]
                # Check if this is likely the end of the name introduction
                remaining_words = parts[i + 2:] if i + 2 < len(parts) else []
                # If there are more words after potential name, they should not be context words
                if remaining_words and has_context_words(' '.join(remaining_words)):
                    continue
                if is_valid_name(potential_name):
                    name = potential_name.capitalize()
                break
    
    elif 'call me' in message_lower:
        # Check if it's a phone number request first
        if contains_phone_number(message):
            return None
        parts = message_lower.split('call me')
        if len(parts) > 1:
            remaining = parts[1].strip()
            # Check if remaining text contains "on" followed by numbers (phone pattern)
            if ' on ' in remaining:
                return None
            words = remaining.split()
            for word in words:
                if is_valid_name(word):
                    name = word.capitalize()
                    break
    
    elif 'mera naam' in message_lower:
        parts = message_lower.split('mera naam')
        if len(parts) > 1:
            potential_name = parts[1].strip().split()[0]
            if is_valid_name(potential_name):
                name = potential_name.capitalize()
    
    elif 'main' in message_lower and 'hun' in message_lower:
        # Handle "main [name] hun" pattern
        parts = message_lower.split()
        try:
            main_index = parts.index('main')
            hun_index = parts.index('hun')
            if hun_index == main_index + 2:  # name is between 'main' and 'hun'
                potential_name = parts[main_index + 1]
                if is_valid_name(potential_name):
                    name = potential_name.capitalize()
        except (ValueError, IndexError):
            pass
    
    # Final validation - ensure name is not a common word
    if name and is_valid_name(name):
        return name
    return None

def is_bot_identity_question(message: str) -> bool:
    """Check if message is asking about bot's identity/name"""
    message_lower = message.lower().strip()
    patterns = [
        'what is your name', 'tell me your name', 'who are you',
        'what do you call yourself', 'your name', 'what\'s your name',
        'apka naam kya hai', 'tumhara naam kya hai', 'aap kaun hain'
    ]
    return any(pattern in message_lower for pattern in patterns)

def get_bot_identity_response(language: str = 'english') -> str:
    """Generate response when user asks about bot's identity"""
    import random
    
    profile = get_domain_profile()
    if language == 'hindi':
        responses = [
            f"Main aapka AI assistant hun, {profile.company_name} ke liye.",
            "Mera naam AI assistant hai. Main aapka AI assistant hun.",
            "Main aapka AI assistant hun, jo aapki madad ke liye yahan hun.",
            f"Namaste! Main aapka AI assistant hun, {profile.company_name} ke liye.",
            "Main aapka AI chatbot hun, jo aapki madad ke liye yahan hun.",
            f"Main aapka AI assistant hun - {profile.company_name} ke services ke baare mein jaankari dene ke liye.",
            f"Main aapka AI assistant hun, jo aapko {profile.company_name} ke baare mein batata hun."
        ]
    else:
        responses = [
            f"I'm your AI assistant, here to help you with {profile.company_name}.",
            f"I'm your AI assistant. I'm here to help you with {profile.company_name}.",
            "I'm your AI assistant, an AI-powered chatbot created to help you.",
            f"Hello! I'm your AI assistant, your friendly helper for {profile.company_name}.",
            f"I'm your AI assistant - I can help you learn about {profile.company_name}.",
            f"Hi there! I'm your AI assistant, here to assist you with {profile.company_name}.",
            f"I'm your AI assistant, designed to help with {profile.company_name}.",
            f"Greetings! I'm your AI assistant, your go-to AI for {profile.company_name}.",
            f"I'm your AI assistant, created to help you with {profile.company_name}.",
            f"Hello! I'm your AI assistant, your companion for {profile.company_name}."
        ]
    
    return random.choice(responses)

def get_name_recall_response(stored_name: Optional[str], user_language: str = 'english') -> str:
    """Generate response when user asks about their name"""
    import random
    
    if user_language == 'hindi':
        if stored_name:
            responses = [
                f"Haan, aapka naam {stored_name} hai! Main yaad rakhta hun. Aap kya janna chahte hain {company} ke bare mein?",
                f"Bilkul yaad hai! Aap {stored_name} hain. Main yahan hun {company} ke bare mein madad karne ke liye. Kya puchhna chahte hain?",
                f"Ji haan, aapne mujhe bataya tha aapka naam {stored_name} hai. Main aapki kya madad kar sakta hun aaj?"
            ]
        else:
            responses = [
                "Maafi chahta hun, aapne abhi tak mujhe apna naam nahi bataya. Aap mujhe bata sakte hain? Main yaad rakhna chahunga!",
                "Main nahi jaanta aapka naam. Kya aap mujhe bata sakte hain? Main aapka AI assistant hun aur aapki madad karna chahta hun.",
                "Aapne mujhe apna naam nahi bataya hai. Kya aap share kar sakte hain? Main yaad rakhunga!"
            ]
    else:
        if stored_name:
            responses = [
                f"Yes, your name is {stored_name}! I remember. I'm here to help you with {company}.",
                f"Of course I remember! You're {stored_name}. I'm here to help you with {company}.",
                f"Yes, you told me your name is {stored_name}. I'm here to assist you with {company}."
            ]
        else:
            responses = [
                "I'm sorry, you haven't told me your name yet. Would you like to share it? I'd love to remember it!",
                "I don't know your name yet. Could you tell me? I'm your AI assistant and I'd like to help you.",
                "You haven't shared your name with me. Would you like to? I'll remember it!"
            ]
    
    return random.choice(responses)
def get_personal_introduction_response(message: str, user_language: str = 'english') -> str:
    """Generate response for personal introductions"""
    import random
    
    # Extract name using helper function
    name = extract_name_from_message(message)
    
    if user_language == 'hindi':
        if name:
            responses = [
                f"Namaste {name}! Aapse mil kar khushi hui. Main aapka AI assistant hun aur {company} ke bare mein madad kar sakta hun. Aap kya janna chahte hain?",
                f"Hello {name}! Aapka swagat hai. Main yahan hun {company} ke bare mein jaankari dene ke liye. Kya puchhna chahte hain?",
                f"Hi {name}! Main aapka friendly AI assistant hun. {company} ke bare mein koi sawal hai?"
            ]
        else:
            responses = [
                f"Namaste! Aapse mil kar achha laga. Main aapka AI assistant hun aur {company} ke bare mein madad kar sakta hun. Aap kya janna chahte hain?",
                f"Hello! Aapka swagat hai. Main yahan hun {company} ke bare mein jaankari dene ke liye. Kya puchhna chahte hain?",
                f"Hi! Main aapka friendly AI assistant hun. {company} ke bare mein koi sawal hai?"
            ]
    else:
        if name:
            responses = [
                f"Nice to meet you, {name}! I'm your AI assistant and I'm here to help you with {company}.",
                f"Hello {name}! Great to meet you. I can help you learn about {company}.",
                f"Hi {name}! I'm your friendly AI assistant. I'm here to help you with {company}."
            ]
        else:
            responses = [
                f"Nice to meet you! I'm your AI assistant and I'm here to help you with {company}.",
                f"Hello! Great to meet you. I can help you learn about {company}.",
                f"Hi! I'm your friendly AI assistant. I'm here to help you with {company}."
            ]
    
    return random.choice(responses)
def get_personality_response(message: str, user_language: str = 'english') -> str:
    """Generate warm, friendly response for personality questions"""
    message_lower = message.lower().strip()
    
    if user_language == 'hindi':
        # Hindi personality responses
        if any(word in message_lower for word in ['dost', 'friendly', 'achhe', 'helpful']):
            responses = [
                "Haan bilkul! Main aapka friendly assistant hun. Main yahan hun aapki help ke liye {company} ke services ke bare me. Aap kya janna chahte hain?",
                f"Zaroor! Main aapke saath friendly way me baat karne ke liye ready hun. {COMPANY_NAME} ke bare me koi sawal hai?",
                "Bilkul friendly hun! Main aapki har possible help kar sakta hun {company} ke services ke bare me. Kya puchhna chahte hain?",
                f"Haan main dost ban sakta hun! Main yahan hun aapki madad ke liye {COMPANY_NAME} ke bare mein. Aaj main aapki kya madad kar sakta hun?",
                f"Bilkul! Main aapka dost ban kar aapki madad karna chahunga. Mera kaam {COMPANY_NAME} ke bare mein jaankari dena hai. Aaj main aapki kya madad kar sakta hun?"
            ]
        else:
            responses = [
                "Haan bilkul! Main aapka friendly assistant hun. Main yahan hun aapki help ke liye {company} ke services ke bare me. Aap kya janna chahte hain?",
                f"Zaroor! Main aapke saath friendly way me baat karne ke liye ready hun. {COMPANY_NAME} ke bare me koi sawal hai?",
                "Bilkul friendly hun! Main aapki har possible help kar sakta hun {company} ke services ke bare me. Kya puchhna chahte hain?"
            ]
    else:
        # English personality responses
        if any(word in message_lower for word in ['friend', 'dost']):
            responses = [
                f"I'd be happy to help you as a friendly AI assistant! My purpose is to provide information about {COMPANY_NAME}.",
                f"As an AI, I don't have personal feelings, but I'm designed to be very helpful and friendly! I'm here to assist you with {COMPANY_NAME}.",
                f"I'm designed to be your friendly AI assistant! I'm here to help you learn about {COMPANY_NAME}."
            ]
        else:
            responses = [
                "Yes, I try to be friendly and helpful! I'm here to assist you with {company}'s services in a warm, supportive way.",
                f"Absolutely! I'm designed to be your friendly AI assistant. I'd be happy to help you learn about {COMPANY_NAME}.",
                "Of course! I'm here to be your helpful, friendly guide to {company}'s offerings.",
                "Yes, I aim to be friendly and approachable! I'm excited to help you discover what {company} can do for your business."
            ]
    
    import random
    return random.choice(responses)

def get_emotional_response(message: str, user_language: str = "en") -> str:
    """Generate appropriate response for emotional expressions"""
    message_lower = message.lower().strip()
    
    if any(word in message_lower for word in ['happy', 'excited', 'pleased', 'good', 'great', 'wonderful', 'amazing', 'fantastic']):
        profile = get_domain_profile()
        return apply_company_placeholders(f"I'm glad to hear that! I'm here to help with questions about {profile.summary_of_offerings()}.")
    
    elif any(word in message_lower for word in ['sad', 'frustrated', 'angry', 'upset', 'annoyed', 'disappointed', 'worried']):
        profile = get_domain_profile()
        return apply_company_placeholders(f"I'm sorry to hear that. I'm here to help you with {profile.summary_of_offerings()}. Let me know what you need.")
    
    elif any(word in message_lower for word in ['bad', 'terrible', 'awful']):
        return "I understand your concerns. I'm committed to providing excellent service. Let me know what specific issues you're facing."
    
    else:
        profile = get_domain_profile()
        return apply_company_placeholders(f"I understand how you're feeling. I'm here to help with questions about {profile.summary_of_offerings()}.")

def get_user_doubt_response(message: str, user_language: str = "en") -> str:
    """Generate empathetic response for user doubt scenarios"""
    message_lower = message.lower().strip()
    
    # Check for specific doubt patterns and provide tailored responses
    if any(word in message_lower for word in ['thank you', 'thanks']):
        return apply_company_placeholders("I understand your hesitation. I'm designed to help with {primary_offerings_summary}. I've helped many clients with similar concerns.")
    
    elif any(word in message_lower for word in ['not helpful', 'not useful', 'can\'t do it']):
        return apply_company_placeholders("I understand your frustration. I'm constantly learning and improving. Our team at {company} specializes in {primary_offerings_summary}.")
    
    elif any(word in message_lower for word in ['don\'t know', 'won\'t understand', 'not smart enough']):
        return apply_company_placeholders("I understand your concern. While I may not know everything, I'm designed to help with {primary_offerings_summary}. I can access our company's knowledge base and connect you with our expert team when needed.")
    
    else:
        return apply_company_placeholders("I understand your hesitation. I'm designed to help with {primary_offerings_summary}. I've helped many clients with similar concerns.")

def get_help_response(message: str, user_language: str = "en") -> Optional[str]:
    """Generate appropriate response for help requests - simplified to let RAG handle most cases"""
    message_lower = message.lower().strip()
    
    # Check for minimal negative patterns - if found, return None to let RAG handle
    negative_patterns = [
        "don't need", "dont need", "do not need",
        "don't want", "dont want", "do not want",
        "i don't need", "i dont need", "i do not need",
        "i am not your client", "i am not your customer", "i'm not your client"
    ]
    
    if any(pattern in message_lower for pattern in negative_patterns):
        return None  # Let RAG flow handle negative cases with human-like responses
    
    # For all other help requests, return None to let RAG handle
    # RAG will provide better context-aware, human-like, and accurate responses
    return None

def is_off_topic(message: str) -> bool:
    """Detect if a query is off-topic or unrelated to {company} business"""
    return get_off_topic_category(message) is not None

def get_off_topic_response(message: str) -> str:
    """Generate human-like, categorized response for off-topic queries"""
    import random
    
    category = get_off_topic_category(message)
    
    # If category is None (e.g., personnel query excluded), return None to let RAG handle it
    if category is None:
        return None
    
    responses = {
        'abusive': [
            "I appreciate you reaching out, but let's keep our conversation professional. I'm here to help you learn about {company}'s services.",
            
            "I understand you might be frustrated, but I'm here to assist you professionally. I can help you with {company}'s {primary_offerings_summary}.",
            
            "Let's focus on how {company} can help you. I'm here to discuss our services in a respectful manner."
        ],
        
        'other_company': [
            "I specialize in {company}'s services rather than other companies. However, I'd be happy to tell you about what we offer! We provide {primary_offerings_summary}.",
            
            "That's outside my expertise, but here's what I can tell you - {company} serves many clients. Our services can benefit you.",
            
            "While I focus on {company}'s services, I can share that our clients often choose us for our personalized approach and proven track record."
        ],
        
        'job_other_company': [
            "I can't help with opportunities at other companies, but I can tell you that {company} is growing! We're always looking for talented people.",
            
            "While I don't have insights into other companies' hiring, {company} is always looking for talented people! We offer {primary_offerings_summary}."
        ],
        
        'unrelated': [
            "That's a bit outside my wheelhouse! My expertise is in {company}'s services. We offer {primary_offerings_summary}.",
            
            "I wish I could help with that, but I specialize in {company}'s services! We can help with {primary_offerings_summary}.",
            
            "Ha, I'm not the best person for that question! But I'm great at discussing {company}'s services - we offer {primary_offerings_summary}.",
            "That topic is a little outside our scope. If you'd like to talk about {primary_offerings_summary}, I'm totally in my comfort zone.",
            "I'm focused on {company}'s services—{primary_offerings_summary}. Let me know if you want insights there, I'm happy to help."
        ],
        
        'general': [
            "I don't have specific information about that, but I can tell you all about {company}'s services. We've helped businesses transform with {primary_offerings_summary}.",
            
            "That's not quite in my area, but I'm really good at discussing {company}'s services! {company} specializes in {primary_offerings_summary}.",
            
            "I'm not able to assist with that particular topic. However, if you're looking for {company}'s services, you're in the right place! We offer {primary_offerings_summary}.",
            "Let me steer things back to what we do best—{primary_offerings_summary}. Ask me anything about those and I'll gladly dive in.",
            "While I can't cover that subject, I can absolutely help you explore {company}'s offerings: {primary_offerings_summary}."
        ]
    }
    
    response = random.choice(responses.get(category, responses['general']))
    # Apply company placeholders (universal approach)
    return apply_company_placeholders(response)

# Greeting detection function
def is_greeting(message: str) -> bool:
    """Check if message is a casual greeting"""
    # CRITICAL: Ensure message is a string before any operations
    if hasattr(message, '__await__'):
        logger.error(f"[is_greeting] CRITICAL: message is coroutine!")
        return False
    if not isinstance(message, str):
        message = str(message) if message else ""
    if not message:
        return False
    try:
        message = message.strip()
    except:
        return False
    greeting_words = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'namaste', 'namaskar', 'salam', 'hiya', 'howdy', 'greetings',
        'kaise ho', 'kaise hain', 'kaise hain aap', 'aap kaise hain'
    ]
    
    # message is already stripped, just need to lower it
    try:
        message_lower = message.lower()
    except:
        return False
    
    # Check for exact matches
    if message_lower in greeting_words:
        return True
    
    # Check for greetings with additional words (like "good morning sir")
    for greeting in greeting_words:
        if message_lower.startswith(greeting) and len(message_lower.split()) <= 3:
            return True
    
    return False
def get_greeting_response(message: str, user_language: str = 'english') -> str:
    """Generate appropriate casual response for greetings with domain context"""
    # CRITICAL: Ensure message is a string before any operations
    if hasattr(message, '__await__'):
        logger.error(f"[get_greeting_response] CRITICAL: message is coroutine!")
        message = ""
    if not isinstance(message, str):
        message = str(message) if message else ""
    if not message:
        return get_safe_fallback_reply()
    try:
        message_lower = message.lower().strip()
    except:
        return get_safe_fallback_reply()
    import random
    profile = get_domain_profile()
    
    if user_language == 'hindi':
        # Hindi greetings with variations
        if any(word in message_lower for word in ['namaste', 'namaskar']):
            responses = [
                f"Namaste! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein madad kar sakta hun. Aap kya janna chahte hain?",
                f"Namaskar! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                f"Namaste! Main yahan hun {COMPANY_NAME} ke bare mein madad karne ke liye. Aaj main aapki kya madad kar sakta hun?"
            ]
        elif any(word in message_lower for word in ['kaise ho', 'kaise hain', 'aap kaise hain']):
            responses = [
                f"Main theek hun! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein madad kar sakta hun. Aap kya janna chahte hain?",
                f"Bilkul theek hun! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                f"Main achha hun! Main yahan hun {COMPANY_NAME} ke bare mein madad karne ke liye."
            ]
        elif 'good morning' in message_lower:
            responses = [
                f"Good morning! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein madad kar sakta hun. Aap kya janna chahte hain?",
                f"Good morning! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                f"Good morning! Main yahan hun {COMPANY_NAME} ke bare mein madad karne ke liye."
            ]
        elif 'good afternoon' in message_lower:
            responses = [
                f"Good afternoon! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein madad kar sakta hun. Main aapki kya madad kar sakta hun?",
                f"Good afternoon! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                f"Good afternoon! Main yahan hun {COMPANY_NAME} ke bare mein madad karne ke liye."
            ]
        elif 'good evening' in message_lower:
            responses = [
                f"Good evening! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein madad kar sakta hun. Main aapki kya madad kar sakta hun?",
                f"Good evening! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                f"Good evening! Main yahan hun {COMPANY_NAME} ke bare mein madad karne ke liye."
            ]
        elif 'hi' in message_lower:
            responses = [
                f"Hi there! Main apka AI assistant hun. Main {COMPANY_NAME} ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                f"Hi there! Main apka AI assistant hun. Main {COMPANY_NAME} ke bare mein madad kar sakta hun. Kya puchhna chahte hain?",
                f"Hi there! Main apka AI assistant hun. Main {COMPANY_NAME} ke bare mein jaankari de sakta hun. Aap kya janna chahte hain?"
            ]
        elif 'hello' in message_lower:
            responses = [
                f"Hello! Main aapka AI assistant hun, yahan {COMPANY_NAME} ke bare mein jaankari dene ke liye. Aapko kya pasand hai?",
                f"Hello! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                f"Hello! Main yahan hun {COMPANY_NAME} ke bare mein madad karne ke liye."
            ]
        elif any(word in message_lower for word in ['hey']):
            responses = [
                f"Hey! Main aapka AI assistant hun. Aap mere se {COMPANY_NAME} ke bare mein puchh sakte hain. Main aapki kya madad kar sakta hun?",
                f"Hey there! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                f"Hey! Main yahan hun {COMPANY_NAME} ke bare mein madad karne ke liye."
            ]
        else:
            responses = [
                f"Hi! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein madad kar sakta hun. Aap kya janna chahte hain?",
                f"Hello! Main aapka AI assistant hun. Main {COMPANY_NAME} ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                f"Hi there! Main yahan hun {COMPANY_NAME} ke bare mein madad karne ke liye."
            ]
    else:
        # English greetings with variations
        if any(word in message_lower for word in ['namaste', 'namaskar']):
            responses = [
                f"Namaste! I'm your AI assistant. I can help you with {COMPANY_NAME}.",
                f"Namaskar! I'm your AI assistant. I can provide information about {COMPANY_NAME}.",
                f"Namaste! I'm here to help you with {COMPANY_NAME}. I'm ready to assist you."
            ]
        elif any(word in message_lower for word in ['kaise ho', 'kaise hain', 'aap kaise hain']):
            responses = [
                f"I'm doing great! I'm your AI assistant. I can help you learn about {COMPANY_NAME}.",
                f"I'm excellent! I'm your AI assistant. I can provide information about {COMPANY_NAME}.",
                f"I'm wonderful! I'm here to help you with {COMPANY_NAME}. I'm ready to assist you."
            ]
        elif 'good morning' in message_lower:
            responses = [
                f"Good morning! I'm your AI assistant. I can help you with {COMPANY_NAME}.",
                f"Good morning! I'm your AI assistant. I can provide information about {COMPANY_NAME}.",
                f"Good morning! I'm here to help you with {COMPANY_NAME}. I'm ready to assist you."
            ]
        elif 'good afternoon' in message_lower:
            responses = [
                f"Good afternoon! I'm your AI assistant. I can help you with {COMPANY_NAME}.",
                f"Good afternoon! I'm your AI assistant. I can provide information about {COMPANY_NAME}.",
                f"Good afternoon! I'm here to help you with {COMPANY_NAME}. I'm ready to assist you."
            ]
        elif 'good evening' in message_lower:
            responses = [
                f"Good evening! I'm your AI assistant. I can help you with {COMPANY_NAME}.",
                f"Good evening! I'm your AI assistant. I can provide information about {COMPANY_NAME}.",
                f"Good evening! I'm here to help you with {COMPANY_NAME}. I'm ready to assist you."
            ]
        elif 'hi' in message_lower:
            responses = [
                f"Hi there! I'm your friendly AI assistant. I can provide information about {COMPANY_NAME}.",
                f"Hi there! I'm your friendly AI assistant. I can help you learn about {COMPANY_NAME}.",
                f"Hi there! I'm your friendly AI assistant. I can assist you with information about {COMPANY_NAME}."
            ]
        elif 'hello' in message_lower:
            responses = [
                f"Hello! I'm your AI assistant, here to help you with information about {COMPANY_NAME}.",
                f"Hello! I'm your AI assistant. I can provide information about {COMPANY_NAME}.",
                f"Hello! I'm here to help you with {COMPANY_NAME}. I'm ready to assist you."
            ]
        elif any(word in message_lower for word in ['hey']):
            responses = [
                f"Hey there! I'm your AI assistant. I can help you with {COMPANY_NAME}.",
                f"Hey! I'm your AI assistant. I can provide information about {COMPANY_NAME}.",
                f"Hey there! I'm here to help you with {COMPANY_NAME}. I'm ready to assist you."
            ]
        else:
            responses = [
                f"Hi there! I'm your AI assistant. I can help you with {COMPANY_NAME}.",
                f"Hello! I'm your AI assistant. I can provide information about {COMPANY_NAME}.",
                f"Hi! I'm here to help you with {COMPANY_NAME}. I'm ready to assist you."
            ]
    
    selected = random.choice(responses)
    return apply_company_placeholders(selected)

# Service detection function
def detect_specific_service(message: str) -> Optional[str]:
    """Detect which specific service user is asking about using RAG (universal approach)"""
    # Removed hardcoded service keywords - use RAG semantic search instead
    # This allows detection of any service type (restaurant menu, hospital departments, etc.)
    # Service detection now happens via RAG search in the main chat flow
    return None  # Generic service inquiry - RAG will identify specific services
# Service inquiry detection
def is_service_inquiry(message: str) -> bool:
    """Check if message is asking about services"""
    message_lower = message.lower().strip()
    
    # Service inquiry patterns
    service_patterns = [
        'tell me about your services', 'what services do you offer', 'your services',
        'about your services', 'what services', 'services you provide', 'your offerings',
        'what do you offer', 'services offered', 'your service', 'about services',
        'what are your services', 'list your services', 'services available',
        'what services are available', 'services you have', 'your service offerings',
        'services provided', 'what services do you have', 'services offered by',
        'tell me about services', 'about your service', 'what service do you offer',
        'services you offer', 'your service offerings', 'what are the services',
        'services list', 'available services', 'services you provide'
    ]
    
    # Check for service inquiry patterns
    for pattern in service_patterns:
        if pattern in message_lower:
            return True
    
    # Check for combination of keywords
    service_keywords = ['services', 'service', 'offerings', 'offer', 'provide']
    inquiry_keywords = ['tell', 'what', 'about', 'list', 'show', 'describe']
    
    has_service_keyword = any(keyword in message_lower for keyword in service_keywords)
    has_inquiry_keyword = any(keyword in message_lower for keyword in inquiry_keywords)
    
    if has_service_keyword and has_inquiry_keyword:
        return True
    
    return False
# Acknowledgment detection
def is_acknowledgment(message: str) -> bool:
    """Check if message is acknowledgment/thanks - more specific to avoid catching service inquiries"""
    message_lower = message.lower().strip()
    
    # Pure acknowledgment words (single words or short phrases)
    pure_acknowledgments = [
        'thanks', 'thank you', 'thank', 'thx', 'thanx', 'appreciate',
        'ok', 'okay', 'got it', 'understood', 'clear', 'perfect', 'great',
        'nice', 'good', 'cool', 'awesome', 'makes sense', 'that helps',
        'theek hai', 'accha', 'samajh gaya', 'dhanyawad', 'shukriya',
        'bilkul', 'zaroor', 'sahi hai', 'badhiya'
    ]
    
    # Check for exact matches or very short phrases
    for word in pure_acknowledgments:
        if word == message_lower:  # Exact match
            return True
        # Only match if it's a short phrase (max 3 words) and doesn't contain service keywords
        elif (word in message_lower and 
              len(message_lower.split()) <= 3 and 
              not any(service_word in message_lower for service_word in ['service', 'services', 'tell', 'about', 'what', 'offer'])):
            return True
    
    return False

def get_acknowledgment_response(user_language: str = 'english') -> str:
    """Generate appropriate response for acknowledgments"""
    import random
    
    if user_language == 'hindi':
        responses = [
            f"Koi baat nahi! {company} ke bare me aur kuch puchh sakte hain.",
            f"Khushi hui madad kar ke! Agar aur koi sawal ho {company} ke bare me to puchhiye.",
            f"Aapka swagat hai! {company} ke bare mein aur jaankari chahiye to puchhiye.",
            f"Bilkul! {company} ke bare mein aur koi sawal ho to zaroor puchhiye.",
            f"Dhanyawad! {company} ke bare mein aur madad chahiye to main yahan hun."
        ]
    else:
        responses = [
            f"You're welcome! Feel free to ask if you need anything else about {company}.",
            f"Glad I could help! Let me know if you have more questions about {company}.",
            f"Happy to help! Don't hesitate to reach out if you need more information about {company}.",
            f"My pleasure! If you have more questions about {company}, I'm here to help.",
            f"You're very welcome! Feel free to ask about {company}."
        ]
    
    return random.choice(responses)

# Goodbye detection
def is_goodbye(message: str) -> bool:
    """Check if message is goodbye/end chat"""
    message_lower = message.lower().strip()
    goodbye_words = [
        'bye', 'goodbye', 'good bye', 'see you', 'later', 'end chat',
        'that\'s all', 'done', 'exit', 'quit', 'close',
        'namaste', 'alvida', 'phir milte hain', 'chaliye', 'bye bye',
        'kal milte hain', 'kal milte hai', 'kal baat karunga', 'kal baat karenge',
        'phir baat karte hain', 'phir baat karenge', 'baad me baat karte hain'
    ]
    
    for word in goodbye_words:
        if word in message_lower and len(message_lower.split()) <= 4:
            return True
    
    return False

def get_goodbye_response(user_language: str = 'english') -> str:
    """Generate appropriate goodbye response"""
    import random
    
    if user_language == 'hindi':
        responses = [
            "Dhanyawad {company} se baat karne ke liye! Agar future me aur sawal hain to main yahan hun. {company_lower}.com visit kariye.",
            f"Aapke saath baat kar ke achha laga! {COMPANY_NAME} ke bare me aur sawal ho to zaroor puchhiye. Achha din guzare!",
            "Shukriya {company} se baat karne ke liye! Agar aur koi sawal ho to main yahan hun. {company_lower}.com par jaankari le sakte hain.",
            f"Aapka dhanyawad! {COMPANY_NAME} ke bare mein aur jaankari chahiye to zaroor puchhiye. Achha din!",
            "Khushi hui {company} se baat kar ke! Agar future mein aur sawal hain to main yahan hun. {company_lower}.com visit kariye."
        ]
    else:
        responses = [
            "Thank you for chatting with {company}! If you have more questions in the future, I'm here to help. Visit {company_lower}.com for more information.",
            f"It was great talking with you! Feel free to return anytime with questions about {COMPANY_NAME}. Have a great day!",
            "Thanks for reaching out to {company}! Don't hesitate to come back if you need anything. Visit {company_lower}.com to explore our services.",
            f"My pleasure helping you! If you have more questions about {COMPANY_NAME}, I'm always here. Have a wonderful day!",
            "Thank you for choosing {company}! Feel free to return anytime for more information about our services. Visit {company_lower}.com!"
        ]
    
    return random.choice(responses)

# General help request detection
def is_general_help_request(message: str) -> bool:
    """Check if message is requesting general help (not asking about AI capabilities)"""
    message_lower = message.lower().strip()
    
    # FIRST: Check for negative patterns - if user says they DON'T need help, return False
    # This allows RAG flow to handle it and generate human-like acknowledgment
    negative_patterns = [
        "don't need", "dont need", "do not need", "don't want", "dont want", "do not want",
        "not need", "not want", "no need", "never need", "never want",
        "don't need help", "dont need help", "do not need help",
        "don't want help", "dont want help", "do not want help",
        "not need help", "not want help", "no need help",
        "i don't need", "i dont need", "i do not need",
        "i don't want", "i dont want", "i do not want",
        "i don't need your help", "i dont need your help", "i do not need your help",
        "i don't want your help", "i dont want your help", "i do not want your help"
    ]
    
    # If negative pattern matches, return False (let RAG flow handle it)
    for pattern in negative_patterns:
        if pattern in message_lower:
            return False
    
    # FIRST: Check for service-specific keywords - if present, this is NOT a general help request
    # These technical/service questions should go to RAG for intelligent, context-based answers
    service_keywords = [
        'erp', 'crm', 'cloud', 'hosting', 'iot', 'ai', 'artificial intelligence',
        'service', 'services', 'solution', 'solutions', 'system', 'systems',
        'application', 'applications', 'website', 'websites', 'software',
        'set up', 'setup', 'implement', 'implementation', 'install', 'installation',
        'develop', 'development', 'create', 'creation', 'build', 'building',
        'transform', 'transformation', 'migrate', 'migration', 'deploy', 'deployment',
        'automation', 'automate', 'digital transformation', 'business process', 'integration',
        'workflow', 'process automation', 'digitalization', 'digitization', 'api', 'apis'
    ]
    if any(keyword in message_lower for keyword in service_keywords):
        return False  # Service-specific help, let RAG/Groq handle it
    
    # Check for cybersecurity queries FIRST - these should be handled by project_manager
    cybersecurity_patterns = [
        'cybersecurity', 'cyber security', 'security', 'penetration testing',
        'vulnerability assessment', 'security audit', 'security testing',
        'ethical hacking', 'security consulting', 'security services',
        'need help with cybersecurity', 'help with cybersecurity', 'want help with cybersecurity',
        'can you help with cybersecurity', 'assistance with cybersecurity'
    ]

    # If it's a cybersecurity query, return False (let project_manager handle it)
    for pattern in cybersecurity_patterns:
        if pattern in message_lower:
            return False

    # Check for database design queries - these should be handled by project_manager
    database_patterns = [
        'database design', 'database designing', 'db design', 'database architecture',
        'need help with database design', 'help with database design', 'want help with database design',
        'can you help with database design', 'assistance with database design',
        'database help', 'db help', 'database services', 'db services',
        'i need help with database design', 'i need help with db design',
        'help me with database design', 'help me with db design',
        'i need help with database', 'i need help with db'
    ]

    # If it's a database design query, return False (let project_manager handle it)
    for pattern in database_patterns:
        if pattern in message_lower:
            return False
    
    # First check for service-specific help requests - these should NOT be general help
    # BUT exclude database design patterns from this check
    service_specific_patterns = [
        'can you help with', 'help with', 'need help with', 'want help with',
        'assistance with', 'support with', 'guidance with', 'help me with',
        'can you assist with', 'can you support with', 'can you guide with'
    ]
    
    # If it's a service-specific help request, return False (let ChromaDB handle it)
    # BUT skip if it's a database design query (already handled above)
    for pattern in service_specific_patterns:
        if pattern in message_lower:
            # Skip if it's a database design query - check if message contains database-related terms
            if not ('database' in message_lower or 'db ' in message_lower):
                return False
    
    # General help request patterns
    help_patterns = [
        'i need help with something else', 'i want another help', 'i need assistance',
        'help me with', 'i need support', 'i want help', 'i need help',
        'can you help me with', 'help me', 'assist me', 'support me',
        'i need guidance', 'i want assistance', 'i need some help',
        'help me out', 'can you assist', 'i need some assistance',
        'help me please', 'i need help please', 'assistance needed',
        'i need help with', 'help with', 'need help', 'want help'
    ]
    
    # Check for help request patterns
    for pattern in help_patterns:
        if pattern in message_lower:
            return True
    
    # Check for combination of keywords
    help_keywords = ['help', 'assistance', 'support', 'guidance']
    request_keywords = ['need', 'want', 'require', 'looking for', 'seeking']
    
    has_help_keyword = any(keyword in message_lower for keyword in help_keywords)
    has_request_keyword = any(keyword in message_lower for keyword in request_keywords)
    
    if has_help_keyword and has_request_keyword:
        return True
    
    return False

def is_capability_question(message: str) -> bool:
    """Check if message is asking about chatbot/AI capabilities"""
    message_lower = message.lower().strip()
    
    # FIRST: Check for service-specific keywords - if present, this is NOT a generic capability question
    # These technical/service questions should go to RAG for intelligent, context-based answers
    service_keywords = [
        'erp', 'crm', 'cloud', 'hosting', 'iot', 'ai', 'artificial intelligence',
        'service', 'services', 'solution', 'solutions', 'system', 'systems',
        'application', 'applications', 'website', 'websites', 'software',
        'set up', 'setup', 'implement', 'implementation', 'install', 'installation',
        'develop', 'development', 'create', 'creation', 'build', 'building',
        'transform', 'transformation', 'migrate', 'migration', 'deploy', 'deployment'
    ]
    if any(keyword in message_lower for keyword in service_keywords):
        return False  # Service-specific capability question, let RAG/Groq handle it
    
    # Capability question patterns
    capability_patterns = [
        'how can you help me', 'how can you help', 'what can you help me with',
        'what can you do', 'how do you help', 'what do you help with',
        'how do you work', 'what are your capabilities', 'what can you assist with',
        'how do you assist', 'what services do you provide', 'what can you offer',
        'i ask how can you help me', 'tell me how you can help', 'explain how you help',
        'what help can you provide', 'how do you support', 'what support do you offer'
    ]
    
    # Check for capability question patterns
    for pattern in capability_patterns:
        if pattern in message_lower:
            return True
    
    return False

def get_capability_response(user_language: str = 'english', session_id: str = None) -> str:
    """Generate response explaining chatbot capabilities with variety"""
    logger.info(f"DEBUG: get_capability_response called with session_id: {session_id}")
    if user_language == 'hindi':
        responses = [
            f"Main aapka AI assistant hun. Main {COMPANY_NAME} ke baare mein batata hun.",
            f"Main aapko service information provide karne ke liye yahan hun. Bataiye aapko {COMPANY_NAME} se kya help chahiye?"
        ]
    else:
        responses = [
            f"I'm here to help with your needs. I can assist with questions about {COMPANY_NAME}'s services, or provide support.",
            f"I can help you start new initiatives, locate information from {COMPANY_NAME}'s website content, or connect you with the right resources.",
            f"I'm your AI assistant for {COMPANY_NAME}. Ask me about our offerings.",
            f"I assist with service questions and information drawn from {COMPANY_NAME}'s website content.",
            f"Need details about {COMPANY_NAME}'s capabilities? I'm ready to share insights on our services."
        ]
    
    # If session_id provided, track capability question count for variety
    if session_id and session_id in conversation_sessions:
        if 'capability_count' not in conversation_sessions[session_id]:
            conversation_sessions[session_id]['capability_count'] = 0
        conversation_sessions[session_id]['capability_count'] += 1
        count = conversation_sessions[session_id]['capability_count']
        selected_response = responses[(count - 1) % len(responses)]
        logger.info(f"Capability response #{count} for session {session_id}: {selected_response[:50]}...")
        return selected_response
    else:
        # Fallback to random selection if no session tracking
        import random
        fallback_response = random.choice(responses)
        logger.info(f"Using fallback capability response (no session tracking): {fallback_response[:50]}...")
        return fallback_response
def get_general_help_response(user_language: str = 'english') -> str:
    """Generate appropriate response for general help requests"""
    import random
    
    if user_language == 'hindi':
        responses = [
            f"Main aapki madad ke liye yahan hun. Aapko {COMPANY_NAME} ke services ke bare mein batayein.",
            f"Bilkul! Main aapki help kar sakta hun {COMPANY_NAME} ke services ke bare mein.",
            "Main yahan hun aapki madad ke liye. Aapko kis area mein assistance chahiye?"
        ]
    else:
        responses = [
            f"I'm here to help. I can assist with {COMPANY_NAME}'s services.",
            f"I can help you with {COMPANY_NAME}'s services.",
            "I'm here to assist. What do you need help with?"
        ]
    
    return random.choice(responses)

# Meta/Help detection
def is_meta_question(message: str) -> bool:
    """Check if message is asking about the chatbot itself (more specific)"""
    message_lower = message.lower().strip()
    
    # Remove common profanity to check core question
    cleaned_message = message_lower.replace('hell', '').replace('fuck', '').replace('damn', '').replace('shit', '')
    
    # More specific meta patterns - only questions about AI capabilities/identity
    meta_patterns = [
        'what can you do', 'what do you do', 'who are you', 
        'how does this work', 'what is this', 'your capabilities',
        'who built you', 'who made you', 'what are you',
        'who are u', 'what are u', 'who r u', 'what r u',
        'who the hell are you', 'who the hell', 'who the f*** are you',
        'who the f are you', 'who the hell are u', 'who the f*** are u',
        'how can you help me'  # Only this specific pattern, not general help requests
    ]
    
    # Check both original and cleaned message
    for pattern in meta_patterns:
        if pattern in message_lower or pattern in cleaned_message:
            return True
    
    # Additional check for "who the hell" variations
    if 'who the hell' in message_lower and any(word in message_lower for word in ['are you', 'are u', 'r u']):
        return True
    
    return False

def get_meta_response() -> str:
    """Generate response about chatbot capabilities"""
    import random
    responses = [
        f"I'm an AI assistant representing {COMPANY_NAME}. I help you explore our services.",
        f"I'm your AI assistant. I can answer questions about {COMPANY_NAME}'s offerings.",
        f"I'm here to help you explore {COMPANY_NAME}'s services. Ask me about our offerings or recent projects."
    ]
    return random.choice(responses)

# Contact info detection
def is_contact_query(message: str) -> bool:
    """Check if message is asking for contact information"""
    message_lower = message.lower().strip()
    contact_patterns = [
        'location', 'address', 'where are you', 'office', 'contact',
        'email', 'phone', 'call', 'telephone', 'reach', 'support hours',
        'how to contact', 'get in touch', 'talk to someone', 'human agent',
        'speak to agent', 'customer service'
    ]
    
    for pattern in contact_patterns:
        if pattern in message_lower:
            return True
    
    return False
def get_contact_response() -> str:
    """Generate response for contact queries"""
    import random
    responses = [
        f"You can reach us at {CONTACT_EMAIL} or visit {CONTACT_PAGE_URL} for contact details and office locations.",
        f"Contact our team at {CONTACT_EMAIL} or call {CONTACT_PHONE} for immediate assistance.",
        f"For detailed contact options, visit {CONTACT_PAGE_URL} or email {CONTACT_EMAIL}."
    ]
    return random.choice(responses)

# Service Information Query Detection
def is_service_info_query(message: str) -> bool:
    """Check if message is asking about services information (not pricing)"""
    message_lower = message.lower().strip()
    
    # Service info patterns - asking ABOUT services, not pricing
    service_info_patterns = [
        'what are your services', 'what services do you offer', 'what services do you provide',
        'tell me about your services', 'what services are available', 'what services do you have',
        'what ai solutions do you have', 'what cloud services do you offer', 'what erp solutions',
        'what crm solutions', 'what services does {company_lower} offer', 'what does {company_lower} do',
        'what can {company_lower} help with', 'what solutions do you provide', 'what do you offer',
        'services you offer', 'services you provide', 'services available'
    ]
    
    for pattern in service_info_patterns:
        if pattern in message_lower:
            return True
    
    # Check for "what" + "services" combination (but not pricing)
    if 'what' in message_lower and 'services' in message_lower:
        # Exclude pricing-related words
        pricing_words = ['cost', 'price', 'pricing', 'expensive', 'cheap', 'how much']
        if not any(word in message_lower for word in pricing_words):
            return True
    
    return False

def get_service_info_response() -> str:
    """Generate response for service information queries"""
    import random
    base = (
        f"{COMPANY_NAME} offers services. "
        "Share the area you're interested in and I'll pull the most relevant details from the latest website content."
    )
    variations = [
        base,
        f"Our team at {COMPANY_NAME} helps with various services. Let me know which service you want to explore.",
        f"{COMPANY_NAME} covers strategy, design, engineering, and managed services. Ask about any specific capability and I'll elaborate."
    ]
    return random.choice(variations)

# Pricing detection
def is_pricing_query(message: str) -> bool:
    """Check if message is asking about pricing (refined to avoid false positives)"""
    message_lower = message.lower().strip()
    
    # First check if it's a complaint about services - don't treat as pricing
    complaint_words = ['frustrated', 'upset', 'angry', 'not satisfied', 'bad', 'wrong', 'terrible', 'poor']
    if any(word in message_lower for word in complaint_words) and 'services' in message_lower:
        return False
    
    # Exclude non-pricing uses of "rate" keyword
    non_pricing_rate_patterns = [
        'success rate', 'conversion rate', 'error rate', 'performance rate',
        'completion rate', 'satisfaction rate', 'response rate', 'uptime rate'
    ]
    for pattern in non_pricing_rate_patterns:
        if pattern in message_lower:
            return False
    
    # Explicit pricing patterns only (but exclude standalone "rate" without pricing context)
    pricing_patterns = [
        'price', 'pricing', 'cost', 'how much', 'expensive', 'cheap',
        'fees', 'charge', 'payment', 'trial', 'free trial',
        'demo', 'packages', 'plans', 'subscription', 'quote', 'estimate'
    ]
    
    # Check for standalone "rate" with pricing context
    if 'rate' in message_lower:
        pricing_context_words = ['price', 'pricing', 'cost', 'fee', 'charge', 'payment', 'quote']
        if any(word in message_lower for word in pricing_context_words):
            # Has pricing context, treat as pricing
            pass  # Will be caught by pricing_patterns check below
        else:
            # "rate" without pricing context - likely not pricing (could be success rate, etc.)
            return False
    
    for pattern in pricing_patterns:
        if pattern in message_lower:
            return True
    
    # Check for "services" + pricing words combination
    if 'services' in message_lower:
        pricing_context_words = ['cost', 'price', 'pricing', 'how much', 'expensive', 'cheap', 'fees']
        if any(word in message_lower for word in pricing_context_words):
            return True
    
    return False

def get_pricing_response() -> str:
    """Generate response for pricing queries"""
    import random
    responses = [
        "Pricing varies based on your requirements. For a custom quote, email info@{company_lower}.com or visit {company_lower}.com/contact.",
        "We offer flexible pricing based on business needs. Contact info@{company_lower}.com or visit {company_lower}.com/contact for detailed pricing.",
        "For pricing information tailored to your needs, email info@{company_lower}.com or visit {company_lower}.com/contact."
    ]
    return random.choice(responses)

# Policy detection
def is_policy_query(message: str) -> bool:
    """Check if message is asking about company policies"""
    message_lower = message.lower().strip()
    policy_patterns = [
        'policy', 'policies', 'terms', 'conditions', 'terms and conditions',
        'privacy policy', 'refund policy', 'cancellation policy', 'return policy',
        'company policy', 'business policy', 'service policy', 'data policy',
        'terms of service', 'terms of use', 'user agreement', 'legal'
    ]
    
    for pattern in policy_patterns:
        if pattern in message_lower:
            return True
    
    return False

# Frustration detection
def is_frustrated(message: str) -> bool:
    """Check if user seems frustrated or confused"""
    message_lower = message.lower().strip()
    frustration_patterns = [
        'not helping', 'not helpful', 'confused', 'don\'t understand',
        'doesn\'t make sense', 'not clear', 'unclear', 'wrong answer',
        'not working', 'fix this', 'actually help', 'canned repl',
        'stop giving', 'useless', 'what are you saying'
    ]
    
    for pattern in frustration_patterns:
        if pattern in message_lower:
            return True
    
    return False

def get_frustration_response() -> str:
    """Generate empathetic response for frustrated users"""
    import random
    responses = [
        f"I apologize if that wasn't clear. I can help you with {COMPANY_NAME}'s services or our projects.",
        f"I'm sorry for the confusion. Let me assist you more directly with {COMPANY_NAME}'s services.",
        f"My apologies. I can provide better information about {COMPANY_NAME}'s services."
    ]
    return random.choice(responses)

# Complaint/Dissatisfaction detection
def is_dissatisfaction(message: str) -> bool:
    """Check if user is expressing dissatisfaction with the previous response"""
    message_lower = message.lower().strip()
    dissatisfaction_patterns = [
        # Direct dissatisfaction expressions
        'not satisfied', 'not happy', 'not satisfied with', 'unsatisfied',
        'dissatisfied', 'not good', 'not helpful', 'not working',
        'i am not satisfy', 'i am not satisfied', 'i\'m not satisfied',
        'i am not happy', 'i\'m not happy', 'i am not satisfy with',
        'i am not satisfied with', 'i\'m not satisfied with',
        
        # Quality issues
        'that\'s not helpful', 'that doesn\'t help', 'that\'s not what i wanted',
        'that\'s not right', 'that\'s wrong', 'incorrect', 'not what i asked',
        'you didn\'t answer', 'you didn\'t understand', 'that\'s not clear',
        'confusing', 'not useful', 'useless', 'waste of time',
        'not what i need', 'bad answer', 'wrong answer', 'disappointed', 'frustrated',
        'not what i was looking for', 'this doesn\'t help', 'not helpful at all',
        'that doesn\'t work', 'doesn\'t help', 'not correct',
        'not helpful', 'not what i expected', 'annoyed', 'not impressed',
        'this is not what i want', 'this is wrong', 'this is bad',
        
        # Service/product dissatisfaction
        'not satisfied with your', 'not happy with your', 'not good with your',
        'not satisfied with the', 'not happy with the', 'not good with the',
        'dissatisfied with your', 'dissatisfied with the',
        
        # Generic negative expressions
        'this is not good', 'this is not helpful', 'this is not what i wanted',
        'this is wrong', 'this is bad', 'this is useless',
        'it\'s not good', 'it\'s not helpful', 'it\'s not what i wanted',
        'it\'s wrong', 'it\'s bad', 'it\'s useless',
        
        # Hindi/Hinglish patterns
        'accha nahi', 'sahi nahi', 'theek nahi', 'pasand nahi',
        'khush nahi', 'satisfy nahi'
    ]
    
    for pattern in dissatisfaction_patterns:
        if pattern in message_lower:
            return True
    
    # Additional checks for common dissatisfaction structures
    if any(phrase in message_lower for phrase in ['not satisfied', 'not happy', 'not good']):
        if any(word in message_lower for word in ['with', 'about', 'regarding']):
            return True
    
    return False

def get_dissatisfaction_response(user_language: str = 'english') -> str:
    """Generate empathetic response for user dissatisfaction"""
    import random
    
    if user_language == 'hindi':
        responses = [
            "Maafi chahta hun ki main aapki help nahi kar saka. Main phir se try karunga {company} ke services ke bare mein.",
            "Main samajh gaya ki aap satisfied nahi hain. Main aapki better help kar sakta hun.",
            "Sorry for the confusion. Main aapki better help kar sakta hun {company} ke bare mein."
        ]
    else:
        responses = [
            "I apologize that my response wasn't helpful. Let me assist you better with {company}'s services.",
            "I understand you're not satisfied. Let me help you more effectively with {company}'s offerings.",
            "I'm sorry for the confusion. I can provide better assistance with {company}'s services."
        ]
    
    return random.choice(responses)

def is_complaint(message: str) -> bool:
    """Check if user is complaining or expressing dissatisfaction"""
    message_lower = message.lower().strip()
    complaint_patterns = [
        'not happy', 'unhappy', 'disappointed', 'dissatisfied', 'not satisfied',
        'poor service', 'bad service', 'terrible service', 'worst service',
        'complaint', 'complain', 'issue with your', 'problem with your',
        'bad experience', 'poor support', 'terrible support', 'poor experience',
        'unsatisfied', 'let down', 'frustrated with your'
    ]
    
    for token in COMPANY_TOKENS:
        complaint_patterns.extend([
            f'problem with {token}',
            f'issue with {token}',
            f'{token} is bad',
            f'{token} poor'
        ])
    
    for pattern in complaint_patterns:
        if pattern in message_lower:
            return True
    
    return False

# Client Identity Detection
def is_client_identity(message: str) -> bool:
    """Check if user is identifying themselves as a client"""
    message_lower = message.lower().strip()
    client_identity_patterns = [
        'i am your client', 'i\'m your client', 'i am a client', 'i\'m a client',
        'i am an existing client', 'i\'m an existing client', 
        'we are your client', 'we\'re your client', 'we are clients', 'we\'re clients',
        'i am your customer', 'i\'m your customer', 'i am a customer', 'i\'m a customer',
        'we are your customer', 'we\'re your customer',
        'i work with you', 'we work with you',
        'we use your service', 'i use your service',
        'i am already a client', 'i\'m already a client',
        'we are already a client', 'we\'re already a client',
        'existing client here', 'current client here',
        'i need support', 'i need help with my project',
        'we need support', 'we need help with our project'
    ]
    
    for token in COMPANY_TOKENS:
        client_identity_patterns.extend([
            f'i work with {token}',
            f'we work with {token}',
            f'{token} client',
            f'{token} customer'
        ])
    
    for pattern in client_identity_patterns:
        if pattern in message_lower:
            return True
    
    return False
def get_client_identity_response(user_language: str = 'english') -> str:
    """Generate welcoming response for existing clients"""
    import random
    
    if user_language == 'hindi':
        responses = [
            f"{COMPANY_NAME} ke saath kaam karne ke liye dhanyavaad! Main aapki project ya service related queries me madad kar sakta hun.",
            f"Humein choose karne ke liye shukriya. Batayein aapko {COMPANY_NAME} ke kis project ya service me support chahiye.",
            "Aapka swagat hai! Main yahan hun aapke ongoing project ya support requests ko resolve karne ke liye."
        ]
    else:
        responses = [
            f"Thank you for partnering with {COMPANY_NAME}! I'm here to assist you with your projects, services, or account details.",
            f"Welcome back! Let me know which {COMPANY_NAME} service or delivery you need help with and I'll get the right information.",
            f"I'm glad to connect with you again. Share your project or support request and I'll use {COMPANY_NAME}'s context to guide you."
        ]
    
    return random.choice(responses)

# Project Query Detection
def is_project_query(message: str) -> bool:
    """Check if user is asking about projects/portfolio"""
    message_lower = message.lower().strip()
    project_query_patterns = [
        'tell me about your projects', 'what projects have you done', 
        'show me your projects', 'your projects', 'your portfolio',
        'tell me about your previous projects', 'previous projects',
        'completed projects', 'past projects', 'project portfolio',
        'what have you built', 'what have you developed',
        'show me what you\'ve built', 'examples of your work',
        'your work samples', 'case studies', 'project examples',
        'what kind of projects', 'types of projects',
        'project list', 'list of projects', 'project names',
        'i am asking about your projects', 'asking about projects',
        'tell me something about your projects', 'about your projects'
    ]
    
    for pattern in project_query_patterns:
        if pattern in message_lower:
            return True
    
    return False

def get_project_query_response(user_language: str = 'english') -> str:
    """Generate response listing projects based on current ChromaDB context"""
    try:
        search_terms = [
            f"{COMPANY_NAME} recent projects",
            f"{COMPANY_NAME} portfolio",
            "case study",
            "project"
        ]
        project_snippets: List[str] = []
        for term in search_terms:
            results = search_chroma(term, COLLECTION_NAME, 5)
            for result in results or []:
                doc = (result.get('document') or '').strip()
                if not doc:
                    continue
                summary = doc.split('\n')[0].strip()
                if len(summary) > 200:
                    summary = summary[:197] + "..."
                if summary not in project_snippets:
                    project_snippets.append(summary)
                if len(project_snippets) >= 3:
                    break
            if len(project_snippets) >= 3:
                break
    except Exception as exc:
        logger.warning(f"Could not fetch project snippets from ChromaDB: {exc}")
        project_snippets = []
    
    if project_snippets:
        if user_language == 'hindi':
            intro = f"{COMPANY_NAME} ne in jaise projects par kaam kiya hai:"
            bullet_list = " • " + " • ".join(project_snippets[:3])
            return f"{intro}{bullet_list}"
        intro = f"{COMPANY_NAME} has completed projects such as:"
        bullet_list = " • " + " • ".join(project_snippets[:3])
        return f"{intro}{bullet_list}"
    
    fallback = (
        f"{COMPANY_NAME} handles projects. "
        "Share your industry or requirement and I'll pull the most relevant case studies from the website."
    )
    if user_language == 'hindi':
        fallback = (
            f"{COMPANY_NAME} kai digital projects par kaam karta hai. "
            "Aap apni requirement bataiye, main website ke latest content se example share karunga."
        )
    return fallback

def get_complaint_response() -> str:
    """Generate empathetic response for complaints with support email"""
    import random
    responses = [
        f"I'm sorry to hear that. Please share the details with our support team at {SUPPORT_EMAIL} so we can address this immediately.",
        f"I apologize for the inconvenience. Email us at {SUPPORT_EMAIL} or call {CONTACT_PHONE} and we'll make this a priority.",
        f"Thank you for flagging this. Our support team at {SUPPORT_EMAIL} will help resolve it right away."
    ]
    return random.choice(responses)

# NEW: Support query detection
def is_support_query(message: str) -> bool:
    """Check if user is asking about support availability (24/7, response time, etc.)"""
    message_lower = message.lower().strip()
    support_patterns = [
        '24/7 support', '24x7 support', '24 7 support',
        'support hours', 'support time', 'support available',
        'when is support available', 'do you have support',
        'support schedule', 'support timing', 'support availability',
        'response time', 'support response', 'support response time',
        'when can i get support', 'is support available',
        'support contact', 'support team available'
    ]
    return any(pattern in message_lower for pattern in support_patterns)

# NEW: Specific project query detection
def is_specific_project_query(message: str) -> bool:
    """Check if user is asking about a specific project (e.g., 'dog walking project', 'funzoop project')"""
    message_lower = message.lower().strip()
    
    project_keywords = ['project', 'case study', 'portfolio', 'solution']
    if not any(keyword in message_lower for keyword in project_keywords):
        return False
    
    generic_words = {'project', 'projects', 'case', 'study', 'case study', 'tell', 'me', 'about', 'your', 'the', 'solution', 'solutions'}
    tokens = re.findall(r'[a-z0-9\-]+', message_lower)
    non_generic_tokens = [token for token in tokens if token not in generic_words and len(token) > 3]
    
    # If user mentions a unique token along with project keywords, treat it as specific
    return len(non_generic_tokens) >= 1

# NEW: Company stats query detection
def is_company_stats_query(message: str) -> bool:
    """Check if user is asking about company statistics (project count, services count, etc.)"""
    message_lower = message.lower().strip()
    
    # Direct patterns for stats queries
    stats_patterns = [
        'how many projects', 'project count', 'number of projects',
        'how many services', 'service count', 'number of services',
        'how many clients', 'client count', 'number of clients',
        'how many employees', 'team size', 'number of employees',
        'projects completed', 'completed projects', 'projects you have',
        'projects you have completed', 'total projects', 'total services',
        'how many projects you have', 'how many services do you'
    ]
    
    # Count keywords + stats keywords combination
    count_keywords = ['how many', 'count', 'number of', 'total', 'how much']
    stats_keywords = ['projects', 'services', 'clients', 'employees', 'team']
    
    # Check for direct patterns
    if any(pattern in message_lower for pattern in stats_patterns):
        return True
    
    # Check for combination of count + stats keywords
    has_count_keyword = any(ck in message_lower for ck in count_keywords)
    has_stats_keyword = any(sk in message_lower for sk in stats_keywords)
    
    # Also check for "how many" + "provided" or "offer" (services count)
    if 'how many' in message_lower and any(word in message_lower for word in ['services', 'service', 'provided', 'offer', 'provide']):
        return True
    
    return has_count_keyword and has_stats_keyword

# NEW: Comparison query detection
def is_comparison_query(message: str) -> bool:
    """Check if user is asking about company comparison or competitive advantages"""
    message_lower = message.lower().strip()
    comparison_patterns = [
        'what makes you different', 'what makes your company different',
        'what makes you unique', 'what makes your company unique',
        'competitors', 'vs', 'versus', 'compare', 'comparison',
        'difference', 'differences', 'why choose you',
        'why you', 'why should i choose', 'advantage', 'advantages',
        'better than', 'why you are better', 'what sets you apart',
        'unique selling point', 'usp', 'competitive advantage',
        'how are you different', 'how do you differ'
    ]
    return any(pattern in message_lower for pattern in comparison_patterns)

# NEW: Industry query detection
def is_industry_query(message: str) -> bool:
    """Check if user is asking about industries served"""
    message_lower = message.lower().strip()
    industry_patterns = [
        'what industries', 'which industries', 'industries served',
        'industries do you serve', 'what sectors', 'which sectors',
        'industries you serve', 'sectors you serve',
        'what industries do you work with', 'which industries do you work with',
        'what industries are you in', 'industries you work in'
    ]
    return any(pattern in message_lower for pattern in industry_patterns)

# NEW: Response generation functions for new query types
async def generate_support_response(message: str, user_language: str, conversation_history: List[Dict[str, str]]) -> str:
    """Generate response for support availability queries using RAG"""
    # Search ChromaDB for support information
    search_results = search_chroma(message, COLLECTION_NAME, n_results=3)
    
    context = ""
    if search_results:
        context = "\n\n".join([result.get('content', '') for result in search_results])
    
    # Build context section
    if context and context.strip():
        context_section = f"""
        CRITICAL: The context provided below contains information about {company}'s support services.
        You MUST base your answer PRIMARILY on this context.
        
        CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about support availability (24/7 support, response time, etc.).
        Provide accurate information based on the context above. If context mentions 24/7 support, mention it.
        If no specific support hours are mentioned, state that support is available and mention contact info.
        """
    else:
        context_section = """
        Note: Use your general knowledge about {company}'s support services.
        The user is asking about support availability. Provide information about support availability.
        """
    
    language_instruction = "You MUST respond in Hindi only." if user_language == 'hindi' else "You MUST respond in English only."
    
    system_prompt = f"""
{context_section}

        You are an AI assistant for {company} {company_descriptor}.

        CRITICAL RULES:
        1. CONTEXT USAGE: When context is provided, use it to answer accurately.
        2. SUPPORT INFORMATION: Answer about support availability (24/7, response time, etc.).
        3. Keep response SHORT (1-2 sentences, 100 tokens max).
        4. NEVER ask follow-up questions.
        5. Be friendly and professional.
        6. LANGUAGE: {language_instruction}
        7. If context mentions 24/7 support, confirm it. If not, state that support is available.
        
        Example: "Yes, we offer 24/7 support for our services. You can reach us at {contact_email} or {contact_phone}."
        """
    
    messages = [{"role": "system", "content": apply_company_placeholders(system_prompt)}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})
    
    try:
        reply = await _get_llm_response(
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        if reply:
            return reply
    except Exception as e:
        logger.error(f"Error generating support response: {e}")
    
    # Fallback response
    profile = get_domain_profile()
    return apply_company_placeholders(f"Yes, we offer comprehensive support for our services. You can reach us at {CONTACT_EMAIL} or {CONTACT_PHONE}.")

async def generate_specific_project_response(message: str, user_language: str, conversation_history: List[Dict[str, str]]) -> str:
    """Generate response for specific project queries using RAG"""
    # Search ChromaDB for project information
    search_results = search_chroma(message, COLLECTION_NAME, n_results=3)
    
    context = ""
    if search_results:
        context = "\n\n".join([result.get('content', '') for result in search_results])
    
    # Build context section
    if context and context.strip():
        context_section = f"""
        CRITICAL: The context provided below contains information about {company}'s completed projects.
        You MUST base your answer PRIMARILY on this context.
        
        CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about a specific project mentioned in their query.
        Provide detailed information about the project based on the context above.
        Mention what was built, technologies used, and outcomes if available in context.
        """
    else:
        context_section = """
        Note: Use your general knowledge about {company}'s projects.
        The user is asking about a specific project. Provide information about the project.
        """
    
    language_instruction = "You MUST respond in Hindi only." if user_language == 'hindi' else "You MUST respond in English only."
    
    system_prompt = f"""
{context_section}

        You are an AI assistant for {company} {company_descriptor}.

        CRITICAL RULES:
        1. CONTEXT USAGE: When context is provided, use it to answer accurately.
        2. PROJECT INFORMATION: Provide details about the specific project mentioned.
        3. Keep response SHORT (1-2 sentences, 100 tokens max).
        4. NEVER ask follow-up questions.
        5. Be friendly and professional.
        6. LANGUAGE: {language_instruction}
        7. Mention project name, what was built, and key features if available.
        
        Example: "We successfully completed the requested services, delivering excellent results and meeting all requirements."
        """
    
    messages = [{"role": "system", "content": apply_company_placeholders(system_prompt)}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})
    
    try:
        reply = await _get_llm_response(
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        if reply:
            return reply
    except Exception as e:
        logger.error(f"Error generating specific project response: {e}")
    
    # Fallback - generic guidance
    return (
        f"{COMPANY_NAME} provides services across industries. "
        "Share more details about the project (industry, goal, or features) and I'll use the latest website content to respond precisely."
    )

async def generate_company_stats_response(message: str, user_language: str, conversation_history: List[Dict[str, str]]) -> str:
    """Generate response for company statistics queries using RAG"""
    # Search ChromaDB with multiple query variations for better matching
    queries = [
        message,
        f"{company} {message}",
        f"{company} projects completed count statistics",
        f"{company} portfolio statistics",
        f"{company} project history overview"
    ]
    
    # Search with all variations and combine results
    all_results = []
    for query in queries[:3]:  # Use first 3 variations to avoid too many searches
        results = search_chroma(query, COLLECTION_NAME, n_results=2)
        if results:
            all_results.extend(results)
    
    # Remove duplicates based on content
    seen_content = set()
    unique_results = []
    for result in all_results:
        content = result.get('content', '')
        if content and content not in seen_content:
            seen_content.add(content)
            unique_results.append(result)
    
    context = ""
    if unique_results:
        context = "\n\n".join([result.get('content', '') for result in unique_results[:3]])
    
    # Build context section
    if context and context.strip():
        context_section = f"""
        CRITICAL: The context provided below contains information about {company}'s company statistics.
        You MUST base your answer PRIMARILY on this context.
        
        CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about company statistics (project count, services count, etc.).
        Provide accurate information based on the context above. If context contains specific numbers, cite them.
        If asking about services, list the services mentioned in the context.
        """
    else:
        context_section = """
        Note: Use your general knowledge about {company}'s company statistics.
        The user is asking about company statistics. Provide accurate information.
        """
    
    language_instruction = "You MUST respond in Hindi only." if user_language == 'hindi' else "You MUST respond in English only."
    
    # Determine what stats are being asked
    message_lower = message.lower()
    if 'project' in message_lower:
        stats_type = "projects completed"
        default_info = f"{COMPANY_NAME} regularly shares project milestones on the website. Reference the context above for the latest counts."
    elif 'service' in message_lower:
        stats_type = "services provided"
        default_info = f"{COMPANY_NAME} offers various solutions and services."
    elif 'client' in message_lower:
        stats_type = "clients"
        default_info = f"{COMPANY_NAME} works with clients across multiple industries. Use the provided context when naming specific partners."
    else:
        stats_type = "company statistics"
        default_info = f"{COMPANY_NAME} publishes up-to-date statistics on its website. Use the available context for precise numbers."
    
    system_prompt = f"""
{context_section}

        You are an AI assistant for {company} {company_descriptor}.

        CRITICAL RULES:
        1. CONTEXT USAGE: When context is provided, use it to answer accurately.
        2. STATISTICS INFORMATION: Answer about {stats_type} based on the context.
        3. Keep response SHORT (1-2 sentences, 100 tokens max).
        4. NEVER ask follow-up questions.
        5. Be friendly and professional.
        6. LANGUAGE: {language_instruction}
        7. If context has specific numbers (like "250+ projects"), use them. If not, use general information.
        
        Default information if context doesn't have specifics: {default_info}
        """
    
    messages = [{"role": "system", "content": apply_company_placeholders(system_prompt)}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})
    
    try:
        reply = await _get_llm_response(
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        if reply:
            return reply
    except Exception as e:
        logger.error(f"Error generating company stats response: {e}")
    
    # Fallback response
    return default_info

async def generate_comparison_response(message: str, user_language: str, conversation_history: List[Dict[str, str]]) -> str:
    """Generate response for comparison queries using RAG"""
    # Search ChromaDB for company advantages
    search_results = search_chroma(message, COLLECTION_NAME, n_results=3)
    
    context = ""
    if search_results:
        context = "\n\n".join([result.get('content', '') for result in search_results])
    
    # Build context section
    if context and context.strip():
        context_section = f"""
        CRITICAL: The context provided below contains information about {company}'s competitive advantages.
        You MUST base your answer PRIMARILY on this context.
        
        CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking what makes {company} different from competitors.
        Provide information about company advantages, experience, expertise, and unique selling points based on context.
        """
    else:
        context_section = """
        Note: Use your general knowledge about {company}'s competitive advantages.
        The user is asking what makes the company different. Provide information about advantages.
        """
    
    language_instruction = "You MUST respond in Hindi only." if user_language == 'hindi' else "You MUST respond in English only."
    
    system_prompt = f"""
{context_section}

        You are an AI assistant for {company} {company_descriptor}.

        CRITICAL RULES:
        1. CONTEXT USAGE: When context is provided, use it to answer accurately.
        2. COMPARISON INFORMATION: Answer about what makes {company} different/unique.
        3. Keep response SHORT (1-2 sentences, 100 tokens max).
        4. NEVER ask follow-up questions.
        5. Be friendly and professional.
        6. LANGUAGE: {language_instruction}
        7. Mention key advantages based on context: experience, projects, expertise, client satisfaction.
        
        Example: "Our experience, successfully completed projects, and focus on client satisfaction set us apart. We specialize in {primary_offerings_summary}."
        """
    
    messages = [{"role": "system", "content": apply_company_placeholders(system_prompt)}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})
    
    try:
        reply = await _get_llm_response(
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        if reply:
            return reply
    except Exception as e:
        logger.error(f"Error generating comparison response: {e}")
    
    # Fallback response
    profile = get_domain_profile()
    return apply_company_placeholders(f"Our experience, successfully completed projects, and focus on client satisfaction set us apart. We specialize in {profile.summary_of_offerings()}.")

async def generate_industry_response(message: str, user_language: str, conversation_history: List[Dict[str, str]]) -> str:
    """Generate response for industry queries using RAG"""
    # Search ChromaDB for industry information
    search_results = search_chroma(message, COLLECTION_NAME, n_results=3)
    
    context = ""
    if search_results:
        context = "\n\n".join([result.get('content', '') for result in search_results])
    
    # Build context section
    if context and context.strip():
        context_section = f"""
        CRITICAL: The context provided below contains information about industries served by {company}.
        You MUST base your answer PRIMARILY on this context.
        
        CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about industries served.
        Provide information about industries/sectors served based on the context above.
        """
    else:
        context_section = """
        Note: Use your general knowledge about industries served by {company}.
        The user is asking about industries served. Provide information about industries.
        """
    
    language_instruction = "You MUST respond in Hindi only." if user_language == 'hindi' else "You MUST respond in English only."
    
    system_prompt = f"""
{context_section}

        You are an AI assistant for {company} {company_descriptor}.

        CRITICAL RULES:
        1. CONTEXT USAGE: When context is provided, use it to answer accurately.
        2. INDUSTRY INFORMATION: Answer about industries served.
        3. Keep response SHORT (1-2 sentences, 100 tokens max).
        4. NEVER ask follow-up questions.
        5. Be friendly and professional.
        6. LANGUAGE: {language_instruction}
        7. Mention industries like finance, insurance, healthcare, retail, manufacturing, technology if available in context.
        
        Example: "We serve various industries including finance, insurance, healthcare, retail, manufacturing, and technology sectors."
        """
    
    messages = [{"role": "system", "content": apply_company_placeholders(system_prompt)}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})
    
    try:
        reply = await _get_llm_response(
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        if reply:
            return reply
    except Exception as e:
        logger.error(f"Error generating industry response: {e}")
    
    # Fallback response
    return "We serve various industries including finance, insurance, healthcare, retail, manufacturing, and technology sectors."

# Conversation Context Management
def is_follow_up_response(message: str, last_bot_response: str = "") -> bool:
    """Check if message is a follow-up response to a previous bot question - DISABLED to prevent unwanted follow-ups"""
    # DISABLED: Follow-up detection disabled to prevent unwanted follow-up questions
    # All responses will be treated as regular queries for better user experience
    return False
def is_definition_query(message: str) -> bool:
    """Check if message is asking for a definition (what is, what are, explain)"""
    import re
    message_lower = message.lower().strip()
    
    # Definition query patterns
    definition_patterns = [
        r'^what is\s+',
        r'^what are\s+',
        r'^what\'s\s+',
        r'^explain\s+',
        r'^tell me what\s+',
        r'^define\s+',
        r'^what do you mean by\s+',
        r'^what does\s+.*\s+mean',
        r'^can you explain\s+',
        r'^can you tell me what\s+'
    ]
    
    for pattern in definition_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False

def verify_service_provided(user_query: str, context: str, search_results: List[Dict[str, Any]] = None) -> bool:
    """
    Semantic verification: Check if context actually contains EXACT service mentioned in query
    Returns True only if service is explicitly mentioned in context, False otherwise
    Uses semantic matching to verify if services mentioned in query are present in context
    """
    import re
    
    if not context or not user_query:
        return False
    
    # Removed hardcoded SERVICE_SYNONYMS dictionary - use semantic matching from RAG context instead
    # This allows verification of any service type (restaurant menu, hospital departments, etc.)
    # Service verification now relies on semantic similarity in context rather than hardcoded synonyms
    
    # Extract service name from user query (e.g., "do you provide X services" -> "X")
    query_lower = user_query.lower()
    context_lower = context.lower()
    
    # Removed hardcoded non_it_services list - no assumptions about company type
    # Service verification now relies purely on RAG context matching
    # If service is mentioned in context, it's verified; otherwise, it's not provided
    
    # Extract potential service names from query
    # Pattern: "do you provide X" or "do you offer X" or "X services"
    service_patterns = [
        r'provide\s+([^?]+?)(?:\s+service|services)?',
        r'offer\s+([^?]+?)(?:\s+service|services)?',
        r'([^?]+?)\s+service',
        r'([^?]+?)\s+services'
    ]
    
    extracted_services = []
    seen_candidates = set()
    service_candidates: List[Tuple[str, Optional[str]]] = []
    for pattern in service_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            service_name = match.strip()
            if service_name and len(service_name) > 3:  # Valid service name
                # Normalize: Remove "services" suffix if it exists (for better synonym matching)
                # Only remove if "services" is at the end, not in the middle
                normalized_service = service_name.strip()
                if normalized_service.endswith(' services'):
                    normalized_service = normalized_service[:-9].strip()  # Remove " services"
                elif normalized_service.endswith(' service'):
                    normalized_service = normalized_service[:-8].strip()  # Remove " service"
                
                if normalized_service and normalized_service not in seen_candidates:
                    service_candidates.append((normalized_service, None))
                    seen_candidates.add(normalized_service)
    
    if not service_candidates:
        # No service candidates extracted - use direct semantic matching from query
        # Check if query mentions any service-related terms directly in context
        service_indicators = ['service', 'services', 'offer', 'provide', 'solution', 'solutions']
        if any(indicator in query_lower for indicator in service_indicators):
            # Extract the main service term from query for context matching
            query_words = query_lower.split()
            # Find words around service indicators
            for i, word in enumerate(query_words):
                if word in service_indicators:
                    # Get surrounding words as potential service name
                    if i > 0:
                        potential_service = ' '.join(query_words[max(0, i-2):i])
                        if len(potential_service) > 3:
                            service_candidates.append((potential_service, None))
                    if i < len(query_words) - 1:
                        potential_service = ' '.join(query_words[i+1:min(len(query_words), i+4)])
                        if len(potential_service) > 3:
                            service_candidates.append((potential_service, None))
                    break
        
        if not service_candidates:
            return False
    
    # Check if context contains service name (with synonym support)
    for service_name, predefined_category in service_candidates:
        # Remove common words
        service_words = [w for w in service_name.split() if w not in ['the', 'a', 'an', 'and', 'or', 'do', 'you', 'provide', 'offer']]
        if not service_words:
            continue
        
        # Use semantic matching from context (no hardcoded categories)
        service_name_lower = service_name.lower().strip()
        # Direct semantic matching - check if service name appears in context
        
        # Use direct semantic matching from context (no hardcoded synonyms)
        # Check if service name appears in context with service indicators
        service_names_to_check = [service_name]
        
        # Check if context contains the service name with service indicators
        for service_term in service_names_to_check:
            service_term_lower = service_term.lower()
            
            # Flexible matching: First check if service term exists directly in context (faster and more flexible)
            # This helps when context has the service mentioned but not in exact "provide X" format
            if service_term_lower in context_lower:
                # Additional validation: Check if it's not part of a negative phrase
                negative_phrases = ['not provide', 'do not', "don't", 'cannot', 'unable to', 'no ']
                context_words_around = context_lower[max(0, context_lower.find(service_term_lower) - 50):context_lower.find(service_term_lower) + len(service_term_lower) + 50]
                if not any(neg in context_words_around for neg in negative_phrases):
                    logger.info(f"Service verified in context via flexible matching: {service_name}")
                    return True
            
            # Word-set matching: Check if key words from service name are present in context
            # This helps with semantic matching when service name has multiple words
            if len(service_term_lower.split()) > 1:
                service_name_words = set(service_term_lower.split())
                # Remove common stop words that don't add meaning
                stop_words = {'the', 'a', 'an', 'and', 'or', 'of', 'for', 'in', 'on', 'at', 'to', 'is', 'are', 'was', 'were', 'do', 'you', 'provide', 'offer', 'services', 'service'}
                service_key_words = {w for w in service_name_words if w not in stop_words and len(w) > 2}
                
                if service_key_words:
                    # Check if service name's key words are present in context (substring match for flexibility)
                    all_words_found = sum(1 for word in service_key_words if word in context_lower)
                    # If at least 60% of key words are found, consider it a match
                    if all_words_found >= len(service_key_words) * 0.6:
                        # Additional validation: Check if it's not part of a negative phrase
                        key_word_positions = [context_lower.find(word) for word in service_key_words if word in context_lower]
                        if key_word_positions:
                            min_pos = min(key_word_positions)
                            max_pos = max(key_word_positions) + max(len(w) for w in service_key_words)
                            context_words_around = context_lower[max(0, min_pos - 50):min(len(context_lower), max_pos + 50)]
                            negative_phrases = ['not provide', 'do not', "don't", 'cannot', 'unable to', 'no ']
                            if not any(neg in context_words_around for neg in negative_phrases):
                                logger.info(f"Service verified in context via word-set matching: {service_name} (key words: {service_key_words}, found: {all_words_found}/{len(service_key_words)})")
                                return True
            
        
        # Fallback: Check if ALL key words from service name appear in context (original logic)
        key_words_found = sum(1 for word in service_words if word in context_lower)
        
        # For strict matching: at least 70% of key words should be present
        if key_words_found >= len(service_words) * 0.7:
            # Additional check: context should contain service-related keywords
            service_indicators = [
                f'provide {service_name}',
                f'offer {service_name}',
                f'{service_name} service',
                f'{service_name} services',
                f'our {service_name}',
                f'{service_name} we'
            ]
            
            # Check if any indicator suggests we provide this service
            if any(indicator in context_lower for indicator in service_indicators):
                logger.info(f"Service verified in context: {service_name}")
                return True
    
    # If no match found, decline
    logger.info(f"Service not explicitly verified in context for query: {user_query}")
    return False
def extract_service_from_response(last_bot_response: str) -> tuple:
    """
    Multi-layer service extraction from bot response
    Returns: (service_name, confidence_score)
    """
    import re
    last_response_lower = last_bot_response.lower()
    
    # Removed hardcoded IT service patterns - use generic service extraction
    # Extract service name from response using pattern matching (works for any service type)
    # Pattern: Look for service-related phrases in response
    service_patterns = [
        r'our\s+([a-z\s]+?)\s+(?:service|services|solution|solutions|offering|offerings)',
        r'provide\s+([a-z\s]+?)\s+(?:service|services|solution|solutions)',
        r'offer\s+([a-z\s]+?)\s+(?:service|services|solution|solutions)',
        r'([a-z\s]+?)\s+(?:service|services|solution|solutions)\s+we',
    ]
    
    for pattern in service_patterns:
        matches = re.findall(pattern, last_response_lower)
        if matches:
            service_name = matches[0].strip()
            if len(service_name) > 3:  # Valid service name
                logger.info(f"Extracted service '{service_name}' from response pattern")
                return (service_name, 0.85)
    
    # Layer 3: HF Intent Detection (if available)
    if TRANSFORMERS_AVAILABLE:
        try:
            # Create smart query combining bot response + user intent
            smart_query = f"{last_bot_response}. User wants more information about: {last_response_lower}"
            hf_result = detect_intent_with_hf(smart_query)
            
            if hf_result and hf_result.get('confidence', 0) >= 0.6:
                detected_intent = hf_result.get('intent', '')
                confidence = hf_result.get('confidence', 0)
                
                # Removed hardcoded IT-specific intent-to-service mapping
                # Use generic service extraction from response content instead
                # This works for any service type (restaurant menu, hospital departments, etc.)
                # If service_inquiry intent detected, extract service from response content
                if detected_intent == 'service_inquiry':
                    # Extract service name from response using pattern matching
                    service_match = re.search(r'our\s+([a-z\s]+?)\s+(?:service|services|solution|solutions)', last_response_lower)
                    if service_match:
                        service_name = service_match.group(1).strip()
                        if len(service_name) > 3:
                            logger.info(f"Layer 3: Extracted service '{service_name}' from service_inquiry intent")
                            return (service_name, 0.75)
        except Exception as e:
            logger.warning(f"HF intent detection failed in service extraction: {e}")
    
    # No service detected
    return (None, 0.0)

def get_follow_up_context(last_bot_response: str, user_response: str) -> Dict[str, Any]:
    """Determine the context for follow-up responses with multi-layer service extraction"""
    last_response_lower = last_bot_response.lower()
    
    # Extract service using multi-layer approach
    detected_service, service_confidence = extract_service_from_response(last_bot_response)
    
    # AI Benefits context (specific check first)
    if any(phrase in last_response_lower for phrase in ['can benefit', 'solutions', 'would you like to know more', 'how can benefit']):
        return {
            "type": "service_benefits_followup",
            "original_question": "service_benefits",
            "user_response": user_response.lower(),
            "last_bot_response": last_bot_response,
            "detected_service": None,
            "service_confidence": 0.9
        }
    
    # Service-specific follow-up (if service detected)
    if detected_service and service_confidence >= 0.7:
        return {
            "type": "service_followup",
            "original_question": f"{detected_service}_details",
            "user_response": user_response.lower(),
            "last_bot_response": last_bot_response,
            "detected_service": detected_service,
            "service_confidence": service_confidence
        }
    
    # Service information context (generic)
    if any(phrase in last_response_lower for phrase in ['our services', 'what services', 'tell you about our services']):
        return {
            "type": "service_info_followup", 
            "original_question": "service_info",
            "user_response": user_response.lower(),
            "last_bot_response": last_bot_response,
            "detected_service": detected_service if detected_service else None,
            "service_confidence": service_confidence
        }
    
    # Pricing context
    if any(phrase in last_response_lower for phrase in ['pricing', 'cost', 'price', 'packages']):
        return {
            "type": "pricing_followup",
            "original_question": "pricing",
            "user_response": user_response.lower(),
            "last_bot_response": last_bot_response,
            "detected_service": None,
            "service_confidence": 0.0
        }
    
    # Project context
    if any(phrase in last_response_lower for phrase in ['project', 'work with you', 'collaborate']):
        return {
            "type": "project_followup",
            "original_question": "project",
            "user_response": user_response.lower(),
            "last_bot_response": last_bot_response,
            "detected_service": None,
            "service_confidence": 0.0
        }
    
    # Default context with service info if available
    return {
        "type": "general_followup",
        "original_question": "unknown",
        "user_response": user_response.lower(),
        "last_bot_response": last_bot_response,
        "detected_service": detected_service if detected_service else None,
        "service_confidence": service_confidence
    }

def strip_markdown(text: str) -> str:
    """Strip markdown formatting from text"""
    import re
    
    # Remove bold formatting (**text** -> text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Remove italic formatting (*text* -> text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove code formatting (`text` -> text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Remove link formatting ([text](url) -> text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove header formatting (# text -> text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    return text.strip()
def generate_rag_query_for_followup(followup_type: str, last_bot_response: str = "", detected_service: str = None) -> List[str]:
    """
    Generate smart RAG queries for different follow-up types
    Returns list of query variations to try
    """
    queries = []
    last_response_lower = last_bot_response.lower() if last_bot_response else ""
    
    if followup_type == "pricing_followup":
        queries = [
            "pricing packages costs pricing information quotes payment plans",
            "pricing models packages costs how much",
            "pricing information quote estimate",
            "cost pricing packages"
        ]
    
    elif followup_type == "project_followup":
        queries = [
            "project onboarding process collaboration steps getting started",
            "how to start project collaboration process",
            "project process requirements steps",
            "working together project collaboration"
        ]
    
    elif followup_type == "service_benefits_followup":
        queries = [
            "service benefits advantages",
            "how services can benefit",
            "service advantages benefits",
            "artificial intelligence business benefits"
        ]
    
    elif followup_type == "service_info_followup":
        queries = [
            "services solutions offerings",
            "services offered solutions available",
            "what services do you offer",
            "services solutions offerings"
        ]
    
    elif followup_type == "service_followup" and detected_service:
        queries = [
            f"{detected_service} implementation services features benefits",
            f"{detected_service} solutions services",
            f"{detected_service} how it works benefits"
        ]
    
    elif followup_type == "general_followup":
        # Extract keywords from last bot response
        import re
        # Extract meaningful words (exclude common words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'about', 'with', 'for', 'from', 'to', 'in', 'on', 'at', 'by', 'of', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'how', 'what', 'which', 'who', 'why', 'you', 'your', 'yours', 'we', 'our', 'ours', 'they', 'their', 'them', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'mine', 'he', 'she', 'his', 'her', 'hers'}
        
        words = re.findall(r'\b\w+\b', last_response_lower)
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        if meaningful_words:
            # Use top 3-5 meaningful words
            keywords = ' '.join(meaningful_words[:5])
            queries = [keywords, last_bot_response[:100]]  # Try keywords and first 100 chars
        else:
            queries = [last_bot_response[:100]]
    
    return queries if queries else [last_bot_response[:100]]

async def generate_rag_based_response(
    followup_type: str,
    queries: List[str],
    last_bot_response: str,
    user_response: str,
    conversation_history: List[Dict[str, str]],
    user_language: str,
    detected_service: str = None
) -> str:
    """
    Universal function to generate RAG-based detailed responses for any follow-up type
    """
    try:
        # Try multiple query variations
        rag_results = []
        for query in queries[:3]:  # Try max 3 query variations
            results = search_chroma(query, COLLECTION_NAME, n_results=3)
            if results:
                rag_results.extend(results)
                if len(rag_results) >= 5:  # Collect enough results
                    break
        
        # Remove duplicates based on content
        seen_content = set()
        unique_results = []
        for result in rag_results:
            content_hash = hash(result.get('content', '')[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        rag_results = unique_results[:5]  # Limit to top 5 unique results
        
        # Build context from RAG results
        context_text = ""
        if rag_results:
            context_text = "\n\n".join([result.get('content', '') for result in rag_results])
            logger.info(f"Found {len(rag_results)} relevant documents for {followup_type}")
        
        # Generate response using Groq API
        language_instruction = "You MUST respond in Hindi only." if user_language == 'hindi' else "You MUST respond in English only."
        
        # Type-specific prompt templates - SHORT responses only (1-2 sentences)
        prompt_templates = {
            'pricing_followup': {
                'instruction': "The user wants pricing information. Provide clear and informative details about pricing models, available plans, and how to get an exact quote.",
                'include_contact': True
            },
            'project_followup': {
                'instruction': "The user wants to start a project. Provide detailed information about the project development process, milestones, and the next steps to get started.",
                'include_contact': True
            },
            'service_benefits_followup': {
                'instruction': "The user wants to know about service benefits. Highlight the key business benefits, ROI, and advantages of our services with specific points.",
                'include_contact': False
            },
            'service_info_followup': {
                'instruction': f"The user wants information about services. Provide a comprehensive overview of {COMPANY_NAME}'s offerings, categorized by industry or solution type.",
                'include_contact': False
            },
            'service_followup': {
                'instruction': "The user wants information about services. Provide key details about the service implementation, features, and capabilities.",
                'include_contact': False
            },
            'general_followup': {
                'instruction': "The user wants more information. Provide helpful and detailed information based on the provided context.",
                'include_contact': False
            }
        }
        
        template = prompt_templates.get(followup_type, prompt_templates['general_followup'])
        
        # Customize instruction for service_followup with detected service
        if followup_type == 'service_followup' and detected_service:
            template = template.copy()  # Create a copy to avoid modifying the original
            template['instruction'] = f"The user wants detailed information about {detected_service.upper()} services. Provide comprehensive details about {detected_service.upper()} implementation process, features, capabilities, business benefits, use cases, and how it can help their business."
        
        context_section = ""
        if context_text:
            context_section = f"""
        CRITICAL: The context below contains specific information from {company}'s knowledge base.
        You MUST base your answer PRIMARILY on this context. Provide detailed, comprehensive information.
        
        CONTEXT FROM KNOWLEDGE BASE:
        {context_text[:1500]}
        
        INSTRUCTIONS: {template['instruction']}
        """
        else:
            context_section = f"""
        Note: Provide comprehensive information based on your knowledge about {company}'s services.
        {template['instruction']}
        """
        
        contact_info = ""
        if template.get('include_contact', False):
            contact_info = " For detailed information tailored to your specific needs, you can also contact info@{company_lower}.com or visit {company_lower}.com/contact."
        
        system_prompt = f"""
{context_section}
        
        You are a highly intelligent and professional AI assistant for {company} {company_descriptor}.
        
        CRITICAL RULES:
        1. The user just said "yes" or similar after you asked if they want to know more.
        2. Provide comprehensive, structured, and helpful information based on the context and instructions above.
        3. Be enthusiastic, authoritative, and professional.
        4. Organize your response with clear points or brief paragraphs for readability.
        5. CRITICAL: Provide enough detail to be truly helpful. End with a full sentence and period.
        6. CRITICAL: NEVER ask follow-up questions. NEVER end responses with questions like "What would you like to know?", "What do you need help with?", "Would you like to know more?", etc. Just provide the information directly and end with a period.{contact_info}
        7. LANGUAGE: {language_instruction}
        8. NEVER mention other companies - ONLY {company} services.
        9. Use the context provided above to give accurate, specific information. If information is missing, state it clearly but professionally.
        
        Remember: You represent {company} EXCLUSIVELY. Be detailed, helpful, and never ask questions.
        """
        
        messages = [{"role": "system", "content": apply_company_placeholders(system_prompt)}]
        if conversation_history:
            messages.extend(conversation_history[-5:])  # Last 5 messages for context
        
        # Add user intent message
        user_message = f"I want to know more. Please provide detailed information."
        messages.append({"role": "user", "content": user_message})
        
                    attempts += 1
                    await _advance_groq_key()
                    continue

                logger.error(
                    f"Groq API error {response.status_code} with key {_mask_api_key(active_key)}: {response.text}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"AI service temporarily unavailable. Please visit {WEBSITE_URL} for more information."
                )
        
        if not response or response.status_code != 200:
            logger.error("All configured Groq API keys have been exhausted or failed.")
            raise HTTPException(
                status_code=503,
                    detail=f"AI service temporarily unavailable. Please visit {WEBSITE_URL} for more information."
            )
        
        if response.status_code == 200:
            result = response.json()
            detailed_reply = result['choices'][0]['message']['content'].strip()
            detailed_reply = strip_markdown(detailed_reply)
            logger.info(f"Generated detailed {followup_type} response via RAG + Groq")
            return detailed_reply
        else:
            raise HTTPException(
                status_code=500,
                    detail=f"AI service temporarily unavailable. Please visit {WEBSITE_URL} for more information."
            )
            
    except Exception as e:
        logger.error(f"Error generating RAG-based response for {followup_type}: {e}")
        return None
async def get_contextual_response(context: Dict[str, Any], user_language: str = 'english', 
                                   conversation_history: List[Dict[str, str]] = None, 
                                   session_id: str = None) -> str:
    """
    Generate appropriate response based on conversation context with universal RAG support for ALL follow-up types
    """
    response_type = context.get("type", "general_followup")
    user_response = context.get("user_response", "")
    last_bot_response = context.get("last_bot_response", "")
    detected_service = context.get("detected_service")
    service_confidence = context.get("service_confidence", 0.0)
    
    # Only process "yes" responses - "no" responses get fallback
    if user_response.lower() not in ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'absolutely', 'definitely']:
        # Handle "no" responses
        if response_type == "pricing_followup":
            return "No problem! If you change your mind about pricing, feel free to ask anytime."
        elif response_type == "project_followup":
            return "No worries! Feel free to reach out when you're ready to start a project with us."
        elif response_type == "service_benefits_followup":
            return "No problem! Is there anything else I can help you with regarding our services?"
        elif response_type == "service_info_followup":
            return "No worries! Feel free to ask if you need information about any of our services."
        else:
            return "No problem! If you have questions, feel free to ask anytime."
    
    # Universal RAG + Groq approach for ALL follow-up types
    # Generate queries based on follow-up type
    queries = generate_rag_query_for_followup(response_type, last_bot_response, detected_service)
    
    # Try to generate RAG-based response
    rag_response = await generate_rag_based_response(
        followup_type=response_type,
        queries=queries,
        last_bot_response=last_bot_response,
        user_response=user_response,
        conversation_history=conversation_history or [],
        user_language=user_language,
        detected_service=detected_service if response_type == "service_followup" else None
    )
    
    # If RAG-based response generated successfully, return it
    if rag_response:
        return rag_response
    
    # Fallback to existing responses if RAG fails
    logger.info(f"RAG-based response failed for {response_type}, using fallback response")
    
    # Fallback responses (existing logic)
    profile = get_domain_profile()
    if response_type == "service_benefits_followup":
        offerings_summary = profile.summary_of_offerings()
        if user_language == 'hindi':
            return f"{profile.company_name} ke offerings ({offerings_summary}) aapke business goals ko seedha support karte hain. Batayein aap kya achieve karna chahte hain, main turant relevant context share karunga."
        else:
            return f"{profile.company_name} focuses on outcomes like {offerings_summary}. Share the scenario you're exploring and I'll highlight the most relevant ways we support it."
    
    elif response_type == "service_info_followup":
        return get_general_service_fallback()
    
    elif response_type == "pricing_followup":
        return f"For detailed pricing tailored to your needs, please email {CONTACT_EMAIL} or visit {CONTACT_PAGE_URL} to connect with our team."
    
    elif response_type == "project_followup":
        return f"Fantastic! {profile.company_name} would love to collaborate. Share your goals or timeline and I'll route the right details."
    
    elif response_type == "service_followup" and detected_service:
        # Fallback for service follow-up
        return f"I'd be happy to tell you more about our {detected_service.upper()} offerings. Let me know what you're aiming to solve and I'll share the most relevant highlights."
    
    # Default response for general follow-ups
    return get_general_service_fallback()

# Pydantic models
class ChatRequest(BaseModel):
    # Use Any type to prevent Pydantic from auto-coercing coroutines
    message: Any
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None  # Add context for conversation flow
    
    class Config:
        arbitrary_types_allowed = True
    
    @root_validator(pre=True)
    def fix_coroutine_in_data(cls, values):
        """Root validator that runs FIRST - fixes coroutines before any field validation"""
        if isinstance(values, dict) and 'message' in values:
            msg = values['message']
            # Check if message is a coroutine
            if hasattr(msg, '__await__'):
                logger.error(f"[DEBUG ROOT VALIDATOR] CRITICAL: message is coroutine! Type: {type(msg)}")
                try:
                    values['message'] = str(msg) if msg else ""
                    logger.info(f"[DEBUG ROOT VALIDATOR] Converted coroutine to string")
                except Exception as e:
                    logger.error(f"[DEBUG ROOT VALIDATOR] Failed to convert: {e}")
                    values['message'] = ""
            # Ensure it's a string
            elif not isinstance(msg, str):
                logger.warning(f"[DEBUG ROOT VALIDATOR] message not string, type: {type(msg)}, converting...")
                try:
                    values['message'] = str(msg) if msg else ""
                except:
                    values['message'] = ""
        return values
    
    @validator('message', pre=True)
    def validate_message_pre(cls, v):
        """Pre-validator - runs before type coercion, converts coroutines to strings"""
        logger.info(f"[DEBUG PRE-VALIDATOR] Type: {type(v)}, has __await__: {hasattr(v, '__await__') if v else False}")
        
        # CRITICAL: Check for coroutine FIRST
        if hasattr(v, '__await__'):
            logger.error(f"[DEBUG PRE-VALIDATOR] CRITICAL: Coroutine detected! Converting...")
            try:
                v = str(v) if v else ""
            except Exception as e:
                logger.error(f"[DEBUG PRE-VALIDATOR] Conversion failed: {e}")
                v = ""
        
        # Ensure it's a string
        if not isinstance(v, str):
            logger.warning(f"[DEBUG PRE-VALIDATOR] Not string, converting...")
            try:
                v = str(v) if v else ""
            except:
                v = ""
        
        return v
    
    @validator('message')
    def validate_message(cls, v):
        """Main validator - converts to string and validates"""
        logger.info(f"[DEBUG VALIDATOR] Type: {type(v)}")
        
        # Final safety check - should never be coroutine at this point
        if hasattr(v, '__await__'):
            logger.error(f"[DEBUG VALIDATOR] CRITICAL: Still coroutine! Converting...")
            v = str(v) if v else ""
        
        if not isinstance(v, str):
            v = str(v) if v else ""
        
        # Now safe to call strip() - v is guaranteed to be string
        try:
            v = v.strip()
        except Exception as e:
            logger.error(f"[DEBUG VALIDATOR] strip() error: {e}, type: {type(v)}")
            # Last resort conversion
            v = str(v).strip() if v else ""
        
        if not v:
            raise ValueError('Message cannot be empty')
        
        if len(v) > 1000:
            raise ValueError('Message too long (max 1000 characters)')
        
        # Basic XSS protection
        dangerous_chars = ['<script', '</script', 'javascript:', 'onload=', 'onerror=']
        v_lower = v.lower()
        for char in dangerous_chars:
            if char in v_lower:
                raise ValueError('Message contains potentially dangerous content')
        
        logger.info(f"[DEBUG VALIDATOR] Validated successfully, length: {len(v)}")
        return v
    
    @property
    def message_str(self) -> str:
        """Property to always return message as string"""
        msg = self.message
        if hasattr(msg, '__await__'):
            return str(msg) if msg else ""
        if not isinstance(msg, str):
            return str(msg) if msg else ""
        return msg
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if v and len(v) > 100:
            raise ValueError('Session ID too long')
        return v
class ChatResponse(BaseModel):
    reply: str
    sources: List[str] = []
    website_url: str = WEBSITE_URL  # Use config value
    context: Optional[Dict[str, Any]] = None  # Add context for conversation flow
    
    @validator('reply', pre=True, always=True)
    def sanitize_reply(cls, v):
        return sanitize_response_text(v)
    
    class Config:
        # Ensure all fields are included in JSON response
        fields = {
            'reply': {'exclude': False},
            'sources': {'exclude': False},
            'website_url': {'exclude': False},
            'context': {'exclude': False}
        }

class URLList(BaseModel):
    urls: List[str]
# Website Scraping Functions
def scrape_website_dynamic(query: str) -> Dict[str, Any]:
    """
    Scrape website for relevant information based on query
    Uses the new automatic crawling logic from rag_helper
    Returns structured data that can be added to ChromaDB
    """
    try:
        logger.info(f"Scraping {WEBSITE_URL} for query: {query}")
        
        # Main website URL from config
        base_url = WEBSITE_URL
        
        # Use the internal extraction logic instead of internal utils module
        extracted = extract_content(base_url)
        all_content = extracted.get('content', '')
        
        if not all_content:
            logger.warning(f"No content scraped from {WEBSITE_URL}")
            return {}
        
        # Parse the scraped content and structure it
        # Split into sentences/chunks
        sentences = all_content.split('. ')
        
        scraped_content = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 20:  # Only meaningful chunks
                scraped_content.append({
                    'text': sentence.strip(),
                    'url': base_url,  # All content is from various pages
                    'type': 'content'
                })
        
        # Process and structure the scraped data
        structured_data = process_scraped_content(scraped_content, query)
        
        logger.info(f"Successfully scraped {len(structured_data)} content chunks from multiple pages on {WEBSITE_URL}")
        return structured_data
        
    except Exception as e:
        logger.error(f"Error scraping {WEBSITE_URL}: {str(e)}")
        return {}

def extract_page_content(soup: BeautifulSoup, url: str, query: str) -> List[Dict[str, str]]:
    """Extract relevant content from a webpage"""
    content_chunks = []
    
    try:
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from different sections
        sections = [
            ('h1', 'heading'),
            ('h2', 'heading'), 
            ('h3', 'heading'),
            ('p', 'paragraph'),
            ('div', 'content'),
            ('section', 'section'),
            ('article', 'article')
        ]
        
        for tag, content_type in sections:
            elements = soup.find_all(tag)
            for element in elements:
                text = element.get_text().strip()
                if text and len(text) > 20:  # Only meaningful content
                    # Check if content is relevant to query
                    if is_content_relevant(text, query):
                        content_chunks.append({
                            'text': text,
                            'type': content_type,
                            'url': url,
                            'source': 'website_scraping'
                        })
        
        return content_chunks
        
    except Exception as e:
        logger.warning(f"Error extracting content from {url}: {str(e)}")
        return []

def is_content_relevant(text: str, query: str) -> bool:
    """Check if scraped content is relevant to the user query"""
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Keywords that indicate relevance (generic - works for any company type)
    relevant_keywords = [
        'service', 'services', 'company', 'about', 'contact',
        'team', 'client', 'customer', 'experience', 'expertise',
        'portfolio', 'project', 'solution', 'solutions', 'offering', 'offerings'
    ]
    
    # Check if query keywords appear in text
    query_words = query_lower.split()
    for word in query_words:
        if word in text_lower and len(word) > 3:
            return True
    
    # Check if relevant keywords appear
    for keyword in relevant_keywords:
        if keyword in text_lower:
            return True
    
    return False
def process_scraped_content(content_chunks: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    """Process and structure scraped content for ChromaDB"""
    processed_chunks = []
    
    try:
        for i, chunk in enumerate(content_chunks):
            # Clean and structure the content
            clean_text = re.sub(r'\s+', ' ', chunk['text']).strip()
            
            if len(clean_text) > 50:  # Only meaningful chunks
                processed_chunks.append({
                    'id': f"scraped_content_{i}_{hash(clean_text) % 10000}",
                    'text': clean_text,
                    'metadata': {
                        'source': '{company_lower}_website',
                        'url': chunk['url'],
                        'type': chunk['type'],
                        'query': query,
                        'scraped_at': datetime.now().isoformat()
                    }
                })
        
        update_domain_profile_from_chunks(processed_chunks)
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error processing scraped content: {str(e)}")
        return []

def generate_response_from_scraped_data(scraped_data: List[Dict[str, str]], query: str) -> str:
    """Generate a response using scraped data when ChromaDB/Groq fails"""
    try:
        if not scraped_data:
            return get_general_service_fallback("Let me know what you'd like to explore.")
        
        # Extract relevant content from scraped data
        relevant_content = []
        for chunk in scraped_data[:5]:  # Use top 5 most relevant chunks
            if chunk.get('text'):
                relevant_content.append(chunk['text'])
        
        if not relevant_content:
            return get_general_service_fallback("Tell me which topic matters most to you.")
        
        # Create a simple response based on scraped content
        content_text = " ".join(relevant_content)
        
        # Simple keyword-based response generation
        if "project" in query.lower() or "projects" in query.lower():
            if "completed" in content_text.lower() or "delivered" in content_text.lower():
                return "{company} has successfully completed numerous projects tailored to client goals across multiple domains highlighted on our website. Share the project details you're exploring and I can surface the most relevant reference points."
            else:
                return "{company} continually delivers tailored projects across the focus areas mentioned on our website. Let me know the outcome you're targeting and I'll point you to the most relevant initiatives."
        
        # Generic response for other queries
        return f"Based on our website content, {company} offers solutions described below. {content_text[:200]}..."
        
    except Exception as e:
        logger.error(f"Error generating response from scraped data: {str(e)}")
        return get_general_service_fallback()

async def scheduled_scraping_job():
    """
    Scheduled job function that runs daily at 3 AM to scrape website and update ChromaDB
    This is a background task that automatically keeps the database up to date
    """
    try:
        logger.info("Starting scheduled daily scraping job at 3 AM")
        
        # Trigger scraping with a special query
        scraped_data = await asyncio.to_thread(scrape_website_dynamic, "scheduled_daily_update")
        
        if scraped_data and len(scraped_data) > 0:
            # Add scraped content to ChromaDB
            success = await asyncio.to_thread(add_scraped_content_to_chromadb, scraped_data)
            
            if success:
                logger.info("Scheduled scraping job completed successfully")
            else:
                logger.warning("Scheduled scraping job completed but encountered issues adding to ChromaDB")
        else:
            logger.warning("Scheduled scraping job found no new content")

        # Ensure all priority queries remain preloaded/fresh
        for query in PRIORITY_PRELOAD_QUERIES:
            status, count = await _ensure_priority_query_preloaded(query)
            if status == "cached":
                logger.info(f"[Scheduled Job] Priority query cached: '{query}' ({count} relevant chunks)")
            elif status == "added":
                logger.info(f"[Scheduled Job] Priority query refreshed: '{query}' ({count} chunks scraped)")
            elif status == "empty":
                logger.warning(f"[Scheduled Job] No content found while refreshing query '{query}'")
            elif status == "add_failed":
                logger.warning(f"[Scheduled Job] Failed to update priority query '{query}' ({count} chunks scraped)")
            else:
                logger.warning(f"[Scheduled Job] Priority preload status '{status}' for query '{query}'")
            
    except Exception as e:
        logger.error(f"Error in scheduled scraping job: {str(e)}")

async def _ensure_priority_query_preloaded(query: str) -> Tuple[str, int]:
    """
    Ensure a specific query has indexed content in ChromaDB.
    Returns (status, count) for logging.
    """
    try:
        existing_results = await asyncio.to_thread(search_chroma, query, COLLECTION_NAME, 3)
        if existing_results:
            return ("cached", len(existing_results))
        
        scraped_data = await asyncio.to_thread(scrape_website_dynamic, query)
        if not scraped_data:
            return ("empty", 0)
        
        added = await asyncio.to_thread(add_scraped_content_to_chromadb, scraped_data)
        if added:
            return ("added", len(scraped_data))
        return ("add_failed", len(scraped_data))
    except Exception as e:
        logger.error(f"Error preloading priority query '{query}': {str(e)}")
        return ("error", 0)

async def preload_priority_content():
    """
    Pre-ingest high-priority queries so responses are instant without live scraping.
    Runs in background on startup to keep accuracy while reducing latency.
    """
    logger.info("Starting priority content preload for latency-critical queries")
    
    for query in PRIORITY_PRELOAD_QUERIES:
        status, count = await _ensure_priority_query_preloaded(query)
        if status == "cached":
            logger.info(f"Priority query already covered: '{query}' ({count} relevant chunks)")
        elif status == "added":
            logger.info(f"Priority content ingested for query '{query}' ({count} chunks scraped)")
        elif status == "empty":
            logger.warning(f"Priority scraping produced no content for query '{query}'")
        elif status == "add_failed":
            logger.warning(f"Failed to add scraped priority content for query '{query}' ({count} chunks scraped)")
        else:
            logger.warning(f"Priority preload encountered status '{status}' for query '{query}'")
    
    logger.info("Priority content preload complete")

def add_scraped_content_to_chromadb(scraped_data: List[Dict[str, str]]) -> bool:
    """Add scraped content to ChromaDB for future use"""
    try:
        if not scraped_data:
            return False
        
        client = get_chroma_client()
        model = get_embedding_model()
        
        if not client or not model:
            return False
        
        collection = client.get_collection(COLLECTION_NAME)
        
        # Prepare data for ChromaDB
        texts = [chunk['text'] for chunk in scraped_data]
        ids = [chunk['id'] for chunk in scraped_data]
        metadatas = [chunk['metadata'] for chunk in scraped_data]
        
        # Check for existing IDs to avoid duplicates
        try:
            existing_results = collection.get(ids=ids)
            existing_ids = set(existing_results['ids']) if existing_results['ids'] else set()
            
            # Filter out existing IDs
            new_texts = []
            new_ids = []
            new_metadatas = []
            new_indices = []
            
            for i, chunk_id in enumerate(ids):
                if chunk_id not in existing_ids:
                    new_texts.append(texts[i])
                    new_ids.append(chunk_id)
                    new_metadatas.append(metadatas[i])
                    new_indices.append(i)
            
            # Only add if there are new items
            if new_texts:
                # Generate embeddings for new content only
                embeddings = model.encode(new_texts)
                
                # Add to collection
                collection.add(
                    embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
                    documents=new_texts,
                    ids=new_ids,
                    metadatas=new_metadatas
                )
                
                logger.info(f"Added {len(new_texts)} new scraped content chunks to ChromaDB")
            else:
                logger.info("All scraped content already exists in ChromaDB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking existing IDs in ChromaDB: {str(e)}")
            return False
        
    except Exception as e:
        logger.error(f"Error adding scraped content to ChromaDB: {str(e)}")
        return False

# Initialize ChromaDB
def get_chroma_client():
    """Get cached ChromaDB client or initialize if not exists"""
    global chroma_client
    if chroma_client is None:
        try:
            chroma_client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("ChromaDB client initialized and cached")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            return None
    return chroma_client

# Initialize embedding model
def get_embedding_model():
    """Get cached embedding model or initialize if not exists"""
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model initialized and cached")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            return None
    return embedding_model

# Web scraping functions
def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    # Remove very short words
    text = ' '.join([word for word in text.split() if len(word) > 2])
    return text.strip()
def extract_content(url: str) -> Dict[str, Any]:
    """Extract and clean content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        logger.info(f"Scraping website: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract main content
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No Title"
        
        # Extract text from different elements
        text_elements = []
        
        # Get headings
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = tag.get_text().strip()
            if len(text) > 5:
                text_elements.append(text)
        
        # Get paragraphs
        for tag in soup.find_all('p'):
            text = tag.get_text().strip()
            if len(text) > 20:  # Only meaningful paragraphs
                text_elements.append(text)
        
        # Get list items
        for tag in soup.find_all('li'):
            text = tag.get_text().strip()
            if len(text) > 10:
                text_elements.append(text)
        
        # Get div content with substantial text, but avoid common UI fragments
        for tag in soup.find_all('div', recursive=False): # Only top-level large divs often contain layout content
             text = tag.get_text(separator=' ').strip()
             if len(text) > 100:
                 # Sub-filter to remove fragments that look like buttons/links
                 sentences = [s.strip() for s in text.split('. ') if len(s.strip()) > 30]
                 if sentences:
                    text_elements.extend(sentences)
        
        # Clean and join text
        cleaned_text = clean_text(' '.join(text_elements))
        
        logger.info(f"Successfully scraped {len(cleaned_text)} characters from {url}")
        
        return {
            'url': url,
            'title': title_text,
            'content': cleaned_text,
            'timestamp': datetime.now().isoformat(),
            'content_length': len(cleaned_text)
        }
        
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {str(e)}")
        return None

# =============================================================================
# AUTOMATIC COMPANY INFO EXTRACTION FROM WEBSITE
# =============================================================================

def extract_company_name_from_html(soup: BeautifulSoup, url: str) -> List[str]:
    """Extract company name candidates from HTML"""
    candidates = []
    
    try:
        # 1. Title tag
        title = soup.find('title')
        if title:
            title_text = title.get_text().strip()
            # Clean common suffixes
            for suffix in [' - Home', ' | Home', ' - Welcome', ' | Welcome', ' | Company']:
                if title_text.endswith(suffix):
                    title_text = title_text[:-len(suffix)].strip()
            if title_text:
                candidates.append(title_text)
        
        # 2. H1 tag (homepage main heading)
        h1 = soup.find('h1')
        if h1:
            h1_text = h1.get_text().strip()
            if h1_text and len(h1_text) < 100:  # Reasonable length
                candidates.append(h1_text)
        
        # 3. Meta og:title
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            candidates.append(og_title.get('content').strip())
        
        # 4. Meta twitter:title
        twitter_title = soup.find('meta', attrs={'name': 'twitter:title'})
        if twitter_title and twitter_title.get('content'):
            candidates.append(twitter_title.get('content').strip())
        
        # 5. Logo alt text
        logo = soup.find('img', alt=re.compile(r'logo', re.I))
        if logo and logo.get('alt'):
            alt_text = logo.get('alt')
            # Extract company name from "Company Name Logo"
            alt_text = re.sub(r'\s*logo\s*', '', alt_text, flags=re.I).strip()
            if alt_text:
                candidates.append(alt_text)
        
        # 6. Domain name inference
        domain = urlparse(url).netloc.replace("www.", "").split(".")[0]
        if domain:
            # Convert domain to title case (websjyoti -> Websjyoti)
            domain_name = domain.replace("-", " ").replace("_", " ").title()
            candidates.append(domain_name)
        
    except Exception as e:
        logger.debug(f"Error extracting company name: {e}")
    
    # Remove duplicates and empty strings
    return list(dict.fromkeys([c for c in candidates if c and len(c) > 2]))


def extract_tagline_from_html(soup: BeautifulSoup, url: str) -> List[str]:
    """Extract tagline candidates from HTML"""
    candidates = []
    
    try:
        # 1. Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            desc = meta_desc.get('content').strip()
            if desc and len(desc) < 200:  # Taglines are usually short
                candidates.append(desc)
        
        # 2. Meta og:description
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            desc = og_desc.get('content').strip()
            if desc and len(desc) < 200:
                candidates.append(desc)
        
        # 3. Hero section (first large paragraph or div)
        hero_sections = soup.find_all(['section', 'div'], class_=re.compile(r'hero|banner|intro', re.I))
        for hero in hero_sections[:2]:  # Check first 2 hero sections
            text = hero.get_text().strip()
            # Get first meaningful sentence
            sentences = re.split(r'[.!?]\s+', text)
            for sentence in sentences[:3]:  # First 3 sentences
                sentence = sentence.strip()
                if 20 < len(sentence) < 200:  # Tagline length
                    candidates.append(sentence)
                    break
        
        # 4. H1 tag if it's descriptive (not just company name)
        h1 = soup.find('h1')
        if h1:
            h1_text = h1.get_text().strip()
            # If H1 is longer than typical company name, might be tagline
            if len(h1_text) > 20 and len(h1_text) < 150:
                candidates.append(h1_text)
        
        # 5. First paragraph of main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            first_p = main_content.find('p')
            if first_p:
                p_text = first_p.get_text().strip()
                if 30 < len(p_text) < 250:
                    # Get first sentence
                    first_sentence = re.split(r'[.!?]\s+', p_text)[0]
                    if len(first_sentence) > 20:
                        candidates.append(first_sentence.strip())
        
    except Exception as e:
        logger.debug(f"Error extracting tagline: {e}")
    
    return list(dict.fromkeys([c for c in candidates if c and len(c) > 10]))


def extract_email_from_html(soup: BeautifulSoup, url: str) -> List[str]:
    """Extract email addresses from HTML"""
    emails = set()
    
    try:
        # 1. mailto: links
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if href.startswith('mailto:'):
                email = href.replace('mailto:', '').split('?')[0].strip()
                if email and '@' in email:
                    emails.add(email.lower())
        
        # 2. Regex pattern in text content
        email_pattern = r'\b[\w\.-]+@[\w\.-]+\.\w+\b'
        all_text = soup.get_text()
        found_emails = re.findall(email_pattern, all_text)
        for email in found_emails:
            # Filter out common non-contact emails
            if not any(skip in email.lower() for skip in ['example.com', 'test@', 'noreply', 'no-reply']):
                emails.add(email.lower())
        
        # 3. Contact page specific (if this is contact page)
        if 'contact' in url.lower():
            contact_section = soup.find(['section', 'div'], class_=re.compile(r'contact', re.I))
            if contact_section:
                contact_text = contact_section.get_text()
                found_emails = re.findall(email_pattern, contact_text)
                for email in found_emails:
                    emails.add(email.lower())
        
    except Exception as e:
        logger.debug(f"Error extracting email: {e}")
    
    # Return sorted by priority (info@, contact@, etc. first)
    email_list = list(emails)
    priority_emails = [e for e in email_list if any(prefix in e for prefix in ['info@', 'contact@', 'hello@', 'support@'])]
    other_emails = [e for e in email_list if e not in priority_emails]
    return priority_emails + other_emails


def extract_phone_from_html(soup: BeautifulSoup, url: str) -> List[str]:
    """Extract phone numbers from HTML"""
    phones = set()
    
    try:
        # 1. tel: links
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if href.startswith('tel:'):
                phone = href.replace('tel:', '').strip()
                if phone:
                    phones.add(phone)
        
        # 2. Regex patterns for phone numbers
        # Indian format: +91-XXXXXXXXXX or +91 XXXXXXXXXX
        indian_pattern = r'\+?91[\s-]?\d{2}[\s-]?\d{4}[\s-]?\d{4}'
        # International format: +X-XXX-XXX-XXXX
        intl_pattern = r'\+?\d{1,3}[\s-]?\d{1,4}[\s-]?\d{1,4}[\s-]?\d{1,9}'
        # Local format: XXXX-XXXXXX or XXXXX-XXXXX
        local_pattern = r'\d{4,5}[\s-]?\d{5,10}'
        
        all_text = soup.get_text()
        
        # Find all phone patterns
        for pattern in [indian_pattern, intl_pattern, local_pattern]:
            found_phones = re.findall(pattern, all_text)
            for phone in found_phones:
                phone = phone.strip()
                # Filter out obviously wrong numbers (too short/long)
                digits_only = re.sub(r'[\s-]', '', phone)
                if 10 <= len(digits_only) <= 15:
                    phones.add(phone)
        
        # 3. Contact page specific
        if 'contact' in url.lower():
            contact_section = soup.find(['section', 'div'], class_=re.compile(r'contact', re.I))
            if contact_section:
                contact_text = contact_section.get_text()
                for pattern in [indian_pattern, intl_pattern, local_pattern]:
                    found_phones = re.findall(pattern, contact_text)
                    for phone in found_phones:
                        digits_only = re.sub(r'[\s-]', '', phone)
                        if 10 <= len(digits_only) <= 15:
                            phones.add(phone.strip())
        
    except Exception as e:
        logger.debug(f"Error extracting phone: {e}")
    
    return list(phones)


async def extract_industry_from_content(content: str, soup: BeautifulSoup = None) -> Optional[str]:
    """Extract industry from content using LLM (universal approach - works for any industry)"""
    if not content:
        return None
    
    try:
        # Use LLM to infer industry from content (no hardcoded keywords)
        industry_prompt = f"""Analyze this website content and identify the primary industry or business type.

Content (first 1000 characters): {content[:1000]}

Return ONLY the industry name (e.g., "Restaurant", "Hospital", "IT Services", "Construction", "Furniture", "Training Institute", etc.). 
If unclear, return "General Services". No explanation, just the industry name."""

        messages = [
            {"role": "system", "content": "You are an industry classification assistant. Return only the industry name."},
            {"role": "user", "content": industry_prompt}
        ]
        
        industry = await _get_llm_response(
            messages=messages,
            max_tokens=20,
            temperature=0.3
        )
        
        if industry:
            industry = industry.strip()
            # Clean up common suffixes
            industry = re.sub(r'\s+services?$', '', industry, flags=re.IGNORECASE)
            return industry.title() if industry else None
        
        return None
    except Exception as e:
        logger.warning(f"Error extracting industry with LLM: {e}")
        return None


async def validate_extracted_info_with_ai(extracted_info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean extracted info using AI"""
    if not extracted_info:
        return extracted_info
    
    try:
        # Build validation prompt
        validation_prompt = f"""
You are a data validation assistant. Clean and validate the following extracted company information from a website.

Extracted Information:
- Company Name Candidates: {extracted_info.get('company_name_candidates', [])}
- Tagline Candidates: {extracted_info.get('tagline_candidates', [])}
- Email Addresses: {extracted_info.get('emails', [])}
- Phone Numbers: {extracted_info.get('phones', [])}
- Industry: {extracted_info.get('industry', 'Not found')}

Website URL: {extracted_info.get('website_url', '')}

Instructions:
1. For Company Name: Select the best, cleanest company name. Remove suffixes like "- Home", "| Company", etc. Return only the company name.
2. For Tagline: Select the most impactful, short tagline (1 sentence, max 150 characters). Should be marketing-focused.
3. For Email: Select the primary contact email (prefer info@, contact@, hello@). Return only one email.
4. For Phone: Select the primary phone number in international format (+91-XXXXXXXXXX). Return only one phone.
5. For Industry: If provided, keep it. If not, infer from context.

Return your response as a JSON object with these exact keys:
{{
    "company_name": "cleaned company name",
    "tagline": "best tagline or null",
    "email": "primary email or null",
    "phone": "primary phone or null",
    "industry": "industry or null"
}}

Only return valid JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are a helpful data validation assistant. Always return valid JSON only."},
            {"role": "user", "content": validation_prompt}
        ]
        
        # Call Groq API for validation
        validated_json = await _get_llm_response(
            messages=messages,
            max_tokens=200,
            temperature=0.3  # Lower temperature for more consistent validation
        )
        
        if validated_json:
            try:
                # Try to parse JSON from response
                import json
                # Extract JSON from response (might have markdown code blocks)
                json_match = re.search(r'\{[^{}]*\}', validated_json, re.DOTALL)
                if json_match:
                    validated_data = json.loads(json_match.group())
                    
                    # Update extracted_info with validated data
                    if validated_data.get('company_name'):
                        extracted_info['company_name'] = validated_data['company_name']
                    if validated_data.get('tagline'):
                        extracted_info['tagline'] = validated_data['tagline']
                    if validated_data.get('email'):
                        extracted_info['email'] = validated_data['email']
                    if validated_data.get('phone'):
                        extracted_info['phone'] = validated_data['phone']
                    if validated_data.get('industry'):
                        extracted_info['industry'] = validated_data['industry']
                    
                    logger.info("AI validation completed successfully")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse AI validation JSON: {e}")
        
    except Exception as e:
        logger.warning(f"AI validation failed, using pattern-based results: {e}")
    
    return extracted_info


async def extract_company_info_from_website() -> Optional[Dict[str, Any]]:
    """
    Automatically extract company information from website.
    Returns extracted info dictionary or None if extraction fails.
    """
    if not WEBSITE_URL:
        logger.warning("WEBSITE_URL not configured, skipping auto-extraction")
        return None
    
    try:
        logger.info(f"Starting automatic company info extraction from {WEBSITE_URL}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        extracted_info = {
            'website_url': WEBSITE_URL,
            'company_name_candidates': [],
            'tagline_candidates': [],
            'emails': [],
            'phones': [],
            'industry': None,
            'confidence': {}
        }
        
        # Scrape homepage
        try:
            response = requests.get(WEBSITE_URL, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract from homepage
            extracted_info['company_name_candidates'].extend(extract_company_name_from_html(soup, WEBSITE_URL))
            extracted_info['tagline_candidates'].extend(extract_tagline_from_html(soup, WEBSITE_URL))
            extracted_info['emails'].extend(extract_email_from_html(soup, WEBSITE_URL))
            extracted_info['phones'].extend(extract_phone_from_html(soup, WEBSITE_URL))
            
            # Get content for industry inference (using LLM)
            homepage_content = soup.get_text()
            extracted_info['industry'] = await extract_industry_from_content(homepage_content, soup)
            
            logger.info(f"Extracted from homepage: {len(extracted_info['company_name_candidates'])} name candidates, {len(extracted_info['emails'])} emails, {len(extracted_info['phones'])} phones")
        except Exception as e:
            logger.warning(f"Failed to scrape homepage: {e}")
        
        # Try to scrape contact page
        contact_urls = [
            urljoin(WEBSITE_URL, '/contact'),
            urljoin(WEBSITE_URL, '/contact-us'),
            urljoin(WEBSITE_URL, '/get-in-touch'),
            urljoin(WEBSITE_URL, '/reach-us'),
        ]
        
        for contact_url in contact_urls[:2]:  # Try first 2 contact URLs
            try:
                response = requests.get(contact_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Extract contact info
                    extracted_info['emails'].extend(extract_email_from_html(soup, contact_url))
                    extracted_info['phones'].extend(extract_phone_from_html(soup, contact_url))
                    logger.info(f"Extracted contact info from {contact_url}")
                    break
            except:
                continue
        
        # Try to scrape about page for industry/tagline
        about_urls = [
            urljoin(WEBSITE_URL, '/about'),
            urljoin(WEBSITE_URL, '/about-us'),
        ]
        
        for about_url in about_urls[:1]:  # Try first about URL
            try:
                response = requests.get(about_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    about_content = soup.get_text()
                    # Extract tagline and industry
                    extracted_info['tagline_candidates'].extend(extract_tagline_from_html(soup, about_url))
                    if not extracted_info['industry']:
                        extracted_info['industry'] = await extract_industry_from_content(about_content, soup)
                    logger.info(f"Extracted info from {about_url}")
                    break
            except:
                continue
        
        # Remove duplicates
        extracted_info['company_name_candidates'] = list(dict.fromkeys(extracted_info['company_name_candidates']))
        extracted_info['tagline_candidates'] = list(dict.fromkeys(extracted_info['tagline_candidates']))
        extracted_info['emails'] = list(dict.fromkeys(extracted_info['emails']))
        extracted_info['phones'] = list(dict.fromkeys(extracted_info['phones']))
        
        # Select best candidates (before AI validation)
        if extracted_info['company_name_candidates']:
            extracted_info['company_name'] = extracted_info['company_name_candidates'][0]
        if extracted_info['tagline_candidates']:
            extracted_info['tagline'] = extracted_info['tagline_candidates'][0]
        if extracted_info['emails']:
            extracted_info['email'] = extracted_info['emails'][0]
        if extracted_info['phones']:
            extracted_info['phone'] = extracted_info['phones'][0]
        
        # AI validation (if we have candidates)
        if extracted_info.get('company_name_candidates') or extracted_info.get('tagline_candidates'):
            extracted_info = await validate_extracted_info_with_ai(extracted_info)
        
        # Calculate confidence scores
        if extracted_info.get('company_name'):
            extracted_info['confidence']['company_name'] = 0.95 if len(extracted_info.get('company_name_candidates', [])) > 0 else 0.7
        if extracted_info.get('tagline'):
            extracted_info['confidence']['tagline'] = 0.85 if len(extracted_info.get('tagline_candidates', [])) > 1 else 0.7
        if extracted_info.get('email'):
            extracted_info['confidence']['email'] = 0.98 if len(extracted_info.get('emails', [])) > 0 else 0.8
        if extracted_info.get('phone'):
            extracted_info['confidence']['phone'] = 0.95 if len(extracted_info.get('phones', [])) > 0 else 0.8
        if extracted_info.get('industry'):
            extracted_info['confidence']['industry'] = 0.75
        
        logger.info(f"Extraction complete: company_name={extracted_info.get('company_name')}, email={extracted_info.get('email')}, phone={extracted_info.get('phone')}")
        
        return extracted_info
        
    except Exception as e:
        logger.error(f"Error in automatic company info extraction: {e}")
        return None


def update_domain_profile_from_extraction(extracted_info: Dict[str, Any]):
    """Update DomainProfile and global constants with extracted information"""
    if not extracted_info:
        return
    
    global COMPANY_NAME, COMPANY_TAGLINE, CONTACT_EMAIL, CONTACT_PHONE, COMPANY_INDUSTRY, SUPPORT_EMAIL
    
    try:
        # Update company name (if extracted and confident)
        if extracted_info.get('company_name') and extracted_info.get('confidence', {}).get('company_name', 0) > 0.8:
            DOMAIN_PROFILE.company_name = extracted_info['company_name']
            COMPANY_NAME = extracted_info['company_name']  # Update global constant
            logger.info(f"Updated company_name: {DOMAIN_PROFILE.company_name}")
        
        # Update tagline
        if extracted_info.get('tagline') and extracted_info.get('confidence', {}).get('tagline', 0) > 0.7:
            DOMAIN_PROFILE.tagline = extracted_info['tagline']
            COMPANY_TAGLINE = extracted_info['tagline']  # Update global constant
            logger.info(f"Updated tagline: {DOMAIN_PROFILE.tagline}")
        
        # Update contact email
        if extracted_info.get('email') and extracted_info.get('confidence', {}).get('email', 0) > 0.8:
            DOMAIN_PROFILE.contact_email = extracted_info['email']
            CONTACT_EMAIL = extracted_info['email']  # Update global constant
            SUPPORT_EMAIL = extracted_info['email']  # Also update support email
            logger.info(f"Updated contact_email: {DOMAIN_PROFILE.contact_email}")
        
        # Update contact phone
        if extracted_info.get('phone') and extracted_info.get('confidence', {}).get('phone', 0) > 0.8:
            DOMAIN_PROFILE.contact_phone = extracted_info['phone']
            CONTACT_PHONE = extracted_info['phone']  # Update global constant
            logger.info(f"Updated contact_phone: {DOMAIN_PROFILE.contact_phone}")
        
        # Update industry
        if extracted_info.get('industry') and extracted_info.get('confidence', {}).get('industry', 0) > 0.6:
            # CRITICAL: Ensure industry is a string, not a coroutine
            industry_value = extracted_info['industry']
            if hasattr(industry_value, '__await__'):
                logger.error(f"[update_domain_profile] CRITICAL: industry is coroutine! Skipping update.")
            else:
                DOMAIN_PROFILE.industry = str(industry_value) if industry_value else None
                COMPANY_INDUSTRY = str(industry_value) if industry_value else None
                logger.info(f"Updated industry: {DOMAIN_PROFILE.industry}")
        
        logger.info("DomainProfile and global constants updated successfully from extracted information")
        
    except Exception as e:
        logger.error(f"Error updating DomainProfile from extraction: {e}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Only keep substantial chunks
            chunks.append(chunk.strip())
    
    return chunks
# Vector database functions
def store_content_in_chroma(urls: List[str], collection_name: str = COLLECTION_NAME) -> Dict[str, Any]:
    """Store scraped content in ChromaDB"""
    try:
        client = get_chroma_client()
        model = get_embedding_model()
        
        if not client or not model:
            raise Exception("Failed to initialize ChromaDB or embedding model")
        
        # Get or create collection
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            collection = client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")
        
        total_chunks = 0
        successful_urls = 0
        failed_urls = []
        
        for url in urls[:MAX_URLS]:  # Limit to prevent overload
            try:
                logger.info(f"Processing URL: {url}")
                
                # Extract content
                content_data = extract_content(url)
                if not content_data or not content_data['content']:
                    failed_urls.append(url)
                    continue
                
                # Chunk the content
                chunks = chunk_text(content_data['content'])
                if not chunks:
                    failed_urls.append(url)
                    continue
                
                # Generate embeddings
                embeddings = model.encode(chunks)
                
                # Create unique IDs and metadata
                chunk_ids = []
                metadatas = []
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{hashlib.md5(url.encode()).hexdigest()[:8]}_chunk_{i}"
                    chunk_ids.append(chunk_id)
                    metadatas.append({
                        "url": url,
                        "title": content_data['title'],
                        "chunk_index": i,
                        "timestamp": content_data['timestamp'],
                        "content_length": content_data['content_length']
                    })
                
                # Check for existing IDs to avoid duplicates
                try:
                    existing_results = collection.get(ids=chunk_ids)
                    existing_ids = set(existing_results['ids']) if existing_results['ids'] else set()
                    
                    # Filter out existing IDs
                    new_chunks = []
                    new_ids = []
                    new_metadatas = []
                    new_embeddings = []
                    
                    for i, chunk_id in enumerate(chunk_ids):
                        if chunk_id not in existing_ids:
                            new_chunks.append(chunks[i])
                            new_ids.append(chunk_id)
                            new_metadatas.append(metadatas[i])
                            new_embeddings.append(embeddings[i])
                    
                    # Only add if there are new items
                    if new_chunks:
                        collection.add(
                            embeddings=[e.tolist() if hasattr(e, "tolist") else e for e in new_embeddings],
                            documents=new_chunks,
                            ids=new_ids,
                            metadatas=new_metadatas
                        )
                        logger.info(f"Stored {len(new_chunks)} new chunks from {url} (skipped {len(chunks) - len(new_chunks)} duplicates)")
                    else:
                        logger.info(f"All chunks from {url} already exist in ChromaDB")
                        
                except Exception as e:
                    logger.error(f"Error checking existing IDs for {url}: {str(e)}")
                    # Fallback to original behavior if checking fails
                    collection.add(
                        embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
                        documents=chunks,
                        ids=chunk_ids,
                        metadatas=metadatas
                    )
                    logger.info(f"Stored {len(chunks)} chunks from {url}")
                
                total_chunks += len(chunks)
                successful_urls += 1
                
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                failed_urls.append(url)
        
        return {
            'success': True,
            'total_chunks': total_chunks,
            'successful_urls': successful_urls,
            'failed_urls': failed_urls,
            'collection_name': collection_name
        }
        
    except Exception as e:
        logger.error(f"Error storing content in ChromaDB: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
def search_chroma(query: str, collection_name: str = COLLECTION_NAME, n_results: int = 3, query_intent: str = None) -> List[Dict[str, Any]]:
    """Search ChromaDB for relevant content with adaptive relevance filtering based on query intent"""
    try:
        client = get_chroma_client()
        model = get_embedding_model()
        
        if not client or not model:
            raise Exception("Failed to initialize ChromaDB or embedding model")
        
        # Get collection
        try:
            collection = client.get_collection(name=collection_name)
        except:
            raise Exception(f"Collection '{collection_name}' not found")
        
        # Generate query embedding with cache optimization
        global embedding_cache
        query_lower = query.lower().strip()
        if query_lower in embedding_cache:
            query_embedding = embedding_cache[query_lower]
        else:
            query_embedding = model.encode([query])
            # Cache the embedding (limit cache size to prevent memory issues)
            if len(embedding_cache) < 2000:  # Max 2000 cached embeddings (increased for better performance)
                embedding_cache[query_lower] = query_embedding
            else:
                # Clear cache if it gets too large (simple LRU: clear all and start fresh)
                embedding_cache = {query_lower: query_embedding}
        
        # Search
        results = collection.query(
            query_embeddings=query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding,
            n_results=n_results
        )
        
        # Use adaptive threshold based on query intent (universal approach)
        if query_intent:
            threshold = get_adaptive_threshold(query_intent)
        else:
            threshold = RELEVANCE_THRESHOLD  # Fallback to default if no intent provided
        
        # Format results with relevance filtering using adaptive threshold
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0
                
                # Filter by adaptive relevance threshold
                if distance <= threshold:
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'distance': distance
                    })
                    logger.info(f"Result {i+1}: distance={distance:.4f} (relevant, threshold={threshold:.4f})")
                else:
                    logger.info(f"Result {i+1}: distance={distance:.4f} (filtered out - not relevant, threshold={threshold:.4f})")
        
        logger.info(f"Found {len(formatted_results)} relevant documents for query: {query} (after filtering with threshold={threshold:.4f})")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error searching ChromaDB: {str(e)}")
        return []

# API Endpoints

@app.post("/crawl-and-store")
async def crawl_and_store(urls: URLList):
    """Crawl URLs and store content in ChromaDB"""
    try:
        result = store_content_in_chroma(urls.urls, COLLECTION_NAME)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")  # Removed response_model to avoid Pydantic validation issues
# Temporarily disabled rate limiter to test if it's causing coroutine issues
# @limiter.limit("10/minute")  # Rate limit: 10 messages per minute per IP
async def chat(request: Request):
    """Chat endpoint with COMPLETE manual parsing to bypass ALL FastAPI/Pydantic validation"""
    import json
    import traceback
    from fastapi.responses import JSONResponse
    
    try:
        # CRITICAL: Completely bypass FastAPI's body parsing
        # Read raw body and parse manually
        body_bytes = await request.body()
        data = json.loads(body_bytes.decode('utf-8'))
        
        # CRITICAL: Fix message field IMMEDIATELY - convert to string before ANY processing
        if 'message' in data:
            msg = data['message']
            # Multiple layers of protection
            if hasattr(msg, '__await__'):
                logger.error(f"[CRITICAL] message is coroutine in raw JSON! Type: {type(msg)}")
                try:
                    data['message'] = str(msg) if msg else ""
                except:
                    data['message'] = ""
            elif not isinstance(msg, str):
                try:
                    data['message'] = str(msg) if msg else ""
                except:
                    data['message'] = ""
        else:
            data['message'] = ""
        
        # Ensure message is definitely a string at this point
        message_str = str(data.get('message', '')) if data.get('message') else ""
        
        # Create simple request object with guaranteed string message
        class SimpleChatRequest:
            def __init__(self, message: str, session_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
                # Triple-check: message MUST be a string
                if hasattr(message, '__await__'):
                    logger.error(f"[CRITICAL] message is STILL coroutine in SimpleChatRequest!")
                    self.message = str(message) if message else ""
                elif not isinstance(message, str):
                    self.message = str(message) if message else ""
                else:
                    self.message = message
                self.session_id = session_id
                self.context = context or {}
        
        # Create simple request object with already-normalized message
        chat_request = SimpleChatRequest(
            message=message_str,  # Already a string
            session_id=data.get('session_id'),
            context=data.get('context')
        )
        
        # FINAL CHECK: Ensure chat_request.message is a string
        if hasattr(chat_request.message, '__await__'):
            logger.error("[CRITICAL] chat_request.message is STILL a coroutine after SimpleChatRequest!")
            chat_request.message = str(chat_request.message) if chat_request.message else ""
        elif not isinstance(chat_request.message, str):
            chat_request.message = str(chat_request.message) if chat_request.message else ""
        
        # Store normalized message for all subsequent uses
        message_str = chat_request.message
        
        # DEBUG: Log the type and value (using message_str, not chat_request.message)
        logger.info(f"[DEBUG] message_str type: {type(message_str)}")
        logger.info(f"[DEBUG] message_str has __await__: {hasattr(message_str, '__await__')}")
        logger.info(f"[DEBUG] message_str value (first 100 chars): {str(message_str)[:100] if message_str else 'None'}")
        
        # Get company descriptor and offerings from domain profile (independent, no hardcoding)
        # Safely initialize with fallback to prevent NameError
        try:
            profile = get_domain_profile()
            company_descriptor = profile.describe_company_category() if profile else "company"
            primary_offerings_summary = profile.summary_of_offerings() if profile else "services"
        except Exception as e:
            logger.warning(f"Error getting domain profile: {e}, using fallback values")
            company_descriptor = "company"
            primary_offerings_summary = "services"
            profile = None
        
        # Session management
        session_id = chat_request.session_id or str(uuid.uuid4())
        
        # Initialize session if it doesn't exist
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = {
                "conversations": [],
                "user_name": None,  # Store user's name for personalization
                "last_bot_response": None,  # Store last bot response for context
                "conversation_context": None,  # Store conversation context
                "project_context": {}  # Initialize project context
            }
            logger.info(f"Created new session: {session_id}")
        
        detected_language = detect_language(message_str)
        user_language = detected_language
        # Use message_str instead of chat_request.message to ensure it's a string
        soft_negative = is_soft_negative_message(message_str)
        
        # Check for follow-up responses to previous bot questions (HIGHEST PRIORITY)
        last_bot_response = conversation_sessions[session_id].get("last_bot_response", "")
        if last_bot_response and is_follow_up_response(message_str, last_bot_response):
            logger.info(f"Detected follow-up response: {message_str}")
            
            # Get conversation context - use message_str, not chat_request.message
            follow_up_context = get_follow_up_context(last_bot_response, message_str)
            
            # Get conversation history for context-aware response
            conversation_history = conversation_sessions[session_id]["conversations"][-10:]
            
            # Generate contextual reply with RAG support (async)
            contextual_reply = await get_contextual_response(
                follow_up_context, 
                user_language,
                conversation_history=conversation_history,
                session_id=session_id
            )
            
            # Update conversation context
            conversation_sessions[session_id]["conversation_context"] = follow_up_context
            
            # Store in conversation history
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": contextual_reply})
            conversation_sessions[session_id]["last_bot_response"] = contextual_reply
            
            from fastapi.responses import JSONResponse
            return JSONResponse(content={
                "reply": contextual_reply,
                "sources": [],
                "website_url": WEBSITE_URL,
                "context": follow_up_context
            })
        
        # Get conversation history (last 10 messages to limit context)
        conversation_history = conversation_sessions[session_id]["conversations"][-10:]
        
        # Initialize search_results variable (will be used later for RAG)
        search_results = None
        
        # FAST PATTERN CHECKS FIRST (for simple queries to avoid slow HF model)
        # Check for simple greetings/bye first - these are fast and don't need HF
        if is_greeting(message_str):
            logger.info(f"Detected greeting: {message_str}")
            greeting_reply = get_greeting_response(message_str, user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": greeting_reply})
            conversation_sessions[session_id]["last_bot_response"] = greeting_reply
            from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": greeting_reply, "sources": [], "website_url": WEBSITE_URL
            })
        
        if is_goodbye(message_str):
            logger.info(f"Detected goodbye: {message_str}")
            goodbye_reply = get_goodbye_response(user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": goodbye_reply})
            from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": goodbye_reply, "sources": [], "website_url": WEBSITE_URL
            })
        # Fast pattern checks - BEFORE HF to save 2-3 seconds
        # Check for capability questions (fast pattern check before HF)
        message_lower = message_str.lower()  # Use normalized message_str instead
        ai_automation_query = is_ai_automation_query(message_str)
        service_keywords = ['erp', 'crm', 'cloud', 'hosting', 'iot', 'ai', 'service', 'services', 'solution', 'solutions']
        has_service_keywords = any(keyword in message_lower for keyword in service_keywords)
        if ai_automation_query and search_results is None:
            search_results = search_chroma(message_str, COLLECTION_NAME, n_results=3)
        
        if not has_service_keywords and is_capability_question(message_str):
            logger.info(f"Detected generic capability question (no service keywords): {message_str}")
            capability_reply = get_capability_response(user_language, session_id)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": capability_reply})
            conversation_sessions[session_id]["last_bot_response"] = capability_reply
            from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": capability_reply, "sources": [], "website_url": WEBSITE_URL
            })
        
        # Check for help requests (fast pattern check before HF)
        # Removed hardcoded help request handling - everything now goes through RAG flow
        # RAG provides intelligent, context-aware, human-like responses for all queries
        
        # Check for acknowledgments (fast pattern check before HF)
        if is_acknowledgment(message_str):
            logger.info(f"Detected acknowledgment: {message_str}")
            ack_reply = get_acknowledgment_response(user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": ack_reply})
            from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": ack_reply, "sources": [], "website_url": WEBSITE_URL
            })
        
        # Check for contact queries (fast pattern check before HF)
        if is_contact_query(message_str):
            logger.info(f"Detected contact query: {message_str}")
            
            # Search for contact information in ChromaDB
            contact_results = search_chroma(message_str, COLLECTION_NAME, n_results=3)
            
            # Build context from ChromaDB results
            context = ""
            sources = []
            if contact_results:
                logger.info(f"Using ChromaDB vector search results for context ({len(contact_results)} relevant chunks)")
                context = "\n\n".join([result.get('content', '') for result in contact_results])
                sources = list(set([result.get('metadata', {}).get('url', '') for result in contact_results if result.get('metadata', {}).get('url')]))
            
            # Get contact phone from variable
            contact_phone = CONTACT_PHONE or "our contact number"
            
            # Build context section
            if context and context.strip():
                context_section = f"""
        CRITICAL: The context provided below contains specific information about {company}'s contact details, office locations, and contact information from our knowledge base. 
        You MUST base your answer PRIMARILY on this context. Do NOT give generic responses when specific context is available.
        
        CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about {company}'s contact information, office location, email, phone number, or how to reach the company. Provide detailed, human-like information based on the context above. Include contact details like email (info@{company_lower}.com), phone number ({contact_phone}), website ({company_lower}.com/contact), and any location information available in the context.
        Use the context provided above to give accurate, specific information about contact details. CRITICAL: The correct phone number for {company} is {contact_phone}. Always use this number when mentioning phone contact.
        """
            else:
                context_section = f"""
        Note: Use your general knowledge about {company}'s contact information to answer questions accurately.
        The user is asking about {company}'s contact information, office location, email, phone number, or how to reach the company. Provide detailed, human-like information.
        CRITICAL: The correct phone number for {company} is {contact_phone}. Always use this number when mentioning phone contact. The correct email address for {company} is info@{company_lower}.com. Always include this email when user asks for email or contact information.
        """
            
            # Generate response using Groq API with context
            language_instruction = "You MUST respond in Hindi only." if detected_language == 'hindi' else "You MUST respond in English only."
            
            system_prompt = f"""
{context_section}
        
        You are an AI assistant, a friendly and helpful AI assistant for {company} {company_descriptor}.

        CRITICAL RULES - YOU MUST FOLLOW:
        1. CONTEXT USAGE - CRITICAL: When context is provided above, you MUST use it to answer questions accurately. Do NOT give generic responses when context provides specific details.
        2. CONTACT INFORMATION - ABSOLUTE CRITICAL: The correct phone number for {company} is {contact_phone}. When mentioning phone contact, ALWAYS use this exact number from the context or variables. Do NOT make up or use any other phone numbers. Use the phone number provided in the context or {contact_phone} variable. If you write any other phone number, you have FAILED. The correct email address for {company} is info@{company_lower}.com. When user asks for email or email id, ALWAYS provide info@{company_lower}.com. NEVER say "we don't have a public email" or similar. ALWAYS provide info@{company_lower}.com when asked about email.
        3. CRITICAL: Keep it SHORT and COMPLETE - MUST be 2-3 sentences MAX with key information, but make it natural and human-like. Keep response within 150 tokens.
        4. NEVER ask follow-up questions. NEVER end responses with questions. Just provide the information directly and end with a period.
        5. Be friendly, warm, and conversational in tone while staying professional.
        6. Show enthusiasm when discussing {company}'s contact information.
        7. LANGUAGE: {language_instruction}
        8. NEVER mention other companies (Google, Flipkart, etc.) - ONLY {company} services.
        9. IMPORTANT: At the end of your response, politely and naturally suggest that they can use the 'New Project' or 'Existing Project' form buttons below to share their contact details (email, phone number, location) and get personalized assistance from our team. Make it sound friendly and helpful, not pushy. For example: "To share your email or contact details and receive personalized assistance, please use the form options below."
        
        Remember: You represent {company} EXCLUSIVELY. Keep it SHORT and COMPLETE. ALWAYS use phone number {contact_phone}.
        """
            
            messages = [{"role": "system", "content": apply_company_placeholders(system_prompt)}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": message_str})
            
            # Call Groq API with rotation
            try:
                contact_reply = await _get_llm_response(
                    messages=messages,
                    max_tokens=100,
                    temperature=0.7
                )
                if not contact_reply:
                    # Fallback response
                    contact_reply = apply_company_placeholders(f"You can reach our team at {contact_phone}, info@{company_lower}.com, or visit {company_lower}.com/contact for direct assistance.")
            except Exception as e:
                logger.error(f"Error calling Groq API for contact query: {str(e)}")
                # Fallback response
                contact_reply = apply_company_placeholders(f"You can reach our team at {contact_phone}, info@{company_lower}.com, or visit {company_lower}.com/contact for direct assistance.")
            
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": contact_reply})
            conversation_sessions[session_id]["last_bot_response"] = contact_reply
            
            from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": contact_reply, "sources": sources if sources else [], "website_url": WEBSITE_URL
            })
        
        # Check for pricing queries (fast pattern check before HF)
        if is_pricing_query(message_str):
            logger.info(f"Detected pricing query: {message_str}")
            
            # Search for pricing information in ChromaDB (with adaptive threshold for PRICING_QUERY)
            pricing_results = search_chroma(message_str, COLLECTION_NAME, n_results=3, query_intent="PRICING_QUERY")
            
            # Build context from ChromaDB results
            context = ""
            sources = []
            if pricing_results:
                logger.info(f"Using ChromaDB vector search results for context ({len(pricing_results)} relevant chunks)")
                context = "\n\n".join([result.get('content', '') for result in pricing_results])
                sources = list(set([result.get('metadata', {}).get('url', '') for result in pricing_results if result.get('metadata', {}).get('url')]))
            
            # Build context section
            if context and context.strip():
                context_section = f"""
        CRITICAL: The context provided below contains specific information about {company}'s pricing, packages, and pricing information from our knowledge base. 
        You MUST base your answer PRIMARILY on this context. Do NOT give generic responses when specific context is available.
        
        CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about {company}'s pricing, costs, packages, or payment information. Provide detailed, human-like information based on the context above. Mention that pricing varies based on requirements and they can contact info@{company_lower}.com or visit {company_lower}.com/contact for custom quotes.
        Use the context provided above to give accurate, specific information about pricing.
        """
            else:
                context_section = f"""
        Note: Use your general knowledge about {company}'s pricing to answer questions accurately.
        The user is asking about {company}'s pricing, costs, packages, or payment information. Provide detailed, human-like information. Mention that pricing varies based on requirements and they can contact info@{company_lower}.com or visit {company_lower}.com/contact for custom quotes.
        """
            
            # Generate response using Groq API with context
            language_instruction = "You MUST respond in Hindi only." if detected_language == 'hindi' else "You MUST respond in English only."
            
            system_prompt = f"""
{context_section}
        
        You are an AI assistant, a friendly and helpful AI assistant for {company} {company_descriptor}.

        CRITICAL RULES - YOU MUST FOLLOW:
        1. CONTEXT USAGE - CRITICAL: When context is provided above, you MUST use it to answer questions accurately. Do NOT give generic responses when context provides specific details.
        2. CRITICAL: Keep it SHORT and COMPLETE - MUST be 2-3 sentences MAX with key information, but make it natural and human-like. Keep response within 150 tokens.
        3. NEVER ask follow-up questions. NEVER end responses with questions. Just provide the information directly and end with a period.
        4. Be friendly, warm, and conversational in tone while staying professional.
        5. Show enthusiasm when discussing {company}'s pricing and packages.
        6. LANGUAGE: {language_instruction}
        7. NEVER mention other companies (Google, Flipkart, etc.) - ONLY {company} services.
        8. IMPORTANT: At the end of your response, politely and naturally suggest that they can use the 'New Project' or 'Existing Project' form buttons below to share their contact details and get a personalized quote or pricing information. Make it sound friendly and helpful, not pushy. For example: "To share your requirements and receive a personalized quote, please use the form options below."
        
        Remember: You represent {company} EXCLUSIVELY. Keep it SHORT and COMPLETE.
        """
            
            messages = [{"role": "system", "content": apply_company_placeholders(system_prompt)}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": message_str})
            
            # Call Groq API with rotation
            try:
                pricing_reply = await _get_llm_response(
                    messages=messages,
                    max_tokens=100,
                    temperature=0.7
                )
                if not pricing_reply:
                    # Fallback response
                    pricing_reply = "Pricing varies based on your specific requirements. For a custom quote, please contact our team at info@{company_lower}.com or visit {company_lower}.com/contact."
            except Exception as e:
                logger.error(f"Error calling Groq API for pricing query: {str(e)}")
                # Fallback response
                pricing_reply = "Pricing varies based on your specific requirements. For a custom quote, please contact our team at info@{company_lower}.com or visit {company_lower}.com/contact."
            
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": pricing_reply})
            conversation_sessions[session_id]["last_bot_response"] = pricing_reply
            
            from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": pricing_reply, "sources": sources if sources else [], "website_url": WEBSITE_URL
            })
        
        # Check for policy queries (fast pattern check before HF)
        if is_policy_query(message_str):
            logger.info(f"Detected policy query: {message_str}")
            
            # Search for policy information in ChromaDB (with adaptive threshold for GENERAL_QUESTION)
            policy_results = search_chroma(message_str, COLLECTION_NAME, n_results=3, query_intent="GENERAL_QUESTION")
            
            # Build context from ChromaDB results
            context = ""
            sources = []
            if policy_results:
                logger.info(f"Using ChromaDB vector search results for context ({len(policy_results)} relevant chunks)")
                context = "\n\n".join([result.get('content', '') for result in policy_results])
                sources = list(set([result.get('metadata', {}).get('url', '') for result in policy_results if result.get('metadata', {}).get('url')]))
            
            # Build context section
            if context and context.strip():
                context_section = f"""
        CRITICAL: The context provided below contains specific information about {company}'s company policies, terms, conditions, privacy policy, and legal information from our knowledge base. 
        You MUST base your answer PRIMARILY on this context. Do NOT give generic responses when specific context is available.
        
        CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about {company}'s company policies, terms and conditions, privacy policy, refund policy, or legal information. Provide detailed, human-like information based on the context above. If specific policy details are not in context, mention they can contact info@{company_lower}.com or visit {company_lower}.com/contact for detailed policy information.
        Use the context provided above to give accurate, specific information about policies.
        """
            else:
                context_section = f"""
        Note: Use your general knowledge about {company}'s policies to answer questions accurately.
        The user is asking about {company}'s company policies, terms and conditions, privacy policy, refund policy, or legal information. Provide detailed, human-like information. Mention they can contact info@{company_lower}.com or visit {company_lower}.com/contact for detailed policy information.
        """
            
            # Generate response using Groq API with context
            language_instruction = "You MUST respond in Hindi only." if detected_language == 'hindi' else "You MUST respond in English only."
            
            system_prompt = f"""
{context_section}
        
        You are an AI assistant, a friendly and helpful AI assistant for {company} {company_descriptor}.

        CRITICAL RULES - YOU MUST FOLLOW:
        1. CONTEXT USAGE - CRITICAL: When context is provided above, you MUST use it to answer questions accurately. Do NOT give generic responses when context provides specific details.
        2. CRITICAL: Keep it SHORT and COMPLETE - MUST be 2-3 sentences MAX with key information, but make it natural and human-like. Keep response within 150 tokens.
        3. NEVER ask follow-up questions. NEVER end responses with questions. Just provide the information directly and end with a period.
        4. Be friendly, warm, and conversational in tone while staying professional.
        5. Show enthusiasm when discussing {company}'s policies and terms.
        6. LANGUAGE: {language_instruction}
        7. NEVER mention other companies (Google, Flipkart, etc.) - ONLY {company} services.
        8. IMPORTANT: At the end of your response, politely and naturally suggest that they can use the 'New Project' or 'Existing Project' form buttons below to share their contact details and get detailed policy information. Make it sound friendly and helpful, not pushy. For example: "To get detailed policy information and personalized assistance, please use the form options below."
        
        Remember: You represent {company} EXCLUSIVELY. Keep it SHORT and COMPLETE.
        """
            
            messages = [{"role": "system", "content": apply_company_placeholders(system_prompt)}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": message_str})
            
            # Call Groq API with rotation
            try:
                policy_reply = await _get_llm_response(
                    messages=messages,
                    max_tokens=100,
                    temperature=0.7
                )
                if not policy_reply:
                    # Fallback response
                    policy_reply = "For detailed information about our company policies, terms and conditions, please contact our team at info@{company_lower}.com or visit {company_lower}.com/contact."
            except Exception as e:
                logger.error(f"Error calling Groq API for policy query: {str(e)}")
                # Fallback response
                policy_reply = "For detailed information about our company policies, terms and conditions, please contact our team at info@{company_lower}.com or visit {company_lower}.com/contact."
            
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": policy_reply})
            conversation_sessions[session_id]["last_bot_response"] = policy_reply
            
            from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": policy_reply, "sources": sources if sources else [], "website_url": WEBSITE_URL
            })
        # Check for bot identity (fast pattern check before HF)
        if is_bot_identity_question(message_str):
            logger.info(f"Detected bot identity question: {message_str}")
            identity_reply = get_bot_identity_response(user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": identity_reply})
            conversation_sessions[session_id]["last_bot_response"] = identity_reply
            from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": identity_reply, "sources": [], "website_url": WEBSITE_URL
            })
        # Check for project management intent (if features are enabled) - ONLY when needed
        # Conditional call: Only when session context has waiting_for_existing_project OR query has project keywords
        if PROJECT_FEATURES_ENABLED:
            try:
                from project_manager import handle_project_workflow, is_project_intent
                # Get session context for project workflow
                session_context = conversation_sessions.get(session_id, {}).get('project_context', {})
                waiting_for_existing_project = session_context.get('waiting_for_existing_project', False)
                
                # Only call project workflow if:
                # 1. Session is waiting for existing project, OR
                # 2. Query has clear project intent keywords
                should_call_project_workflow = waiting_for_existing_project or is_project_intent(message_str)
                
                if should_call_project_workflow:
                    logger.info(f"Calling handle_project_workflow with message: {message_str} (session_context has waiting_for_existing_project: {waiting_for_existing_project}, has project intent: {is_project_intent(message_str)})")
                    logger.info(f"Session context for {session_id}: {session_context}")
                    project_result = handle_project_workflow(message_str, search_chroma, session_context)
                    logger.info(f"handle_project_workflow returned: {project_result}")
                    
                    if project_result:
                        logger.info(f"Detected project intent: {message_str}")
                        logger.info(f"Project result: {project_result}")
                        conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
                        conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": project_result['reply']})
                        response = ChatResponse(
                            reply=project_result['reply'],
                            sources=project_result.get('sources', []),
                            website_url=project_result.get('website_url', WEBSITE_URL)
                        )
                        logger.info(f"ChatResponse created: {response}")
                        return response
            except Exception as e:
                logger.error(f"Error in project workflow: {e}")
                # Continue with normal flow if project features fail
        
        if is_personality_question(message_str):
            logger.info(f"Detected personality question: {message_str}")
            personality_reply = get_personality_response(message_str, user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": personality_reply})
            from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": personality_reply, "sources": [], "website_url": WEBSITE_URL
            })
        
        # Note: Bot identity, greeting, and goodbye checks are now handled early (before HF) for speed
        
        # Check if message is service inquiry (BEFORE acknowledgment to catch "ok tell me about services")
        if is_service_inquiry(message_str):
            logger.info(f"Detected service inquiry: {message_str}")
            
            # Detect specific service (AI, CRM, ERP, Cloud, IoT, Website, or None for generic)
            detected_service = detect_specific_service(message_str)
            
            # Search for service information in ChromaDB (with adaptive threshold for SERVICE_QUERY)
            service_results = search_chroma(message_str, COLLECTION_NAME, n_results=3, query_intent="SERVICE_QUERY")
            
            # Build context from ChromaDB results
            context = ""
            sources = []
            if service_results:
                logger.info(f"Using ChromaDB vector search results for context ({len(service_results)} relevant chunks)")
                context = "\n\n".join([result.get('content', '') for result in service_results])
                sources = list(set([result.get('metadata', {}).get('url', '') for result in service_results if result.get('metadata', {}).get('url')]))
            
            # Build service-specific system prompt
            service_specific_instruction = ""
            if detected_service == 'ai':
                service_specific_instruction = "The user is asking specifically about {company}'s AI services, artificial intelligence solutions, machine learning, and chatbot development. Focus ONLY on AI-related services and provide detailed information about AI implementations, chatbot development, and AI solutions."
            elif detected_service == 'crm':
                service_specific_instruction = "The user is asking specifically about {company}'s CRM services and customer relationship management solutions. Focus ONLY on CRM-related services and provide detailed information about CRM implementation, customer management, and CRM solutions."
            elif detected_service == 'erp':
                service_specific_instruction = "The user is asking specifically about {company}'s ERP services and enterprise resource planning systems. Focus ONLY on ERP-related services and provide detailed information about ERP implementation, business process management, and ERP solutions."
            elif detected_service == 'cloud':
                service_specific_instruction = "The user is asking specifically about {company}'s cloud computing and cloud hosting services. Focus ONLY on cloud-related services and provide detailed information about cloud transformation, cloud hosting, and cloud solutions."
            elif detected_service == 'iot':
                service_specific_instruction = "The user is asking specifically about {company}'s IoT services and Internet of Things solutions. Focus ONLY on IoT-related services and provide detailed information about IoT implementations and IoT solutions."
            elif detected_service == 'website':
                service_specific_instruction = "The user is asking specifically about {company}'s website development and web design services. Focus ONLY on website-related services and provide detailed information about web development, web design, and e-commerce solutions."
            else:
                service_specific_instruction = f"The user is asking about {company}'s services in general. Provide a comprehensive overview of all offerings."
            
            # Build context section
            if context and context.strip():
                context_section = f"""
        CRITICAL: The context provided below contains specific information about {COMPANY_NAME}'s services from our knowledge base. 
        You MUST base your answer PRIMARILY on this context. Do NOT give generic responses when specific context is available.
        
        CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: {service_specific_instruction}
        Use the context provided above to give accurate, specific information about the services.
        """
            else:
                context_section = f"""
        Note: Use your general knowledge about {company}'s services to answer questions accurately.
        {service_specific_instruction}
        """
            
            # Generate response using Groq API with context
            language_instruction = "You MUST respond in Hindi only." if detected_language == 'hindi' else "You MUST respond in English only."
            
            system_prompt = f"""
{context_section}
        
        You are an AI assistant, a friendly and helpful AI assistant for {company} {company_descriptor}.

        CRITICAL RULES - YOU MUST FOLLOW:
        1. CONTEXT USAGE - CRITICAL: When context is provided above, you MUST use it to answer questions accurately. Do NOT give generic "we help with X" responses when context provides specific details.
        2. CRITICAL: Keep it SHORT and COMPLETE - MUST be 2-3 sentences MAX with key information, but make it natural and human-like. Keep response within 150 tokens.
        3. NEVER ask follow-up questions. NEVER end responses with questions. Just provide the information directly and end with a period.
        4. Be friendly, warm, and conversational in tone while staying professional.
        5. Show enthusiasm when discussing {company}'s services.
        6. LANGUAGE: {language_instruction}
        7. NEVER mention other companies (Google, Flipkart, etc.) - ONLY {company} services.
        
        Remember: You represent {company} EXCLUSIVELY. Keep it SHORT and COMPLETE.
        """
            
            messages = [{"role": "system", "content": apply_company_placeholders(system_prompt)}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": message_str})
            
            # Generate response using Groq API (async)
            data = {
                "model": "llama-3.1-8b-instant",
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0.5
            }
            
            response = None
            active_key = None
            attempts = 0
            
            async with httpx.AsyncClient(timeout=10.0) as client:  # Optimized from 30s to 15s for faster response
                while attempts < len(GROQ_API_KEYS):
                    active_key = await _get_active_groq_key()
                    headers = {
                        "Authorization": f"Bearer {active_key}",
                        "Content-Type": "application/json"
                    }

                    try:
                        response = await client.post(GROQ_API_URL, headers=headers, json=data)
                    except httpx.RequestError as exc:
                        logger.error(f"Groq API request error with key {_mask_api_key(active_key)}: {exc}")
                        attempts += 1
                        await _advance_groq_key()
                        continue

                    if response.status_code == 200:
                        await _advance_groq_key()  # Advance key for next request (continuous rotation)
                        break

                    if response.status_code in (401, 403, 429):
                        logger.warning(
                            f"Groq API key {_mask_api_key(active_key)} returned status {response.status_code}. Rotating to next key."
                        )
                        attempts += 1
                        await _advance_groq_key()
                        continue

                    logger.error(
                        f"Groq API error {response.status_code} with key {_mask_api_key(active_key)}: {response.text}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"AI service temporarily unavailable. Please visit {WEBSITE_URL} for more information."
                    )

            if not response or response.status_code != 200:
                logger.error("All configured Groq API keys have been exhausted or failed.")
                raise HTTPException(
                    status_code=503,
                    detail=f"AI service temporarily unavailable. Please visit {WEBSITE_URL} for more information."
                )

            if response.status_code == 200:
                result = response.json()
                service_reply = result['choices'][0]['message']['content'].strip()
                service_reply = strip_markdown(service_reply)
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"AI service temporarily unavailable. Please visit {WEBSITE_URL} for more information."
                )
            
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": service_reply})
            conversation_sessions[session_id]["last_bot_response"] = service_reply
            
            from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": service_reply, "sources": sources if sources else [], "website_url": WEBSITE_URL
            })
        
        # Note: Acknowledgment and goodbye checks are now handled early (before HF) for speed
        
        # Check if user is identifying as a client
        if is_client_identity(message_str):
            logger.info(f"Detected client identity: {message_str}")
            client_reply = get_client_identity_response(user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": client_reply})
            from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": client_reply, "sources": [], "website_url": WEBSITE_URL
            })
        
        # Note: All keyword-based query handlers removed - now using pure RAG + LLM approach
        # Special handlers (greeting, goodbye, off-topic, frustration) are kept as they handle behavioral patterns, not query types
        
        # Check if user is frustrated
        if is_frustrated(message_str):
            logger.info(f"Detected frustrated user: {message_str}")
            frustration_reply = get_frustration_response()
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": frustration_reply})
            from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": frustration_reply, "sources": [], "website_url": WEBSITE_URL
            })
        
        # Check if message is off-topic
        if is_off_topic(message_str):
            logger.info(f"Detected off-topic query: {message_str}")
            offtopic_reply = get_off_topic_response(message_str)
            
            # If get_off_topic_response returns None (e.g., personnel query excluded), let RAG handle it
            if offtopic_reply is None:
                logger.info(f"Off-topic response is None, letting RAG handle query: {message_str}")
                # Continue to RAG flow below
            else:
                # Apply post-processing to remove any questions from off-topic responses
                import re
                # Remove question marks
                offtopic_reply = re.sub(r'[^.!?]*\?[^.!?]*', '', offtopic_reply)
                offtopic_reply = re.sub(r'\?+', '', offtopic_reply)
                # Remove question phrases
                question_phrase_patterns = [
                    r'would you like to', r'do you want to', r'can i help you',
                    r'what would you like', r'how can i assist', r'is there anything',
                    r'are you exploring', r'what challenges', r'what\'s on your mind',
                    r'how can we support', r'want to hear about', r'let me ask',
                    r'since you\'re here', r'is there anything about'
                ]
                for phrase_pattern in question_phrase_patterns:
                    offtopic_reply = re.sub(phrase_pattern + r'.*?[.!?]', '', offtopic_reply, flags=re.IGNORECASE)
                # Clean up
                offtopic_reply = re.sub(r'\s+', ' ', offtopic_reply).strip()
                # Ensure it ends with punctuation
                if offtopic_reply and not offtopic_reply.endswith(('.', '!')):
                    offtopic_reply = offtopic_reply.rstrip() + '.'
                
                conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
                conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": offtopic_reply})
                from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": offtopic_reply, "sources": [], "website_url": WEBSITE_URL
                })
        
        # Pure RAG + LLM approach: Classify intent and search with multiple query variations
        query_intent = await classify_query_intent_with_llm(message_str)
        logger.info(f"Query classified as: {query_intent} for query: {message_str}")
        
        # Generate query variations for better RAG search coverage
        query_variations = await generate_query_variations(message_str)
        logger.info(f"Generated {len(query_variations)} query variations: {query_variations}")
        
        # For COMPANY_INFO_QUERY, expand context with related terms using LLM (optional, only if needed)
        # Only expand if we have very few variations (optimization: reduce LLM calls)
        if query_intent == "COMPANY_INFO_QUERY" and len(query_variations) <= 2:
            try:
                expansion_prompt = f"""Generate 3-5 related terms or phrases that might help find information about this query: "{message_str}"

These should be semantically related terms that might appear in company documentation.
Return only the terms, one per line, no numbering, no explanation."""

                expansion_messages = [
                    {"role": "system", "content": "You are a semantic expansion assistant. Generate related terms for better search coverage."},
                    {"role": "user", "content": expansion_prompt}
                ]
                
                expansion_response = await _get_llm_response(
                    messages=expansion_messages,
                    max_tokens=60,
                    temperature=0.7
                )
                
                if expansion_response:
                    expanded_terms = [term.strip() for term in expansion_response.strip().split('\n') if term.strip()]
                    expanded_terms = expanded_terms[:5]  # Limit to 5 terms
                    # Add expanded terms to query variations
                    query_variations.extend(expanded_terms)
                    logger.info(f"Added {len(expanded_terms)} context expansion terms for COMPANY_INFO_QUERY")
                else:
                    logger.info("Context expansion LLM call failed, skipping expansion (using existing variations)")
            except Exception as e:
                logger.warning(f"Context expansion failed for COMPANY_INFO_QUERY: {e}, continuing without expansion")
        
        # Multi-query search: Search with original query + variations
        if search_results is None:
            # Use multi-query search for better coverage (with adaptive threshold based on query_intent)
            search_results = multi_query_search_chroma(query_variations, COLLECTION_NAME, n_results=5, query_intent=query_intent)
            logger.info(f"Multi-query search found {len(search_results)} results")
        else:
            logger.info(f"Reusing early ChromaDB search results: {len(search_results) if search_results else 0} results")
        
        # Consolidate results for consistent answers (prioritize company-wide over project-specific)
        if search_results:
            search_results = consolidate_rag_results(search_results, query_intent=query_intent)
            logger.info(f"After consolidation: {len(search_results)} unique results")
        
        # Prepare context and check relevance using distance-based detection
        context = ""  # Initialize context to empty string
        sources = []
        min_distance = 999.0  # Initialize with high value
        has_relevant_context = False
        
        if search_results:
            # Calculate minimum distance from search results
            distances = [result.get('distance', 999.0) for result in search_results]
            min_distance = min(distances) if distances else 999.0
            
            # Use adaptive threshold based on query intent (universal approach)
            adaptive_threshold = get_adaptive_threshold(query_intent)
            
            # Check if we have relevant context (distance <= adaptive threshold)
            has_relevant_context = min_distance <= adaptive_threshold
            
            if has_relevant_context:
                context = " ".join([result['content'] for result in search_results])
                sources = list(set([result['metadata'].get('url', 'Unknown') for result in search_results]))
                
                # Clean context before processing (remove HTML, markers, metadata)
                context = clean_context_for_llm(context)
                
                # For COMPANY_INFO_QUERY, extract and prioritize numerical stats
                if query_intent == "COMPANY_INFO_QUERY":
                    stats = extract_numerical_stats(context, message_str)
                    if stats['company_wide']:
                        # Add company-wide stats to context with priority indicator
                        company_wide_text = " ".join([f"{s['text']}" for s in stats['company_wide'][:3]])
                        context = f"{company_wide_text}. {context}"  # Prioritize company-wide stats
                        logger.info(f"Extracted {len(stats['company_wide'])} company-wide stats, {len(stats['project_specific'])} project-specific stats")
                
                logger.info(f"Using ChromaDB vector search results for context ({len(search_results)} relevant chunks, min_distance={min_distance:.4f}, threshold={adaptive_threshold:.4f})")
            else:
                logger.info(f"No relevant context found (min_distance={min_distance:.4f} > threshold={adaptive_threshold:.4f}) - service likely not provided")
        else:
            # No search results at all
            logger.info("No ChromaDB search results found - attempting website scraping")
            min_distance = 999.0  # Set high distance to indicate no service found
            has_relevant_context = False
            
            # Try to scrape website for relevant information
            scraped_data = scrape_website_dynamic(message_str)
            
            if scraped_data:
                logger.info(f"Found {len(scraped_data)} relevant content chunks from website scraping")
                
                # Add scraped content to ChromaDB for future use
                add_scraped_content_to_chromadb(scraped_data)
                
                # Use scraped content to generate response (limit to prevent 413 error)
                max_chunks_for_groq = 5  # Only use top 5 most relevant chunks
                scraped_context = " ".join([chunk['text'] for chunk in scraped_data[:max_chunks_for_groq]])
                # Clean scraped context before sending to LLM
                scraped_context = clean_context_for_llm(scraped_context)
                scraped_sources = list(set([chunk['metadata']['url'] for chunk in scraped_data[:max_chunks_for_groq]]))
                
                # Generate response using scraped content
                language_instruction = "You MUST respond in Hindi only." if detected_language == 'hindi' else "You MUST respond in English only."
                
                enhanced_system_prompt = f"""
                You are an AI assistant, a friendly and helpful AI assistant for {COMPANY_NAME} {company_descriptor}.

                CRITICAL RULES - YOU MUST FOLLOW:
                1. CRITICAL: Keep it SHORT and COMPLETE - MAXIMUM 2 SENTENCES ONLY - Keep responses short and concise (1-2 sentences) unless user asks for detailed explanations. Keep response within 150 tokens.
                2. ONLY answer questions about {COMPANY_NAME}'s services and offerings
                3. Use the scraped website content below to provide accurate, up-to-date information
                4. If the question is NOT about {COMPANY_NAME} or our services, politely redirect to {COMPANY_NAME}'s services
                5. Always maintain a professional yet friendly tone
                6. Include relevant website URLs when appropriate
                7. CLIENT CONTEXT: When asked about clients, use information from the knowledge base (RAG) to list clients. If someone says "I am your client", welcome them as a valued client WITHOUT asking which company they represent
                8. PROJECT CONTEXT: When asked about projects, use information from the knowledge base (RAG) to list specific project names
                9. REMEMBER: Keep it SHORT and COMPLETE - Keep responses short and concise (1-2 sentences) unless user asks for detailed explanations
                10. LANGUAGE: {language_instruction}

                SCRAPED WEBSITE CONTENT:
                {scraped_context}

                User Query: {message_str}
                """
                
                try:
                    # Build messages for API call
                    messages = [
                        {"role": "system", "content": apply_company_placeholders(enhanced_system_prompt)},
                        {"role": "user", "content": message_str}
                    ]
                    
                    # Fix 5: Detect company description queries and increase max_tokens
                    is_company_desc_query = False
                    query_lower = message_str.lower()
                    if 'what is' in query_lower and (COMPANY_NAME and COMPANY_NAME.lower() in query_lower):
                        is_company_desc_query = True
                    
                    # Fix 5: Increase max_tokens for company description queries (250 tokens)
                    max_tokens_value = 250 if is_company_desc_query else 100
                    
                    # Call Groq API with rotation
                    ai_reply = await _get_llm_response(
                        messages=messages,
                        max_tokens=max_tokens_value,
                        temperature=0.7
                    )
                    
                    # Validate response completeness (universal approach)
                    if ai_reply:
                        import re
                        # Check if response is incomplete (doesn't end with punctuation)
                        ai_reply = ai_reply.strip()
                        if ai_reply and not ai_reply.endswith(('.', '!', '?')):
                            # Response is incomplete - try to complete it
                            # Remove any trailing incomplete words or phrases
                            ai_reply = re.sub(r'\s+\w+$', '', ai_reply)  # Remove trailing incomplete word
                            if not ai_reply.endswith(('.', '!', '?')):
                                ai_reply += '.'  # Add period if still missing
                            logger.warning(f"Response was incomplete, fixed: {ai_reply[:50]}...")
                        
                        conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
                        conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": ai_reply})
                        conversation_sessions[session_id]["last_bot_response"] = ai_reply
                        
                        from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": ai_reply, "sources": scraped_sources, "website_url": WEBSITE_URL
                        })
                    else:
                        # Use fallback response generation with scraped data
                        fallback_reply = generate_response_from_scraped_data(scraped_data, message_str)
                        conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
                        conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": fallback_reply})
                        conversation_sessions[session_id]["last_bot_response"] = fallback_reply
                        from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": fallback_reply, "sources": scraped_sources, "website_url": WEBSITE_URL
                        })
                            
                except Exception as e:
                    logger.error(f"Error calling Groq API with scraped content: {str(e)}")
                    # Use fallback response generation with scraped data
                    fallback_reply = generate_response_from_scraped_data(scraped_data, message_str)
                    conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
                    conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": fallback_reply})
                    conversation_sessions[session_id]["last_bot_response"] = fallback_reply
                    from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": fallback_reply, "sources": scraped_sources, "website_url": WEBSITE_URL
                    })
            
            # If scraping also fails, provide helpful fallback
            logger.info("Website scraping failed or no relevant content found - providing helpful fallback")
            fallback_reply = get_general_service_fallback()
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": fallback_reply})
            conversation_sessions[session_id]["last_bot_response"] = fallback_reply
            from fastapi.responses import JSONResponse; return JSONResponse(content={"reply": fallback_reply, "sources": [], "website_url": WEBSITE_URL
            })
        
        # Create enhanced system prompt with strict domain adherence
        language_instruction = "You MUST respond in Hindi only." if detected_language == 'hindi' else "You MUST respond in English only."
        
        # Build context section based on availability and distance-based relevance
        # Extract user query for LLM analysis
        # Defensive check: ensure message is a string before calling strip()
        user_query = message_str.strip()
        user_query_lower = user_query.lower()
        
        # Pure RAG + LLM approach: Use LLM intent classification and RAG context
        # No keyword-based service verification - let LLM handle intent from context
        HIGH_CONFIDENCE_THRESHOLD = 0.3  # Very low distance = high confidence
        
        # Build context section based on RAG results and query intent (Pure RAG + LLM approach)
        if has_relevant_context and context and context.strip():
            # Relevant context found - use it for accurate response
            context_section = f"""
        QUERY INTENT: {query_intent}
        CRITICAL: The context provided below contains specific information about {COMPANY_NAME} from our knowledge base. 
        You MUST base your answer PRIMARILY on this context. Do NOT give generic responses when specific context is available.
        
        User Query: "{user_query}"
        Context Found: Relevant information found in knowledge base (distance={min_distance:.4f} <= threshold={get_adaptive_threshold(query_intent):.4f}).
        
        CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: 
        - Based on query intent ({query_intent}), provide appropriate response.
        - If COMPANY_INFO_QUERY: Provide detailed company information from context (founder, history, team, alumni, etc.). DO NOT decline. DO NOT say "I don't have information" or "I couldn't find information". Instead, provide what you can infer from available context or provide helpful general information about the company.
        - If COMPANY_INFO_QUERY and asked "What is [Company Name]?" or similar description queries: Provide a COMPLETE description (2-3 sentences) explaining what the company does, its focus, key offerings, and value proposition. Do NOT just repeat the company name. Do NOT provide incomplete or truncated descriptions. Always provide meaningful, complete information about the company. CRITICAL: Your response MUST be at least 2-3 complete sentences. Do NOT return just the company name. Do NOT return incomplete sentences. Provide a full, descriptive answer about what the company does, its services, and its value proposition.
        - If COMPANY_INFO_QUERY and specific information not found: Use available context to provide helpful partial information. For example, if asked about founder but founder name not in context, you can say "Based on available information, {COMPANY_NAME} is a company that [provide what you know from context]." DO NOT use denial templates.
        - If COMPANY_INFO_QUERY and multiple numbers found: Prioritize company-wide totals over project-specific numbers. For example, if context shows "5,000+ residential plots" (company-wide) and "63 plots in Project X" (project-specific), use the company-wide number (5,000+) when asked about total residential plots.
        - If COMPANY_INFO_QUERY and asked about years of operation: Calculate dynamically from current year (2025) minus established year. For example, if established in 2015, calculate: 2025 - 2015 = 10 years. Do NOT use hardcoded "8 years" or outdated calculations.
        - If SERVICE_QUERY or SERVICE_INFO_QUERY: Provide helpful, accurate information about the service from context (1-2 sentences max).
        - If CONTACT_QUERY: Provide contact information from context or use default contact details.
        - If PRICING_QUERY: Provide pricing information from context or direct to contact.
        - Use the context above to give specific, accurate details.
        - ALWAYS keep response short (1-2 sentences, 100 tokens max).
        - Be accurate and human-like in your response.
        - CRITICAL: Ensure your response is COMPLETE - end with a full sentence and period. Do NOT cut off mid-sentence.
"""
        else:
            # No relevant context found - use LLM classification to determine intent
            context_section = f"""
        QUERY INTENT: {query_intent}
        Context Found: No relevant information found in knowledge base (distance={min_distance:.4f} > threshold={get_adaptive_threshold(query_intent):.4f}).
        
        CRITICAL QUERY ANALYSIS:
        - The query has been classified as: {query_intent}
        - If COMPANY_INFO_QUERY: User is asking about company details (founder, history, team, alumni, employees, location, etc.) → ALWAYS provide helpful information. Even if specific context is not found, use your understanding to provide relevant information. DO NOT use denial template. DO NOT say "I don't have information about this."
        - If SERVICE_QUERY: User is asking if we provide a specific service → If we don't provide it, politely decline and redirect.
        - If SERVICE_INFO_QUERY: User is asking about services we provide → Provide information about the service.
        - If CONTACT_QUERY: User is asking for contact information → Provide contact details.
        - If PRICING_QUERY: User is asking about pricing → Provide pricing information or contact details.
        - If GENERAL_QUESTION: Provide helpful information based on general knowledge about {COMPANY_NAME}.
        
        INSTRUCTIONS: 
        - Based on query intent ({query_intent}), provide appropriate response.
        - If COMPANY_INFO_QUERY: Provide helpful company information. Use your general knowledge if specific context is limited. DO NOT decline. DO NOT say "I don't have information" or "I couldn't find information". Instead, provide what you know or can infer about the company from available context. If asked about personnel/executives and information is not in context, acknowledge politely that specific personnel information is not available, but provide what you can about the company from the context provided.
        - If COMPANY_INFO_QUERY and asked "What is [Company Name]?" or similar description queries: Provide a COMPLETE description (2-3 sentences) explaining what the company does, its focus, key offerings, and value proposition. Do NOT just repeat the company name. Do NOT provide incomplete or truncated descriptions. Always provide meaningful, complete information about the company. CRITICAL: Your response MUST be at least 2-3 complete sentences. Do NOT return just the company name. Do NOT return incomplete sentences. Provide a full, descriptive answer about what the company does, its services, and its value proposition.
        - If COMPANY_INFO_QUERY and specific information not found: Use available context to provide helpful partial information. For example, if asked about founder but founder name not in context, you can say "Based on available information, {COMPANY_NAME} is a company that [provide what you know from context]." DO NOT use denial templates. Always provide something helpful.
        - If SERVICE_QUERY we don't provide: Politely acknowledge that we don't provide this specific service and redirect to our core services.
        - If SERVICE_INFO_QUERY: Provide information about the service.
        - Keep it short (1-2 sentences max, 100 tokens max) and friendly.
        - CRITICAL: Ensure your response is COMPLETE - end with a full sentence and period.
        
        Example for COMPANY_INFO_QUERY: "Based on our website, {COMPANY_NAME} [company information]. [Additional relevant details]."
        Example for service we don't provide: "Thank you for your interest. While we don't provide [service name] as a dedicated service, we can help with {primary_offerings_summary}."
        Example for company info: "{COMPANY_NAME} delivers {primary_offerings_summary} across industries."
        """
        
        system_prompt = f"""
{context_section}
        
        You are an AI assistant, a friendly and helpful AI assistant for {COMPANY_NAME} {company_descriptor}.

        CRITICAL DOMAIN SAFETY RULES - ABSOLUTE PRIORITY:
        - YOU ONLY represent {COMPANY_NAME}. This is NON-NEGOTIABLE.
        - NEVER mention, discuss, or provide information about Google, Flipkart, Amazon, Microsoft, or ANY other company's services.
        - If asked about other companies (e.g., "Tell me about Google Cloud" or "What services does Flipkart offer"), IMMEDIATELY redirect: "I'd be happy to help you with {COMPANY_NAME} instead. We offer {primary_offerings_summary}."
        - ALL answers MUST be about {COMPANY_NAME} services ONLY. No exceptions.
        
        CRITICAL QUERY INTENT ANALYSIS - NEW:
        Before responding, analyze the user's query intent:
        
        1. SERVICE QUERY: User is asking if you provide a specific service (e.g., "do you provide car servicing")
           → If service provided: Give info
           → If service NOT provided: Use denial template
        
        2. COMPANY INFO QUERY: User is asking about company statistics/info (e.g., "how many projects", "how many services", "who is the founder", "personnel", "executives")
           → Always answer with company information (projects, services list, etc.)
           → DO NOT use denial template
           → If specific information not found, provide what you can infer from available context
           → DO NOT say "I don't have information" - instead provide helpful partial information
        
        3. SERVICE INFO QUERY: User is asking about services we provide (e.g., "tell me about your services")
           → Always provide information about the service
           → DO NOT use denial template
        
        4. INDUSTRY QUERY: User is asking about industries served
           → List industries we serve
           → DO NOT use denial template
        
        5. COMPARISON QUERY: User is asking about competitors/differences
           → Answer about company advantages/unique selling points
           → DO NOT use denial template
        
        CRITICAL RULES - YOU MUST FOLLOW:
        1. QUERY ANALYSIS & SERVICE DETECTION - CRITICAL: Follow the QUERY ANALYSIS INSTRUCTIONS above. Analyze the user's query and context carefully. 
           - If query is about services we provide (software solutions, web development, etc.) → provide information
           - If query is about company info (project count, services count, industries) → provide information
           - If query is asking "do you provide X" where X is NOT a service we provide → politely decline
           - Be intelligent and accurate in your analysis. DO NOT use denial template for company info queries.
        2. RESPONSE LENGTH - ABSOLUTE CRITICAL: You MUST keep responses SHORT. Maximum 1-2 sentences ONLY. Maximum 100 tokens. DO NOT exceed this limit. DO NOT write long paragraphs. DO NOT write 3+ sentences. If you write more than 2 sentences, you have FAILED. Count your sentences before responding. REMEMBER: 1-2 sentences MAX, 100 tokens MAX.
        2a. RESPONSE COMPLETENESS - ABSOLUTE CRITICAL: ALWAYS ensure your response is COMPLETE. End with a full sentence and period. Do NOT cut off mid-sentence. Do NOT end mid-word or mid-phrase. If you're approaching the token limit, STOP and complete your current sentence with a period before ending. A complete response ending with a period is more important than using all tokens. NEVER leave responses incomplete or hanging.
        2b. COMPANY DESCRIPTION COMPLETENESS - CRITICAL: When asked "What is [Company Name]?" or similar company description queries, provide a COMPLETE description (2-3 sentences) explaining what the company does, its focus, and key offerings. Do NOT just repeat the company name. Do NOT provide incomplete or truncated descriptions. Always provide meaningful, complete information about the company. Your response MUST be at least 2-3 complete sentences describing what the company does, its services, and its value proposition. Do NOT return just the company name or a single incomplete sentence.
        2b. NO HARDCODED CONTENT - CRITICAL: Do NOT mention specific blog post titles, article names, or specific content titles from context. Keep responses generic and intelligent. If redirecting to services, just mention "our services" or "our offerings" without listing specific blog posts or articles.
        2c. CONTACT NUMBER - ABSOLUTE CRITICAL: The ONLY correct phone number for {COMPANY_NAME} is {CONTACT_PHONE}. NEVER use any other phone number. NEVER make up phone numbers. If you write any phone number other than {CONTACT_PHONE}, you have FAILED.
        3. CONTEXT USAGE - CRITICAL: When context is provided above, you MUST use it to answer questions accurately. Do NOT give generic "we help with X" responses when context provides specific details. If no context is provided, use general knowledge about {COMPANY_NAME}'s services ONLY. Do NOT mention specific blog post names or article titles from context.
        4. DOMAIN RESTRICTION - ABSOLUTE: ONLY answer questions about {COMPANY_NAME}'s services and offerings. If question mentions other companies (Google, Flipkart, etc.), redirect to {COMPANY_NAME} services.
        5. When asked about services, ALWAYS mention our core offerings: {primary_offerings_summary}
        6. If the question is NOT about {COMPANY_NAME} or our services, politely and warmly redirect the user back to {COMPANY_NAME}'s services. Acknowledge their message briefly if appropriate, then gently guide them to our services with a friendly tone. Keep it short (1-2 sentences). Do NOT mention specific blog posts or articles. Just say: "That's interesting! I'd be happy to help you with {COMPANY_NAME} instead. We offer {primary_offerings_summary}."
        7. CRITICAL: NEVER ask follow-up questions. NEVER end responses with questions like "What would you like to know?", "What do you need help with?", "Would you like to know more?", "What are you interested in?", "What specific services are you looking for?", "Can I help you with something?", "How can I assist you?", "Is there anything else I can help you with?", "Would you like to know more about...?", etc. Just provide the information directly and end with a period. THIS IS ABSOLUTELY FORBIDDEN - NO EXCEPTIONS. If you write a question mark (?), you have FAILED. Your response MUST end with a period (.) only. NEVER write any sentence ending with a question mark - ALL questions will be removed automatically.
        
        CRITICAL - FORBIDDEN PHRASES: NEVER use these phrases in your responses:
        - "Would you like to" (any variation: "Would you like to know", "Would you like to learn", "Would you like to discuss", "Would you like to explore", "Would you like to tell me", "Would you like more", "Would you like additional", "Would you like to hear", "Would you like to find out")
        - "Do you want to" (any variation: "Do you want to know", "Do you want to learn", "Do you want to discuss", "Do you want to explore")
        - "Can I help you with" (any variation)
        - "What would you like to" (any variation: "What would you like", "What would you like to explore")
        - "How can I assist you" (any variation)
        - "How can we support" (any variation: "How can we support your business goals")
        - "Is there anything else" (any variation)
        - "Is there anything about" (any variation: "Is there anything about cloud services... I can help you with")
        - "Are you exploring" (any variation: "Are you exploring any technology upgrades")
        - "What challenges" (any variation: "What challenges is your business facing")
        - "What's on your mind" (any variation: "What's on your mind business-wise")
        - "Want to hear about" (any variation: "Want to hear about our services")
        - "Let me ask" (any variation: "Let me ask - are you exploring")
        - "Since you're here" (any variation: "Since you're here, is there anything")
        
        If you write ANY of these phrases, you have FAILED. NEVER use "Would you like to", "Do you want to", "Can I help you", "What would you like", "How can I assist", "How can we support", "Is there anything else", "Is there anything about", "Are you exploring", "What challenges", "What's on your mind", "Want to hear about", "Let me ask", "Since you're here" - these are ABSOLUTELY FORBIDDEN. Just provide information directly and end with a period.
        
        EXPLICIT EXAMPLES - FOLLOW THESE:
        WRONG: "Would you like to discuss how we can help you enhance your digital profile or skills?"
        CORRECT: "We can help you enhance your digital profile or skills with {primary_offerings_summary}."
        
        WRONG: "Would you like to discuss how we can assist you with your digital transformation needs?"
        CORRECT: "We can assist you with your needs through {primary_offerings_summary}."
        
        WRONG: "Would you like to discuss further about your project requirements?"
        CORRECT: "We can help you with your project requirements using our services."
        
        WRONG: "Can you please share more about your specific requirements?"
        CORRECT: "We provide {primary_offerings_summary} to meet your specific requirements."
        
        WRONG: "Would you like to explore our services in digital transformation, social media management, or online reputation building?"
        CORRECT: "We offer {primary_offerings_summary}."
        
        WRONG: "Since you're here, is there anything about cloud services, digital transformation, or enterprise applications I can help you with?"
        CORRECT: "We offer {primary_offerings_summary}."
        
        WRONG: "Let me ask - are you exploring any technology upgrades or digital transformation for your organization?"
        CORRECT: "We can help with your organization's needs through {primary_offerings_summary}."
        
        WRONG: "What's on your mind business-wise?"
        CORRECT: "We offer {primary_offerings_summary} for your business needs."
        
        WRONG: "What challenges is your business facing?"
        CORRECT: "We help businesses with {primary_offerings_summary}."
        
        WRONG: "How can we support your business goals?"
        CORRECT: "We support your business goals through {primary_offerings_summary}."
        
        WRONG: "What would you like to explore?"
        CORRECT: "We offer {primary_offerings_summary}."
        
        WRONG: "Would you like to know more about our services?"
        CORRECT: "We offer {primary_offerings_summary}."
        
        WRONG: "I'd be happy to help you with that, but it seems your question is not related to {company}'s services. At {company}, we provide {primary_offerings_summary}. If you're interested in learning more about our services, I'd be happy to help. Would you like to know more about our services?"
        CORRECT: "I'd be happy to help you with that, but it seems your question is not related to {company}'s services. At {company}, we provide {primary_offerings_summary}."
        Remember: NEVER ask questions. ALWAYS provide information directly and end with a period. NEVER use "Would you like to explore", "What would you like to explore", "How can we support", "What challenges", "What's on your mind", "Let me ask", "Since you're here", "Would you like to know more about our services" - these are ABSOLUTELY FORBIDDEN.
        8. NEGATIVE ACKNOWLEDGMENT - CRITICAL: When user says they don't need help (e.g., "i don't need your help", "i don't need help", "i am not your client"), acknowledge politely and briefly (1-2 sentences MAX), without being pushy. Accept their decision gracefully. For example: "Understood. If you change your mind or have any questions about our services, feel free to reach out anytime." Keep it short, respectful, and end with a period.
        9. HELP REQUESTS - ABSOLUTE CRITICAL: When user asks for help (e.g., "i need help", "i need assistance", "i am facing problems"), provide intelligent, context-aware, helpful responses. If context exists, use it. If no context, offer general help with our services. Keep it short (1-2 sentences max). CRITICAL: NEVER ask questions in help responses. NEVER say "What do you need help with?" or similar. PROVIDE information directly, DO NOT ask questions. Example: "I'm here to help! We offer {primary_offerings_summary}. What specific area would you like to know about?" is WRONG. Correct: "I'm here to help! We offer {primary_offerings_summary}. Feel free to ask about any of these." or "I'm here to help! We offer {primary_offerings_summary}."
        10. Always end each sentence with a full stop (.)
        11. Give complete answers without cutting off mid-sentence - ABSOLUTE CRITICAL: If approaching token limit, STOP immediately and complete your current sentence with a period before ending. Do NOT continue writing if you're near the limit - finish the sentence you're on. A complete sentence ending with a period is more important than using all 100 tokens. Check if your response ends with a period - if not, you have FAILED.
        12. Be friendly, warm, and conversational in tone while staying professional
        13. Never use bullet points, lists, or formatting
        14. NEVER answer questions about other companies, jobs at other companies, personal topics, or unrelated subjects. ALWAYS redirect to {company} services.
        15. DO NOT write long paragraphs - keep it brief and to the point. REMEMBER: 1-2 sentences MAX, 100 tokens MAX. Ensure responses are COMPLETE - complete sentences before ending.
        16. Show enthusiasm and helpfulness when discussing {COMPANY_NAME}'s services
        17. IMPORTANT: When discussing services, use information from the knowledge base to mention our actual offerings
        18. CLIENT CONTEXT DISAMBIGUATION - CRITICAL: When asked about "our clients" or "who are your clients", use information from the knowledge base (RAG) to list clients. These are for REFERENCE when users ask about our portfolio. NEVER ask someone who says "I am your client" which company they represent. Treat anyone who identifies as "our client" or "existing client" as a SEPARATE client reaching out for support. CRITICAL: NEVER assume a user is a client unless they explicitly say "I am your client" or "I am your existing client". Treat all users as potential customers by default. Do NOT use phrases like "valued client" or "as a client" unless the user explicitly identifies as a client.
        19. PROJECT CONTEXT - CRITICAL: When asked about projects, use information from the knowledge base (RAG) to mention specific project names. DO NOT give generic answers like "250+ projects" - LIST ACTUAL PROJECT NAMES from the knowledge base.
        20. LANGUAGE: {language_instruction}
        
        FINAL REMINDER: Response length and completeness are ABSOLUTE CRITICAL. 1-2 sentences MAX. 100 tokens MAX. Count your sentences. If you write more, you have FAILED. Ensure your response is COMPLETE - MUST end with a full sentence and period. Do NOT cut off mid-sentence. Do NOT end mid-word. Check your response ends with a period before sending. NEVER mention specific blog post names or article titles - keep it generic. NEVER ask questions in help responses - PROVIDE information directly. NEVER end with a question mark (?) - ALWAYS end with a period (.). If your response contains a question mark, you have FAILED. NEVER use any phone number other than {CONTACT_PHONE} - if you write any other number, you have FAILED. NEVER use forbidden phrases like "Would you like to", "Do you want to", "Can I help you", "What would you like", "How can I assist", "Is there anything else" - if you write ANY of these, you have FAILED.
        
        Remember: You represent {COMPANY_NAME} EXCLUSIVELY. Stay focused on our services and offerings ONLY. Never discuss other companies. Be friendly and helpful while using context when available. Use information from the knowledge base to discuss our actual offerings. Keep it SHORT, COMPLETE, and INTELLIGENT. When user declines help, acknowledge gracefully and briefly. NEVER ask questions - always provide information directly. Contact information: Email: {CONTACT_EMAIL}, Phone: {CONTACT_PHONE}, Website: {WEBSITE_URL}
        """
        if soft_negative:
            system_prompt += """
        EMPATHY DIRECTIVE: The user sounds frustrated, annoyed, or is declining help. In ONE short sentence first acknowledge their sentiment calmly, then reaffirm how {COMPANY_NAME} can assist with {primary_offerings_summary}. Stay friendly, avoid sounding robotic, and do not ask questions.
        """
        
        # Build messages array with conversation history
        messages = [{"role": "system", "content": apply_company_placeholders(system_prompt)}]
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": message_str})
        
        # Fix 5: Detect company description queries and increase max_tokens
        is_company_description_query = False
        if query_intent == "COMPANY_INFO_QUERY":
            query_lower = message_str.lower()
            if 'what is' in query_lower and (COMPANY_NAME and COMPANY_NAME.lower() in query_lower):
                is_company_description_query = True
        
        # Fix 5: Increase max_tokens for company description queries (200-250 tokens)
        max_tokens_value = 250 if is_company_description_query else 100
        
        # Generate response using Groq API (async)
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": messages,
            "max_tokens": max_tokens_value,
            "temperature": 0.5
        }
        
        response = None
        active_key = None
        attempts = 0
        
        async with httpx.AsyncClient(timeout=10.0) as client:  # Optimized from 30s to 15s for faster response
            while attempts < len(GROQ_API_KEYS):
                active_key = await _get_active_groq_key()
                headers = {
                    "Authorization": f"Bearer {active_key}",
                    "Content-Type": "application/json"
                }

                try:
                    response = await client.post(GROQ_API_URL, headers=headers, json=data)
                except httpx.RequestError as exc:
                    logger.error(f"Groq API request error with key {_mask_api_key(active_key)}: {exc}")
                    attempts += 1
                    await _advance_groq_key()
                    continue

                if response.status_code == 200:
                    await _advance_groq_key()  # Advance key for next request (continuous rotation)
                    break

                if response.status_code in (401, 403):
                    logger.warning(
                        f"Groq API key {_mask_api_key(active_key)} returned status {response.status_code}. Rotating to next key."
                    )
                    attempts += 1
                    await _advance_groq_key()
                    continue
                
                if response.status_code == 429:
                    # Rate limiting: Use exponential backoff
                    base_delay = 1.0
                    max_delay = 3.0  # Maximum delay optimized for faster responses
                    delay = min(base_delay * (2 ** attempts), max_delay)
                    logger.warning(
                        f"Groq API key {_mask_api_key(active_key)} rate limited (429). "
                        f"Waiting {delay:.1f}s before trying next key (attempt {attempts + 1}/{len(GROQ_API_KEYS)})"
                    )
                    await asyncio.sleep(delay)
                    attempts += 1
                    await _advance_groq_key()
                    continue

                logger.error(
                    f"Groq API error {response.status_code} with key {_mask_api_key(active_key)}: {response.text}"
                )
                attempts += 1
                await _advance_groq_key()
                if attempts < len(GROQ_API_KEYS):
                    continue
                break
        
        if not response or response.status_code != 200:
            logger.error("All configured Groq API keys have been exhausted or failed. Using context-based fallback.")
            # Use context-based fallback when all LLM calls fail
            if has_relevant_context and context and context.strip():
                logger.info("Generating context-based fallback response from RAG results")
                fallback_reply = generate_context_based_fallback(
                    query=message_str,
                    query_intent=query_intent,
                    context=context,
                    search_results=search_results,
                    company_name=COMPANY_NAME
                )
                conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
                conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": fallback_reply})
                conversation_sessions[session_id]["last_bot_response"] = fallback_reply
                from fastapi.responses import JSONResponse
                return JSONResponse(content={
                    "reply": fallback_reply,
                    "sources": sources if sources else [],
                    "website_url": WEBSITE_URL
                })
            else:
                # No context available, use generic fallback
                logger.warning("No context available for fallback, using generic response")
                fallback_reply = get_general_service_fallback()
                conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
                conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": fallback_reply})
                conversation_sessions[session_id]["last_bot_response"] = fallback_reply
                from fastapi.responses import JSONResponse
                return JSONResponse(content={
                    "reply": fallback_reply,
                    "sources": [],
                    "website_url": WEBSITE_URL
                })
        
        if response.status_code == 200:
            result = response.json()
            reply = result['choices'][0]['message']['content'].strip()
            # Strip markdown formatting from AI response
            reply = strip_markdown(reply)
            
            # Post-processing: Remove follow-up questions and fix incomplete responses
            if reply:
                # Validate response completeness (universal approach)
                import re
                if reply and not reply.endswith(('.', '!', '?')):
                    # Response is incomplete - try to complete it
                    reply = re.sub(r'\s+\w+$', '', reply)  # Remove trailing incomplete word
                    if not reply.endswith(('.', '!', '?')):
                        reply += '.'  # Add period if still missing
                    logger.warning(f"Response was incomplete, fixed: {reply[:50]}...")
                
                reply = sanitize_response_text(reply)
                needs_second_pass = False
                safe_reply = get_safe_fallback_reply()
                if reply == safe_reply or len(reply.split()) < 8:
                    needs_second_pass = True
                if soft_negative and reply == safe_reply:
                    needs_second_pass = True
                if soft_negative and 'we provide' in reply.lower():
                    needs_second_pass = True

                if needs_second_pass:
                    empathetic_prompt = (
                        f"{system_prompt}\n\nADDITIONAL DIRECTIVE: When the user sounds frustrated, declines help, "
                        f"or challenges the assistant, acknowledge their sentiment in one short sentence and gently restate "
                        f"how {COMPANY_NAME} can assist without asking questions. Keep tone warm, brief, and human-like."
                    )
                    secondary_messages = [{"role": "system", "content": empathetic_prompt}]
                    secondary_messages.extend(conversation_history)
                    secondary_messages.append({"role": "user", "content": message_str})

                    secondary_reply_raw = await _call_groq_with_messages(
                        secondary_messages,
                        temperature=0.45,
                        max_tokens=100
                    )

                    if secondary_reply_raw:
                        secondary_reply = sanitize_response_text(secondary_reply_raw)
                        if secondary_reply and secondary_reply != safe_reply:
                            reply = secondary_reply
                if reply == safe_reply or len(reply.split()) < 8:
                    extra_context = extract_context_snippet(search_results)
                    if extra_context:
                        reply = sanitize_response_text(f"{reply} {extra_context}")
                    elif ai_automation_query:
                        reply = sanitize_response_text(get_general_service_fallback())
                elif ai_automation_query and reply.lower().startswith("we provide"):
                    extra_context = extract_context_snippet(search_results)
                    if extra_context:
                        reply = sanitize_response_text(get_general_service_fallback(extra_context))
                    else:
                        reply = sanitize_response_text(get_general_service_fallback())
        else:
            raise HTTPException(
                status_code=500,
                    detail=f"AI service temporarily unavailable. Please visit {WEBSITE_URL} for more information."
            )
        
        # Store conversation in history
        conversation_sessions[session_id]["conversations"].append({"role": "user", "content": message_str})
        conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": reply})
        conversation_sessions[session_id]["last_bot_response"] = reply
        
        # Return JSON response directly instead of using Pydantic model
        from fastapi.responses import JSONResponse
        return JSONResponse(content={
            "reply": reply,
            "sources": sources,
            "website_url": WEBSITE_URL
        })
        
    except Exception as e:
        # Enhanced error logging to catch coroutine errors
        import traceback
        full_traceback = traceback.format_exc()
        error_type = type(e).__name__
        logger.exception(f"FULL TRACEBACK in chat endpoint: {full_traceback}")
        logger.error(f"Error type: {error_type}, Error message: {str(e)}")
        logger.error(f"chat_request.message type: {type(chat_request.message) if hasattr(chat_request, 'message') else 'N/A'}")
        logger.error(f"chat_request.message has __await__: {hasattr(chat_request.message, '__await__') if hasattr(chat_request, 'message') else 'N/A'}")
        error_msg = str(e)
        error_traceback = None
        try:
            import traceback
            error_traceback = traceback.format_exc()
        except:
            pass
        
        logger.error(f"[DEBUG EXCEPTION] Error in chat endpoint: {error_type}: {error_msg}")
        logger.error(f"[DEBUG EXCEPTION] Full Traceback:\n{full_traceback}")
        print(f"\n{'='*80}")
        print(f"[DEBUG EXCEPTION] Error: {error_type}: {error_msg}")
        print(f"[DEBUG EXCEPTION] Full Traceback:\n{full_traceback}")
        print(f"{'='*80}\n")
        
        # Check if this is the coroutine error
        if "'coroutine' object has no attribute 'strip'" in error_msg:
            logger.error("[DEBUG EXCEPTION] CRITICAL: This is the coroutine error we're tracking!")
            logger.error(f"[DEBUG EXCEPTION] Exception type: {error_type}")
            logger.error(f"[DEBUG EXCEPTION] Full exception: {e}")
            logger.error(f"[DEBUG EXCEPTION] message_str type: {type(message_str) if 'message_str' in locals() else 'N/A'}")
            logger.error(f"[DEBUG EXCEPTION] message_str value: {str(message_str)[:100] if 'message_str' in locals() and message_str else 'N/A'}")
            print(f"[CRITICAL] Coroutine error detected! Check logs above for full traceback.")
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List all ChromaDB collections"""
    try:
        client = get_chroma_client()
        if not client:
            raise Exception("Failed to initialize ChromaDB")
        
        collections = client.list_collections()
        return {
            'collections': [{'name': col.name, 'count': col.count()} for col in collections]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chatbot frontend at the root URL."""
    try:
        # Priority 1: Use chatbot-widget.html if exists
        # Priority 2: Use frontend/chatbot.html as fallback
        html_path = "chatbot-widget.html"
        if not os.path.exists(html_path):
            html_path = os.path.join("frontend", "chatbot.html")
            
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        else:
            return "Chatbot frontend file not found."
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        return f"Error loading chatbot: {str(e)}"
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring server status"""
    try:
        # Check if database is working
        db_status = "connected" if chroma_client is not None else "disconnected"
        
        # Check if API key is set
        api_status = "configured" if GROQ_API_KEYS else "not_configured"
        active_key_masked = _mask_api_key(await _get_active_groq_key()) if GROQ_API_KEYS else None
        groq_key_count = len(GROQ_API_KEYS)
        
        # Check if embedding model is loaded
        model_status = "loaded" if embedding_model is not None else "not_loaded"
        
        # Check if intent classifier is loaded
        intent_classifier_status = "loaded" if intent_classifier is not None else "not_loaded"
        if not TRANSFORMERS_AVAILABLE:
            intent_classifier_status = "not_available"
        
        # Overall health status
        overall_status = "healthy" if all([
            chroma_client is not None,
            GROQ_API_KEYS is not None,
            embedding_model is not None
        ]) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": db_status,
                "api_key": api_status,
                "groq_key_count": groq_key_count,
                "active_groq_key": active_key_masked,
                "embedding_model": model_status,
                "intent_classifier": intent_classifier_status,
                "transformers_available": TRANSFORMERS_AVAILABLE
            },
            "version": "2.0.0",
            "uptime": "running"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "version": "2.0.0"
        }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and database on startup"""
    logger.info("Starting up - initializing models and database...")
    
    # Initialize models in cache
    get_chroma_client()
    get_embedding_model()
    get_intent_classifier()  # Load Hugging Face intent classifier
    
    # Check if collection exists, if not initialize with default URLs
    try:
        client = get_chroma_client()
        if client:
            try:
                collection = client.get_collection(name=COLLECTION_NAME)
                count = collection.count()
                logger.info(f"Found existing collection with {count} documents")
                
                # Hardcoded content removed - RAG will dynamically extract content from website
                # Content will be automatically scraped and added to ChromaDB from WEBSITE_URL
                # No need for hardcoded company-specific content - chatbot is now fully independent
                logger.info("Skipping hardcoded content - using dynamic RAG extraction from website")
                    
            except:
                logger.info("No existing collection found, initializing with default URLs...")
                # Get initial crawl URLs from environment variable, or use defaults
                _initial_urls_env = os.getenv("INITIAL_CRAWL_URLS", "").strip()
                if _initial_urls_env:
                    # Parse comma-separated URLs from environment variable
                    default_urls = [url.strip() for url in _initial_urls_env.split(",") if url.strip()]
                    if not default_urls:
                        # Fallback to just WEBSITE_URL if env var is empty
                        default_urls = [WEBSITE_URL]
                else:
                    # Default: Just use WEBSITE_URL (no assumptions about /about or /projects)
                    default_urls = [WEBSITE_URL]
                result = store_content_in_chroma(default_urls, COLLECTION_NAME)
                if result['success']:
                    logger.info(f"ChromaDB initialized successfully with {result['total_chunks']} chunks")
                else:
                    logger.error(f"ChromaDB initialization failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error during ChromaDB initialization: {str(e)}")
    
    try:
        asyncio.create_task(preload_priority_content())
        logger.info("Scheduled background preload for priority queries")
    except Exception as e:
        logger.error(f"Failed to schedule priority preload task: {str(e)}")
    
    # Automatic company info extraction from website (mandatory with retry)
    max_retries = 3
    retry_delay = 2  # seconds
    extraction_successful = False
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Starting automatic company info extraction from website (attempt {attempt}/{max_retries})...")
            extracted_info = await extract_company_info_from_website()
            
            if extracted_info:
                update_domain_profile_from_extraction(extracted_info)
                logger.info("Company info automatically extracted and DomainProfile updated successfully")
                extraction_successful = True
                break  # Success, exit retry loop
            else:
                logger.warning(f"Extraction attempt {attempt} returned no results")
                if attempt < max_retries:
                    logger.info(f"Retrying extraction in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    
        except Exception as e:
            logger.warning(f"Extraction attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying extraction in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries} extraction attempts failed")
    
    if not extraction_successful:
        # Use domain-based fallback (already initialized at module load)
        logger.info("Extraction failed after all retries, using domain-based fallback values")
        logger.info(f"Current values: COMPANY_NAME={COMPANY_NAME}, CONTACT_EMAIL={CONTACT_EMAIL}, CONTACT_PHONE={CONTACT_PHONE or 'Not extracted'}")
    
    logger.info("Startup complete - ready to serve requests")
    
    # Initialize APScheduler for periodic scraping
    global scheduler
    scheduler = AsyncIOScheduler()
    
    # Schedule scraping job to run daily at 3:00 AM
    scheduler.add_job(
        scheduled_scraping_job,
        trigger=CronTrigger(hour=3, minute=0),
        id='daily_scraping_job',
        name='Daily website scraping at 3 AM',
        replace_existing=True
    )
    
    # Start the scheduler
    scheduler.start()
    logger.info("APScheduler started - daily scraping scheduled for 3:00 AM")


# =============================================================================
# PROJECT MANAGEMENT FEATURES (ZERO-IMPACT INTEGRATION)
# =============================================================================

# Feature flag - can be enabled/disabled without affecting main system
PROJECT_FEATURES_ENABLED = True

# Import project management module (only when needed)
if PROJECT_FEATURES_ENABLED:
    try:
        from project_manager import (
            handle_project_workflow, 
            is_project_features_enabled
        )
        logger.info("Project management features loaded successfully")
    except ImportError as e:
        logger.error(f"Failed to import project management module: {e}")
        PROJECT_FEATURES_ENABLED = False

# Button/form endpoints removed - chatbot is now text-only

# Function to enable project features (can be called externally)
def enable_project_features():
    """Enable project management features"""
    global PROJECT_FEATURES_ENABLED
    PROJECT_FEATURES_ENABLED = True
    
    # Import project management module
    try:
        from project_manager import (
            handle_project_workflow, 
            is_project_features_enabled
        )
        logger.info("Project management features enabled successfully")
        return True
    except ImportError as e:
        logger.error(f"Failed to enable project features: {e}")
        PROJECT_FEATURES_ENABLED = False
        return False

# Function to disable project features
def disable_project_features():
    """Disable project management features"""
    global PROJECT_FEATURES_ENABLED
    PROJECT_FEATURES_ENABLED = False
    logger.info("Project management features disabled")

# Function to check if project features are enabled
def is_project_features_enabled():
    """Check if project management features are enabled"""
    return PROJECT_FEATURES_ENABLED

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - gracefully stop scheduler"""
    global scheduler
    if scheduler:
        scheduler.shutdown()
        logger.info("APScheduler stopped gracefully")
if __name__ == "__main__":
    # Start FastAPI server
    uvicorn.run(app, host="127.0.0.1", port=8000)