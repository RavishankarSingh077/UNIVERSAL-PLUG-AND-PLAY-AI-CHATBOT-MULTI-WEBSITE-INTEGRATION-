#!/usr/bin/env python3
"""Company-agnostic project management helpers for the chatbot."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_FEATURES_ENABLED = True


def _get_company_context() -> Dict[str, str]:
    """Fetch configuration from app_chromadb without creating circular imports."""
    try:
        import app_chromadb  # type: ignore
        profile = app_chromadb.get_domain_profile()

        return {
            "company": app_chromadb.COMPANY_NAME,
            "company_lower": app_chromadb.COMPANY_NAME.lower(),
            "website_url": app_chromadb.WEBSITE_URL,
            "contact_email": app_chromadb.CONTACT_EMAIL,
            "contact_phone": app_chromadb.CONTACT_PHONE,
            "collection": getattr(app_chromadb, "COLLECTION_NAME", None),
            "profile": profile,
        }
    except Exception as exc:  # pragma: no cover - fallback
        logger.debug("Falling back to default company context: %s", exc)
        return {
            "company": "Our Company",
            "company_lower": "our company",
            "website_url": "https://example.com",
            "contact_email": "info@example.com",
            "contact_phone": "+1-000-000-0000",
            "collection": None,
            "profile": None,
        }


def _render(text: str) -> str:
    ctx = _get_company_context()
    replacements = {
        "{company}": ctx["company"],
        "{company_lower}": ctx["company_lower"],
        "{website}": ctx["website_url"],
        "{contact_email}": ctx["contact_email"],
        "{contact_phone}": ctx["contact_phone"],
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text


def _search_company_context(query: str, limit: int = 3) -> List[str]:
    ctx = _get_company_context()
    try:
        import app_chromadb  # type: ignore

        collection_name = ctx["collection"] or getattr(app_chromadb, "COLLECTION_NAME", None)
        if not collection_name:
            return []
        results = app_chromadb.search_chroma(query, collection_name, limit)
    except Exception as exc:  # pragma: no cover - rely on fallback
        logger.debug("Context search failed for '%s': %s", query, exc)
        return []

    snippets: List[str] = []
    for result in results or []:
        doc = (result.get("document") or "").strip()
        if not doc:
            continue
        sentence = re.split(r"[\.!?]\s", doc)[0].strip()
        if sentence and sentence not in snippets:
            snippets.append(sentence[:240])
        if len(snippets) >= limit:
            break
    return snippets


def is_project_intent(message: str) -> bool:
    message_lower = message.lower().strip()
    intent_patterns = [
        "start a project",
        "new project",
        "work with you",
        "begin a project",
        "collaborate",
        "partner",
        "hire you",
        "engage your team",
        "build a website",
        "build an app",
        "need development",
        "project help",
    ]
    return any(pattern in message_lower for pattern in intent_patterns)


def handle_project_workflow(message: str, session_context: Optional[Dict] = None) -> Dict:
    ctx = _get_company_context()
    profile = ctx.get("profile")
    snippets = _search_company_context(f"{ctx['company']} projects services", limit=2)

    if snippets:
        intro = f"Great! {ctx['company']} handles projects. Here's how we work: "
        reply = intro + " ".join(snippets)
    else:
        summary = profile.summary_of_offerings() if profile else "services aligned with your needs"
        reply = (
            f"Great! {ctx['company']} delivers {summary}. "
            f"Share your context or timeline and I'll surface the most relevant information from {ctx['website_url']} "
            "or connect you with the right team."
        )

    reply += f" You can also email {ctx['contact_email']} or call {ctx['contact_phone']} to continue the conversation."

    return {
        'reply': reply,
        'sources': [],
        'website_url': ctx['website_url']
    }


def is_business_info_query(message: str) -> bool:
    message_lower = message.lower().strip()
    business_keywords = [
        'team size', 'employees', 'people on your team',
        'location', 'office', 'address', 'where are you',
        'certifications', 'awards', 'recognition',
        'experience', 'years in business', 'how long have you been operating',
        'portfolio', 'clients', 'case studies',
        'company info', 'about your company', 'company details',
        # Company history and founder queries
        'founder', 'founders', 'who started', 'who created', 'who established', 'established by', 'started by',
        'who founded', 'founded by', 'who is the founder', 'who are the founders',
        # Alumni and student queries
        'alumni', 'alumnus', 'alumna', 'graduates', 'students', 'former students', 'past students',
        'who studied', 'who trained', 'trained students', 'successful students',
        # Company history
        'history', 'background', 'origin', 'how it started', 'when started', 'when established',
        'company history', 'our history', 'about us', 'our story'
    ]
    return any(keyword in message_lower for keyword in business_keywords)


def generate_business_info_response(message: str) -> Dict:
    ctx = _get_company_context()
    profile = ctx.get("profile")
    message_lower = message.lower().strip()

    def contextual_reply(query: str, fallback: str) -> str:
        snippets = _search_company_context(query, limit=1)
        if snippets:
            return snippets[0]
        return fallback

    if any(keyword in message_lower for keyword in ['team size', 'employees', 'people']):
        reply = contextual_reply(
            f"{ctx['company']} team size",
            f"Learn more about {ctx['company']}'s team on our website at {ctx['website_url']}."
        )
    elif any(keyword in message_lower for keyword in ['location', 'where are you', 'office', 'address']):
        reply = contextual_reply(
            f"{ctx['company']} location address",
            f"You can reach us via {ctx['website_url']}. For more information, email {ctx['contact_email']}."
        )
    elif any(keyword in message_lower for keyword in ['certifications', 'awards', 'recognition']):
        reply = contextual_reply(
            f"{ctx['company']} certifications awards",
            f"Learn more about {ctx['company']}'s certifications and recognition on our website at {ctx['website_url']}."
        )
    elif any(keyword in message_lower for keyword in ['experience', 'how long', 'years']):
        reply = contextual_reply(
            f"{ctx['company']} experience history",
            f"Learn more about {ctx['company']}'s experience and history on our website at {ctx['website_url']}."
        )
    elif any(keyword in message_lower for keyword in ['portfolio', 'clients', 'case studies']):
        reply = contextual_reply(
            f"{ctx['company']} portfolio clients",
            f"Learn more about {ctx['company']}'s portfolio and clients on our website at {ctx['website_url']}."
        )
    else:
        reply = contextual_reply(
            f"{ctx['company']} overview",
            (profile.summary_sentence() if profile else f"Learn more about {ctx['company']} on our website at {ctx['website_url']}.")
        )

    return {
        'reply': reply,
        'sources': [],
        'website_url': ctx['website_url']
    }


def is_project_features_enabled() -> bool:
    return PROJECT_FEATURES_ENABLED
