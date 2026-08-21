from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.core.config import get_settings


def _require_api_key() -> str:
    settings = get_settings()
    # Gemini API keys are API-key credentials, not OAuth bearer/access tokens.
    # Prefer the current GEMINI_API_KEY name and support GOOGLE_API_KEY only
    # for backwards compatibility.
    key = settings.gemini_api_key or settings.google_api_key
    if not key:
        raise RuntimeError(
            "Gemini API key is not configured. Set GEMINI_API_KEY in .env "
            "(a Google AI Studio Gemini API key, not a Google OAuth access token)."
        )

    # A common cause of ACCESS_TOKEN_TYPE_UNSUPPORTED is putting a short-lived
    # OAuth access token (typically beginning with ya29.) in the API-key field.
    if key.strip().lower().startswith("ya29."):
        raise RuntimeError(
            "Invalid Gemini credential: an OAuth access token was supplied where "
            "a Gemini API key is required. Create a Gemini API key in Google AI "
            "Studio and set GEMINI_API_KEY in .env."
        )
    return key.strip()


@lru_cache
def get_generation_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.generation_model,
        google_api_key=_require_api_key(),
    )


@lru_cache
def get_router_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.router_model,
        google_api_key=_require_api_key(),
    )
