"""
Central configuration for ACES. Single source of truth for tunable values —
change a setting here instead of hunting through multiple files.
"""

# --- Scraper settings ---
MAX_CONCURRENT_BROWSERS = 2
DEFAULT_TIMEOUT_MS = 60000

# --- LLM model selection (per provider) ---
GEMINI_MODEL = "gemini-3.5-flash-lite"
GROQ_MODEL = "openai/gpt-oss-20b"
OPENROUTER_MODEL = "openrouter/free"

# --- LLM call behavior ---
LLM_TIMEOUT_SECONDS = 20
LLM_PROVIDER_ORDER = ["gemini", "groq", "openrouter"]

# --- Extraction ---
HTML_TRUNCATE_SINGLE = 8000
HTML_TRUNCATE_LIST = 40000

# --- Strategy memory ---
STRATEGY_TIMEOUT_INCREMENT_MS = 30000
STRATEGY_MAX_TIMEOUT_MS = 120000