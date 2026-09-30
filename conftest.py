# Root conftest — ensures project root is on sys.path for all tests.
# Also keeps the LLM cache from writing to ~/.aces during tests.
import os

os.environ.setdefault("ACES_LLM_CACHE_DISABLED", "1")# Root conftest — ensures project root is on sys.path for all tests.