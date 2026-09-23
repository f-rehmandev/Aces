import logging
import time
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from storage.db import get_client

logger = logging.getLogger("audit_logger")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def log_event(event: str, provider: str = None, query: str = None, details: dict = None):
    """Records an event to Supabase audit_logs — non-blocking best-effort, never raises."""
    try:
        client = get_client()
        client.table("audit_logs").insert({
            "event": event,
            "provider": provider,
            "query": query,
            "details": details or {},
        }).execute()
    except Exception as e:
        logger.warning(f"Audit log failed (non-fatal): {e}")


class Timer:
    """Simple context manager to measure and log how long a task_runner run took."""
    def __init__(self, query: str):
        self.query = query
        self.start = None

    def __enter__(self):
        self.start = time.time()
        log_event("run_started", query=self.query)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = round(time.time() - self.start, 2)
        if exc_type:
            log_event("run_failed", query=self.query, details={"elapsed_seconds": elapsed, "error": str(exc_val)})
        else:
            log_event("run_completed", query=self.query, details={"elapsed_seconds": elapsed})


if __name__ == "__main__":
    with Timer("test query"):
        time.sleep(1)
        print("Simulated work done.")