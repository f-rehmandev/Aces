# syntax=docker/dockerfile:1.7
# ============================================================================
# ACES — multi-stage production image
#
# One image serves both roles:
#     api       — FastAPI server + in-process WorkerPool + SchedulerLoop
#     worker    — Worker-only mode (no HTTP server), for horizontal scaling
#     shell     — Debug shell
#
# The entrypoint picks the mode from the container command.
# ============================================================================

# ---------------------------------------------------------------------------
# Base layer: system deps for Playwright / lxml / general tooling
# ---------------------------------------------------------------------------
# Pinned to Bookworm (Debian 12). The floating `python:3.11-slim` tag
# now resolves to Debian 13 (trixie), where two font packages that
# Playwright's `--with-deps` script expects have been renamed. Bookworm
# is the last Debian version Playwright 1.47 officially supports.
FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Playwright's browser binaries live here so both the root install
    # step and the non-root runtime user can find them.
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies (separate layer so source edits don't reinstall)
# ---------------------------------------------------------------------------
COPY requirements.txt requirements-scraping.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt -r requirements-scraping.txt

# ---------------------------------------------------------------------------
# Playwright's Chromium + system libs. Installed as root, made world-
# readable so the non-root runtime user can launch it.
# ---------------------------------------------------------------------------
RUN playwright install --with-deps chromium \
    && chmod -R 755 /ms-playwright

# ---------------------------------------------------------------------------
# SeleniumBase UC Mode's chromedriver. Downloaded at build time so the
# first runtime request doesn't pay a 30-second download cost.
#
# SeleniumBase bundles its driver under the package directory; making it
# world-readable lets the non-root runtime user launch it.
# ---------------------------------------------------------------------------
RUN sbase install chromedriver \
    && find /usr/local/lib/python3.11/site-packages/seleniumbase -type f \
       -exec chmod 644 {} \; \
    && find /usr/local/lib/python3.11/site-packages/seleniumbase -type d \
       -exec chmod 755 {} \;

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------------
# Defensive: normalize line endings on any shell scripts. If someone
# re-saves entrypoint.sh on Windows, CRLF creeps in and bash dies with
# the classic "bad interpreter ^M" error. This makes the build
# self-healing.
# ---------------------------------------------------------------------------
RUN find /app -type f -name "*.sh" -exec sed -i 's/\r$//' {} + \
    && chmod +x /app/docker/entrypoint.sh

# ---------------------------------------------------------------------------
# Non-root runtime user
# ---------------------------------------------------------------------------
RUN useradd --create-home --shell /bin/bash --uid 1000 aces \
    && chown -R aces:aces /app
USER aces

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
EXPOSE 8000

# Healthcheck hits the API's root endpoint. `curl` is in the base image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/ || exit 1

ENTRYPOINT ["/bin/bash", "/app/docker/entrypoint.sh"]
CMD ["api"]