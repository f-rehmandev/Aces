"""
ACES — Autonomous Cognitive Extraction System
Streamlit front-end (src/ui/app.py)

Structure:
    1. Imports + page config
    2. Theme CSS (user-supplied design)
    3. AUTH GATE — stops the script unless signed in
    4. Session state defaults
    5. Helper functions (all defined before use)
    6. Plan review renderer
    7. Module-level UI: top bar → hero → intake → handlers → results

Every backend call goes through src.ui.runner (the bridge). This file
never touches the pipeline directly.
"""

import html
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.runner import (       # noqa: E402
    build_spec_for_preview,
    run_ui_task,
    run_ui_task_from_spec,
)
from src.ui.auth import (         # noqa: E402
    get_context,
    get_session,
    render_auth_screen,
    render_user_chip,
)
from src.core.task_spec import TaskSpec, Target, FieldSpec  # noqa: E402


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ACES — Autonomous Cognitive Extraction System",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# ACES PREMIUM DARK THEME (user-supplied design — unchanged)
# ============================================================

ACES_CSS = r"""
<style>

:root {
    --aces-bg: #1E2329;
    --aces-bg-soft: #222830;
    --aces-panel: #252B33;
    --aces-panel-hover: #2A313A;
    --aces-border: rgba(148, 163, 184, 0.14);
    --aces-border-strong: rgba(148, 163, 184, 0.22);

    --aces-text: #D8DEE8;
    --aces-text-soft: #B2BAC7;
    --aces-muted: #87919F;
    --aces-dim: #697381;

    --aces-accent: #8190FF;
    --aces-accent-soft: rgba(129, 144, 255, 0.14);
    --aces-accent-purple: #A78BFA;
    --aces-success: #68C89B;
    --aces-warning: #D7B66A;
    --aces-danger: #D98787;

    --aces-radius: 8px;
    --aces-radius-lg: 12px;

    --aces-shadow:
        0 8px 30px rgba(8, 12, 18, 0.18),
        0 2px 8px rgba(8, 12, 18, 0.10);

    --aces-font:
        Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;

    --aces-mono:
        "JetBrains Mono", "Fira Code", "SFMono-Regular",
        Consolas, "Liberation Mono", monospace;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: var(--aces-bg) !important;
    color: var(--aces-text) !important;
    font-family: var(--aces-font) !important;
}

body { overflow-x: hidden; }

.block-container {
    max-width: 1500px !important;
    padding-top: 1.15rem !important;
    padding-bottom: 3rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

header[data-testid="stHeader"] {
    background: rgba(30, 35, 41, 0.82) !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-bottom: 1px solid rgba(148, 163, 184, 0.06);
}

#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
[data-testid="stToolbar"] { visibility: hidden !important; }

[data-testid="stVerticalBlock"] { gap: 0.65rem; }

.aces-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    margin-bottom: 2rem;
}

.aces-brand { display: flex; align-items: center; gap: 0.72rem; }

.aces-logo {
    width: 34px; height: 34px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 9px;
    color: #D9DEFF;
    background: linear-gradient(145deg, rgba(129,144,255,0.23), rgba(167,139,250,0.12));
    border: 1px solid rgba(129, 144, 255, 0.25);
    font-size: 15px; font-weight: 700;
}

.aces-brand-name {
    font-size: 0.98rem; font-weight: 650;
    letter-spacing: -0.015em; color: var(--aces-text);
}

.aces-brand-subtitle { font-size: 0.74rem; color: var(--aces-muted); margin-top: 2px; }

.aces-status {
    display: inline-flex; align-items: center; gap: 0.46rem;
    padding: 0.38rem 0.72rem; border-radius: 999px;
    border: 1px solid rgba(104, 200, 155, 0.16);
    background: rgba(104, 200, 155, 0.06);
    color: #A8DCC5; font-size: 0.72rem; font-weight: 600;
}

.aces-status.running {
    border-color: rgba(215, 182, 106, 0.22);
    background: rgba(215, 182, 106, 0.08);
    color: #E4CD91;
}

.aces-status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--aces-success);
}

.aces-status.running .aces-status-dot {
    background: var(--aces-warning);
}

.aces-hero { max-width: 900px; margin: 0 auto; text-align: center; padding-bottom: 2.15rem; }

.aces-eyebrow {
    display: inline-block; margin-bottom: 0.85rem;
    color: #9EA8B7; font-size: 0.72rem; font-weight: 650;
    letter-spacing: 0.14em; text-transform: uppercase;
}

.aces-title {
    margin: 0; color: #DCE2EC;
    font-size: clamp(2rem, 3.5vw, 3.1rem);
    line-height: 1.05; letter-spacing: -0.045em; font-weight: 660;
}

.aces-title span { color: #929EFF; }

.aces-description {
    max-width: 675px; margin: 1.15rem auto 0;
    color: var(--aces-muted); font-size: 0.93rem; line-height: 1.72;
}

[data-testid="stTextArea"] { margin-top: 0.75rem; }
[data-testid="stTextArea"] label { display: none !important; }

[data-testid="stTextArea"] textarea {
    min-height: 120px !important;
    background: #252B33 !important;
    color: var(--aces-text) !important;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    border-radius: var(--aces-radius-lg) !important;
    padding: 1.1rem 1.2rem !important;
    font-family: var(--aces-font) !important;
    font-size: 0.95rem !important; line-height: 1.65 !important;
    resize: vertical !important;
    transition: border-color 160ms ease, background 160ms ease !important;
}

[data-testid="stTextArea"] textarea::placeholder { color: #6F7986 !important; }

[data-testid="stTextArea"] textarea:focus {
    outline: none !important;
    border-color: rgba(129, 144, 255, 0.55) !important;
    background: #282F38 !important;
}

.aces-field-label {
    margin: 0.7rem 0 0.35rem 0;
    color: var(--aces-muted); font-size: 0.73rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.09em;
}

[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stCheckbox"] label {
    color: var(--aces-muted) !important; font-size: 0.78rem !important;
}

[data-testid="stTextInput"] input {
    background: #252B33 !important; color: var(--aces-text) !important;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    border-radius: 8px !important; padding: 0.72rem 0.9rem !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: rgba(129, 144, 255, 0.55) !important;
}

.stButton > button, .stDownloadButton > button,
button[kind="secondary"], button[kind="primary"] {
    min-height: 40px; border-radius: 8px !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    background: #2A313A !important;
    color: #D7DDE7 !important;
    font-family: var(--aces-font) !important;
    font-size: 0.84rem !important; font-weight: 600 !important;
    transition: border-color 120ms ease, background 120ms ease !important;
}

.stButton > button:hover, .stDownloadButton > button:hover,
button[kind="secondary"]:hover {
    border-color: rgba(129, 144, 255, 0.42) !important;
    background: #2F3742 !important;
    color: #E3E7F0 !important;
}

button[kind="primary"] {
    background: #5A67D8 !important;
    border-color: #5A67D8 !important;
    color: #F5F6FB !important;
}

button[kind="primary"]:hover {
    background: #6B77E0 !important;
    border-color: #6B77E0 !important;
}

.aces-section-heading {
    margin: 2rem 0 0.85rem;
    color: #AEB7C4; font-size: 0.76rem; font-weight: 650;
    text-transform: uppercase; letter-spacing: 0.11em;
}

.aces-metric {
    min-height: 108px; padding: 1rem 1.05rem;
    background: #252B33;
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 8px;
}

.aces-metric-label {
    color: #7F8997; font-size: 0.72rem; font-weight: 620;
    text-transform: uppercase; letter-spacing: 0.07em;
}

.aces-metric-value {
    margin-top: 0.62rem; color: #DCE2EA;
    font-size: 1.6rem; line-height: 1; font-weight: 660; letter-spacing: -0.03em;
}

.aces-metric-sub { margin-top: 0.55rem; color: #717B88; font-size: 0.72rem; }
.aces-metric-positive { color: #87CDAA; }
.aces-metric-purple { color: #AEB5FF; }

.aces-terminal {
    margin-top: 0.45rem; overflow: hidden;
    background: #20252C;
    border: 1px solid rgba(148, 163, 184, 0.11);
    border-radius: 10px;
}

.aces-terminal-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.72rem 0.9rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.08);
    background: #222830;
}

.aces-terminal-title {
    display: flex; align-items: center; gap: 0.5rem;
    color: #AEB7C5; font-family: var(--aces-mono); font-size: 0.72rem;
}

.aces-terminal-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--aces-success);
}

.aces-terminal-status { color: #65707D; font-family: var(--aces-mono); font-size: 0.65rem; }

.aces-log { padding: 0.95rem 1rem 1.1rem; font-family: var(--aces-mono); font-size: 0.72rem; line-height: 1.75; }

.aces-log-row {
    display: grid; grid-template-columns: 74px 92px minmax(0, 1fr);
    gap: 0.6rem; padding: 0.15rem 0;
}

.aces-log-time { color: #56606D; }
.aces-log-kind { color: #8D98A6; }
.aces-log-message { color: #B8C0CB; }
.aces-log-success { color: #86CEA9; }
.aces-log-warning { color: #D6B873; }
.aces-log-info { color: #99A5FF; }
.aces-log-error { color: #D58F8F; }

[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid rgba(148, 163, 184, 0.10) !important;
    border-radius: 9px !important; overflow: hidden;
}

[data-testid="stExpander"] details { background: transparent !important; }

[data-testid="stExpander"] summary {
    color: #B5BEC9 !important; font-size: 0.81rem !important; font-weight: 610 !important;
}

[data-testid="stAlert"] {
    background: #252B33 !important;
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
    border-radius: 8px !important; color: var(--aces-text) !important;
}

.aces-badge-row { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem; }

.aces-badge {
    display: inline-flex; align-items: center;
    padding: 0.31rem 0.58rem; border-radius: 999px;
    font-size: 0.66rem; font-weight: 650; border: 1px solid transparent;
}

.aces-badge-new { color: #96D8B4; background: rgba(104,200,155,0.08); border-color: rgba(104,200,155,0.15); }
.aces-badge-changed { color: #AEB6FF; background: rgba(129,144,255,0.08); border-color: rgba(129,144,255,0.15); }
.aces-badge-removed { color: #DB9A9A; background: rgba(217,135,135,0.07); border-color: rgba(217,135,135,0.14); }

hr { margin: 1.6rem 0 !important; border: none !important; border-top: 1px solid rgba(148,163,184,0.08) !important; }

[data-testid="stDataFrame"] { border-radius: 9px !important; overflow: hidden !important; }

.aces-footnote {
    color: var(--aces-dim); font-size: 0.72rem; text-align: center; margin-top: 0.5rem;
}

.aces-plan-card {
    padding: 1.1rem 1.3rem;
    background: #252B33;
    border: 1px solid rgba(129, 144, 255, 0.22);
    border-left: 3px solid var(--aces-accent);
    border-radius: 8px;
    margin-bottom: 1.2rem;
}

.aces-plan-eyebrow {
    color: #99A5FF;
    font-size: 0.7rem;
    font-weight: 650;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.aces-plan-prompt {
    margin-top: 0.55rem;
    color: #DCE2EC;
    font-size: 0.95rem;
    line-height: 1.6;
    font-style: italic;
}

.aces-plan-note {
    margin-top: 0.8rem;
    color: var(--aces-muted);
    font-size: 0.78rem;
    line-height: 1.55;
}

@media (max-width: 900px) {
    .block-container { padding-left: 1.15rem !important; padding-right: 1.15rem !important; }
    .aces-topbar { margin-bottom: 1.6rem; }
    .aces-title { font-size: 2rem; }
    .aces-log-row { grid-template-columns: 62px 76px minmax(0, 1fr); }
}

</style>
"""

st.markdown(ACES_CSS, unsafe_allow_html=True)


# ============================================================
# AUTH GATE — nothing below renders unless a user is signed in
# ============================================================

if get_session() is None:
    render_auth_screen()
    st.stop()

# Resolve the tenant context once per rerun; cached in session_state.
CLIENT_CONTEXT = get_context()


# ============================================================
# SESSION STATE DEFAULTS
# ============================================================

_DEFAULTS = {
    "aces_prompt": "",
    "aces_url": "",
    "aces_client_id": "default",
    "aces_running": False,
    "aces_pending_spec": None,
    "aces_spec_error": "",
    "aces_grid_areas": [],              # neighborhoods for lead-gen grid mode
    "aces_grid_error": "",
    "aces_metrics": {"records": "—", "quality": "—", "confidence": "—", "urls": "—"},
    "aces_changes": {"new": 0, "changed": 0, "removed": 0},
    "aces_preview": None,
    "aces_run_mode": None,
    "aces_workbook_path": None,
    "aces_receipt": None,
    "aces_logs": [
        {"time": datetime.now().strftime("%H:%M:%S"), "kind": "INIT",
         "message": "ACES workspace initialized.", "level": "info"},
        {"time": datetime.now().strftime("%H:%M:%S"), "kind": "READY",
         "message": "Waiting for an extraction request.", "level": "success"},
    ],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ============================================================
# HELPERS
# ============================================================

def add_log(kind: str, message: str, level: str = "info") -> None:
    st.session_state.aces_logs.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "kind": kind.upper(),
        "message": message,
        "level": level,
    })


def render_logs() -> str:
    rows = []
    for event in st.session_state.aces_logs[-40:]:
        message = html.escape(event["message"])
        kind = html.escape(event["kind"])
        ts = html.escape(event["time"])
        rows.append(
            f'<div class="aces-log-row">'
            f'<div class="aces-log-time">{ts}</div>'
            f'<div class="aces-log-kind">{kind}</div>'
            f'<div class="aces-log-message aces-log-{event["level"]}">{message}</div>'
            f'</div>'
        )
    return (
        '<div class="aces-terminal">'
        '<div class="aces-terminal-head">'
        '<div class="aces-terminal-title">'
        '<span class="aces-terminal-dot"></span>'
        'ACES execution trace'
        '</div>'
        '<div class="aces-terminal-status">live / observable events</div>'
        '</div>'
        f'<div class="aces-log">{"".join(rows)}</div>'
        '</div>'
    )


def metric_card(label: str, value: str, subtitle: str, value_class: str = "") -> str:
    return (
        f'<div class="aces-metric">'
        f'<div class="aces-metric-label">{html.escape(label)}</div>'
        f'<div class="aces-metric-value {value_class}">{html.escape(str(value))}</div>'
        f'<div class="aces-metric-sub">{html.escape(subtitle)}</div>'
        f'</div>'
    )


def render_badges(changes: dict) -> str:
    chips = "".join(
        f'<span class="aces-badge aces-badge-{kind}">{changes[kind]} {label}</span>'
        for kind, label in (("new", "New"), ("changed", "Changed"), ("removed", "Removed"))
    )
    return f'<div class="aces-badge-row">{chips}</div>'


def _result_to_state(result) -> None:
    """Copy a UiRunResult into session state so the UI can render it."""
    st.session_state.aces_metrics = {
        "records": f"{result.record_count:,}",
        "quality": (f"{result.quality_score:.2f}"
                    if result.quality_score is not None else "—"),
        "confidence": (f"{result.confidence_mean:.2f}"
                       if result.confidence_mean is not None else "—"),
        "urls": str(result.source_count),
    }
    # For grid/lead runs, "changed" reflects existing leads; for pipeline
    # runs it reflects modified records. Same badge row either way.
    existing = getattr(result, "existing_count", 0)
    st.session_state.aces_changes = {
        "new": result.new_count,
        "changed": existing if existing else result.changed_count,
        "removed": result.removed_count,
    }
    st.session_state.aces_preview = (
        pd.DataFrame(result.records) if result.records else None
    )
    st.session_state.aces_run_mode = result.mode
    st.session_state.aces_workbook_path = result.workbook_path
    st.session_state.aces_receipt = result.receipt_signature
    st.session_state.aces_logs = [
        {"time": e.time, "kind": e.kind, "message": e.message, "level": e.level}
        for e in result.trace
    ]


def _run_with_bridge(prompt: str, url: str, output_format: str,
                     min_sources: int) -> None:
    """Prompt → plan → run. Used by the "Run now" shortcut."""
    st.session_state.aces_running = True
    with st.spinner("ACES is running the pipeline…"):
        result = run_ui_task(
            prompt,
            url=url,
            client_id=CLIENT_CONTEXT.client_id,
            output_format=output_format,
            min_sources=int(min_sources),
            context=CLIENT_CONTEXT,
        )
    _result_to_state(result)
    st.session_state.aces_running = False


def _run_from_spec(spec: TaskSpec, grid_areas=None,
                   discovery_mode: bool = False) -> None:
    st.session_state.aces_running = True
    with st.spinner("Executing approved plan…"):
        result = run_ui_task_from_spec(
            spec,
            client_id=CLIENT_CONTEXT.client_id,
            context=CLIENT_CONTEXT,
            grid_areas=grid_areas or None,
            discovery_mode=discovery_mode,
        )
    _result_to_state(result)
    st.session_state.aces_running = False

def _render_plan_review(spec: TaskSpec):
    """
    Show an editable view of the TaskSpec ACES extracted.
    Returns (approve_clicked, cancel_clicked, edited_values).
    """
    st.markdown('<div class="aces-section-heading">Review the plan</div>',
                unsafe_allow_html=True)

    prompt_safe = html.escape(spec.natural_language_prompt or "(no prompt recorded)")
    st.markdown(
        '<div class="aces-plan-card">'
        '<div class="aces-plan-eyebrow">ACES understood your request as</div>'
        f'<div class="aces-plan-prompt">“{prompt_safe}”</div>'
        '<div class="aces-plan-note">'
        'This is a proposal — edit anything before you approve. '
        'Nothing is fetched until you click <strong>Approve &amp; run</strong>.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        targets_text = st.text_area(
            "Target URLs (one per line)",
            value="\n".join(spec.target.start_urls),
            height=110,
            key="plan_targets",
        )
        fields_text = st.text_input(
            "Fields to extract (comma-separated)",
            value=", ".join(spec.field_names),
            key="plan_fields",
        )
    with col2:
        objectives = ["extract", "monitor", "compare", "lead_gen"]
        obj_index = (objectives.index(spec.objective)
                     if spec.objective in objectives else 0)
        objective = st.selectbox("Objective", objectives, index=obj_index,
                                 key="plan_objective")
        min_records = st.number_input(
            "Minimum records",
            min_value=1, max_value=1_000_000,
            value=max(1, int(spec.quality.min_records)),
            key="plan_min_records",
        )
        formats = ["xlsx", "csv", "json", "jsonl", "parquet"]
        fmt_index = formats.index(spec.output.format) if spec.output.format in formats else 0
        output_format = st.selectbox("Output format", formats,
                                     index=fmt_index, key="plan_output_format")

    # ---- Google Maps neighborhoods (lead-gen only) ----
    grid_areas_text = ""
    discovery_mode = False
    grid_applicable = (
        spec.objective == "lead_gen"
        and bool(spec.constraints.geography)
    )
    if grid_applicable:
        with st.expander("Google Maps neighborhoods (lead-gen grid search)",
                          expanded=True):
            st.caption(
                "One neighborhood per line. Each is scraped separately "
                "and the results are merged + deduped by place_id. "
                "Add or remove areas freely — up to ~25 works well."
            )
            current_areas = st.session_state.get("aces_grid_areas") or []
            if st.session_state.get("aces_grid_error"):
                st.warning(
                    f"Could not auto-generate neighborhoods: "
                    f"{st.session_state['aces_grid_error']}"
                )
            grid_areas_text = st.text_area(
                "Neighborhoods",
                value="\n".join(current_areas),
                height=180,
                key="plan_grid_areas",
                label_visibility="collapsed",
            )

        st.markdown(
            '<div style="height:0.4rem"></div>',
            unsafe_allow_html=True,
        )
        discovery_mode = st.checkbox(
            "Discovery mode — only return NEW leads (per client)",
            value=False,
            key="plan_discovery_mode",
            help=(
                "ON:  only leads this client has NEVER seen before. "
                "Existing leads are filtered out. Quantity may be lower "
                "than a normal run because repeats are dropped.\n\n"
                "OFF: every lead is returned, each tagged NEW or EXISTING "
                "in the output sheet."
            ),
        )
        if discovery_mode:
            st.warning(
                "⚠ Discovery mode is ON. Only NEW leads will be returned "
                "for this client. The quantity of leads may be lower than "
                "a normal run because anything already seen is filtered out."
            )
        else:
            st.caption(
                "Monitoring mode: every lead will be returned and each "
                "one will be tagged NEW or EXISTING in the output sheet."
            )

    with st.expander("Budget & compliance"):
        max_usd = st.number_input(
            "Maximum budget (USD)",
            min_value=0.0, max_value=1000.0, step=0.05,
            value=float(spec.budget.max_usd),
            key="plan_budget",
        )
        declared = st.checkbox(
            "I confirm I'm authorized to collect this data",
            value=bool(spec.compliance.user_authorization_declared),
            key="plan_declared",
        )

    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        approve = st.button("Approve & run", type="primary", use_container_width=True)
    with col_b:
        cancel = st.button("Cancel plan", use_container_width=True)

    return approve, cancel, {
        "targets_text": targets_text,
        "fields_text": fields_text,
        "objective": objective,
        "min_records": min_records,
        "output_format": output_format,
        "grid_areas_text": grid_areas_text,
        "grid_applicable": grid_applicable,
        "discovery_mode": discovery_mode,
        "max_usd": max_usd,
        "declared": declared,
    }


def _apply_edits(spec: TaskSpec, edited: dict) -> TaskSpec:
    """Merge user edits back into the TaskSpec before running."""
    new_targets = [u.strip() for u in edited["targets_text"].splitlines() if u.strip()]
    if new_targets:
        spec.target.start_urls = new_targets

    new_fields = [f.strip() for f in edited["fields_text"].split(",") if f.strip()]
    if new_fields:
        spec.fields = [FieldSpec(name=f) for f in new_fields]

    spec.objective = edited["objective"]
    spec.quality.min_records = int(edited["min_records"])
    spec.output.format = edited["output_format"]
    spec.budget.max_usd = float(edited["max_usd"])
    spec.compliance.user_authorization_declared = bool(edited["declared"])
    return spec


# ============================================================
# TOP BAR (signed-in user is guaranteed here)
# ============================================================

status_class = "running" if st.session_state.aces_running else ""
status_label = "Running…" if st.session_state.aces_running else "System Ready"

st.markdown(
    f'<div class="aces-topbar">'
    f'<div class="aces-brand">'
    f'<div class="aces-logo">A</div>'
    f'<div>'
    f'<div class="aces-brand-name">ACES</div>'
    f'<div class="aces-brand-subtitle">Autonomous Cognitive Extraction System</div>'
    f'</div>'
    f'</div>'
    f'<div style="text-align:right;">'
    f'<div class="aces-status {status_class}">'
    f'<span class="aces-status-dot"></span>{status_label}'
    f'</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# Signed-in user chip: email · workspace · role + sign-out
_chip_left, _chip_right = st.columns([3, 1])
with _chip_right:
    render_user_chip()

# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="aces-hero">'
    '<span class="aces-eyebrow">AI-native data operations</span>'
    '<h1 class="aces-title">Turn a sentence into <span>validated data.</span></h1>'
    '<p class="aces-description">'
    'Describe what you need in plain language. ACES discovers the structure, '
    'extracts, triangulates across sources, checks its own trust score, and '
    'hands you a signed, auditable dataset — not just a scrape.'
    '</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# INTAKE FORM
# ============================================================

st.markdown('<div class="aces-field-label">What do you want to collect?</div>',
            unsafe_allow_html=True)

prompt = st.text_area(
    "prompt",
    key="aces_prompt",
    placeholder='e.g. "Find 15 leads for a website maker, pizza shops niche, need phone/address/email/website status"',
    label_visibility="collapsed",
)

with st.expander("Advanced options"):
    col1, col2 = st.columns(2)
    with col1:
        url = st.text_input(
            "Start URL (optional)",
            key="aces_url",
            placeholder="https://example.com/category/laptops",
        )
    with col2:
        output_format = st.selectbox("Output format", ["XLSX", "CSV", "JSON"])
        min_sources = st.number_input(
            "Minimum independent sources", min_value=1, max_value=10, value=1,
        )

col_preview, col_run, col_clear = st.columns([2, 1, 1])
with col_preview:
    preview_clicked = st.button("Preview plan", type="primary", use_container_width=True)
with col_run:
    run_clicked = st.button("Run now", use_container_width=True)
with col_clear:
    clear_clicked = st.button("Clear", use_container_width=True)


# ============================================================
# INTENT HANDLERS
# ============================================================

if clear_clicked:
    st.session_state.aces_pending_spec = None
    st.session_state.aces_spec_error = ""
    st.session_state.aces_metrics = {"records": "—", "quality": "—",
                                      "confidence": "—", "urls": "—"}
    st.session_state.aces_changes = {"new": 0, "changed": 0, "removed": 0}
    st.session_state.aces_preview = None
    st.session_state.aces_run_mode = None
    st.session_state.aces_workbook_path = None
    st.session_state.aces_receipt = None
    st.session_state.aces_logs = st.session_state.aces_logs[:2]
    st.rerun()

if preview_clicked:
    if not prompt.strip():
        st.warning("Describe what you want to collect before previewing.")
    else:
        st.session_state.aces_spec_error = ""
        st.session_state.aces_grid_areas = []
        st.session_state.aces_grid_error = ""
        with st.spinner("ACES is reading your request…"):
            spec, err = build_spec_for_preview(
                prompt, url=url, client_id=CLIENT_CONTEXT.client_id,
                output_format=output_format, min_sources=int(min_sources),
            )
        if err or spec is None:
            st.session_state.aces_spec_error = err or "unknown error"
            st.session_state.aces_pending_spec = None
        else:
            st.session_state.aces_pending_spec = spec

            # If this is a lead-gen task with a location, preview neighborhoods
            if spec.objective == "lead_gen" and spec.constraints.geography:
                entity = spec.entities[0].entity_name if spec.entities else ""
                city = spec.constraints.geography
                from src.ui.runner import build_grid_preview
                areas, gerr = build_grid_preview(entity, city, max_areas=15)
                st.session_state.aces_grid_areas = areas
                st.session_state.aces_grid_error = gerr
        st.rerun()

if run_clicked:
    if not prompt.strip():
        st.warning("Describe what you want to collect before running.")
    else:
        st.session_state.aces_pending_spec = None
        _run_with_bridge(prompt, url, output_format, int(min_sources))
        st.rerun()

if st.session_state.aces_spec_error:
    st.error(f"Could not build a plan: {st.session_state.aces_spec_error}")
    st.caption("Check the prompt, or the LLM providers in your .env file.")


# ============================================================
# PLAN REVIEW (only when a spec is pending)
# ============================================================

if st.session_state.aces_pending_spec is not None:
    _pending = st.session_state.aces_pending_spec
    approve, cancel, edited = _render_plan_review(_pending)

    if cancel:
        st.session_state.aces_pending_spec = None
        st.session_state.aces_grid_areas = []
        st.rerun()

    if approve:
        final_spec = _apply_edits(_pending, edited)

        # Parse the neighborhoods textarea into a list
        grid_areas: list[str] = []
        if edited.get("grid_applicable"):
            raw = edited.get("grid_areas_text") or ""
            grid_areas = [a.strip() for a in raw.splitlines() if a.strip()]

        _run_from_spec(
            final_spec,
            grid_areas=grid_areas,
            discovery_mode=bool(edited.get("discovery_mode", False)),
        )
        st.session_state.aces_pending_spec = None
        st.session_state.aces_grid_areas = []
        st.rerun()
# ============================================================
# RESULTS
# ============================================================

st.markdown('<div class="aces-section-heading">Run summary</div>',
            unsafe_allow_html=True)
m = st.session_state.aces_metrics
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.markdown(metric_card("Records extracted", m["records"],
                         "rows returned by this run"), unsafe_allow_html=True)
mc2.markdown(metric_card("Quality score", m["quality"],
                         "quality gate result", "aces-metric-positive"),
             unsafe_allow_html=True)
mc3.markdown(metric_card("Mean confidence", m["confidence"],
                         "source-weighted consensus", "aces-metric-purple"),
             unsafe_allow_html=True)
mc4.markdown(metric_card("Sources crawled", m["urls"],
                         "independent domains"), unsafe_allow_html=True)

st.markdown(render_badges(st.session_state.aces_changes), unsafe_allow_html=True)

if st.session_state.aces_run_mode == "demo":
    st.info("This run used the demo simulation. See the trace below for why.")

if st.session_state.aces_receipt:
    st.caption(f"Signed receipt: {st.session_state.aces_receipt[:24]}…")

st.markdown('<div class="aces-section-heading">Execution trace</div>',
            unsafe_allow_html=True)
st.markdown(render_logs(), unsafe_allow_html=True)

if st.session_state.aces_preview is not None:
    st.markdown('<div class="aces-section-heading">Data preview</div>',
                unsafe_allow_html=True)
    st.dataframe(st.session_state.aces_preview,
                 use_container_width=True, hide_index=True)

    if st.session_state.aces_workbook_path:
        wp = Path(st.session_state.aces_workbook_path)
        if wp.exists():
            with open(wp, "rb") as f:
                st.download_button(
                    label="⬇ Download workbook",
                    data=f.read(),
                    file_name=wp.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=False,
                )

st.markdown(
    '<hr />'
    '<p class="aces-footnote">'
    'Every fetched page is treated as untrusted input — hidden content is '
    'stripped before extraction and model output is validated before it '
    'becomes data.'
    '</p>',
    unsafe_allow_html=True,
)