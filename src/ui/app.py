import asyncio
import sys as _sys
if _sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from assistant import find_best_prices
from diff.diff_engine import get_previous_run, compute_diff
from diff.excel_writer import write_excel
from storage.db import save_run_results

st.set_page_config(page_title="ACES - Price Finder", page_icon="🔍", layout="centered")

st.markdown("""
<style>
    .main .block-container { max-width: 720px; padding-top: 2.5rem; }
    .aces-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0; }
    .aces-subtitle { color: #888; font-size: 1rem; margin-top: 0.2rem; margin-bottom: 2rem; }
    .result-card {
        background: #1e1e1e; border: 1px solid #333; border-radius: 12px;
        padding: 1.1rem 1.4rem; margin-bottom: 0.9rem;
    }
    .result-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 0.15rem; }
    .result-source { font-size: 0.8rem; color: #888; margin-bottom: 0.5rem; }
    .result-price { font-size: 1.4rem; font-weight: 700; color: #4ade80; }
    .badge {
        display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; margin-top: 0.4rem;
    }
    .badge-new { background: #14532d; color: #4ade80; }
    .badge-changed { background: #713f12; color: #fbbf24; }
    .badge-removed { background: #7f1d1d; color: #f87171; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="aces-title">🔍 ACES Price Finder</div>', unsafe_allow_html=True)
st.markdown('<div class="aces-subtitle">Describe what you want. Get real results from multiple sources, ranked.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Search", placeholder="e.g. wireless mouse", label_visibility="collapsed")
with col2:
    max_sources = st.selectbox("Sources", [2, 3, 4, 5], index=1, label_visibility="collapsed")

search_clicked = st.button("Find Best Prices", type="primary", use_container_width=True)

if search_clicked and query:
    with st.spinner(f"Searching, scraping, and comparing prices for '{query}'..."):
        all_results = asyncio.run(find_best_prices(query, max_sources=max_sources))

        if all_results:
            previous = get_previous_run(query)
            diffed_all = compute_diff(all_results, previous)
            save_run_results(query, diffed_all)

            top_results = diffed_all[:max_sources]

            safe_name = "".join(c if c.isalnum() else "_" for c in query)[:40]
            excel_path = f"output_{safe_name}.xlsx"
            write_excel(top_results, excel_path)

            st.session_state["results"] = top_results
            st.session_state["excel_path"] = excel_path
        else:
            st.session_state["results"] = []
            st.session_state["excel_path"] = None

if st.session_state.get("results"):
    results = st.session_state["results"]
    st.success(f"Found {len(results)} results")

    for r in results:
        status = r.get("diff_status", "")
        badge_html = ""
        if status == "New":
            badge_html = '<span class="badge badge-new">NEW</span>'
        elif "Price Changed" in status:
            badge_html = f'<span class="badge badge-changed">{status.upper()}</span>'
        elif status == "Removed":
            badge_html = '<span class="badge badge-removed">REMOVED</span>'

        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">{r.get('title', 'Unknown')}</div>
            <div class="result-source">{r.get('source_url', '')}</div>
            <div class="result-price">{r.get('price', 'N/A')}</div>
            {badge_html}
        </div>
        """, unsafe_allow_html=True)

    excel_path = st.session_state.get("excel_path")
    if excel_path and os.path.exists(excel_path):
        with open(excel_path, "rb") as f:
            excel_bytes = f.read()
        st.download_button(
            label="⬇ Download Excel",
            data=excel_bytes,
            file_name=excel_path,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

elif "results" in st.session_state and not st.session_state["results"]:
    st.warning("No results found. Try a different search term.")