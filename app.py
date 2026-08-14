from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from fabric_migrator import MigrationEngine  # noqa: E402
from fabric_migrator.notebook_io import NotebookValidationError  # noqa: E402

st.set_page_config(
    page_title="RANCH (BETA) · Fabric Data Agent Migration",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

components.html(
    """
<script>
  (() => {
    const param = new URLSearchParams(window.location.search).get("scoutTheme");
    const theme =
      param || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", theme);
  })();
</script>
""",
    height=0,
)

st.markdown(
    """
<style>
:root {
  color-scheme: light;
  --cp-bg: #f7f4ef;
  --cp-bg-elevated: #fcfbf8;
  --cp-surface: #ffffff;
  --cp-surface-soft: #f5f5f5;
  --cp-border: #dedede;
  --cp-border-strong: #919191;
  --cp-text: #242424;
  --cp-text-muted: #5c5c5c;
  --cp-text-soft: #6f6f6f;
  --cp-accent: #b11f4b;
  --cp-accent-hover: #9a1a41;
  --cp-accent-soft: rgba(177, 31, 75, 0.08);
  --cp-accent-fg: #ffffff;
  --cp-success: #16a34a;
  --cp-danger: #dc2626;
  --cp-warning: #f59e0b;
  --cp-link: #0078d4;
  --cp-shadow: 0 18px 48px rgba(0, 0, 0, 0.12);
  --cp-overlay: rgba(255, 255, 255, 0.8);
  --cp-panel: rgba(255, 255, 255, 0.86);
  --cp-panel-strong: rgba(255, 255, 255, 0.96);
  --cp-sheen: rgba(255, 255, 255, 0.55);
  --cp-highlight: rgba(177, 31, 75, 0.12);
}
html[data-theme="dark"] {
  color-scheme: dark;
  --cp-bg: #3d3b3a;
  --cp-bg-elevated: #343231;
  --cp-surface: #292929;
  --cp-surface-soft: #2e2e2e;
  --cp-border: #474747;
  --cp-border-strong: #5f5f5f;
  --cp-text: #dedede;
  --cp-text-muted: #919191;
  --cp-text-soft: #b0b0b0;
  --cp-accent: #fd8ea1;
  --cp-accent-hover: #fb7b91;
  --cp-accent-soft: rgba(253, 142, 161, 0.14);
  --cp-accent-fg: #1a1a1a;
  --cp-success: #4ade80;
  --cp-danger: #f87171;
  --cp-warning: #fbbf24;
  --cp-link: #4da6ff;
  --cp-shadow: 0 18px 48px rgba(0, 0, 0, 0.32);
  --cp-overlay: rgba(41, 41, 41, 0.88);
  --cp-panel: rgba(41, 41, 41, 0.72);
  --cp-panel-strong: rgba(41, 41, 41, 0.96);
  --cp-sheen: rgba(255, 255, 255, 0.04);
  --cp-highlight: rgba(253, 142, 161, 0.12);
}
html, body, [class*="css"], .stApp {
  font-family: "Segoe UI", Aptos, Calibri, -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
  background: var(--cp-bg);
  color: var(--cp-text);
}
.block-container {
  max-width: 1180px;
  padding-top: 2.25rem;
  padding-bottom: 4rem;
}
.hero {
  background: var(--cp-bg-elevated);
  border: 1px solid var(--cp-border);
  border-left: 6px solid var(--cp-accent);
  border-radius: 16px;
  box-shadow: 0 0 2px var(--cp-border), 0 1px 2px var(--cp-border);
  padding: 28px 32px;
  margin-bottom: 24px;
}
.brand-row {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 10px;
}
.wordmark {
  color: var(--cp-text);
  font-size: clamp(2.8rem, 6vw, 4.6rem);
  font-weight: 800;
  letter-spacing: -0.07em;
  line-height: 0.95;
}
.beta-stamp {
  border: 2px solid var(--cp-accent);
  border-radius: 0.625rem;
  color: var(--cp-accent);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  padding: 6px 9px;
  transform: rotate(-2deg);
}
.eyebrow {
  color: var(--cp-accent);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.hero h1, .hero h2 {
  color: var(--cp-text);
  font-size: clamp(1.1rem, 2vw, 1.45rem);
  font-weight: 600;
  letter-spacing: -0.015em;
  line-height: 1.3;
  margin: 0 0 14px;
}
.ranch-initial {
  color: var(--cp-accent);
  font-size: 1.12em;
  font-weight: 800;
}
.hero p {
  color: var(--cp-text-muted);
  font-size: 1.05rem;
  line-height: 1.6;
  margin: 0;
  max-width: 760px;
}
.hero .tagline {
  color: var(--cp-text);
  font-size: 1.08rem;
  font-weight: 600;
  margin-bottom: 6px;
}
.trust-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 20px;
}
.source-row {
  align-items: center;
  border-top: 1px solid var(--cp-border);
  display: flex;
  flex-wrap: wrap;
  gap: 8px 16px;
  margin-top: 20px;
  padding-top: 16px;
}
.source-label {
  color: var(--cp-text-muted);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.source-row a {
  color: var(--cp-link);
  font-size: 0.86rem;
  font-weight: 600;
  text-decoration: none;
}
.source-row a:hover {
  text-decoration: underline;
}
.beta-warning {
  background: var(--cp-accent-soft);
  border: 1px solid var(--cp-accent);
  border-radius: 16px;
  color: var(--cp-text);
  display: grid;
  gap: 4px;
  grid-template-columns: auto 1fr;
  margin: 0 0 24px;
  padding: 16px 18px;
}
.warning-mark {
  color: var(--cp-accent);
  font-size: 1.25rem;
  font-weight: 800;
  line-height: 1.2;
  padding-right: 6px;
}
.warning-title {
  font-weight: 750;
}
.warning-copy {
  color: var(--cp-text-muted);
  font-size: 0.9rem;
  line-height: 1.5;
}
.trust-pill {
  background: var(--cp-surface);
  border: 1px solid var(--cp-border);
  border-radius: 0.625rem;
  color: var(--cp-text-muted);
  font-size: 0.82rem;
  padding: 7px 10px;
}
.score-card {
  background: var(--cp-surface);
  border: 1px solid var(--cp-border);
  border-radius: 16px;
  box-shadow: 0 0 2px var(--cp-border), 0 1px 2px var(--cp-border);
  padding: 20px;
  min-height: 160px;
}
.score-value {
  color: var(--cp-accent);
  font-size: 3rem;
  font-weight: 700;
  letter-spacing: -0.05em;
  line-height: 1;
}
.score-label {
  color: var(--cp-text-muted);
  font-size: 0.82rem;
  margin-top: 8px;
}
.status-banner {
  background: var(--cp-accent-soft);
  border: 1px solid var(--cp-accent);
  border-radius: 0.625rem;
  color: var(--cp-text);
  margin: 16px 0 22px;
  padding: 12px 16px;
}
.ready-card {
  align-items: center;
  animation: ready-arrive 520ms ease-out both;
  background: var(--cp-surface);
  border: 1px solid var(--cp-success);
  border-radius: 16px;
  box-shadow: 0 0 2px var(--cp-border), 0 1px 2px var(--cp-border);
  display: grid;
  gap: 14px;
  grid-template-columns: auto 1fr;
  margin: 4px 0 18px;
  overflow: hidden;
  padding: 16px 18px;
  position: relative;
}
.ready-card::after {
  animation: ready-sweep 900ms 180ms ease-out both;
  background: var(--cp-accent-soft);
  content: "";
  inset: 0 auto 0 0;
  pointer-events: none;
  position: absolute;
  width: 100%;
}
.ready-icon {
  align-items: center;
  background: var(--cp-accent-soft);
  border: 1px solid var(--cp-accent);
  border-radius: 0.625rem;
  color: var(--cp-accent);
  display: flex;
  font-size: 1.25rem;
  font-weight: 800;
  height: 46px;
  justify-content: center;
  position: relative;
  width: 46px;
  z-index: 1;
}
.ready-copy {
  position: relative;
  z-index: 1;
}
.ready-title {
  color: var(--cp-text);
  font-size: 1rem;
  font-weight: 750;
}
.ready-detail {
  color: var(--cp-text-muted);
  font-size: 0.86rem;
  line-height: 1.45;
  margin-top: 2px;
}
@keyframes ready-arrive {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
@keyframes ready-sweep {
  from {
    transform: translateX(0);
  }
  to {
    transform: translateX(101%);
  }
}
.finding {
  background: var(--cp-surface);
  border: 1px solid var(--cp-border);
  border-radius: 0.625rem;
  margin: 8px 0;
  padding: 12px 14px;
}
.finding-title {
  color: var(--cp-text);
  font-weight: 650;
}
.finding-meta {
  color: var(--cp-text-muted);
  font-size: 0.8rem;
  margin-top: 4px;
}
.privacy {
  color: var(--cp-text-muted);
  font-size: 0.84rem;
  line-height: 1.5;
  margin-top: 8px;
}
code, pre, .stCode {
  font-family: Consolas, "Courier New", Courier, monospace !important;
}
a {
  color: var(--cp-link);
}
div[data-testid="stFileUploader"] {
  background: var(--cp-surface);
  border: 1px solid var(--cp-border);
  border-radius: 16px;
  padding: 12px;
}
div.stButton > button, div.stDownloadButton > button {
  border-radius: 0.625rem;
  min-height: 44px;
}
div.stDownloadButton > button[kind="primary"] {
  animation: download-pulse 2.4s 1.1s ease-in-out infinite;
  background: var(--cp-accent);
  border-color: var(--cp-accent);
  color: var(--cp-accent-fg);
  overflow: hidden;
  position: relative;
  transition: transform 160ms ease, box-shadow 160ms ease;
}
div.stDownloadButton > button[kind="primary"]:hover {
  background: var(--cp-accent-hover);
  border-color: var(--cp-accent-hover);
  color: var(--cp-accent-fg);
  transform: translateY(-2px);
}
div.stDownloadButton > button[kind="primary"]::after {
  animation: download-arrow 1.4s ease-in-out infinite;
  content: "↓";
  display: inline-block;
  font-size: 1.05rem;
  margin-left: 8px;
}
@keyframes download-pulse {
  0%, 100% {
    box-shadow: 0 0 0 0 var(--cp-highlight);
  }
  50% {
    box-shadow: 0 0 0 7px var(--cp-highlight);
  }
}
@keyframes download-arrow {
  0%, 100% {
    transform: translateY(-1px);
  }
  50% {
    transform: translateY(3px);
  }
}
@media (prefers-reduced-motion: reduce) {
  .ready-card,
  .ready-card::after,
  div.stDownloadButton > button[kind="primary"],
  div.stDownloadButton > button[kind="primary"]::after {
    animation: none;
  }
  div.stDownloadButton > button[kind="primary"] {
    transition: none;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


def render_finding(finding: dict[str, object]) -> None:
    location = f"Cell {finding['cell_index']}"
    if finding.get("line_start"):
        location += f", line {finding['line_start']}"
    st.markdown(
        f"""
<div class="finding">
  <div class="finding-title">{finding["title"]}</div>
  <div class="finding-meta">{finding["rule_id"]} · {location} · {finding["confidence"]}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write(finding["message"])
    if finding.get("action"):
        st.caption(f"Action: {finding['action']}")
    if finding.get("original_excerpt") or finding.get("replacement_excerpt"):
        with st.expander("Show redacted change"):
            if finding.get("original_excerpt"):
                st.caption("Before")
                st.code(str(finding["original_excerpt"]), language="python")
            if finding.get("replacement_excerpt"):
                st.caption("After")
                st.code(str(finding["replacement_excerpt"]), language="python")


st.markdown(
    """
<section class="hero">
  <div class="eyebrow">Fabric Data Agent · Assistants → Responses</div>
  <div class="brand-row">
    <div class="wordmark">RANCH</div>
    <div class="beta-stamp">BETA</div>
  </div>
  <h1><span class="ranch-initial">R</span>esponses
  <span class="ranch-initial">A</span>PI
  <span class="ranch-initial">N</span>otebook
  <span class="ranch-initial">C</span>onversion
  <span class="ranch-initial">H</span>elper</h1>
  <p class="tagline">Putting old threads out to pasture—without trampling the rest of your notebook.</p>
  <p>Upload a notebook. RANCH changes only verified Assistants API patterns, shows
  its work, and hands back a migrated copy plus a review report.</p>
  <div class="trust-row">
    <span class="trust-pill">No code runs here</span>
    <span class="trust-pill">No mystery rewrites</span>
    <span class="trust-pill">Original stays put</span>
    <span class="trust-pill">Ambiguous code gets a flag, not a guess</span>
  </div>
  <div class="source-row">
    <span class="source-label">Why now</span>
    <a href="https://community.fabric.microsoft.com/t5/Fabric-Updates-Blog/Prepare-your-Fabric-Data-Agent-integrations-for-Assistants-API/ba-p/5314634" target="_blank" rel="noopener noreferrer">Microsoft retirement announcement</a>
    <a href="https://learn.microsoft.com/fabric/data-science/fabric-data-agent-sdk" target="_blank" rel="noopener noreferrer">Migration guidance</a>
    <a href="https://github.com/microsoft/fabric-samples/tree/main/docs-samples/data-science/data-agent-sdk/responses-api" target="_blank" rel="noopener noreferrer">Official samples</a>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<section class="beta-warning">
  <div class="warning-mark">!</div>
  <div>
    <div class="warning-title">BETA means bring a backup.</div>
    <div class="warning-copy">Keep the current notebook. Test the migrated copy in
    a non-production Fabric workspace and compare representative results before
    deleting, overwriting, or redirecting anything. Migration confidence is not
    runtime proof.</div>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Choose one Jupyter notebook",
    type=["ipynb"],
    accept_multiple_files=False,
    help="Maximum 10 MB and 1,000 cells. The notebook is processed in memory.",
)
st.markdown(
    '<div class="privacy">The app parses notebook JSON but never executes a cell. '
    "Migration confidence measures rule coverage and review risk—not whether the "
    "notebook will run successfully against your Fabric environment.</div>",
    unsafe_allow_html=True,
)

if uploaded is None:
    st.info("Upload an .ipynb notebook to begin the assessment.")
    st.stop()

try:
    result = MigrationEngine().migrate(uploaded.getvalue(), uploaded.name)
except NotebookValidationError as exc:
    st.error(f"Invalid notebook: {exc}")
    st.stop()
except Exception as exc:
    st.error(
        "Migration stopped safely because the notebook triggered an internal "
        f"consistency check: {exc}"
    )
    st.stop()

report = result.report.to_dict()
score = report["score"]
summary = report["summary"]
status_label = str(report["status"]).replace("_", " ").title()

st.markdown(
    f'<div class="status-banner"><strong>{status_label}</strong> · '
    f'{summary["cells_changed"]} of {summary["cells_scanned"]} cells changed · '
    f'{summary["manual_actions"]} manual actions</div>',
    unsafe_allow_html=True,
)

score_col, metrics_col = st.columns([1, 2.2], gap="large")
with score_col:
    st.markdown(
        f"""
<div class="score-card">
  <div class="score-value">{score["overall"]}</div>
  <div class="score-label">Migration confidence / 100</div>
  <div class="score-label">Runtime correctness: {score["runtime_correctness"]}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with metrics_col:
    metric_columns = st.columns(3)
    metric_columns[0].metric("Automatic coverage", f'{score["automatic_coverage"]}%')
    metric_columns[1].metric(
        "Transformation confidence", f'{score["transformation_confidence"]}%'
    )
    metric_columns[2].metric(
        "Preservation confidence", f'{score["preservation_confidence"]}%'
    )
    st.progress(score["overall"] / 100)
    st.caption(
        "Score combines verified rule coverage, unresolved review risk, and "
        "structural preservation. It does not prove service behavior."
    )

st.markdown(
    f"""
<section class="ready-card" aria-live="polite">
  <div class="ready-icon">✓</div>
  <div class="ready-copy">
    <div class="ready-title">Your migrated notebook is ready for review.</div>
    <div class="ready-detail">{summary["cells_changed"]} cells updated · Keep the
    original, then test this copy in a non-production workspace.</div>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

download_notebook, download_report = st.columns(2)
with download_notebook:
    st.download_button(
        "Download migrated notebook",
        data=result.notebook_bytes,
        file_name=result.output_filename,
        mime="application/x-ipynb+json",
        type="primary",
        use_container_width=True,
    )
with download_report:
    st.download_button(
        "Download migration report",
        data=json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name=result.report_filename,
        mime="application/json",
        use_container_width=True,
    )

overview_tab, review_tab, mapping_tab = st.tabs(
    ["Overview & applied changes", "Review & test", "SDK mapping"]
)

with overview_tab:
    columns = st.columns(4)
    columns[0].metric("Cells scanned", summary["cells_scanned"])
    columns[1].metric("Cells changed", summary["cells_changed"])
    columns[2].metric("Rules applied", summary["rules_applied"])
    columns[3].metric("Warnings", summary["warnings"])
    if report["manual_actions"]:
        st.warning(
            "The generated notebook is downloadable, but unresolved code remains "
            "unchanged and must be reviewed."
        )
    elif report["warnings"]:
        st.warning(
            "No unsafe rewrite was applied, but the assessment found runtime or "
            "configuration items that require confirmation."
        )
    elif report["status"] == "no_migration_needed":
        st.success(
            "No Assistants query-plane migration was detected. Management-plane "
            "operations were intentionally preserved."
        )
    else:
        st.success("All detected patterns were handled by verified migration rules.")
    st.subheader("Applied changes")
    if not report["changes"]:
        st.caption("No automatic changes were applied.")
    for finding in report["changes"]:
        render_finding(finding)

with review_tab:
    if report["manual_actions"]:
        st.subheader("Manual findings")
        for finding in report["manual_actions"]:
            render_finding(finding)
    if report["warnings"]:
        st.subheader("Warnings")
        for finding in report["warnings"]:
            render_finding(finding)
    st.subheader("What you should test and confirm")
    for index, item in enumerate(report["user_test_checklist"], start=1):
        st.checkbox(item, key=f"check-{index}")

with mapping_tab:
    mapping = report["sdk_mapping"]
    st.write(
        f'**Observed release:** {mapping["observed_current_release"]}  \n'
        f'**Recommended floor:** {mapping["minimum_recommended_release"]}  \n'
        f'**Mapping last verified:** {mapping["last_verified_utc"]}'
    )
    st.caption(mapping["minimum_reason"])
    st.dataframe(mapping["rules"], use_container_width=True, hide_index=True)
    st.subheader("Evidence")
    for source in mapping["sources"]:
        st.markdown(f"- [{source}]({source})")
