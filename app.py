"""
VeloTech LifeCourse-360 - Multi-Agent Genomic, Household & Bio-Signal Precision Screening Ecosystem.
Team: VeloTech Health AI · Future Health Digital Innovation Challenge · Challenge 04.

Run:  streamlit run app.py
Everything works offline at $0 (deterministic engine + synthetic data).
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from data.synthetic_generator import (
    DATA_DOMAINS, DOMAIN_KEYS, DOMAIN_LABELS, LANGUAGES, CHANNELS, get_persona, get_personas, graph_layout_positions,
    persona_household_graph, persona_options, CUSTOM_ID, set_custom_data,
)
from src import config as cfg
from src import opencv_biosignal as bio
from src import own_data
from src import tutorial
from src.doh_guidelines_kb import DISCLAIMER, get_kb
from src.foundry_agents import (
    PROFILES, WORKFLOW_STATES, Runtime, compliance_matrix, compose_outreach, run_workflow, fairness_snapshot,
)
from src.ml_conformal_engine import FALLBACK_THRESHOLD, get_engine
from src.mcp_server import MCPServerLite
from src.safeguards_evaluation import (
    ASSUMPTIONS, METRIC_LABELS, SUPPRESSED, benchmark_scoring, build_proof_metrics, cohort_table, consent_sweep,
    display_table, equalize_fnr_thresholds, fairlearn_crosscheck, parity_summary, simulate_program, tech_docs,
)

_ICON = Path(__file__).parent / "assets" / "favicon.png"
st.set_page_config(page_title="VeloTech LifeCourse-360 · Future Health", page_icon=Image.open(_ICON) if _ICON.exists() else None,
                   layout="wide", initial_sidebar_state="expanded")
C = cfg.COLORS
ss = st.session_state

# ---- theme: light / dark (remembered in the URL as ?theme=dark) ----
if "dark_mode" not in ss:
    ss["dark_mode"] = str(st.query_params.get("theme", "light")).lower() == "dark"
    ss["dark_mode_sb"] = ss["dark_mode"]
if "palette" not in ss:
    _q = str(st.query_params.get("palette", "Ocean"))
    ss["palette"] = _q if _q in cfg.PALETTE_NAMES else "Ocean"
    ss["palette_sb"] = ss["palette"]
cfg.set_theme("dark" if ss["dark_mode"] else "light", ss["palette"])


def _remember_theme() -> None:
    st.query_params["theme"] = "dark" if ss["dark_mode"] else "light"
    st.query_params["palette"] = ss["palette"]


def _theme_from_main() -> None:
    ss["dark_mode_sb"] = ss["dark_mode"]
    _remember_theme()


def _theme_from_sidebar() -> None:
    ss["dark_mode"] = ss["dark_mode_sb"]
    _remember_theme()


def _palette_from_main() -> None:
    ss["palette_sb"] = ss["palette"]
    _remember_theme()


def _palette_from_sidebar() -> None:
    ss["palette"] = ss["palette_sb"]
    _remember_theme()


def wide_columns(spec):
    """st.columns that stack (instead of squeezing) when the content area is narrow."""
    cols = st.columns(spec)
    cols[0].markdown('<span class="vt-wide2"></span>', unsafe_allow_html=True)
    return cols


def _open_tour() -> None:
    ss["tour_open"] = True
    ss["tour_step"] = 0


def _tour_nav(delta: int) -> None:
    ss["tour_step"] = int(min(max(ss.get("tour_step", 0) + delta, 0), len(tutorial.TOUR) - 1))


def _open_custom() -> None:
    ss["custom_dialog_open"] = True


def _persona_changed() -> None:
    """Choosing 'My own data' for the first time opens the entry form straight away."""
    if ss.get("persona") == CUSTOM_ID and not ss.get("custom_raw"):
        ss["custom_dialog_open"] = True


# --------------------------------------------------------------------------------------
# helpers: version-tolerant Streamlit calls
# --------------------------------------------------------------------------------------
def show_fig(fig, key: str | None = None) -> None:
    opts = {"displayModeBar": False}
    m = fig.layout.margin  # keep titles readable and clear of the legend in both themes
    fig.update_layout(title_font=dict(color=C["heading"], size=15), title_y=0.97, title_yanchor="top", title_x=0.0,
                      margin=dict(l=m.l if m.l is not None else 10, r=m.r if m.r is not None else 10, b=m.b if m.b is not None else 10, t=max(m.t or 0, 78)))
    try:
        st.plotly_chart(fig, width="stretch", key=key, config=opts)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key, config=opts)


def show_df(df: pd.DataFrame, **kw) -> None:
    """Themed HTML table (fully readable in light and dark mode)."""
    st.markdown(cfg.html_table(df), unsafe_allow_html=True)


def show_img(data, caption: str | None = None, width: int | None = None) -> None:
    if width:
        st.image(data, caption=caption, width=width)
        return
    try:
        st.image(data, caption=caption, width="stretch")
    except TypeError:
        st.image(data, caption=caption, use_container_width=True)


def md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def chips(items: list[str], kind: str = "info") -> str:
    return "".join(cfg.chip(i, kind) for i in items)


# --------------------------------------------------------------------------------------
# cached resources
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training calibrated risk models on synthetic Abu Dhabi data…")
def engine():
    return get_engine()


@st.cache_data(show_spinner=False)
def scored_population(_n: int = 0):
    return engine().score_population(engine().pop)


@st.cache_data(show_spinner=False)
def cached_sim(capacity: float):
    return simulate_program(scored_population(), capacity_share=capacity)


@st.cache_data(show_spinner=False)
def cached_throughput():
    return benchmark_scoring(engine())


@st.cache_resource(show_spinner="Evaluating model profiles…")
def cached_router_frame():
    return Runtime.create(engine()).router.evaluate_all()


@st.cache_data(show_spinner=False)
def synth_scan(hr: float, tone: str, noise: float, motion: float, seed: int):
    frames = bio.generate_session(hr_bpm=hr, tone=tone, noise_sd=noise, motion_px=motion, seed=seed)
    r = bio.extract_rppg(frames, 30)
    return r, bio.to_png_bytes(bio.annotate_roi(frames[0], r["roi"]))


eng = engine()
md(cfg.global_css())

# --------------------------------------------------------------------------------------
# SIDEBAR: persona, granular consent (synced with workspace), language/channel, safeguards
# --------------------------------------------------------------------------------------
# Changes made in the "My own data" dialog are applied here, before the sidebar widgets exist.
if "_pending_persona" in ss:
    ss["persona"] = ss.pop("_pending_persona")
if ss.pop("_custom_changed", False):
    for _key in [k for k in list(ss.keys()) if isinstance(k, str) and k.startswith((f"lang_{CUSTOM_ID}", f"chan_{CUSTOM_ID}", f"c_{CUSTOM_ID}_", f"sb_{CUSTOM_ID}_", f"ws_{CUSTOM_ID}_"))]:
        del ss[_key]
set_custom_data(ss.get("custom_raw") or {"who": "Adult"})  # empty entry = 0% data, so the safe fallback shows until the form is filled in
personas = {p["id"]: p for p in get_personas()}
options = persona_options()


def _sync(pid: str, dom: str, src: str) -> None:
    v = ss[f"{src}_{pid}_{dom}"]
    ss[f"c_{pid}_{dom}"] = v
    ss[f"{'ws' if src == 'sb' else 'sb'}_{pid}_{dom}"] = v


def _reset_consent(pid: str, only_ehr: bool = False) -> None:
    p = personas[pid]
    for d in DOMAIN_KEYS:
        v = (d == "ehr_history") if only_ehr else (d in p["consent_default"])
        for pre in ("c", "sb", "ws"):
            ss[f"{pre}_{pid}_{d}"] = v


with st.sidebar:
    md(cfg.build_sidebar_brand())
    st.toggle("Dark mode", key="dark_mode_sb", on_change=_theme_from_sidebar)
    st.selectbox("Color theme", cfg.PALETTE_NAMES, key="palette_sb", on_change=_palette_from_sidebar)
    st.button("Guided tour", key="tour_btn", type="primary", on_click=_open_tour)
    pid = st.selectbox("Choose a person", list(options), format_func=lambda k: options[k], key="persona", on_change=_persona_changed,
                       help="Eight sample people are built in (synthetic). Pick 'My own data' to enter your own details and get a result for them.")
    person = personas[pid]
    if pid == CUSTOM_ID:
        st.button("Edit my data" if ss.get("custom_raw") else "Enter my data", key="own_btn", on_click=_open_custom, type="primary" if not ss.get("custom_raw") else "secondary")
        st.caption("Your entry is used only in this session and is never saved.")
    else:
        st.caption("Sample people are synthetic. Want a result for yourself? Choose 'My own data'.")
    for d in DOMAIN_KEYS:
        if f"c_{pid}_{d}" not in ss:
            for pre in ("c", "sb", "ws"):
                ss[f"{pre}_{pid}_{d}"] = d in person["consent_default"]
    st.markdown("**Granular data consent**")
    st.caption("Untick a domain and the risk, interval and pathway recalculate instantly.")
    for dom in DATA_DOMAINS:
        avail = dom["key"] in person["available_domains"]
        st.checkbox(dom["label"] + ("" if avail else "  (no record on file)"), key=f"sb_{pid}_{dom['key']}", on_change=_sync, args=(pid, dom["key"], "sb"))
    st.button("Restore defaults", on_click=_reset_consent, args=(pid,), key="reset_consent")
    st.button("Withdraw consent (demo)", on_click=_reset_consent, args=(pid, True), key="revoke_consent", help="Keeps only EHR history so you can see the below-40% fallback")
    st.divider()
    lang_default = person["language"] if person["language"] in LANGUAGES else "English"
    language = st.selectbox("Outreach language", LANGUAGES, index=LANGUAGES.index(lang_default), key=f"lang_{pid}")
    channel = st.selectbox("Outreach channel", CHANNELS, index=CHANNELS.index(person["channel"]), key=f"chan_{pid}")
    st.divider()
    k = st.slider("k-anonymity threshold (k)", 5, 25, 10, help="Cohort cells with n < k are suppressed everywhere.")
    threshold = st.slider("Outreach recommendation threshold", 0.10, 0.60, 0.30, 0.01, help="Risk at or above this value is recommended for outreach.")
    biosignal_on = st.toggle("Optional OpenCV bio-signal pre-screener", value=False, help="Contactless pulse (adults) or milestone check (children). Wellness motivation only.")
    st.divider()
    md(cfg.boundary_banner())
    md('<div class="vt-sb-label">Project team</div>' + cfg.build_sidebar_contact())

consent = {d for d in DOMAIN_KEYS if ss.get(f"c_{pid}_{d}", False)}

# --------------------------------------------------------------------------------------
# RUN THE 4-AGENT WORKFLOW FOR THE SELECTED PERSON
# --------------------------------------------------------------------------------------
rt = Runtime.create(eng)
bio_state = ss.get("biosignal") if (biosignal_on and "wearable_biosignal" in consent) else None
res = run_workflow(pid, consent, language, channel, k, bio_state, runtime=rt)
risk, pathway = res.risk, res.pathway
scored = scored_population()
capacity = ss.get("capacity", 0.25)
sim = cached_sim(capacity)
fair = fairness_snapshot(eng, k)
region_table = cohort_table(scored, ["region"], k=k, threshold=threshold)
parity = parity_summary(region_table)
fine_table = cohort_table(scored, ["region", "nationality_group", "occupation_group", "age_band"], k=k, threshold=threshold)
n_suppressed = int(fine_table["suppressed"].sum())

outcome_status = [
    "warning" if risk["fallback"] else "verified",
    "verified" if pathway["items"] else "pending",
    "verified" if len(consent) >= 1 else "warning",
    "verified" if fair["status"] == "ok" else "warning",
    "verified",
]

# --------------------------------------------------------------------------------------
# PAGE SECTIONS 1-3
# --------------------------------------------------------------------------------------
top_l, top_m, top_r = st.columns([8, 2.2, 1.7], vertical_alignment="center")
with top_l:
    md(cfg.build_profile_bar())
with top_m:
    st.selectbox("Color theme", cfg.PALETTE_NAMES, key="palette", on_change=_palette_from_main)
with top_r:
    st.toggle("Dark mode", key="dark_mode", on_change=_theme_from_main)


@st.dialog("Welcome to VeloTech LifeCourse-360", width="large")
def tour_dialog() -> None:
    step = ss.get("tour_step", 0)
    md(tutorial.progress_html(step))
    md(tutorial.step_html(step))
    md('<div style="height:14px"></div>')
    b1, b2, b3 = st.columns([1, 1, 1.3])
    if step > 0:
        b1.button("Back", key="tour_back", on_click=_tour_nav, args=(-1,))
    if step < len(tutorial.TOUR) - 1:
        b2.button("Skip tour", key="tour_skip")
        b3.button("Next", key="tour_next", type="primary", on_click=_tour_nav, args=(1,))
        if ss.get("tour_skip"):
            ss["tour_open"] = False
            st.rerun()
    else:
        if b3.button("Finish", key="tour_finish", type="primary"):
            ss["tour_open"] = False
            st.rerun()


@st.dialog("Use your own data", width="large")
def custom_dialog() -> None:
    md('<div class="vt-tour-intro">Enter the details of the person you want to check. Fill in what you know: anything left blank is treated as '
       'missing, not as zero, so the result stays honest. The more you enter, the narrower the uncertainty range. '
       'Nothing is saved; your entry lives only in this browser session.</div>')
    data, completeness, avail = own_data.render_form(ss.get("custom_raw"))
    box = "vt-tour-see" if completeness >= FALLBACK_THRESHOLD else "vt-fallback"
    md(f'<div class="{box}" style="margin:14px 0 12px 0">{own_data.completeness_notice(completeness, avail)}</div>')
    b1, b2, b3 = st.columns([1.4, 1, 1])
    if b1.button("Save and see my result", key="own_save", type="primary"):
        ss["custom_raw"] = data
        ss["_pending_persona"] = CUSTOM_ID
        ss["_custom_changed"] = True
        st.rerun()
    if b2.button("Clear my data", key="own_clear"):
        ss.pop("custom_raw", None)
        ss["own_ver"] = ss.get("own_ver", 0) + 1
        ss["_pending_persona"] = CUSTOM_ID
        ss["_custom_changed"] = True
        st.rerun()
    if b3.button("Cancel", key="own_cancel"):
        st.rerun()


ss.setdefault("tour_step", 0)
_tour_shown = False
if ss.get("tour_open", True):
    ss["tour_open"] = False
    _tour_shown = True
    tour_dialog()
if ss.get("custom_dialog_open") and not _tour_shown:
    ss["custom_dialog_open"] = False
    custom_dialog()

md(cfg.build_hero())
md(cfg.build_intro())
md(cfg.build_focus_explorer(outcome_status))

# --------------------------------------------------------------------------------------
# LIVE WORKSPACE
# --------------------------------------------------------------------------------------
md('<div id="vt-workspace" style="height:64px"></div>')
md(cfg.section_header("Live workspace", f'{person["name"]} · {person["age_label"]} · {"Well-Child Visit" if person["segment"] == "Well-Child" else "IFHAS"} pathway'))
md(f'<div class="vt-body" style="margin-bottom:8px">{cfg._esc(person["story"])}</div>' + chips(person["tags"], "info"))
md(cfg.boundary_banner())

tabs = st.tabs(["1 · Risk & Uncertainty", "2 · DoH Pathways & Household", "3 · Engagement & OpenCV", "4 · Population Safeguards",
                "5 · Evaluation & Calibration", "6 · Foundry Agents & MCP"])

# ======================================================================================
# TAB 1 - OUTCOME 1
# ======================================================================================
with tabs[0]:
    md(cfg.section_header("Outcome 1", "Transparent risk and care-gap identification", "Calibrated score, SHAP factor attribution and a conformal 95% interval, using only the data domains this person authorized."))
    md(cfg.provenance_badge([(DOMAIN_LABELS[d], d in consent, d in person["available_domains"]) for d in DOMAIN_KEYS]))
    st.write("")
    left, right = wide_columns([1, 1.25])
    with left:
        if risk["fallback"]:
            md(cfg.fallback_banner(risk["completeness"], FALLBACK_THRESHOLD))
            md(cfg.callout("What happens instead", "The baseline DoH age and gender schedule is shown in tab 2. Authorize more data domains in the sidebar to unlock a personalised, explainable score."))
        else:
            p = risk["risk"]
            md(cfg._c(f"""
<div class="vt-callout" style="padding:22px">
<div class="lbl">Calibrated unmet-need risk</div>
<div style="font-size:54px;font-weight:300;color:{C['heading']};line-height:1.05">{p:.1%}</div>
<div style="color:{C['amber']};font-weight:600;font-size:15px;margin:6px 0 10px 0">± {risk['half_width']:.1%} &nbsp;[95% CI: {risk['ci_low']:.1%} – {risk['ci_high']:.1%}]</div>
<div class="txt">Conformal prediction set: <b>{cfg._esc(' / '.join(risk['conformal_set']) or 'empty (abstain)')}</b><br>Interval band: <b>{risk['band']}</b> data completeness</div>
</div>"""))
            md(cfg._c(f'''<div class="vt-statrow">
<div class="vt-stat"><div class="k">Outreach tier</div><div class="v">{risk["tier"]}</div></div>
<div class="vt-stat"><div class="k">Care-gap score</div><div class="v">{risk["care_gap_score"]:.0f} / 100</div></div>
<div class="vt-stat"><div class="k">Months overdue</div><div class="v">{(person["months_overdue"] or 0):.0f}</div></div>
</div>'''))
            if risk.get("red_flags"):
                md(cfg.chip("Clinical red flags escalate urgency", "warn") + chips(risk["red_flags"], "warn"))
            lo, hi = risk["ci_low"] * 100, risk["ci_high"] * 100
            f = go.Figure()
            f.add_shape(type="rect", x0=0, x1=100, y0=-0.25, y1=0.25, fillcolor=C["callout"], line=dict(width=0), layer="below")
            f.add_shape(type="rect", x0=lo, x1=hi, y0=-0.25, y1=0.25, fillcolor=cfg.rgba(C["amber"], 0.28), line=dict(color=C["amber"], width=1.5), layer="below")
            f.add_trace(go.Scatter(x=[risk["risk"] * 100], y=[0], mode="markers", marker=dict(size=16, color=C["series"], line=dict(color="white", width=2)), name="Calibrated risk", hovertemplate="%{x:.1f}%<extra></extra>"))
            for t, lab in ((30, "Priority"), (60, "Urgent")):
                f.add_vline(x=t, line=dict(color=C["hairline"], dash="dot"), annotation_text=lab, annotation_position="top")
            f.update_yaxes(visible=False, range=[-0.6, 0.6])
            f.update_xaxes(range=[0, 100], ticksuffix="%")
            cfg.style_figure(f, 170, legend=False)
            f.update_layout(title="Risk with 95% conformal interval")
            show_fig(f, "ci_fig")
    with right:
        if not risk["fallback"]:
            rows = [r for r in risk["shap"] if abs(r["contribution_pp"]) >= 0.05][:10][::-1]
            colors = [(C["border_strong"] if r["imputed"] else (C["amber"] if r["contribution_pp"] > 0 else C["emerald"])) for r in rows]
            labels = [f'{r["label"]}' + (f' = {r["value"]:.1f}' if r["value"] is not None else " (no data)") for r in rows]
            labels = ["<br>".join(textwrap.wrap(l, 30)) for l in labels]
            vals = [r["contribution_pp"] for r in rows]
            pad = max(1.5, 0.28 * (max(vals + [0]) - min(vals + [0])))
            f = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors, cliponaxis=False,
                                 text=[f'{v:+.1f} pp' for v in vals], textposition="outside", textfont=dict(color=C["ink"]),
                                 hovertemplate="%{y}<br>%{x:+.2f} percentage points<extra></extra>"))
            f.add_vline(x=0, line=dict(color=C["ink"], width=1))
            cfg.style_figure(f, 420, legend=False)
            f.update_layout(title="SHAP attribution (pp vs. baseline)", margin=dict(l=10, r=30, t=44, b=10))
            f.update_xaxes(zeroline=False, range=[min(vals + [0]) - pad * 1.4, max(vals + [0]) + pad], title="Percentage points vs. baseline")
            f.update_yaxes(showgrid=False)
            show_fig(f, "shap_fig")
            st.caption(f'Amber bars raise the flag, green bars lower it, grey bars had no data. Baseline population rate for this pathway: {risk["baseline_rate"]:.1%}. Contributions are exact linear SHAP values in log-odds, shown as marginal percentage points.')
        else:
            md(cfg.callout("Explanation withheld", "With fewer than 40% of data domains authorized, no attribution is shown because it would explain an estimate we have chosen not to make."))
    with st.expander("HL7 FHIR R4 payloads fetched through the MCP server (consent-filtered)", expanded=False):
        t1, t2, t3, t4 = st.tabs(["Patient", "Observation Bundle", "CarePlan (draft proposal)", "EGP lookup"])
        t1.markdown(cfg.json_block(res.ingest["patient"]), unsafe_allow_html=True)
        t2.markdown(cfg.json_block(res.ingest["observations"]), unsafe_allow_html=True)
        t3.markdown(cfg.json_block(pathway["careplan"]), unsafe_allow_html=True)
        t4.markdown(cfg.json_block(res.ingest["egp"]), unsafe_allow_html=True)

# ======================================================================================
# TAB 2 - OUTCOME 2
# ======================================================================================
with tabs[1]:
    md(cfg.section_header("Outcome 2", "Personalized DoH pathway and household synchronization", "Grounded in Well-Child Visit and IFHAS guidance through Foundry IQ hybrid RAG. Timing, fasting, recall and follow-up come straight from the cited guideline."))
    md(cfg.boundary_banner())
    if pathway["baseline_only"]:
        md(cfg.fallback_banner(risk["completeness"], FALLBACK_THRESHOLD))
        md(cfg.callout("Baseline schedule", "Showing the age and gender schedule only. No individual risk extrapolation was used."))
    rows = []
    for i in pathway["items"]:
        kind = {"Urgent": "warn", "Priority": "info", "Routine": "ok"}[i["priority"]]
        e = cfg._esc
        rows.append(f"""<div class="vt-pathrow"><div><div class="h">{e(i['title'])} {cfg.chip(i['priority'], kind)}</div><div class="m">{e(i['rationale'])}</div>
<div class="m"><i>{e(i['citation']['id'])} · {e(i['citation']['snippet'])}</i></div></div>
<div class="m"><b>When:</b> {e(i['window'])}<br><b>Before the visit:</b> {e(i['fasting'])}</div>
<div class="m"><b>Recall:</b> {e(i['recall'])}<br><b>After the visit:</b> {e(i['followup'])}</div></div>""")
    md(cfg._c("".join(rows)))
    cA, cB = wide_columns(2)
    with cA:
        md(cfg.callout("Pre-visit instructions", "<br>".join(cfg._esc(r) for r in pathway["previsit"])))
    with cB:
        md(cfg.callout("Recall intervals", "<br>".join(cfg._esc(r) for r in pathway["recall"]) or "Per the schedule above."))
    st.markdown("#### Household care-gap synchronization")
    g = persona_household_graph(person["household_id"])
    pos = graph_layout_positions(g)
    ex, ey = [], []
    for a, b in g.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    nodes = list(g.nodes(data=True))
    f = go.Figure()
    f.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=C["hairline"], width=2), hoverinfo="skip", showlegend=False))
    f.add_trace(go.Scatter(
        x=[pos[n][0] for n, _ in nodes], y=[pos[n][1] for n, _ in nodes], mode="markers+text",
        text=[f'{d["name"]}<br>{("Well-Child" if d["segment"] == "Well-Child" else "IFHAS")} · {d["months_overdue"]:.0f} mo overdue' for _, d in nodes], textposition="bottom center",
        marker=dict(size=[34 if d["segment"] != "Well-Child" else 26 for _, d in nodes], color=[C["amber"] if d["months_overdue"] > 0 else C["emerald"] for _, d in nodes], line=dict(color="white", width=2)),
        hovertemplate="%{text}<extra></extra>", showlegend=False))
    f.update_xaxes(visible=False, range=[0, 1])
    f.update_yaxes(visible=False, range=[-0.7, 1.4])
    cfg.style_figure(f, 300, legend=False)
    f.update_layout(title=f'Household graph · {person["household_id"]} (amber = overdue)')
    show_fig(f, "hh_graph")
    bundle = pathway["bundle"]
    if bundle:
        md(cfg._c(f"""<div class="vt-callout" style="border:1px solid {C['emerald']}"><div class="lbl" style="color:{C['emerald']}">{cfg._esc(bundle['bundle_name'])}</div>
<div class="txt">{' + '.join(cfg._esc(m['name']) + ' (' + m['pathway'] + ', ' + format(m['months_overdue'], '.0f') + ' mo overdue)' for m in bundle['members'])}<br>
<b>Site:</b> {cfg._esc(bundle['site'])} · <b>Lead contact:</b> {cfg._esc(bundle['lead_contact'])} ({cfg._esc(bundle['preferred_language'])}, {cfg._esc(bundle['preferred_channel'])})<br>{cfg._esc(bundle['scheduling_note'])}</div></div>"""))
        m1, m2, m3 = st.columns(3)
        m1.metric("Separate trips", bundle["separate_trips"])
        m2.metric("Bundled trips", bundle["bundled_trips"])
        m3.metric("Trips saved", bundle["trips_saved"])
    else:
        md(cfg.callout("No synchronized appointment", "This household does not currently have both an overdue adult (IFHAS) and an overdue child (Well-Child). Try Fatima or Omar Al Mansoori, or Aisha and Yousef Al Hammadi."))
    st.markdown("#### Ask the approved DoH guidelines (Foundry IQ hybrid RAG)")
    q = st.text_input("Question", value="When is the M-CHAT-R screening done and what happens if the score is medium risk?" if person["segment"] == "Well-Child" else "Do I need to fast before the blood tests, and when should I book?", key=f"rag_q_{pid}")
    ans = get_kb().answer(q, person)
    md(cfg.callout("Grounded answer (extractive, cited)", cfg._esc(ans["answer"]) + f'<br><br>Groundedness: <b>{ans["groundedness"]["score"]:.0%}</b> of claims supported by cited passages.'))
    show_df(pd.DataFrame([{"Citation": c["id"], "Guideline": c["title"], "Source": c["source"], "Hybrid score": c["score"], "BM25": c["bm25"], "Cosine": c["cosine"]} for c in ans["citations"]]))
    st.caption("Guideline passages are illustrative demo summaries, not official DoH text. " + DISCLAIMER)

# ======================================================================================
# TAB 3 - OUTCOME 3
# ======================================================================================
with tabs[2]:
    md(cfg.section_header("Outcome 3", "Culturally appropriate engagement, consent choice and optional OpenCV pre-screener", "Language, channel and tone adapt to the person. Consent is a real choice: change it here or in the sidebar and everything recalculates."))
    cc1, cc2 = wide_columns([1, 1.4])
    with cc1:
        st.markdown("**Granular consent (synced with the sidebar)**")
        for dom in DATA_DOMAINS:
            st.checkbox(dom["label"], key=f"ws_{pid}_{dom['key']}", on_change=_sync, args=(pid, dom["key"], "ws"))
        st.metric("Authorized data completeness", f'{risk["completeness"]:.0%}')
        if risk["fallback"]:
            md(cfg.chip("AI risk blocked below 40%", "warn"))
        else:
            md(cfg.chip(f'Risk {risk["risk"]:.1%} recalculated live', "ok"))
    with cc2:
        msg = res.outreach["message"]
        if msg is None:
            md(cfg.callout("Dispatch held", "The responsible-AI gate blocked this outreach for human review."))
        else:
            md(cfg._c(f'<div class="vt-eyebrow">CulturaVoice · {cfg._esc(msg["language"])} · {cfg._esc(msg["channel"])}</div>'))
            if msg["title"]:
                st.markdown(f'**{msg["title"]}**')
            md(f'<div class="vt-msg {"vt-rtl" if msg["rtl"] else ""}">{cfg._esc(msg["body"])}</div>')
            md(chips(msg["culture_notes"], "info"))
            mc = res.message_check
            md(chips([f'{"Passed" if v else "Failed"}: {name}' for name, v in mc["checks"].items()], "ok" if mc["ok"] else "warn"))
            st.caption(f'Tone: {msg["tone"]} · {msg["chars"]} characters · message from vetted templates'
                       + (f' · clinician summary by {res.outreach["model"]}' if res.outreach.get("narrative") else ""))
    with st.expander("Compare every channel and language for this person"):
        ch_tabs = st.tabs(CHANNELS)
        for ch, tb in zip(CHANNELS, ch_tabs):
            with tb:
                for lg in ("English", "Arabic (Emirati)", "Arabic (Standard)", "Urdu"):
                    m = compose_outreach(person, pathway["items"], lg, ch, pathway["bundle"], fallback=pathway["baseline_only"])
                    st.markdown(f"**{lg}** ({m['chars']} chars)")
                    md(f'<div class="vt-msg {"vt-rtl" if m["rtl"] else ""}">{cfg._esc(m["body"])}</div>')
    st.divider()
    md(cfg.section_header("Optional", "OpenCV smartphone bio-signal & pediatric visual pre-screener", "Wellness motivation only. Never an input to the clinical score. " + bio.DISCLAIMER))
    if not biosignal_on:
        md(cfg.callout("Pre-screener is off", "Switch on 'Optional OpenCV bio-signal pre-screener' in the sidebar, and keep the bio-signal consent domain ticked, to try it."))
    elif "wearable_biosignal" not in consent:
        md(cfg.callout("Consent needed", "Tick 'Bio-signal pre-screener / wearable' in the consent list. Nothing runs without it."))
    elif person["segment"] == "Well-Child":
        items = bio.milestone_items(person["age_months"])
        st.markdown(f"**Visual milestone check** (age band: {bio.milestone_bracket(person['age_months'])} months). Parents answer from what they see at home.")
        t_pos = st.slider("Gaze-tracking stimulus: move the slider slowly and watch whether your child's eyes follow the dot", 0.0, 1.0, 0.0, 0.02, key=f"stim_{pid}")
        show_img(bio.render_tracking_stimulus(t_pos, dark=cfg.current_theme() == "dark"), width=560)
        answers = {}
        cols = st.columns(2)
        for n, it in enumerate(items):
            answers[it["id"]] = cols[n % 2].checkbox(it["text"], key=f"ms_{pid}_{it['id']}")
        sc = bio.score_milestones(answers, items)
        if sc["yes"] == 0:
            md(cfg.callout("Your summary appears here", "Tick everything you have seen your child do at home, then a short, friendly summary shows up. " + cfg._esc(sc["disclaimer"])))
        else:
            md(cfg._c(f'<div class="vt-callout"><div class="lbl">{sc["yes"]} of {sc["total"]} seen · {cfg._esc(sc["status"])}</div><div class="txt">{cfg._esc(sc["message"])}<br><i>{cfg._esc(sc["disclaimer"])}</i></div></div>'))
            ss["biosignal"] = {"quality": "Good" if sc["fraction"] >= 0.85 else "Fair"}
    else:
        mode = st.radio("Capture mode", ["Synthetic face video (always works)", "Webcam photo (st.camera_input)"], horizontal=True, key=f"bio_mode_{pid}")
        r, frame_png, note = None, None, ""
        if mode.startswith("Synthetic"):
            b1, b2, b3, b4 = st.columns(4)
            hr = b1.slider("True heart rate (bpm)", 48, 140, 74)
            tone = b2.selectbox("Skin tone", ["light", "medium", "deep"], index=1)
            noise = b3.slider("Sensor noise", 1.0, 8.0, 3.0, 0.5)
            motion = b4.slider("Head motion (px)", 0.0, 4.0, 1.5, 0.5)
            r, frame_png = synth_scan(float(hr), tone, float(noise), float(motion), 11)
            note = f"Synthetic face video with an embedded pulse of {hr} bpm."
        else:
            shot = st.camera_input("Take a well-lit photo of your face")
            if shot is not None:
                img = bio.decode_image(shot.getvalue())
                if img is not None:
                    st_info = bio.analyze_still_frame(img)
                    show_img(bio.to_png_bytes(st_info["annotated"]), "Real photo checks: green = face, orange = forehead ROI")
                    md(chips([f'Face found: {st_info["face_found"]}', f'Brightness {st_info["brightness"]}', f'Sharpness {st_info["sharpness"]}'], "ok" if st_info["face_found"] else "warn"))
                    for a in st_info["advice"]:
                        st.caption(a)
                    r = bio.extract_rppg(bio.frames_from_still(img), 30, simulated_pulse=True)
                    frame_png = bio.to_png_bytes(bio.annotate_roi(bio.frames_from_still(img, seconds=0.2)[0], r["roi"]))
                    note = "A single photo cannot contain a pulse: a simulated pulse is injected into your real face pixels so the full pipeline can be demonstrated. Real rPPG needs video."
            else:
                md(cfg.callout("Waiting for a photo", "Take a photo to run the pipeline, or switch to the synthetic video."))
        if r is not None:
            left, right = wide_columns([1, 2])
            with left:
                if frame_png:
                    show_img(frame_png, "Frame with forehead ROI")
            with right:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Heart rate", f'{r["hr_bpm"]} bpm')
                m2.metric("Signal quality", r["quality"], f'{r["snr_db"]} dB SNR')
                m3.metric("Pulse variability", f'{r["rmssd_ms"]} ms' if r["rmssd_ms"] else "n/a")
                m4.metric("Vascular stress proxy", f'{r["stress_index"]}/100')
            st.caption(note)
            f = go.Figure()
            f.add_trace(go.Scatter(x=r["t"], y=(r["raw"] - r["raw"].mean()), name="Raw green channel", line=dict(color=C["border_strong"], width=1)))
            f.add_trace(go.Scatter(x=r["t"], y=r["filtered"], name="Band-passed pulse (0.7–3 Hz)", line=dict(color=C["azure"], width=2)))
            f.add_trace(go.Scatter(x=r["t"][r["peaks"]], y=r["filtered"][r["peaks"]], mode="markers", name="Pulse peaks", marker=dict(color=C["amber"], size=7)))
            f.update_xaxes(title="Seconds")
            cfg.style_figure(f, 280)
            f.update_layout(title="rPPG waveform")
            show_fig(f, "rppg_wave")
            f2 = go.Figure(go.Scatter(x=r["freqs_hz"] * 60, y=r["power"], fill="tozeroy", line=dict(color=C["series"]), fillcolor=cfg.rgba(C["series"], 0.2)))
            f2.add_vline(x=r["hr_bpm"], line=dict(color=C["amber"], dash="dash"), annotation_text=f'{r["hr_bpm"]} bpm')
            f2.update_xaxes(title="Beats per minute")
            cfg.style_figure(f2, 220, legend=False)
            f2.update_layout(title="Pulse spectrum")
            show_fig(f2, "rppg_spec")
            nudge = bio.motivation_message(r["quality"], language)
            md(cfg._c(f'<div class="vt-callout"><div class="lbl">Motivational nudge ({cfg._esc(language)})</div><div class="txt {"vt-rtl" if language != "English" else ""}">{cfg._esc(nudge)}</div></div>'))
            ss["biosignal"] = {"quality": r["quality"], "hr_bpm": r["hr_bpm"]}
    if biosignal_on and "wearable_biosignal" in consent:
        with st.expander("Connected wearable / IoT hook (technology-neutral, hardware welcome)"):
            adapter = bio.SimulatedWearable(person)
            wdf = adapter.read_window(30)
            st.caption(f"Adapter: {adapter.name}. Implement `SensorAdapter.read_window()` for any BLE strap, smartwatch API or clinic kiosk.")
            f = go.Figure()
            f.add_trace(go.Scatter(x=wdf["minute"], y=wdf["hr_bpm"], name="Heart rate (bpm)", line=dict(color=C["series"])))
            f.add_trace(go.Scatter(x=wdf["minute"], y=wdf["rmssd_ms"], name="RMSSD (ms)", line=dict(color=C["azure"])))
            f.add_trace(go.Scatter(x=wdf["minute"], y=(wdf["skin_temp_c"] - 30) * 10, name="Skin temp (index)", line=dict(color=C["amber"])))
            cfg.style_figure(f, 260)
            show_fig(f, "wearable")
            ws = bio.wearable_summary(wdf)
            md(chips([f'HR {ws["mean_hr"]}', f'RMSSD {ws["mean_rmssd"]} ms', f'Min SpO2 {ws["min_spo2"]}%', f'Skin {ws["mean_skin_temp"]} C'], "info") + "<br>" + cfg._esc(ws["flag"]))

# ======================================================================================
# TAB 4 - OUTCOME 4
# ======================================================================================
with tabs[3]:
    md(cfg.section_header("Outcome 4", "Local population safeguards and low-data fallbacks", f"Reach, FPR, FNR, recommendation rate and uptake across Abu Dhabi segments. Cells with n < {k} are suppressed everywhere (k-anonymity)."))
    group_choices = ["region", "nationality_group", "occupation_group", "age_band", "segment", "sex"]
    groups = st.multiselect("Cohort dimensions", group_choices, default=["region", "nationality_group", "occupation_group"], key="groups")
    if not groups:
        groups = ["region"]
    use_adj = False
    if len(groups) == 1:
        use_adj = st.toggle("Apply FNR-parity threshold adjustment (post-processing mitigation)", value=False)
    thr_map = equalize_fnr_thresholds(scored, groups[0], threshold) if use_adj else None
    table = cohort_table(scored, groups, k=k, threshold=threshold, thresholds_by_group=thr_map)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cohort cells", len(table))
    m2.metric("Cells suppressed (n < k)", int(table["suppressed"].sum()))
    _fair_dims = [g_ for g_ in groups if g_ in ("region", "nationality_group", "sex")]
    _eo = parity_summary(cohort_table(scored, _fair_dims, k=k, threshold=threshold))["equalized_odds_difference"] if _fair_dims else None
    m3.metric("Equalised-odds gap", f"{_eo:.1%}" if _eo is not None else "Add region or nationality",
              help="Largest FPR or FNR difference across region, nationality and sex cohorts. Age and occupation differences reflect genuine differences in risk and are excluded from this headline.")
    m4.metric("People scored by AI", f'{(~scored["fallback"]).mean():.0%}')
    show_df(display_table(table, groups))
    if thr_map:
        st.caption("Per-group thresholds: " + ", ".join(f"{g}: {t:.2f}" for g, t in thr_map.items()))
    plot = table[~table["suppressed"] & (table["n"] >= 30)].copy()
    if len(plot):
        plot["cohort"] = plot[groups].astype(str).agg(" · ".join, axis=1)
        metric_pick = st.multiselect("Metrics to plot", list(METRIC_LABELS), default=["fpr", "fnr", "recommendation_rate"], format_func=lambda m: METRIC_LABELS[m])
        f = go.Figure()
        for mname, col in zip(metric_pick, [C["azure"], C["amber"], C["series"], C["emerald"], C["border_strong"], C["hairline"]]):
            f.add_trace(go.Bar(x=plot["cohort"], y=plot[mname] * 100, name=METRIC_LABELS[mname], marker_color=col))
        f.update_layout(barmode="group")
        f.update_yaxes(ticksuffix="%")
        cfg.style_figure(f, 360)
        f.update_layout(title="Parity view (suppressed cohorts hidden)")
        show_fig(f, "parity_fig")
    ps = parity_summary(table)
    md(chips([f'{METRIC_LABELS[m]}: gap {v["gap"]:.1%}' for m, v in ps["metrics"].items() if v["gap"] is not None], "info"))
    st.caption("Gaps across age bands or occupation partly reflect genuine differences in risk. Judge equity mainly on region, nationality and sex.")
    fl = fairlearn_crosscheck(scored, groups[0], threshold)
    st.caption("Fairlearn cross-check: " + (f'selection-rate difference {fl["selection_rate_difference"]:.3f}' if fl else "fairlearn not installed (built-in Fairlearn-compatible metrics used)."))
    st.divider()
    st.markdown("#### Insufficient-data fallback simulator")
    sweep = consent_sweep(eng, person)
    f = go.Figure()
    f.add_trace(go.Scatter(x=sweep["step"], y=sweep["completeness"] * 100, name="Authorized completeness", mode="lines+markers", line=dict(color=C["series"], width=3)))
    f.add_hline(y=40, line=dict(color=C["amber"], dash="dash"), annotation_text="40% fallback line")
    f.update_yaxes(ticksuffix="%", range=[0, 105])
    f.update_xaxes(title="Consent domains withdrawn (lowest weight first)", dtick=1)
    cfg.style_figure(f, 300)
    f.update_layout(title="Completeness as consent is withdrawn")
    show_fig(f, "sweep_fig")
    sw = sweep.copy()
    sw["Risk (95% CI)"] = [("withheld: fallback to DoH baseline" if r.fallback else f"{r.risk:.1%} [{r.ci_low:.1%} – {r.ci_high:.1%}]") for r in sw.itertuples()]
    sw["completeness"] = (sw["completeness"] * 100).round(0).astype(int).astype(str) + "%"
    show_df(sw[["step", "withdrawn", "completeness", "Risk (95% CI)"]].rename(columns={"step": "Step", "withdrawn": "Domain withdrawn", "completeness": "Completeness"}))
    fb = scored.groupby("region").agg(fallback=("fallback", "mean"), n=("fallback", "size")).reset_index()
    fb = fb[fb["n"] >= k]
    f = go.Figure(go.Bar(x=fb["region"], y=fb["fallback"] * 100, marker_color=C["amber"], text=[f"{v:.0%}" for v in fb["fallback"]], textposition="outside"))
    f.update_yaxes(ticksuffix="%")
    cfg.style_figure(f, 260, legend=False)
    f.update_layout(title="Residents on the baseline schedule, by region")
    show_fig(f, "fallback_region")

# ======================================================================================
# TAB 5 - OUTCOME 5
# ======================================================================================
with tabs[4]:
    md(cfg.section_header("Outcome 5", "Outcome-based evaluation and calibration suite", "Held-out test split of the synthetic population. Outcome numbers are modelled expected values using the explicit assumptions at the bottom of this tab."))
    em = eng.metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUC", f'{em["auc"]:.3f}')
    _db = em["brier_calibrated"] - em["brier_uncalibrated"]
    m2.metric("Brier score", f'{em["brier_calibrated"]:.3f}', f"{_db:+.4f} vs raw" if abs(_db) >= 5e-4 else "same as raw", delta_color="inverse" if abs(_db) >= 5e-4 else "off")
    m3.metric("Expected calibration error", f'{em["ece_calibrated"]:.3f}', f'{em["ece_calibrated"] - em["ece_uncalibrated"]:+.4f} vs raw', delta_color="inverse")
    m4.metric("95% interval coverage", f'{em["interval_coverage"]:.1%}')
    m5.metric("Conformal set coverage", f'{em["set_coverage"]:.1%}')
    cL, cR = wide_columns(2)
    with cL:
        f = go.Figure()
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C["hairline"], dash="dash"), name="Perfect calibration"))
        for key, name, col in (("reliability_uncalibrated", "Before recalibration", C["border_strong"]), ("reliability_calibrated", "After recalibration", C["series"])):
            r = em[key]
            f.add_trace(go.Scatter(x=r["mean_predicted"], y=r["observed_rate"], mode="lines+markers", name=name, line=dict(color=col, width=3), marker=dict(size=np.clip(r["n"] / 25, 6, 18))))
        f.update_xaxes(title="Mean predicted risk", range=[0, 1])
        f.update_yaxes(title="Observed rate", range=[0, 1])
        cfg.style_figure(f, 360)
        f.update_layout(title=f'Reliability · ECE {em["ece_calibrated"]:.3f} · Brier {em["brier_calibrated"]:.3f}')
        show_fig(f, "reliab")
    with cR:
        dca = em["decision_curve"]
        f = go.Figure()
        f.add_trace(go.Scatter(x=dca["threshold"], y=dca["model"], name="VeloTech model", line=dict(color=C["series"], width=3)))
        f.add_trace(go.Scatter(x=dca["threshold"], y=dca["treat_all"], name="Invite everyone", line=dict(color=C["amber"], dash="dot")))
        f.add_trace(go.Scatter(x=dca["threshold"], y=dca["treat_none"], name="Invite no one", line=dict(color=C["border_strong"], dash="dash")))
        f.add_vline(x=threshold, line=dict(color=C["azure"], dash="dot"), annotation_text="current threshold")
        f.update_xaxes(title="Threshold probability")
        f.update_yaxes(title="Net clinical benefit", range=[-0.03, max(0.12, float(dca["model"].max()) * 1.2)])
        cfg.style_figure(f, 360)
        f.update_layout(title="Decision curve (net clinical benefit)")
        show_fig(f, "dca")
    st.markdown("#### Programme outcomes at equal capacity: age-rule outreach vs VeloTech prioritization")
    st.slider("Outreach capacity (share of population that can be actively invited)", 0.10, 0.40, 0.25, 0.01, key="capacity")
    b, v = sim["baseline"], sim["velotech"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Early detections", f'{v["detections"]:.0f}', f'{sim["detection_lift"]:+.0%} vs age rule')
    m2.metric("Estimated QALYs gained", f'{v["qalys"]:.1f}', f'{sim["qaly_gain"]:+.1f}')
    m3.metric("Completed Well-Child Visits", f'{v["wcv_completed"]:.0f}', f'{v["wcv_completed"] - b["wcv_completed"]:+.0f}')
    m4.metric("Completed IFHAS screenings", f'{v["ifhas_completed"]:.0f}', f'{v["ifhas_completed"] - b["ifhas_completed"]:+.0f}')
    st.caption("VeloTech directs limited outreach capacity toward people with the greatest unmet need, so some visit counts can fall while early detections rise.")
    c1, c2 = wide_columns(2)
    with c1:
        f = go.Figure()
        f.add_trace(go.Bar(x=["Age-rule outreach", "VeloTech"], y=[b["timely_attended"], v["timely_attended"]], name="Timely attendance", marker_color=C["emerald"]))
        f.add_trace(go.Bar(x=["Age-rule outreach", "VeloTech"], y=[b["overdue_recalled"], v["overdue_recalled"]], name="Overdue recalled", marker_color=C["amber"]))
        f.update_layout(barmode="stack")
        cfg.style_figure(f, 320)
        f.update_layout(title="Timely vs overdue attendance")
        show_fig(f, "timely")
    with c2:
        stages = [s_["stage"] for s_ in sim["funnel_baseline"]][::-1]
        fb_, fv_ = [s_["count"] for s_ in sim["funnel_baseline"]][::-1], [s_["count"] for s_ in sim["funnel_velotech"]][::-1]
        f = go.Figure()
        f.add_trace(go.Bar(y=stages, x=fb_, orientation="h", name="Age-rule", marker_color=C["border_strong"], texttemplate="%{x:,.0f}", textposition="outside", cliponaxis=False, textfont=dict(color=C["ink"])))
        f.add_trace(go.Bar(y=stages, x=fv_, orientation="h", name="VeloTech + navigator", marker_color=C["series"], texttemplate="%{x:,.0f}", textposition="outside", cliponaxis=False, textfont=dict(color=C["ink"])))
        f.update_layout(barmode="group")
        f.update_xaxes(range=[0, max(fb_ + fv_) * 1.18])
        f.update_yaxes(showgrid=False)
        cfg.style_figure(f, 340)
        f.update_layout(title="Loop closure (expected people)")
        show_fig(f, "funnel")
    c3, c4 = wide_columns(2)
    with c3:
        st.metric("Outreach cost per early detection", f'AED {v["cost_per_detection"]:,.0f}', f'{v["cost_per_detection"] - b["cost_per_detection"]:+,.0f} vs age rule', delta_color="inverse")
    with c4:
        reg = pd.DataFrame(sim["reach_by_region"]).reset_index().rename(columns={"index": "region"})
        f = go.Figure()
        f.add_trace(go.Bar(x=reg["region"], y=reg["baseline"] * 100, name="Age-rule", marker_color=C["border_strong"]))
        f.add_trace(go.Bar(x=reg["region"], y=reg["velotech"] * 100, name="VeloTech", marker_color=C["series"]))
        f.update_yaxes(ticksuffix="%")
        f.update_layout(barmode="group")
        cfg.style_figure(f, 260)
        f.update_layout(title="Share of each region invited")
        show_fig(f, "reach_region")
    with st.expander("Explicit modelling assumptions (editable in src/safeguards_evaluation.py)"):
        show_df(pd.DataFrame([{"Assumption": k_, "Value": str(v_)} for k_, v_ in ASSUMPTIONS.items()]))

# ======================================================================================
# TAB 6 - AI-103
# ======================================================================================
with tabs[5]:
    md(cfg.section_header("Microsoft Foundry", "Multi-agent workflow and MCP trace inspector", f"Trace {res.trace_id[:12]}… · {len(res.spans)} spans · {len(res.mcp_calls)} MCP calls · {len(res.tool_calls)} custom tool calls"))
    boxes = []
    for n, s in enumerate(res.state_log):
        col = C["emerald"] if s["status"] == "completed" else C["amber"]
        boxes.append(f'<div style="flex:1;min-width:150px;background:{C["callout"]};border:1px solid {C["hairline"]};border-top:4px solid {col};border-radius:12px;padding:12px 14px">'
                     f'<div style="font-size:11px;font-weight:600;color:{C["slate"]}">{s["state"][:2]} · {s["ms"]} ms</div><div style="font-size:13px;font-weight:600;color:{C["ink"]};margin:2px 0 4px 0">{cfg._esc(s["state"][3:].replace("_", " "))}</div>'
                     f'<div style="font-size:11.5px;color:{C["slate"]}">{cfg._esc(s["agent"].split(":")[0])}</div><div style="font-size:11.5px;color:{C["ink"]};margin-top:6px">{cfg._esc(s["summary"])}</div></div>')
    md(f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 14px 0">{"".join(boxes)}</div>')
    st.markdown("#### 4-Agent swarm")
    ac = st.columns(2)
    for n, a in enumerate(res.agents):
        with ac[n % 2]:
            with st.expander(a["name"]):
                st.write(a["role"])
                st.caption("Instructions: " + a["instructions"])
    st.markdown("#### Responsible AI safeguard gate")
    audit_df = pd.DataFrame(res.audit["checks"])
    audit_df["status"] = audit_df["status"].map({"ok": "PASS", "warn": "WARN", "fail": "FAIL"})
    show_df(audit_df.rename(columns={"check": "Check", "status": "Status", "detail": "Detail"}))
    md(chips([f'Verdict: {res.audit["verdict"]}', f'{res.audit["passed"]}/{res.audit["total"]} checks green'], "ok" if res.audit["verdict"].startswith("passed") else "warn"))
    st.markdown("#### MCP tool calls")
    srv = MCPServerLite()
    with st.expander("Registered MCP tools (tools/list)"):
        md(cfg.json_block(srv.list_tools()))
    if res.mcp_calls:
        show_df(pd.DataFrame([{"#": i + 1, "Tool": c["tool"], "ms": c["ms"], "Error": c["is_error"], "Arguments": json.dumps(c["arguments"])[:90]} for i, c in enumerate(res.mcp_calls)]))
        pick = st.selectbox("Inspect JSON-RPC exchange", range(len(res.mcp_calls)), format_func=lambda i: f'{i + 1}. {res.mcp_calls[i]["tool"]}')
        i1, i2 = wide_columns(2)
        i1.markdown("**Request**")
        i1.markdown(cfg.json_block(res.mcp_calls[pick]["request"], 300), unsafe_allow_html=True)
        i2.markdown("**Response (preview)**")
        i2.markdown(cfg.json_block(res.mcp_calls[pick]["response_preview"], 300), unsafe_allow_html=True)
    st.markdown("#### Custom `@ai_function` tools")
    show_df(pd.DataFrame([{"Tool": t["name"], "Description": t["description"], "Parameters": ", ".join(t["parameters"]["properties"])} for t in res.tool_schemas]))
    show_df(pd.DataFrame(res.tool_calls)[["tool", "summary"]].rename(columns={"tool": "Tool called this run", "summary": "Result"}) if res.tool_calls else pd.DataFrame({"Tool called this run": []}))
    st.markdown("#### OpenTelemetry-compatible execution trace")
    t0 = min(s["start_time_unix_nano"] for s in res.spans)
    ordered = sorted(res.spans, key=lambda s: s["start_time_unix_nano"])
    kind_col = {"SERVER": C["series"], "CLIENT": C["azure"], "INTERNAL": C["border_strong"]}
    f = go.Figure(go.Bar(y=[s["name"] for s in ordered][::-1], x=[max(s["duration_ms"], 0.05) for s in ordered][::-1], base=[(s["start_time_unix_nano"] - t0) / 1e6 for s in ordered][::-1],
                         orientation="h", marker_color=[kind_col[s["kind"]] for s in ordered][::-1], hovertemplate="%{y}<br>%{x:.2f} ms<extra></extra>"))
    f.update_xaxes(title="ms from workflow start")
    f.update_yaxes(showgrid=False)
    cfg.style_figure(f, max(300, 18 * len(ordered)), legend=False)
    f.update_layout(title="Span waterfall (ms from start)")
    show_fig(f, "waterfall")
    with st.expander("Span records"):
        show_df(res.span_frame)
    st.markdown("#### Foundry model router and side-by-side evaluation")
    router_df = cached_router_frame()
    chosen, why = rt.router.route(language)
    md(cfg.callout("Routing decision for this request", f"<b>{cfg._esc(chosen.label)}</b>: {cfg._esc(why)}"))
    show_df(router_df)
    f = go.Figure()
    for metric, col in (("Groundedness", C["series"]), ("Clinical relevance", C["azure"]), ("Cultural appropriateness", C["emerald"])):
        f.add_trace(go.Bar(x=router_df["Model"], y=router_df[metric], name=metric, marker_color=col))
    f.update_layout(barmode="group")
    f.update_yaxes(range=[0, 1])
    cfg.style_figure(f, 300)
    show_fig(f, "router_fig")
    st.caption("Only the deterministic engine is measured offline. Other rows are illustrative priors until a live endpoint is configured (VELOTECH_LLM_MODE=github|foundry).")
    st.markdown("#### AI-103 compliance matrix (live evidence)")
    show_df(compliance_matrix(res, router_df))

# --------------------------------------------------------------------------------------
# PAGE SECTIONS 5-7
# --------------------------------------------------------------------------------------
kpis = [
    ("Outreach tier · " + person["name"].split()[0], risk["tier"], f'Care-gap {risk["care_gap_score"]:.0f}/100' if risk["care_gap_score"] is not None else "Baseline schedule applies"),
    ("Authorized data completeness", f'{risk["completeness"]:.0%}', "AI blocked below 40%" if risk["fallback"] else "Above the 40% safety line"),
    ("Early-detection lift vs age rule", f'{sim["detection_lift"]:+.0%}', f'{sim["qaly_gain"]:+.1f} modelled QALYs'),
    ("Small cells suppressed", str(n_suppressed), f"k = {k} anonymity safeguard"),
]
md('<div style="height:40px"></div>')
md(cfg.build_summary_band(kpis, (f'{sim["detection_lift"]:+.0%}', "Modelled early-detection lift versus age-rule outreach at equal capacity")))
md(cfg.build_who())
md(cfg.build_timeline())
proof = build_proof_metrics(eng.metrics, sim, parity, n_suppressed, cached_throughput(), n_spans=len(res.spans), n_mcp_calls=len(res.mcp_calls), k=k)
md(cfg.build_proof_grid(proof))
md(cfg.build_faq_accordion([tutorial.faq_item()] + tech_docs()))
md(cfg.build_team())
md(cfg.build_cta())
md(cfg.build_footer())
