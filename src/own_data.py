"""
VeloTech LifeCourse-360 - "My own data" form.

Lets anyone type in their own (or a family member's) details and run the same workflow as the sample people.
Blank fields are treated as missing data, never as zero. Nothing is stored: the entry lives in the browser
session only and disappears when the session ends.
"""
from __future__ import annotations

import html

import streamlit as st

from data.synthetic_generator import (
    CHANNELS, DOMAIN_LABELS, LANGUAGES, NATIONALITY_GROUPS, OCCUPATION_GROUPS, REGIONS, build_custom_persona, compute_completeness,
)
from src.ml_conformal_engine import FALLBACK_THRESHOLD

_UNSURE = {"Not sure": None, "No": 0.0, "Yes": 1.0}
_HEAT = {"Not sure": None, "Low (mostly indoors)": 0.1, "Moderate": 0.45, "High (outdoor work or very hot area)": 0.9}


_MODE = {"p": "a"}  # key prefix: adult widgets and child widgets must never share a key


def _k(name: str) -> str:
    return f"own_{_MODE['p']}_{name}_{st.session_state.get('own_ver', 0)}"


def _num(label: str, raw: dict, key: str, lo, hi, step, help: str | None = None):
    """Optional number: an empty box means unknown (None), never zero."""
    v = raw.get(key)
    if v is not None:
        v = int(v) if isinstance(lo, int) else float(v)
    return st.number_input(label, min_value=lo, max_value=hi, value=v, step=step, placeholder="leave blank if unknown",
                           help=help, key=_k(key))


def _choice(label: str, raw: dict, key: str, options: dict[str, float | None]):
    cur = raw.get(key)
    names = list(options)
    idx = next((i for i, n in enumerate(names) if options[n] == cur), 0)
    return options[st.selectbox(label, names, index=idx, key=_k(key))]


def render_form(saved: dict | None) -> tuple[dict, float, list[str]]:
    """Draw the form. Returns (data, completeness, available_domains) for the values currently on screen."""
    saved = saved or {}
    raw = dict(saved.get("features") or {})
    ver = st.session_state.get("own_ver", 0)

    st.markdown("**1 · About the person**")
    c1, c2, c3 = st.columns([1.3, 1, 1])
    name = c1.text_input("Name or label", value=saved.get("name", ""), placeholder="e.g. Me", key=f"own_name_{ver}")
    who = c2.radio("This person is", ["Adult (18+)", "Child (0-5)"], index=1 if saved.get("who") == "Child" else 0, horizontal=True, key=f"own_who_{ver}")
    child = who.startswith("Child")
    _MODE["p"] = "c" if child else "a"
    sex = c3.selectbox("Sex", ["F", "M"], index=1 if saved.get("sex") == "M" else 0, format_func=lambda v: "Female" if v == "F" else "Male", key=f"own_sex_{ver}")
    c4, c5, c6 = st.columns(3)
    if child:
        age = c4.number_input("Age in months", min_value=0, max_value=71, value=int(saved.get("age_months") or 24), step=1, key=f"own_agem_{ver}")
    else:
        age = c4.number_input("Age in years", min_value=18, max_value=100, value=int(saved.get("age_years") or 40), step=1, key=f"own_agey_{ver}")
    nat = c5.selectbox("Nationality group", NATIONALITY_GROUPS, index=NATIONALITY_GROUPS.index(saved["nationality_group"]) if saved.get("nationality_group") in NATIONALITY_GROUPS else 0, key=f"own_nat_{ver}")
    region = c6.selectbox("Region in Abu Dhabi", REGIONS, index=REGIONS.index(saved["region"]) if saved.get("region") in REGIONS else 0, key=f"own_region_{ver}")
    c7, c8, c9 = st.columns(3)
    lang = c7.selectbox("Preferred language", LANGUAGES, index=LANGUAGES.index(saved["language"]) if saved.get("language") in LANGUAGES else 2, key=f"own_lang_{ver}")
    chan = c8.selectbox("Preferred channel", CHANNELS, index=CHANNELS.index(saved["channel"]) if saved.get("channel") in CHANNELS else 0, key=f"own_chan_{ver}")
    if child:
        occ = "Student / child"
        c9.text_input("Occupation group", value=occ, disabled=True, key=f"own_occ_{ver}")
    else:
        occ = c9.selectbox("Occupation group", [o for o in OCCUPATION_GROUPS if o != "Student / child"],
                           index=[o for o in OCCUPATION_GROUPS if o != "Student / child"].index(saved["occupation_group"]) if saved.get("occupation_group") in OCCUPATION_GROUPS and saved.get("occupation_group") != "Student / child" else 0,
                           key=f"own_occ_{ver}")

    st.markdown("**2 · Visit history**")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        last_visit = _num("Months since last check-up", {"v": saved.get("last_visit_months_ago")}, "v", 0, 300, 1, "Used to work out how overdue the next visit is.")
    with h2:
        missed = _num("Missed visits (last 2 years)", raw, "missed_visits_2y", 0, 20, 1)
    if child:
        with h3:
            wfh = _num("Weight-for-height z-score", raw, "bmi_or_wfh_z", -5.0, 5.0, 0.1, "From the growth chart; 0 is average.")
        with h4:
            imm = _num("Immunization gaps", raw, "immunization_gaps", 0, 10, 1, "Number of recommended vaccines not yet given.")
        bmi = sbp = None
    else:
        with h3:
            bmi = _num("BMI (kg/m²)", raw, "bmi_or_wfh_z", 10.0, 70.0, 0.1)
        with h4:
            sbp = _num("Systolic blood pressure (mmHg)", raw, "sbp", 70, 230, 1)
        wfh = imm = None

    st.markdown("**3 · Laboratory results**")
    if child:
        l1, = st.columns(1)
        with l1:
            hb = _num("Hemoglobin (g/dL)", raw, "hemoglobin", 5.0, 20.0, 0.1)
        ldl = a1c = egfr = uacr = None
    else:
        l1, l2, l3, l4, l5 = st.columns(5)
        with l1:
            ldl = _num("LDL (mmol/L)", raw, "ldl", 0.5, 10.0, 0.1)
        with l2:
            a1c = _num("HbA1c (%)", raw, "hba1c", 3.0, 15.0, 0.1)
        with l3:
            egfr = _num("eGFR (mL/min)", raw, "egfr", 5, 140, 1)
        with l4:
            uacr = _num("UACR (mg/g)", raw, "uacr", 0, 3000, 1)
        with l5:
            hb = _num("Hemoglobin (g/dL)", raw, "hemoglobin", 5.0, 20.0, 0.1)

    st.markdown("**4 · Family history**")
    f1, f2 = st.columns(2)
    with f1:
        fcvd = _choice("Early heart disease in the family", raw, "fam_hx_cvd", _UNSURE)
    with f2:
        fdm = _choice("Diabetes in the family", raw, "fam_hx_dm", _UNSURE)

    st.markdown("**5 · Development and behaviour**")
    if child:
        d1, = st.columns(1)
        with d1:
            mchat = _num("M-CHAT-R/F score (0-20)", raw, "mchat_score", 0, 20, 1, "Developmental screening score, if one was done.")
        phq = smoker = None
    else:
        d1, d2 = st.columns(2)
        with d1:
            phq = _num("PHQ-9 score (0-27)", raw, "phq9", 0, 27, 1, "Mental wellbeing questionnaire, if you have done it.")
        with d2:
            smoker = _choice("Tobacco use", raw, "smoker", _UNSURE)
        mchat = None

    st.markdown("**6 · Environment**")
    e1, e2, e3 = st.columns(3)
    with e1:
        heat = _choice("Heat exposure", raw, "heat_exposure", _HEAT)
    with e2:
        pm25 = _num("PM2.5 air quality (µg/m³)", raw, "pm25", 0, 250, 1)
    with e3:
        dist = _num("Distance to nearest clinic (km)", raw, "distance_km", 0.0, 200.0, 0.5)

    st.markdown("**7 · Emirati Genome Program (optional)**")
    if nat == "Emirati":
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            prs_c = _num("Heart-disease risk percentile", raw, "prs_cvd", 0, 100, 1)
        with g2:
            prs_t = _num("Type 2 diabetes risk percentile", raw, "prs_t2d", 0, 100, 1)
        with g3:
            fh = _choice("Familial hypercholesterolemia variant", raw, "fh_variant", _UNSURE)
        with g4:
            carrier = _choice("Metabolic carrier flag", raw, "metabolic_carrier", _UNSURE)
    else:
        st.caption("The Emirati Genome Program covers Emirati nationals only, so no genomic fields are needed here.")
        prs_c = prs_t = fh = carrier = None

    features = {
        "bmi_or_wfh_z": wfh if child else bmi, "sbp": sbp, "missed_visits_2y": missed, "immunization_gaps": imm,
        "ldl": ldl, "hba1c": a1c, "egfr": egfr, "uacr": uacr, "hemoglobin": hb,
        "prs_cvd": prs_c, "prs_t2d": prs_t, "fh_variant": fh, "metabolic_carrier": carrier,
        "fam_hx_cvd": fcvd, "fam_hx_dm": fdm, "phq9": phq, "smoker": smoker, "mchat_score": mchat,
        "heat_exposure": heat, "pm25": pm25, "distance_km": dist,
    }
    data = {
        "who": "Child" if child else "Adult", "name": name, "sex": sex, "nationality_group": nat, "region": region,
        "language": lang, "channel": chan, "occupation_group": occ, "last_visit_months_ago": last_visit, "features": features,
        "age_months": int(age) if child else None, "age_years": None if child else int(age),
    }
    persona = build_custom_persona(data)
    completeness = compute_completeness(persona["available_domains"], persona["available_domains"])
    return data, completeness, persona["available_domains"]


def completeness_notice(completeness: float, avail: list[str]) -> str:
    """Short HTML status line shown above the Save button."""
    pct = f"{completeness:.0%}"
    names = html.escape(", ".join(DOMAIN_LABELS[d] for d in avail) or "nothing yet")
    if completeness >= FALLBACK_THRESHOLD:
        return f"<b>Data completeness: {pct}.</b> Enough to estimate a risk. Using: {names}."
    return (f"<b>Data completeness: {pct}.</b> At least {FALLBACK_THRESHOLD:.0%} is needed for a risk estimate. Fill in more of the sections above "
            f"(visit history and lab results count the most). Until then the app shows the standard age-based schedule. Using: {names}.")
