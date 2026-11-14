"""
VeloTech LifeCourse-360 - Synthetic data layer.

ALL DATA IS SYNTHETIC. No real patient, no real Emirati Genome Program (EGP) record and no real
Malaffi payload is used anywhere in this repository. The generator reproduces plausible *shapes*
(distributions, missingness patterns, household structure) so the prototype is demonstrable at $0.

Provides
--------
* The shared feature / data-domain schema consumed by the ML engine, MCP server and agents.
* 8 hand-written personas (incl. the 4 mandatory test cases).
* `generate_population()` - a few thousand synthetic Abu Dhabi residents with realistic
  missingness, segments and unmet-need labels for calibration / fairness / k-anonymity work.
* NetworkX household kinship graphs and the "Synchronized Family Prevention Appointment" detector.
"""
from __future__ import annotations

import copy
import math
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# 1. SHARED SCHEMA
# --------------------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    "age_years", "is_child", "bmi_or_wfh_z", "sbp", "ldl", "hba1c", "egfr", "uacr", "phq9",
    "mchat_score", "hemoglobin", "fam_hx_cvd", "fam_hx_dm", "smoker", "months_overdue",
    "missed_visits_2y", "immunization_gaps", "heat_exposure", "pm25", "prs_cvd", "prs_t2d",
    "fh_variant", "metabolic_carrier", "distance_km",
]

FEATURE_LABELS: dict[str, str] = {
    "age_years": "Age", "is_child": "Child (Well-Child pathway)", "bmi_or_wfh_z": "BMI / weight-for-height z-score",
    "sbp": "Systolic blood pressure", "ldl": "LDL cholesterol", "hba1c": "HbA1c", "egfr": "Kidney function (eGFR)",
    "uacr": "Urine albumin-creatinine ratio", "phq9": "PHQ-9 mental wellbeing score",
    "mchat_score": "M-CHAT-R/F developmental score", "hemoglobin": "Hemoglobin",
    "fam_hx_cvd": "Family history: early heart disease", "fam_hx_dm": "Family history: diabetes",
    "smoker": "Tobacco use", "months_overdue": "Months overdue for scheduled visit",
    "missed_visits_2y": "Missed visits (last 2 years)", "immunization_gaps": "Immunization gaps",
    "heat_exposure": "Occupational / neighbourhood heat exposure", "pm25": "PM2.5 air quality",
    "prs_cvd": "EGP polygenic risk: cardiovascular (percentile)", "prs_t2d": "EGP polygenic risk: type 2 diabetes (percentile)",
    "fh_variant": "EGP familial hypercholesterolemia variant", "metabolic_carrier": "EGP hereditary metabolic carrier flag",
    "distance_km": "Distance to nearest preventive clinic (km)",
}

# Values that contribute "nothing" when a feature does not apply to a segment
# (e.g. adult-only labs for a toddler). Distinct from *missing* data, which is NaN.
NEUTRAL_VALUES: dict[str, float] = {
    "sbp": 110.0, "ldl": 2.6, "hba1c": 5.2, "egfr": 110.0, "uacr": 6.0, "phq9": 0.0, "mchat_score": 0.0,
    "hemoglobin": 13.5, "smoker": 0.0, "prs_cvd": 50.0, "prs_t2d": 50.0, "fh_variant": 0.0,
    "metabolic_carrier": 0.0, "immunization_gaps": 0.0, "bmi_or_wfh_z": 0.0,
}

# Granular consent domains. `weight` sums to 1.0 and drives data-completeness.
DATA_DOMAINS: list[dict] = [
    {"key": "ehr_history", "label": "EHR & visit history (Malaffi)", "weight": 0.25,
     "features": ["months_overdue", "missed_visits_2y", "immunization_gaps", "bmi_or_wfh_z", "sbp"]},
    {"key": "labs", "label": "Laboratory results", "weight": 0.25,
     "features": ["ldl", "hba1c", "egfr", "uacr", "hemoglobin"]},
    {"key": "genomic_egp", "label": "Emirati Genome Program (EGP)", "weight": 0.12,
     "features": ["prs_cvd", "prs_t2d", "fh_variant", "metabolic_carrier"]},
    {"key": "family_history", "label": "Family history", "weight": 0.12,
     "features": ["fam_hx_cvd", "fam_hx_dm"]},
    {"key": "developmental_behavioral", "label": "Developmental & behavioural", "weight": 0.12,
     "features": ["mchat_score", "phq9", "smoker"]},
    {"key": "environmental_context", "label": "Environmental & geographic context", "weight": 0.10,
     "features": ["heat_exposure", "pm25", "distance_km"]},
    {"key": "wearable_biosignal", "label": "Bio-signal pre-screener / wearable", "weight": 0.04,
     "features": []},
]
DOMAIN_KEYS: list[str] = [d["key"] for d in DATA_DOMAINS]
DOMAIN_LABELS: dict[str, str] = {d["key"]: d["label"] for d in DATA_DOMAINS}
DOMAIN_WEIGHTS: dict[str, float] = {d["key"]: d["weight"] for d in DATA_DOMAINS}
FEATURE_TO_DOMAIN: dict[str, str] = {f: d["key"] for d in DATA_DOMAINS for f in d["features"]}
# Features always available (derived from demographics, not from a consent domain)
DEMOGRAPHIC_FEATURES = ["age_years", "is_child"]

REGIONS = ["Abu Dhabi City", "Al Ain", "Al Dhafra"]
NATIONALITY_GROUPS = ["Emirati", "Arab expat", "South Asian expat", "Other expat"]
OCCUPATION_GROUPS = ["Office / professional", "Outdoor / heat-exposed", "Homemaker / caregiver", "Student / child", "Retired"]
CHANNELS = ["WhatsApp", "SMS", "Mobile App", "Voice Call"]
LANGUAGES = ["Arabic (Emirati)", "Arabic (Standard)", "English", "Urdu"]

REGION_CLINICS = {
    "Abu Dhabi City": "Preventive Care Centre, Abu Dhabi City",
    "Al Ain": "Preventive Care Centre, Al Ain",
    "Al Dhafra": "Preventive Care Centre, Madinat Zayed",
}


def recall_interval_months(age_years: float, segment: str) -> int:
    """Baseline DoH-style recall interval used for overdue calculations (illustrative)."""
    if segment == "Well-Child":
        return 6 if age_years < 2 else 12
    return 36 if age_years < 40 else 24


def age_band(age_years: float, segment: str) -> str:
    if segment == "Well-Child":
        return "0-1 yr" if age_years < 2 else "2-5 yrs"
    if age_years < 40:
        return "18-39"
    if age_years < 55:
        return "40-54"
    if age_years < 65:
        return "55-64"
    return "65+"


# --------------------------------------------------------------------------------------
# 2. PERSONAS (8, hand-written)
# --------------------------------------------------------------------------------------
def _persona(
    pid: str, name: str, age_years: float, sex: str, nationality: str, nat_group: str, language: str,
    channel: str, segment: str, household_id: str, role: str, region: str, occupation: str,
    occupation_group: str, last_visit_months_ago: float | None, features: dict, available: Iterable[str],
    story: str, tags: list[str], consent_off: Iterable[str] = ("wearable_biosignal",),
    age_months: int | None = None,
) -> dict:
    """Assemble a persona; non-applicable features get neutral values, unavailable domains get None."""
    avail = list(available)
    feats: dict[str, float | None] = {}
    child = segment == "Well-Child"
    for col in FEATURE_COLUMNS:
        if col == "age_years":
            feats[col] = float(age_years)
        elif col == "is_child":
            feats[col] = 1.0 if child else 0.0
        elif col in features:
            feats[col] = features[col]
        else:
            feats[col] = NEUTRAL_VALUES.get(col, 0.0)
    # Blank out every feature that belongs to a domain with no data on file
    for col in FEATURE_COLUMNS:
        dom = FEATURE_TO_DOMAIN.get(col)
        if dom and dom not in avail:
            feats[col] = None
    interval = recall_interval_months(age_years, segment)
    if last_visit_months_ago is None:
        overdue = None
    else:
        overdue = max(0.0, float(last_visit_months_ago) - interval)
        if "ehr_history" in avail:
            feats["months_overdue"] = overdue
    consent_default = [d for d in DOMAIN_KEYS if d not in set(consent_off)]
    return {
        "id": pid, "name": name, "age_years": float(age_years),
        "age_months": age_months if age_months is not None else int(round(age_years * 12)),
        "sex": sex, "nationality": nationality, "nationality_group": nat_group, "language": language,
        "channel": channel, "segment": segment, "household_id": household_id, "role": role,
        "region": region, "occupation": occupation, "occupation_group": occupation_group,
        "last_visit_months_ago": last_visit_months_ago, "recall_interval_months": interval,
        "months_overdue": overdue, "features": feats, "available_domains": avail,
        "consent_default": consent_default, "story": story, "tags": tags,
        "age_label": (f"{age_months} months" if child and age_years < 3 and age_months else
                      (f"{int(age_years)} years" if child else f"{int(age_years)} years")),
    }


_ALL_BUT_WEARABLE = ["ehr_history", "labs", "genomic_egp", "family_history", "developmental_behavioral", "environmental_context"]


def _build_personas() -> list[dict]:
    return [
        _persona(
            "P-001", "Fatima Al Mansoori", 34, "F", "Emirati", "Emirati", "Arabic (Emirati)", "WhatsApp",
            "IFHAS", "HH-AM", "Parent / Caregiver", "Abu Dhabi City", "Primary-school teacher",
            "Office / professional", 50,
            dict(bmi_or_wfh_z=29.5, sbp=124, ldl=3.4, hba1c=5.9, egfr=104, uacr=7, phq9=7, hemoglobin=12.1,
                 fam_hx_cvd=0, fam_hx_dm=1, smoker=0, missed_visits_2y=2, heat_exposure=0.2, pm25=38,
                 prs_cvd=55, prs_t2d=82, fh_variant=0, distance_km=4.2),
            _ALL_BUT_WEARABLE,
            "Mother of Omar. Gestational diabetes history, prediabetic HbA1c, last adult screening over four years ago. "
            "Her toddler is also overdue, which makes this household a synchronization candidate.",
            ["Household pair", "Overdue IFHAS", "Prediabetes range", "High EGP T2D polygenic risk"],
        ),
        _persona(
            "P-002", "Omar Al Mansoori", 22 / 12, "M", "Emirati", "Emirati", "Arabic (Emirati)", "WhatsApp",
            "Well-Child", "HH-AM", "Child", "Abu Dhabi City", "Toddler", "Student / child", 11,
            dict(bmi_or_wfh_z=-0.4, hemoglobin=10.4, mchat_score=4, fam_hx_dm=1, fam_hx_cvd=0,
                 missed_visits_2y=1, immunization_gaps=2, heat_exposure=0.1, pm25=38, metabolic_carrier=1,
                 distance_km=4.2),
            _ALL_BUT_WEARABLE,
            "Toddler of 22 months who missed the 18-month Well-Child Visit. Low hemoglobin, a medium-range M-CHAT-R score "
            "and two immunization gaps, plus an EGP hereditary metabolic carrier flag in the family.",
            ["Overdue Well-Child Visit", "M-CHAT-R follow-up", "Anemia flag", "EGP carrier flag"],
            age_months=22,
        ),
        _persona(
            "P-003", "Khalid Al Dhaheri", 27, "M", "Emirati", "Emirati", "Arabic (Standard)", "WhatsApp",
            "IFHAS", "HH-KD", "Adult", "Al Ain", "Software engineer", "Office / professional", 40,
            dict(bmi_or_wfh_z=26.0, sbp=118, ldl=5.3, hba1c=5.2, egfr=112, uacr=5, phq9=3, hemoglobin=14.8,
                 fam_hx_cvd=1, fam_hx_dm=0, smoker=0, missed_visits_2y=0, heat_exposure=0.15, pm25=33,
                 prs_cvd=94, prs_t2d=40, fh_variant=1, distance_km=7.5),
            _ALL_BUT_WEARABLE,
            "Young adult who looks healthy on age and gender rules alone. EGP flags a familial hypercholesterolemia variant, "
            "LDL is markedly raised and his father had a heart attack at 46: a clear case for early lipid review and cascade screening.",
            ["EGP familial hypercholesterolemia", "Family history of early CVD", "Missed by age rules"],
        ),
        _persona(
            "P-004", "Imran Hussain", 38, "M", "Pakistani", "South Asian expat", "Urdu", "SMS",
            "IFHAS", "HH-IH", "Adult", "Al Dhafra", "Construction worker (outdoor)", "Outdoor / heat-exposed", 50,
            dict(bmi_or_wfh_z=27.5, sbp=138, ldl=3.2, hba1c=5.7, egfr=76, uacr=38, phq9=6, hemoglobin=14.8,
                 fam_hx_cvd=0, fam_hx_dm=1, smoker=1, missed_visits_2y=1, heat_exposure=0.95, pm25=62,
                 distance_km=38.0),
            ["ehr_history", "labs", "family_history", "developmental_behavioral", "environmental_context"],
            "Outdoor worker in a high-heat zone, 38 km from the nearest clinic. eGFR has slipped and albuminuria is raised, "
            "a heat-stress nephropathy pattern. No EGP record exists because EGP covers Emirati nationals, so the model must lean on the other domains.",
            ["Climate-worker renal risk", "High heat exposure", "Remote region", "No genomic record"],
        ),
        _persona(
            "P-005", "Mariam Al Ketbi", 52, "F", "Emirati", "Emirati", "Arabic (Emirati)", "Mobile App",
            "IFHAS", "HH-MK", "Adult", "Abu Dhabi City", "Administrator", "Office / professional", 45,
            dict(bmi_or_wfh_z=32.4, sbp=136, ldl=3.9, hba1c=6.3, egfr=82, uacr=22, phq9=11, hemoglobin=12.6,
                 fam_hx_cvd=1, fam_hx_dm=1, smoker=0, missed_visits_2y=2, heat_exposure=0.2, pm25=40,
                 prs_cvd=77, prs_t2d=91, fh_variant=0, distance_km=6.0),
            _ALL_BUT_WEARABLE,
            "Midlife adult with HbA1c in the diabetic range, moderate PHQ-9, strong family history and very high EGP type 2 diabetes risk. "
            "Overdue for the 40-75 IFHAS panel, mammography and colorectal screening.",
            ["IFHAS 40-75", "Moderate PHQ-9", "High T2D polygenic risk", "Mammography overdue"],
        ),
        _persona(
            "P-006", "Aisha Al Hammadi", 68, "F", "Emirati", "Emirati", "Arabic (Emirati)", "Voice Call",
            "IFHAS", "HH-AH", "Grandparent / Caregiver", "Al Ain", "Retired", "Retired", 54,
            dict(bmi_or_wfh_z=30.0, sbp=142, ldl=3.1, hba1c=6.6, egfr=58, uacr=45, phq9=9, hemoglobin=11.9,
                 fam_hx_cvd=0, fam_hx_dm=1, smoker=0, missed_visits_2y=3, heat_exposure=0.1, pm25=33,
                 prs_cvd=70, prs_t2d=88, fh_variant=0, distance_km=9.0),
            _ALL_BUT_WEARABLE,
            "Grandmother and carer for Yousef. Reduced kidney function, raised albuminuria, long gap since screening and no colorectal or "
            "bone-density record. Prefers voice calls in Emirati Arabic.",
            ["Household pair", "Senior screening gaps", "CKD pattern", "Prefers voice"],
        ),
        _persona(
            "P-007", "Yousef Al Hammadi", 4, "M", "Emirati", "Emirati", "Arabic (Emirati)", "Voice Call",
            "Well-Child", "HH-AH", "Child", "Al Ain", "Preschool child", "Student / child", 24,
            dict(bmi_or_wfh_z=1.8, hemoglobin=11.8, mchat_score=0, fam_hx_dm=1, fam_hx_cvd=0,
                 missed_visits_2y=1, immunization_gaps=1, heat_exposure=0.1, pm25=33, metabolic_carrier=0,
                 distance_km=9.0),
            _ALL_BUT_WEARABLE,
            "Four-year-old overdue for vision and audiometry checks and the four-year immunization booster, with a rising weight-for-height z-score. "
            "Lives with his grandmother, who is also overdue.",
            ["Overdue Well-Child Visit", "Vision / audiometry due", "Immunization booster", "Household pair"],
        ),
        _persona(
            "P-008", "Daniel Thompson", 45, "M", "British", "Other expat", "English", "Mobile App",
            "IFHAS", "HH-DT", "Adult", "Abu Dhabi City", "Management consultant", "Office / professional", None,
            dict(bmi_or_wfh_z=None, sbp=None),
            ["ehr_history"],
            "Recent arrival with only a thin EHR stub. Authorized data completeness is far below 40%, so the AI must refuse to extrapolate "
            "and fall back to the baseline DoH age and gender schedule.",
            ["Low-data fallback test case", "New resident", "Sparse EHR"],
        ),
    ]


_PERSONA_CACHE: list[dict] | None = None


# --------------------------------------------------------------------------------------
# 2b. "MY OWN DATA": a person entered by the user (kept in memory for the session only)
# --------------------------------------------------------------------------------------
CUSTOM_ID = "P-CUSTOM"
CUSTOM_LABEL = "My own data · enter your details"
_CUSTOM_DATA: dict | None = None

# which model features each consent domain is built from, per segment, and which are meaningful for whom
ADULT_ONLY = {"sbp", "ldl", "hba1c", "egfr", "uacr", "phq9", "smoker"}
CHILD_ONLY = {"mchat_score", "immunization_gaps"}


def set_custom_data(data: dict | None) -> None:
    """Register (or clear) the user-entered person. `data` comes from the 'My own data' form."""
    global _CUSTOM_DATA
    _CUSTOM_DATA = copy.deepcopy(data) if data is not None else None


def has_custom_data() -> bool:
    return _CUSTOM_DATA is not None


def build_custom_persona(d: dict | None) -> dict:
    """Turn the form output into a persona. Blank fields stay missing (None); a domain counts as
    available only if at least one of its fields was actually filled in."""
    d = d or {}
    child = d.get("who") == "Child"
    segment = "Well-Child" if child else "IFHAS"
    age_months = int(d.get("age_months") or 24) if child else None
    age_years = (age_months / 12.0) if child else float(d.get("age_years") or 40)
    raw = {k: v for k, v in (d.get("features") or {}).items()}
    feats: dict = {}
    for col, v in raw.items():
        if child and col in ADULT_ONLY - {"hemoglobin"}:
            continue
        if not child and col in CHILD_ONLY:
            continue
        feats[col] = None if v is None else float(v)
    last_visit = d.get("last_visit_months_ago")
    last_visit = None if last_visit is None else float(last_visit)
    if last_visit is not None:
        feats["months_overdue"] = None  # recomputed by _persona from last visit
    for col in ("months_overdue",):
        feats.setdefault(col, None)
    avail = []
    for dom in DATA_DOMAINS:
        if dom["key"] == "wearable_biosignal":
            continue
        filled = [c for c in dom["features"] if feats.get(c) is not None]
        if dom["key"] == "ehr_history" and last_visit is not None:
            filled.append("months_overdue")
        if filled:
            avail.append(dom["key"])
    nat = d.get("nationality_group") or "Other expat"
    occ = d.get("occupation_group") or ("Student / child" if child else "Office / professional")
    name = (d.get("name") or "").strip() or "My own data"
    tags = ["Entered by you", "Child 0-5 (Well-Child)" if child else "Adult (IFHAS)"]
    story = (f"Details entered by you for {name}. Only the fields you filled in are used; anything left blank is treated as "
             f"missing, which widens the uncertainty range. Below 40% data completeness the app will not estimate a risk.")
    persona = _persona(
        CUSTOM_ID, name, age_years, d.get("sex") or "F", nat, nat, d.get("language") or "English", d.get("channel") or "WhatsApp",
        segment, "HH-CUSTOM", "Child" if child else "Adult", d.get("region") or "Abu Dhabi City", occ, occ, last_visit, feats, avail,
        story, tags, age_months=age_months,
    )
    persona["is_custom"] = True
    # a custom person starts with every domain they actually filled in consented
    persona["consent_default"] = [k for k in avail]
    return persona


def get_personas() -> list[dict]:
    """Return deep copies of all personas: the 8 built-in synthetic people plus the user's own entry, if any."""
    global _PERSONA_CACHE
    if _PERSONA_CACHE is None:
        _PERSONA_CACHE = _build_personas()
    out = copy.deepcopy(_PERSONA_CACHE)
    if _CUSTOM_DATA is not None:
        out.append(build_custom_persona(_CUSTOM_DATA))
    return out


def get_persona(pid: str) -> dict:
    for p in get_personas():
        if p["id"] == pid:
            return p
    raise KeyError(f"Unknown persona id: {pid}")


def persona_options() -> dict[str, str]:
    """{persona_id: 'Name · age · pathway'} for selectboxes. The last entry is always 'My own data'."""
    out = {}
    for p in get_personas():
        if p["id"] == CUSTOM_ID:
            continue
        pathway = "Well-Child" if p["segment"] == "Well-Child" else "IFHAS"
        out[p["id"]] = f'{p["name"]} · {p["age_label"]} · {pathway}'
    out[CUSTOM_ID] = CUSTOM_LABEL
    return out


# --------------------------------------------------------------------------------------
# 3. DATA COMPLETENESS & EGP RECORDS
# --------------------------------------------------------------------------------------
def compute_completeness(available: Iterable[str], consent: Iterable[str]) -> float:
    """Weighted share of data domains that are both consented AND actually on file (0-1)."""
    avail, cons = set(available), set(consent)
    return float(sum(DOMAIN_WEIGHTS[k] for k in DOMAIN_KEYS if k in avail and k in cons))


def egp_record(person: dict) -> dict:
    """Synthetic Emirati Genome Program lookup (EGP covers Emirati nationals only)."""
    f = person["features"]
    if person["nationality_group"] != "Emirati" or "genomic_egp" not in person["available_domains"]:
        return {"person_id": person["id"], "enrolled": False,
                "note": "No EGP record on file (programme covers Emirati nationals; synthetic demo)."}
    prs_cvd, prs_t2d = f.get("prs_cvd"), f.get("prs_t2d")
    flags = []
    if f.get("fh_variant"):
        flags.append("LDLR-pathway variant associated with familial hypercholesterolemia (synthetic label)")
    if f.get("metabolic_carrier"):
        flags.append("Hereditary metabolic condition carrier flag (synthetic label)")
    if prs_cvd is not None and prs_cvd >= 90:
        flags.append("Polygenic risk for coronary disease in top decile")
    if prs_t2d is not None and prs_t2d >= 90:
        flags.append("Polygenic risk for type 2 diabetes in top decile")
    return {"person_id": person["id"], "enrolled": True, "prs_cvd_percentile": prs_cvd,
            "prs_t2d_percentile": prs_t2d, "flags": flags, "synthetic": True}


# --------------------------------------------------------------------------------------
# 4. POPULATION GENERATOR
# --------------------------------------------------------------------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_population(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Synthetic Abu Dhabi population with segments, labels, availability and consent columns.

    Feature columns hold the TRUE values; `avail_<domain>` and `consent_<domain>` columns hold
    availability and consent. The ML engine masks features accordingly, so consent changes and
    missingness genuinely change the model input.
    """
    rng = np.random.default_rng(seed)
    n_child = int(n * 0.28)
    n_adult = n - n_child
    seg = np.array(["Well-Child"] * n_child + ["IFHAS"] * n_adult)
    is_child = (seg == "Well-Child").astype(float)

    # ---- demographics ----
    age = np.where(is_child == 1, rng.uniform(0.1, 5.9, n), np.clip(rng.normal(41, 13, n), 18, 82))
    sex = rng.choice(["F", "M"], n, p=[0.46, 0.54])
    nat_p_adult, nat_p_child = [0.30, 0.18, 0.32, 0.20], [0.45, 0.17, 0.23, 0.15]
    nat = np.empty(n, dtype=object)
    for i, (mask, p) in enumerate([(is_child == 0, nat_p_adult), (is_child == 1, nat_p_child)]):
        nat[mask] = rng.choice(NATIONALITY_GROUPS, mask.sum(), p=p)
    region = rng.choice(REGIONS, n, p=[0.62, 0.27, 0.11])

    occ = np.empty(n, dtype=object)
    occ[is_child == 1] = "Student / child"
    for i in np.where(is_child == 0)[0]:
        if age[i] >= 62 and rng.random() < 0.55:
            occ[i] = "Retired"
        elif sex[i] == "M":
            p_out = 0.5 if nat[i] == "South Asian expat" else 0.12
            if region[i] == "Al Dhafra":
                p_out += 0.2
            occ[i] = "Outdoor / heat-exposed" if rng.random() < p_out else "Office / professional"
        else:
            occ[i] = "Homemaker / caregiver" if rng.random() < 0.28 else (
                "Outdoor / heat-exposed" if rng.random() < 0.03 else "Office / professional")

    # ---- environment ----
    outdoor = (occ == "Outdoor / heat-exposed")
    heat = np.where(outdoor, rng.beta(8, 2, n), rng.beta(1.5, 6, n))
    heat = np.clip(heat + np.where(region == "Al Dhafra", 0.08, 0.0), 0, 1)
    pm25 = np.clip(np.select([region == "Abu Dhabi City", region == "Al Ain", region == "Al Dhafra"], [38, 32, 46]) + rng.normal(0, 7, n), 12, 95)
    distance = np.clip(np.select([region == "Abu Dhabi City", region == "Al Ain", region == "Al Dhafra"], [5, 8, 32]) * rng.lognormal(0, 0.45, n), 0.5, 140)

    # ---- clinical values (adults) ----
    adult = is_child == 0
    bmi = np.where(adult, rng.normal(28, 5.5, n) + 0.02 * (age - 40), rng.normal(0.1, 1.1, n))
    sbp = np.where(adult, rng.normal(118 + 0.45 * (age - 40), 14, n), NEUTRAL_VALUES["sbp"])
    ldl = np.where(adult, np.clip(rng.normal(3.0 + 0.012 * (age - 40), 0.9, n), 1.0, 7.0), NEUTRAL_VALUES["ldl"])
    hba1c = np.where(adult, np.clip(rng.normal(5.4 + 0.012 * (age - 40), 0.65, n), 4.4, 11), NEUTRAL_VALUES["hba1c"])
    egfr = np.where(adult, np.clip(rng.normal(104 - 0.75 * np.maximum(age - 30, 0) - 10 * heat, 12, n), 25, 135), NEUTRAL_VALUES["egfr"])
    uacr = np.where(adult, np.clip(rng.lognormal(math.log(8), 0.7, n) * (1 + 0.6 * heat), 1, 400), NEUTRAL_VALUES["uacr"])
    phq9 = np.where(adult, np.clip(rng.poisson(3.6, n) + (rng.random(n) < 0.12) * rng.integers(4, 12, n), 0, 27), 0)
    smoker = np.where(adult, (rng.random(n) < np.where(sex == "M", 0.22, 0.04)).astype(float), 0.0)
    hemo = np.where(adult, rng.normal(np.where(sex == "F", 12.9, 14.6), 1.2, n), rng.normal(11.9, 1.1, n))
    mchat = np.where((is_child == 1) & (age * 12 >= 16) & (age * 12 <= 30), np.clip(rng.poisson(1.2, n) + (rng.random(n) < 0.08) * rng.integers(2, 6, n), 0, 20), 0)
    fam_cvd = (rng.random(n) < 0.14).astype(float)
    fam_dm = (rng.random(n) < np.where(nat == "Emirati", 0.38, 0.24)).astype(float)

    # ---- genomic (EGP): populated for everyone, masked later for non-Emirati ----
    prs_cvd = np.where(adult, np.clip(rng.normal(50, 25, n), 1, 99), NEUTRAL_VALUES["prs_cvd"])
    prs_t2d = np.where(adult, np.clip(rng.normal(50, 25, n) + np.where(nat == "Emirati", 6, 0), 1, 99), NEUTRAL_VALUES["prs_t2d"])
    fh = np.where(adult, (rng.random(n) < np.where(nat == "Emirati", 0.012, 0.004)).astype(float), 0.0)
    ldl = np.where(fh == 1, np.maximum(ldl, 4.9 + rng.random(n)), ldl)
    carrier = np.where(is_child == 1, (rng.random(n) < np.where(nat == "Emirati", 0.07, 0.015)).astype(float), 0.0)

    # ---- care-gap history ----
    interval = np.array([recall_interval_months(a, s) for a, s in zip(age, seg)])
    since = interval + rng.gamma(1.4, 7.0, n) - rng.uniform(0, 10, n)
    months_overdue = np.clip(since - interval, 0, 90)
    months_overdue = np.where(rng.random(n) < 0.35, 0.0, months_overdue)
    missed = np.clip(rng.poisson(0.5 + months_overdue / 18, n), 0, 6).astype(float)
    imm_gaps = np.where(is_child == 1, np.clip(rng.poisson(0.2 + months_overdue / 10, n), 0, 5), 0).astype(float)

    # ---- ground-truth unmet-need (elevated unmet or overdue risk) ----
    z_adult = (-2.9 + 0.03 * (age - 40) + 0.5 * months_overdue / 12 + 0.38 * (ldl - 3.0) + 0.55 * (hba1c - 5.5)
               + 0.011 * (prs_cvd - 50) + 0.009 * (prs_t2d - 50) + 1.0 * fh + 0.42 * fam_cvd + 0.28 * fam_dm
               + 0.35 * smoker + 0.04 * (phq9 - 4) + 0.85 * (egfr < 75) + 0.65 * (uacr > 30) + 0.7 * heat
               + 0.012 * (sbp - 120) + 0.025 * (bmi - 26) + 0.22 * missed)
    z_child = (-2.6 + 0.55 * (mchat >= 3) + 0.2 * np.maximum(mchat - 1, 0) + 0.85 * (hemo < 11) + 0.42 * imm_gaps
               + 0.55 * months_overdue / 6 + 0.4 * (np.abs(bmi) > 2) + 0.65 * carrier + 0.35 * fam_dm + 0.22 * missed)
    z_clean = np.where(adult, z_adult, z_child)
    risk_proxy = _sigmoid(z_clean)  # synthetic "adjudicated risk" (no label noise) used to calibrate conformal intervals
    unmet = (rng.random(n) < _sigmoid(z_clean + rng.normal(0, 0.55, n))).astype(int)

    # ---- availability (missingness correlated with geography / occupation) ----
    p_labs = np.clip(0.92 - 0.12 * (region == "Al Dhafra") - 0.12 * outdoor, 0.2, 0.98)
    avail = {
        "ehr_history": rng.random(n) < 0.97,
        "labs": rng.random(n) < p_labs,
        "genomic_egp": (nat == "Emirati") & (rng.random(n) < 0.9),
        "family_history": rng.random(n) < np.where(nat == "Emirati", 0.85, 0.65),
        "developmental_behavioral": rng.random(n) < 0.85,
        "environmental_context": rng.random(n) < 0.95,
        "wearable_biosignal": rng.random(n) < 0.12,
    }
    consent_p = np.where(nat == "South Asian expat", 0.88, 0.94)
    consent = {k: rng.random(n) < consent_p for k in DOMAIN_KEYS}
    consent["wearable_biosignal"] = rng.random(n) < 0.15

    df = pd.DataFrame({
        "person_id": [f"S-{i:05d}" for i in range(n)], "segment": seg, "sex": sex,
        "nationality_group": nat, "region": region, "occupation_group": occ,
        "age_years": age.round(2), "is_child": is_child,
        "age_band": [age_band(a, s) for a, s in zip(age, seg)],
        "bmi_or_wfh_z": bmi.round(2), "sbp": sbp.round(0), "ldl": ldl.round(2), "hba1c": hba1c.round(2),
        "egfr": egfr.round(0), "uacr": uacr.round(1), "phq9": phq9.astype(float), "mchat_score": mchat.astype(float),
        "hemoglobin": hemo.round(1), "fam_hx_cvd": fam_cvd, "fam_hx_dm": fam_dm, "smoker": smoker,
        "months_overdue": months_overdue.round(1), "missed_visits_2y": missed, "immunization_gaps": imm_gaps,
        "heat_exposure": heat.round(2), "pm25": pm25.round(0), "prs_cvd": prs_cvd.round(0), "prs_t2d": prs_t2d.round(0),
        "fh_variant": fh, "metabolic_carrier": carrier, "distance_km": distance.round(1), "unmet_need": unmet,
        "risk_proxy": risk_proxy.round(4),
    })
    for k in DOMAIN_KEYS:
        df[f"avail_{k}"] = avail[k]
        df[f"consent_{k}"] = consent[k]
    df["data_completeness"] = sum(
        DOMAIN_WEIGHTS[k] * (df[f"avail_{k}"] & df[f"consent_{k}"]).astype(float) for k in DOMAIN_KEYS
    ).round(3)

    # preferred channel / language and uptake propensity (what happens if invited)
    df["channel"] = rng.choice(CHANNELS, n, p=[0.46, 0.2, 0.22, 0.12])
    df["language"] = np.where(
        nat == "Emirati", rng.choice(["Arabic (Emirati)", "Arabic (Standard)", "English"], n, p=[0.6, 0.2, 0.2]),
        np.where(nat == "South Asian expat", rng.choice(["Urdu", "English"], n, p=[0.6, 0.4]),
                 np.where(nat == "Arab expat", rng.choice(["Arabic (Standard)", "English"], n, p=[0.7, 0.3]), "English")))
    uptake_logit = (0.1 - 0.025 * distance + 0.5 * (df["channel"] == "WhatsApp") + 0.2 * (df["channel"] == "Voice Call")
                    - 0.3 * (df["occupation_group"] == "Outdoor / heat-exposed") + 0.25 * is_child + rng.normal(0, 0.4, n))
    df["uptake_propensity"] = _sigmoid(uptake_logit).round(3)
    return assign_households(df, rng)


def assign_households(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Attach children to plausible parent households (adult aged 22-50, one household per adult)."""
    df = df.copy()
    df["household_id"] = [f"HH-S{i:05d}" for i in range(len(df))]
    df["household_role"] = np.where(df["is_child"] == 1, "Child", "Adult")
    parents = df.index[(df["is_child"] == 0) & (df["age_years"].between(22, 50))].to_numpy()
    children = df.index[df["is_child"] == 1].to_numpy()
    if len(parents) and len(children):
        chosen = rng.choice(parents, size=len(children), replace=True)
        df.loc[children, "household_id"] = df.loc[chosen, "household_id"].to_numpy()
        df.loc[children, "region"] = df.loc[chosen, "region"].to_numpy()
        df.loc[children, "nationality_group"] = df.loc[chosen, "nationality_group"].to_numpy()
        df.loc[chosen, "household_role"] = "Parent / Caregiver"
    df["household_size"] = df.groupby("household_id")["person_id"].transform("count")
    return df


# --------------------------------------------------------------------------------------
# 5. HOUSEHOLD KINSHIP GRAPH & FAMILY-SYNC DETECTOR
# --------------------------------------------------------------------------------------
def _relation(role_a: str, role_b: str) -> str:
    if "Child" in (role_a, role_b) and ("Parent" in role_a + role_b or "Grandparent" in role_a + role_b):
        return "grandchild-of" if "Grandparent" in role_a + role_b else "parent-of"
    return "household-member"


def build_household_graph(members: list[dict]) -> nx.Graph:
    """Build a NetworkX kinship graph. `members` are persona dicts (or dicts with the same keys)."""
    g = nx.Graph()
    for m in members:
        g.add_node(
            m["id"], name=m["name"], role=m["role"], segment=m["segment"], age_years=m["age_years"],
            months_overdue=(m.get("months_overdue") or 0.0), region=m["region"], language=m["language"],
            channel=m["channel"],
        )
    adults = [m for m in members if m["segment"] != "Well-Child"]
    kids = [m for m in members if m["segment"] == "Well-Child"]
    for a in adults:
        for k in kids:
            g.add_edge(a["id"], k["id"], relation=_relation(a["role"], k["role"]))
    for i, a in enumerate(adults):
        for b in adults[i + 1:]:
            g.add_edge(a["id"], b["id"], relation="household-member")
    for i, a in enumerate(kids):
        for b in kids[i + 1:]:
            g.add_edge(a["id"], b["id"], relation="sibling")
    return g


def persona_household_graph(household_id: str) -> nx.Graph:
    members = [p for p in get_personas() if p["household_id"] == household_id]
    return build_household_graph(members)


def household_sync_bundle(graph: nx.Graph, min_overdue_months: float = 0.5) -> dict | None:
    """Detect an overdue adult (IFHAS) + overdue child (Well-Child) pair and bundle them.

    Returns a "Synchronized Family Prevention Appointment" proposal, or None when the household
    does not have both an overdue adult and an overdue child.
    """
    overdue_adults = [n for n, d in graph.nodes(data=True) if d["segment"] != "Well-Child" and d["months_overdue"] >= min_overdue_months]
    overdue_kids = [n for n, d in graph.nodes(data=True) if d["segment"] == "Well-Child" and d["months_overdue"] >= min_overdue_months]
    if not overdue_adults or not overdue_kids:
        return None
    nodes = graph.nodes
    region = nodes[overdue_adults[0]]["region"]
    lead = max(overdue_adults, key=lambda n: nodes[n]["months_overdue"])
    members = overdue_adults + overdue_kids
    separate_trips = len(members)
    return {
        "bundle_name": "Synchronized Family Prevention Appointment",
        "members": [{"id": n, "name": nodes[n]["name"], "pathway": "Well-Child Visit" if nodes[n]["segment"] == "Well-Child" else "IFHAS",
                     "months_overdue": nodes[n]["months_overdue"]} for n in members],
        "lead_contact": nodes[lead]["name"], "lead_contact_id": lead,
        "preferred_language": nodes[lead]["language"], "preferred_channel": nodes[lead]["channel"],
        "site": REGION_CLINICS.get(region, "Preventive Care Centre (demo site)"),
        "trips_saved": separate_trips - 1, "separate_trips": separate_trips, "bundled_trips": 1,
        "scheduling_note": ("Book back-to-back slots in one visit window: the adult fasting panel first, then the child's growth, "
                            "developmental and immunization checks, so nobody travels twice."),
    }


def graph_layout_positions(graph: nx.Graph) -> dict[str, tuple[float, float]]:
    """Deterministic layout for Plotly rendering (adults on top row, children on bottom row)."""
    adults = sorted(n for n, d in graph.nodes(data=True) if d["segment"] != "Well-Child")
    kids = sorted(n for n, d in graph.nodes(data=True) if d["segment"] == "Well-Child")
    pos: dict[str, tuple[float, float]] = {}
    for row, group, y in ((0, adults, 1.0), (1, kids, 0.0)):
        for i, n in enumerate(group):
            x = (i + 1) / (len(group) + 1)
            pos[n] = (x, y)
    return pos


if __name__ == "__main__":  # quick smoke test: python -m data.synthetic_generator
    pop = generate_population(1500)
    print(pop.shape, "unmet-need prevalence:", round(pop["unmet_need"].mean(), 3))
    print("mean completeness:", round(pop["data_completeness"].mean(), 3))
    for p in get_personas():
        c = compute_completeness(p["available_domains"], p["consent_default"])
        print(f'{p["id"]} {p["name"]:<22} {p["segment"]:<10} overdue={p["months_overdue"]} completeness={c:.0%}')
    bundle = household_sync_bundle(persona_household_graph("HH-AM"))
    print("sync bundle:", bundle["bundle_name"] if bundle else None, [m["name"] for m in bundle["members"]])
