"""
VeloTech LifeCourse-360 - Foundry IQ-style Hybrid RAG over Abu Dhabi DoH preventive guidance.

IMPORTANT: the guideline chunks below are *illustrative paraphrased summaries written for this
hackathon demonstration*. They mirror the structure of the DoH Abu Dhabi Well-Child Visit and IFHAS
programmes and widely used clinical screening conventions (WHO growth/anemia cut-offs, M-CHAT-R/F
scoring, PHQ-9 bands, KDIGO CKD criteria). They are NOT the official text. Before any real
deployment, replace `GUIDELINE_CHUNKS` with the current official DoH documents.

Retrieval = BM25 keyword score + TF-IDF cosine similarity (hybrid fusion), with hard metadata
filters (programme, age, sex), citation snippets and a lightweight groundedness verifier.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from data.synthetic_generator import DOMAIN_WEIGHTS, FEATURE_TO_DOMAIN

DISCLAIMER = (
    "Illustrative summary written for the hackathon demo. Verify against current official DoH Abu Dhabi documents "
    "before any real-world use. Final clinical decisions rest with licensed physicians."
)
SRC_WC = "DoH Abu Dhabi · Well-Child Visit Programme (demo summary)"
SRC_IF = "DoH Abu Dhabi · IFHAS Adult Comprehensive Screening (demo summary)"

# --------------------------------------------------------------------------------------
# 1. GUIDELINE CHUNKS
# --------------------------------------------------------------------------------------
def _g(id_, program, title, text, age_min, age_max, sex="any", timing="", fasting="", recall="", followup="", tags=""):
    return dict(id=id_, program=program, title=title, text=text, age_min_months=age_min, age_max_months=age_max,
                sex=sex, timing=timing, fasting=fasting, recall=recall, followup=followup, tags=tags,
                source=SRC_WC if program == "Well-Child" else SRC_IF)


GUIDELINE_CHUNKS: list[dict] = [
    # ------------------------------ WELL-CHILD (0-5 years) ------------------------------
    _g("WC-01", "Well-Child", "Well-Child Visit schedule and developmental surveillance",
       "Children aged 0 to 5 years follow the Well-Child Visit schedule with contacts about every six months until age two and yearly from two to five. "
       "Each visit reviews growth, development, feeding, sleep, immunization status, oral health and caregiver wellbeing. "
       "A missed visit should trigger recall rather than waiting for the next age-based invitation.",
       0, 71, timing="Visit due at each recall point; book within 30 days of a missed visit.",
       fasting="No fasting needed for routine Well-Child checks.",
       recall="Every 6 months until age 2, then every 12 months to age 5.",
       followup="Document the visit, update the growth chart and record next recall date.",
       tags="well-child visit schedule recall missed overdue surveillance development growth"),
    _g("WC-02", "Well-Child", "Neurodevelopmental screening with M-CHAT-R/F",
       "Toddlers are screened with the M-CHAT-R/F autism screening questionnaire at the 18-month and 24-month visits, with a usable window of 16 to 30 months. "
       "A total score of 0 to 2 is low risk. A score of 3 to 7 is medium risk and needs the structured follow-up interview. A score of 8 or more is high risk and goes straight to referral. "
       "If the follow-up interview still scores 2 or more, the child is referred for a developmental and audiology evaluation.",
       16, 30, timing="Complete at the 18 or 24 month visit; medium-risk follow-up interview within 2 weeks.",
       fasting="No fasting needed.", recall="Repeat at 24 months if the 18-month screen was low risk.",
       followup="Positive follow-up: refer for developmental and audiology evaluation and notify the family physician.",
       tags="m-chat-r autism neurodevelopmental toddler developmental screening referral behavioral"),
    _g("WC-03", "Well-Child", "Growth, BMI and weight-for-height monitoring",
       "Length or height, weight and head circumference are plotted against WHO growth standards at every visit. From age two BMI-for-age is used. "
       "A weight-for-height or BMI z-score above plus 2 flags overweight and below minus 2 flags wasting, both of which need clinician review and feeding or activity counselling.",
       0, 71, timing="Measure at every Well-Child Visit.", fasting="No fasting needed.",
       recall="Every Well-Child Visit.", followup="z-score outside plus or minus 2: clinician review within 30 days.",
       tags="growth bmi weight height z-score overweight wasting nutrition obesity"),
    _g("WC-04", "Well-Child", "Anemia screening",
       "Hemoglobin is checked at about 12 months and whenever risk factors are present such as low intake of iron-rich foods or prematurity. "
       "Hemoglobin below 11.0 g/dL between 6 and 59 months meets the WHO definition of anemia and needs a clinician-led iron assessment.",
       6, 59, timing="Test at 12 months, or within 30 days when risk factors or a low value are known.",
       fasting="No fasting needed; a finger-prick or venous sample is sufficient.",
       recall="Recheck per the treating clinician, typically 4 to 8 weeks after starting treatment.",
       followup="Low hemoglobin: clinician iron assessment and dietary counselling within 30 days.",
       tags="anemia hemoglobin iron blood haemoglobin low"),
    _g("WC-05", "Well-Child", "Vision screening",
       "Vision screening begins at 3 years with visual acuity testing from age 4 where the child can cooperate. Referral to an eye specialist follows any abnormal result or parental concern.",
       36, 71, timing="Due from 3 years; book with the next Well-Child Visit.", fasting="No fasting needed.",
       recall="Yearly to age 5.", followup="Abnormal screen: optometry or ophthalmology referral within 30 days.",
       tags="vision eyes sight acuity screening preschool"),
    _g("WC-06", "Well-Child", "Audiometry and hearing screening",
       "Hearing is screened with audiometry at 4 to 5 years, before school entry, and earlier when there is a risk factor or speech concern. "
       "Newborn hearing results are confirmed in the child's record.",
       48, 71, timing="Due at 4 to 5 years.", fasting="No fasting needed.",
       recall="Once at school entry, sooner when concerns arise.", followup="Failed screen: audiology referral within 30 days.",
       tags="hearing audiometry audiology ears speech preschool"),
    _g("WC-07", "Well-Child", "Catch-up immunizations",
       "The immunization record is reviewed at every visit. Missing doses are given as catch-up according to the national schedule, which includes toddler boosters around 18 months and preschool boosters at 4 to 6 years. "
       "A catch-up plan is written down and booked before the family leaves.",
       0, 71, timing="Catch-up dose at the same visit where the gap is found, or within 14 days.",
       fasting="No fasting needed.", recall="Per national immunization schedule.",
       followup="Record doses given and schedule the next due dose before leaving.",
       tags="immunization vaccine catch-up booster vaccination gap"),
    _g("WC-08", "Well-Child", "Emirati Genome Program hereditary metabolic review",
       "Where the Emirati Genome Program flags a hereditary metabolic carrier status in the family, the Well-Child Visit confirms the newborn screening result is on file and offers genetic counselling and a metabolic review. "
       "Genomic data is only used with the family's authorization.",
       0, 71, timing="Offer at the next Well-Child Visit.", fasting="Metabolic bloods may need a short fast; the clinician confirms.",
       recall="Per genetic counsellor.", followup="Genetic counselling and metabolic clinic referral where indicated.",
       tags="egp emirati genome hereditary metabolic carrier genetic counselling newborn screening genomic"),
    _g("WC-09", "Well-Child", "Oral health",
       "A dental check and fluoride varnish are offered from the first tooth and at least twice a year, with brushing and sugar-intake advice for caregivers.",
       12, 71, timing="Offer at each visit from 12 months.", fasting="No fasting needed.",
       recall="Every 6 months.", followup="Refer to dental services for visible decay.", tags="oral dental teeth fluoride"),
    _g("WC-10", "Well-Child", "Caregiver wellbeing at Well-Child Visits",
       "Caregiver wellbeing is a routine part of the visit. A brief PHQ-2 or PHQ-9 prompt is offered to the parent or guardian and positive answers are linked to adult services.",
       0, 71, timing="At each visit.", fasting="No fasting needed.", recall="Every visit.",
       followup="Positive screen: signpost to IFHAS mental wellbeing review.", tags="caregiver parent mental wellbeing phq depression"),
    _g("WC-11", "Well-Child", "Household synchronization of preventive visits",
       "When a parent or caregiver is overdue for adult screening and a child in the same household is overdue for a Well-Child Visit, the family is offered one combined appointment to reduce travel and missed visits. "
       "Both pathways keep their own clinical content and consent.",
       0, 71, timing="Offer a single combined appointment within 30 days.", fasting="Adult fasting panel first, then child checks.",
       recall="Re-synchronize at each recall.", followup="Confirm attendance of every household member; recall absentees within 14 days.",
       tags="household family synchronization combined appointment parent child overdue bundle"),

    # ------------------------------ IFHAS (adults) ------------------------------
    _g("IF-01", "IFHAS", "IFHAS core screening, ages 18 to 39",
       "Adults aged 18 to 39 receive blood pressure, BMI and waist measurement, tobacco review and a family history review. "
       "Lipid and HbA1c testing is added when risk factors exist such as BMI of 25 or more, a family history of early heart disease or diabetes, or an Emirati Genome Program flag.",
       216, 479, timing="Due every 3 years; bring forward when risk factors or genomic flags exist.",
       fasting="Fast 9 to 12 hours (water allowed) if lipids with triglycerides or fasting glucose are included.",
       recall="Every 36 months if normal.", followup="Any abnormal result: clinician review within 30 days.",
       tags="ifhas adult core screening 18-39 young adult blood pressure bmi lipid hba1c risk factors"),
    _g("IF-02", "IFHAS", "IFHAS core screening, ages 40 to 75",
       "Adults aged 40 to 75 receive blood pressure, BMI, lipid profile, HbA1c or fasting glucose, kidney function with eGFR and urine albumin-creatinine ratio when diabetes, hypertension or heat exposure is present, and a cancer screening review.",
       480, 900, timing="Due every 2 years; yearly with prediabetes or diabetes.",
       fasting="Fast 9 to 12 hours (water allowed) for lipids with triglycerides and fasting glucose.",
       recall="Every 24 months; every 12 months with prediabetes, diabetes or kidney findings.",
       followup="Abnormal result: clinician review within 30 days.",
       tags="ifhas adult core screening 40-75 midlife cardiovascular lipid hba1c renal egfr uacr"),
    _g("IF-03", "IFHAS", "Cardiovascular lipids and familial hypercholesterolemia",
       "LDL cholesterol of 4.9 mmol/L (190 mg/dL) or more raises suspicion of familial hypercholesterolemia, as does a known EGP variant or a first-degree relative with early heart disease (men under 55, women under 65). "
       "Suspected cases are offered early specialist lipid review and cascade testing of first-degree relatives.",
       216, 900, timing="Urgent lipid review within 14 days when LDL is 4.9 mmol/L or more or an FH variant is flagged.",
       fasting="Non-fasting lipids are acceptable; fast 9 to 12 hours when triglycerides are required.",
       recall="Per lipid specialist, usually every 6 to 12 months once treated.",
       followup="Specialist lipid clinic referral and cascade counselling for first-degree relatives.",
       tags="lipid ldl cholesterol familial hypercholesterolemia fh egp variant cascade family history heart"),
    _g("IF-04", "IFHAS", "Diabetes and prediabetes",
       "HbA1c of 5.7 to 6.4 percent is the prediabetes range and 6.5 percent or more is in the diabetes range and must be confirmed with a repeat test. "
       "Prediabetes is rechecked yearly and offered lifestyle support. A history of gestational diabetes or a high polygenic risk score brings screening forward.",
       216, 900, timing="Repeat confirmatory test within 14 days if 6.5 percent or more; otherwise within 30 days.",
       fasting="HbA1c needs no fasting; fasting glucose needs 8 hours without food.",
       recall="Yearly for prediabetes.", followup="Confirmed diabetes: clinician-led care plan and structured education.",
       tags="diabetes prediabetes hba1c glucose gestational t2d polygenic risk"),
    _g("IF-05", "IFHAS", "Kidney health and climate-worker heat-stress panel",
       "Kidney function is checked with eGFR and urine albumin-creatinine ratio (uACR). An eGFR under 60 or a uACR of 30 mg/g or more that persists beyond three months meets chronic kidney disease criteria. "
       "Outdoor workers in high heat receive a heat-stress panel with creatinine and eGFR, uACR, urinalysis and electrolytes before the summer heat season and again after it.",
       216, 900, timing="Priority review within 30 days if eGFR is under 75 or uACR is 30 mg/g or more; repeat at 3 months.",
       fasting="Stay normally hydrated; avoid heavy exercise the day before and avoid fasting for kidney tests.",
       recall="Before and after the summer heat season for outdoor workers; otherwise yearly if abnormal.",
       followup="Persistent abnormality: nephrology advice and a workplace heat-protection review.",
       tags="renal kidney egfr uacr heat stress outdoor worker climate nephropathy hydration albuminuria ckd"),
    _g("IF-06", "IFHAS", "Colorectal cancer screening",
       "Adults aged 45 to 75 at average risk are offered a yearly faecal immunochemical test (FIT) or colonoscopy at intervals set by the clinician. A positive FIT is followed by colonoscopy within 4 to 8 weeks.",
       540, 900, timing="Offer FIT at the IFHAS visit; collect at home.", fasting="No fasting or diet change needed for FIT.",
       recall="FIT every 12 months.", followup="Positive FIT: colonoscopy within 4 to 8 weeks.",
       tags="colorectal bowel cancer fit faecal screening colonoscopy"),
    _g("IF-07", "IFHAS", "Breast cancer screening",
       "Women aged 40 to 74 are offered screening mammography every two years, more often when risk is higher. For comfort, premenopausal women are advised to book in the week after their period and to avoid deodorant or talc on the day.",
       480, 888, sex="F", timing="Book at the IFHAS visit; book within 30 days if overdue.",
       fasting="No fasting needed.", recall="Every 24 months.", followup="Abnormal result: recall for assessment within 2 weeks.",
       tags="breast mammography cancer women screening"),
    _g("IF-08", "IFHAS", "Cervical cancer screening",
       "Women aged 25 to 65 are offered cervical screening every three years. The test is best booked away from menstruation and without intercourse in the previous 48 hours.",
       300, 780, sex="F", timing="Book at the IFHAS visit.", fasting="No fasting needed.",
       recall="Every 36 months.", followup="Abnormal result: colposcopy referral within 4 weeks.",
       tags="cervical pap smear hpv women cancer screening"),
    _g("IF-09", "IFHAS", "Bone density",
       "Bone density testing by DXA is offered to women from 65 and men from 70, and earlier when there are risk factors such as steroid use, prior fracture or early menopause.",
       780, 900, timing="Book within 30 days if overdue.", fasting="No fasting needed; avoid calcium supplements for 24 hours before the scan.",
       recall="Every 24 months with osteopenia; otherwise per clinician.", followup="Low bone density: clinician review and fall-prevention advice.",
       tags="bone density dxa osteoporosis fracture seniors"),
    _g("IF-10", "IFHAS", "Mental wellbeing: PHQ-9 and GAD-7",
       "PHQ-9 and GAD-7 are offered at every IFHAS visit. A PHQ-9 of 10 to 14 is the moderate range and needs clinician follow-up within two weeks. Any positive answer on the self-harm item requires same-day clinician contact.",
       216, 900, timing="At every IFHAS visit; moderate range follow-up within 14 days.",
       fasting="No fasting needed.", recall="Every IFHAS visit, or sooner on request.",
       followup="PHQ-9 of 10 or more: clinician review within 14 days; positive item 9: same-day clinician contact.",
       tags="mental wellbeing phq-9 gad-7 depression anxiety mood behavioral"),
    _g("IF-11", "IFHAS", "Brain health and cognitive baseline",
       "Adults aged 65 and over receive a brief cognitive baseline such as the Mini-Cog or MoCA. Midlife vascular risk reduction (blood pressure, glucose, lipids, activity) is a core brain-health measure.",
       780, 900, timing="Offer at the IFHAS visit from age 65.", fasting="No fasting needed.",
       recall="Every 12 months or on family concern.", followup="Abnormal baseline: memory clinic referral.",
       tags="brain health cognitive dementia memory mini-cog moca seniors"),
    _g("IF-12", "IFHAS", "Closing the screening-to-care loop",
       "Every abnormal screening result is tracked until it is closed. A clinician reviews the result within 30 days, a navigator calls at day 14 if no appointment exists, and unresolved cases escalate at day 30. Completion is recorded back into the record.",
       216, 900, timing="Day 14 navigator contact; day 30 escalation.", fasting="Not applicable.", recall="Until closed.",
       followup="Record referral attendance and outcome; escalate unresolved cases at day 30.",
       tags="follow-up loop closure referral abnormal result care gap navigator"),
    _g("IF-13", "IFHAS", "Genomically informed early screening",
       "A familial hypercholesterolemia variant or a polygenic risk score in the top decile for coronary disease or type 2 diabetes brings lipid and glucose screening forward to age 18 or above, even when age-based rules would not yet invite the person. "
       "Results are discussed with a genetic counsellor and, where relevant, carrier information supports reproductive planning.",
       216, 900, timing="Bring forward the first lipid and HbA1c review to the next available visit.",
       fasting="Fast 9 to 12 hours if triglycerides or fasting glucose are included.", recall="Per risk, usually yearly at first.",
       followup="Genetic counselling and cascade testing offers for first-degree relatives.",
       tags="egp genomic polygenic risk score prs early screening carrier fertility genetic counselling"),
    _g("IF-14", "IFHAS", "Data authorization and insufficient-data fallback",
       "Only data domains the individual has authorized are used. When authorized data completeness is under 40 percent, risk extrapolation is not performed and the baseline age and gender schedule applies, so uncertain results never drive outreach.",
       216, 900, timing="Applies to every scoring request.", fasting="Not applicable.", recall="Re-evaluate when new data is authorized.",
       followup="Offer the person a chance to authorize more data or attend a standard IFHAS visit.",
       tags="consent authorization data completeness fallback governance privacy baseline"),
    _g("IF-15", "IFHAS", "Tobacco and healthy-behaviour support",
       "People who smoke are offered brief advice and cessation support at every visit, with a personal quit plan and follow-up contact within two weeks. Short post-visit micro-habits support lasting change.",
       216, 900, timing="Offer at the IFHAS visit.", fasting="No fasting needed.", recall="Every visit.",
       followup="Cessation follow-up call within 14 days.", tags="tobacco smoking cessation lifestyle behaviour habits"),
]
_CHUNK_BY_ID = {c["id"]: c for c in GUIDELINE_CHUNKS}


# --------------------------------------------------------------------------------------
# 2. HYBRID RETRIEVER (BM25 + TF-IDF cosine)
# --------------------------------------------------------------------------------------
_STOP = set("a an and are as at be by for from has have in is it of on or that the their this to was were will with within than not no if when who after before each".split())
_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-\.]*")


def tokenize(text: str) -> list[str]:
    return [t.strip(".-") for t in _TOKEN.findall(text.lower()) if t.strip(".-") and t.strip(".-") not in _STOP]


@dataclass
class Hit:
    id: str
    title: str
    program: str
    source: str
    score: float
    bm25: float
    cosine: float
    snippet: str
    chunk: dict = field(repr=False)

    def citation(self) -> dict:
        return {"id": self.id, "title": self.title, "source": self.source, "snippet": self.snippet,
                "score": round(self.score, 3), "bm25": round(self.bm25, 3), "cosine": round(self.cosine, 3)}


class DoHGuidelineKB:
    """Local hybrid retriever mimicking a Foundry IQ knowledge base (no network, no cost)."""

    def __init__(self, chunks: list[dict] | None = None, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks or GUIDELINE_CHUNKS
        self.k1, self.b = k1, b
        self._docs = [f'{c["title"]}. {c["text"]} {c["tags"]}' for c in self.chunks]
        self._tokens = [tokenize(d) for d in self._docs]
        self._tf = [Counter(t) for t in self._tokens]
        self._len = np.array([len(t) for t in self._tokens], dtype=float)
        self._avg = float(self._len.mean())
        df: Counter = Counter()
        for t in self._tokens:
            df.update(set(t))
        n = len(self._tokens)
        self._idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self._vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)
        self._mat = self._vec.fit_transform(self._docs)

    # ---- scoring ----
    def _bm25(self, q_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.chunks))
        for i, tf in enumerate(self._tf):
            s = 0.0
            for w in q_tokens:
                if w not in tf:
                    continue
                f = tf[w]
                s += self._idf.get(w, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self._len[i] / self._avg))
            scores[i] = s
        return scores

    def _cosine(self, query: str) -> np.ndarray:
        q = self._vec.transform([query])
        return np.asarray((self._mat @ q.T).todense()).ravel()

    @staticmethod
    def _best_sentence(text: str, q_tokens: set[str], limit: int = 260) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        best = max(sentences, key=lambda s: len(q_tokens & set(tokenize(s))) / (1 + 0.02 * len(s)))
        return best if len(best) <= limit else best[: limit - 1].rstrip() + "…"

    def _allowed(self, c: dict, program: str | None, age_months: float | None, sex: str | None) -> bool:
        if program and c["program"] != program:
            return False
        if age_months is not None and not (c["age_min_months"] <= age_months <= c["age_max_months"]):
            return False
        if sex and c["sex"] not in ("any", sex):
            return False
        return True

    def search(self, query: str, top_k: int = 4, program: str | None = None, age_months: float | None = None,
               sex: str | None = None, alpha: float = 0.5) -> list[Hit]:
        """Hybrid search. alpha weights cosine vs BM25 (after max-normalisation)."""
        q_tokens = tokenize(query)
        bm = self._bm25(q_tokens)
        cs = self._cosine(query)
        bm_n = bm / bm.max() if bm.max() > 0 else bm
        cs_n = cs / cs.max() if cs.max() > 0 else cs
        fused = alpha * cs_n + (1 - alpha) * bm_n
        hits: list[Hit] = []
        for i in np.argsort(-fused):
            c = self.chunks[int(i)]
            if fused[i] <= 0 or not self._allowed(c, program, age_months, sex):
                continue
            hits.append(Hit(c["id"], c["title"], c["program"], c["source"], float(fused[i]), float(bm_n[i]), float(cs_n[i]),
                            self._best_sentence(c["text"], set(q_tokens)), c))
            if len(hits) >= top_k:
                break
        return hits

    def get(self, chunk_id: str) -> dict:
        return _CHUNK_BY_ID[chunk_id]

    def snippet_for(self, chunk_id: str, query: str) -> dict:
        c = self.get(chunk_id)
        return {"id": c["id"], "title": c["title"], "source": c["source"],
                "snippet": self._best_sentence(c["text"], set(tokenize(query or c["title"])))}

    # ---- groundedness ----
    def groundedness(self, answer: str, hits: list[Hit], threshold: float = 0.6) -> dict:
        """Share of answer sentences whose content words are supported by the retrieved chunks."""
        support_tokens: set[str] = set()
        for h in hits:
            support_tokens |= set(tokenize(f'{h.chunk["title"]} {h.chunk["text"]} {h.chunk["timing"]} {h.chunk["followup"]} {h.chunk["recall"]} {h.chunk["fasting"]}'))
        claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if len(tokenize(s)) >= 4]
        details = []
        for s in claims:
            toks = [t for t in tokenize(s) if not t.isdigit()]
            cov = sum(t in support_tokens for t in toks) / max(1, len(toks))
            details.append({"claim": s, "coverage": round(cov, 2), "supported": cov >= threshold})
        score = sum(d["supported"] for d in details) / max(1, len(details))
        return {"score": round(score, 3), "n_claims": len(details), "claims": details}

    def answer(self, question: str, person: dict | None = None, top_k: int = 3) -> dict:
        """Extractive grounded answer with citations (used by the offline engine)."""
        program = None
        age_m = None
        sex = None
        if person:
            program = "Well-Child" if person["segment"] == "Well-Child" else "IFHAS"
            age_m = person["age_years"] * 12
            sex = person["sex"]
        hits = self.search(question, top_k=top_k, program=program, age_months=age_m, sex=sex)
        text = " ".join(h.snippet for h in hits) or "No approved guideline passage matched this question."
        return {"answer": text, "citations": [h.citation() for h in hits], "groundedness": self.groundedness(text, hits)}


_KB: DoHGuidelineKB | None = None


def get_kb() -> DoHGuidelineKB:
    global _KB
    if _KB is None:
        _KB = DoHGuidelineKB()
    return _KB


# --------------------------------------------------------------------------------------
# 3. PATHWAY BUILDER (rules select chunks; chunks supply timing, fasting, recall, follow-up)
# --------------------------------------------------------------------------------------
PRIORITY_ORDER = {"Urgent": 0, "Priority": 1, "Routine": 2}
PRIORITY_WINDOW = {"Urgent": "within 14 days", "Priority": "within 30 days", "Routine": "at the next scheduled recall"}


def _visible_features(person: dict, consent: set[str] | None) -> dict:
    """Feature dict with unconsented or unavailable domains blanked to None."""
    out = {}
    for k, v in person["features"].items():
        dom = FEATURE_TO_DOMAIN.get(k)
        if dom and consent is not None and dom not in consent:
            out[k] = None
        else:
            out[k] = v
    return out


def _item(chunk_id: str, priority: str, rationale: str, person: dict, query: str = "", title: str | None = None) -> dict:
    kb = get_kb()
    c = kb.get(chunk_id)
    overdue = person.get("months_overdue") or 0
    when = c["timing"] or PRIORITY_WINDOW[priority]
    if overdue and priority != "Routine":
        when = f'{when} (currently {overdue:.0f} months overdue)'
    return {
        "id": chunk_id, "title": title or c["title"], "program": c["program"], "priority": priority,
        "rationale": rationale, "window": when, "fasting": c["fasting"], "recall": c["recall"],
        "followup": c["followup"], "citation": kb.snippet_for(chunk_id, query or title or c["title"]),
    }


def recommend_pathway(person: dict, consent: set[str] | None = None, household_overdue: bool = False) -> list[dict]:
    """Personalised, guideline-grounded pathway. Uses only features from consented data domains."""
    f = _visible_features(person, consent)
    items: list[dict] = []
    age_m = person["age_years"] * 12
    overdue = person.get("months_overdue") or 0

    def has(key: str) -> bool:
        return f.get(key) is not None

    if person["segment"] == "Well-Child":
        prio = "Priority" if overdue > 0 else "Routine"
        items.append(_item("WC-01", prio, f"Visit overdue by {overdue:.0f} months; recall now." if overdue else "Routine recall.", person, "missed visit recall schedule"))
        if 16 <= age_m <= 30:
            m = f.get("mchat_score")
            if m is not None and m >= 8:
                items.append(_item("WC-02", "Urgent", f"M-CHAT-R/F score {m:.0f} is in the high-risk band: direct referral.", person, "high risk referral M-CHAT-R score"))
            elif m is not None and m >= 3:
                items.append(_item("WC-02", "Urgent", f"M-CHAT-R/F score {m:.0f} is in the medium-risk band: structured follow-up interview.", person, "medium risk follow-up interview M-CHAT-R score"))
            else:
                items.append(_item("WC-02", "Priority" if overdue > 0 else "Routine", "Toddler is inside the 16 to 30 month M-CHAT-R/F window.", person, "M-CHAT-R/F window 18 and 24 month visit"))
        z = f.get("bmi_or_wfh_z")
        if z is not None and abs(z) >= 1.5:
            items.append(_item("WC-03", "Priority", f"Weight-for-height z-score {z:+.1f} is approaching or beyond the plus or minus 2 review line.", person, "z-score overweight wasting review"))
        else:
            items.append(_item("WC-03", "Routine", "Routine growth monitoring.", person, "growth monitoring WHO standards"))
        hb = f.get("hemoglobin")
        if hb is not None and hb < 11.0 and 6 <= age_m <= 59:
            items.append(_item("WC-04", "Priority", f"Hemoglobin {hb:.1f} g/dL is below the 11.0 anemia threshold.", person, "hemoglobin below 11 anemia"))
        elif 9 <= age_m <= 15:
            items.append(_item("WC-04", "Routine", "Routine 12-month hemoglobin check.", person, "hemoglobin 12 months"))
        if age_m >= 36:
            items.append(_item("WC-05", "Priority" if overdue > 0 else "Routine", "Vision screening is due from 3 years.", person, "vision screening acuity"))
        if age_m >= 48:
            items.append(_item("WC-06", "Priority" if overdue > 0 else "Routine", "Audiometry is due at 4 to 5 years before school entry.", person, "audiometry hearing school entry"))
        gaps = f.get("immunization_gaps")
        if gaps is not None and gaps > 0:
            items.append(_item("WC-07", "Priority", f"{gaps:.0f} immunization gap(s) on file.", person, "catch-up immunization missing doses"))
        if f.get("metabolic_carrier"):
            items.append(_item("WC-08", "Priority", "EGP hereditary metabolic carrier flag in the family.", person, "EGP hereditary metabolic carrier genetic counselling"))
        if age_m >= 12:
            items.append(_item("WC-09", "Routine", "Dental check and fluoride varnish are offered from 12 months.", person, "dental fluoride"))
        items.append(_item("WC-10", "Routine", "Caregiver wellbeing prompt at the visit.", person, "caregiver wellbeing PHQ"))
        if household_overdue:
            items.append(_item("WC-11", "Priority", "A parent or caregiver in this household is also overdue for IFHAS.", person, "household combined appointment parent child"))
    else:
        age = person["age_years"]
        sex = person["sex"]
        core_id = "IF-01" if age < 40 else "IF-02"
        items.append(_item(core_id, "Priority" if overdue > 0 else "Routine",
                           f"IFHAS core screening is {overdue:.0f} months overdue." if overdue else "Core IFHAS screening on schedule.", person, "core screening recall"))
        ldl, fh, fam = f.get("ldl"), f.get("fh_variant"), f.get("fam_hx_cvd")
        if (fh and fh >= 1) or (ldl is not None and ldl >= 4.9):
            why = []
            if fh:
                why.append("EGP familial hypercholesterolemia variant")
            if ldl is not None and ldl >= 4.9:
                why.append(f"LDL {ldl:.1f} mmol/L")
            if fam:
                why.append("family history of early heart disease")
            items.append(_item("IF-03", "Urgent", "; ".join(why).capitalize() + ".", person, "familial hypercholesterolemia LDL cascade"))
            if fh or (f.get("prs_cvd") or 0) >= 90:
                items.append(_item("IF-13", "Priority", "Genomic flag brings lipid and glucose screening forward ahead of age-based invitation.", person, "genomic early screening PRS EGP"))
        elif ldl is not None and ldl >= 3.4 and fam:
            items.append(_item("IF-03", "Priority", f"LDL {ldl:.1f} mmol/L with family history of early heart disease.", person, "lipid LDL family history"))
        a1c, prs_t2d = f.get("hba1c"), f.get("prs_t2d")
        if a1c is not None and a1c >= 6.5:
            items.append(_item("IF-04", "Urgent", f"HbA1c {a1c:.1f}% is in the diabetes range and needs confirmation.", person, "HbA1c 6.5 diabetes range confirm repeat"))
        elif a1c is not None and a1c >= 5.7:
            items.append(_item("IF-04", "Priority", f"HbA1c {a1c:.1f}% is in the prediabetes range." + (" EGP polygenic risk for type 2 diabetes is high." if (prs_t2d or 0) >= 80 else ""), person, "prediabetes HbA1c yearly recheck"))
        egfr, uacr, heat = f.get("egfr"), f.get("uacr"), f.get("heat_exposure")
        renal_why = []
        if egfr is not None and egfr < 75:
            renal_why.append(f"eGFR {egfr:.0f}")
        if uacr is not None and uacr >= 30:
            renal_why.append(f"uACR {uacr:.0f} mg/g")
        if heat is not None and heat >= 0.6:
            renal_why.append(f"high heat exposure index {heat:.2f}")
        if renal_why:
            urgent = (egfr is not None and egfr < 60) and (uacr is not None and uacr >= 30)
            items.append(_item("IF-05", "Urgent" if urgent else "Priority", "Kidney and heat-stress review: " + ", ".join(renal_why) + ".", person, "kidney egfr uacr heat stress outdoor worker panel"))
        if age >= 45:
            items.append(_item("IF-06", "Priority" if overdue > 0 else "Routine", "Colorectal screening is due from age 45.", person, "colorectal FIT yearly"))
        if sex == "F" and 40 <= age <= 74:
            items.append(_item("IF-07", "Priority" if overdue > 0 else "Routine", "Mammography every two years for women 40 to 74.", person, "mammography women every two years"))
        if sex == "F" and 25 <= age <= 65:
            items.append(_item("IF-08", "Priority" if overdue > 0 else "Routine", "Cervical screening every three years for women 25 to 65.", person, "cervical screening every three years"))
        if (sex == "F" and age >= 65) or (sex == "M" and age >= 70):
            items.append(_item("IF-09", "Priority" if overdue > 0 else "Routine", "Bone density is due for this age and sex.", person, "bone density DXA"))
        phq = f.get("phq9")
        if phq is not None and phq >= 10:
            items.append(_item("IF-10", "Priority", f"PHQ-9 of {phq:.0f} is in the moderate range: clinician follow-up within 2 weeks.", person, "PHQ-9 moderate range clinician follow-up"))
        else:
            items.append(_item("IF-10", "Routine", "PHQ-9 and GAD-7 are offered at every IFHAS visit.", person, "PHQ-9 GAD-7 every visit"))
        if age >= 65:
            items.append(_item("IF-11", "Routine", "Cognitive baseline from age 65.", person, "cognitive baseline Mini-Cog MoCA"))
        if f.get("smoker"):
            items.append(_item("IF-15", "Routine", "Tobacco use recorded.", person, "tobacco cessation brief advice"))
        items.append(_item("IF-12", "Routine", "Any abnormal result is tracked until the loop is closed.", person, "close the loop abnormal result day 30"))
    items.sort(key=lambda i: PRIORITY_ORDER[i["priority"]])
    return items


def baseline_schedule(person: dict) -> list[dict]:
    """Age and gender only pathway used by the insufficient-data fallback (no AI extrapolation)."""
    age_m = person["age_years"] * 12
    age, sex = person["age_years"], person["sex"]
    ids: list[tuple[str, str]] = []
    if person["segment"] == "Well-Child":
        ids.append(("WC-01", "Age-based Well-Child Visit schedule."))
        ids.append(("WC-03", "Growth monitoring at every visit."))
        ids.append(("WC-07", "Immunization record review at every visit."))
        if 16 <= age_m <= 30:
            ids.append(("WC-02", "Age-based M-CHAT-R/F window."))
        if age_m >= 36:
            ids.append(("WC-05", "Age-based vision screening."))
        if age_m >= 48:
            ids.append(("WC-06", "Age-based audiometry."))
    else:
        ids.append(("IF-01" if age < 40 else "IF-02", "Age-based IFHAS core screening."))
        if age >= 45:
            ids.append(("IF-06", "Age-based colorectal screening."))
        if sex == "F" and 40 <= age <= 74:
            ids.append(("IF-07", "Age and gender-based mammography."))
        if sex == "F" and 25 <= age <= 65:
            ids.append(("IF-08", "Age and gender-based cervical screening."))
        if (sex == "F" and age >= 65) or (sex == "M" and age >= 70):
            ids.append(("IF-09", "Age and gender-based bone density."))
        ids.append(("IF-10", "PHQ-9 and GAD-7 at every IFHAS visit."))
        ids.append(("IF-14", "Authorize more data to unlock personalised prioritization."))
    return [_item(cid, "Routine", why, person) for cid, why in ids]


def consolidated_previsit_rules(items: list[dict]) -> list[str]:
    """Merge per-item fasting rules into plain-language pre-visit instructions."""
    ids = {i["id"] for i in items}
    rules: list[str] = []
    if ids & {"IF-01", "IF-02", "IF-03", "IF-13", "IF-04"}:
        rules.append("Fast 9 to 12 hours before the visit (water is fine) if the lipid panel with triglycerides or a fasting glucose is included. HbA1c, FIT, eGFR and uACR do not need fasting.")
    if "IF-05" in ids:
        rules.append("Stay normally hydrated before kidney tests and avoid heavy exercise the day before.")
    if "IF-07" in ids:
        rules.append("Mammography: premenopausal women book the week after their period and skip deodorant or talc that day.")
    if "IF-08" in ids:
        rules.append("Cervical screening: avoid menstruation and intercourse for 48 hours beforehand.")
    if "IF-09" in ids:
        rules.append("Bone density: avoid calcium supplements for 24 hours beforehand.")
    if "IF-06" in ids:
        rules.append("FIT kit is collected at home; no diet change is needed.")
    if "WC-08" in ids:
        rules.append("Metabolic review: the clinician will confirm whether a short fast is needed for the child.")
    if not rules:
        rules.append("No fasting is needed for this visit.")
    return rules


def recall_summary(items: list[dict]) -> list[str]:
    seen: list[str] = []
    for i in items:
        wanted = i["priority"] != "Routine" or i["id"] in ("WC-01", "IF-01", "IF-02")
        if i["recall"] and wanted:
            line = f'{i["title"]}: {i["recall"]}'
            if line not in seen:
                seen.append(line)
    return seen[:6]


if __name__ == "__main__":  # smoke test: python -m src.doh_guidelines_kb
    from data.synthetic_generator import get_personas

    kb = get_kb()
    for q, kw in [("toddler autism screening medium risk", {}), ("outdoor worker kidney heat", {}), ("LDL familial hypercholesterolemia", {})]:
        print("\nQ:", q)
        for h in kb.search(q, top_k=3, **kw):
            print(f"  {h.id} score={h.score:.2f} bm25={h.bm25:.2f} cos={h.cosine:.2f} :: {h.title}")
    for p in get_personas():
        pw = recommend_pathway(p, set(p["consent_default"]))
        print(f'\n{p["id"]} {p["name"]}: ' + ", ".join(f'{i["id"]}({i["priority"][0]})' for i in pw))
    ans = kb.answer("When is the M-CHAT-R screening done and what happens if medium risk?", get_personas()[1])
    print("\nGrounded answer:", ans["groundedness"]["score"], ans["answer"][:200])
