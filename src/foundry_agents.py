"""
VeloTech LifeCourse-360 - Microsoft Foundry / Agent Framework orchestration layer.

Implements (all at $0, offline-first):
  AI-103 #1  Model router + side-by-side evaluation (Phi-4, GPT-4o-mini, Jais profile, deterministic engine)
  AI-103 #2  Tool-using generative app (typed tools invoked by agents, surfaced in the UI)
  AI-103 #3  VS Code agent development: OpenTelemetry-style spans, optional stdout trace (VELOTECH_TRACE_STDOUT=1)
  AI-103 #4  Custom tools as typed `@ai_function` callables
  AI-103 #5  MCP tools through `MCPClient` -> `src/mcp_server.py`
  AI-103 #6  Foundry IQ hybrid RAG (`src/doh_guidelines_kb.py`)
  AI-103 #7  Six-state agent-driven workflow
  AI-103 #8  Specialist agent classes in the Microsoft Agent Framework pattern
  AI-103 #9  Four-agent swarm orchestration
  AI-103 #10 Responsible-AI gate (boundary, completeness fallback, calibration, parity, groundedness, k-anonymity)

Live LLMs are optional. Set VELOTECH_LLM_MODE to `github` (needs GITHUB_TOKEN) or `foundry`
(Foundry Local, default http://localhost:5272/v1; Ollama also works by setting FOUNDRY_LOCAL_ENDPOINT=
http://localhost:11434/v1 and FOUNDRY_LOCAL_MODEL). Any failure falls back to the deterministic engine
and is recorded in the trace.
"""
from __future__ import annotations

import contextlib
import inspect
import json
import os
import re
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

import pandas as pd

from data.synthetic_generator import (
    DOMAIN_KEYS, REGION_CLINICS, compute_completeness, get_persona, get_personas, household_sync_bundle,
    persona_household_graph,
)
from src import opencv_biosignal as bio
from src.doh_guidelines_kb import (
    PRIORITY_ORDER, baseline_schedule, consolidated_previsit_rules, get_kb, recommend_pathway, recall_summary, tokenize,
)
from src.ml_conformal_engine import FALLBACK_THRESHOLD, RiskEngine, get_engine
from src.mcp_server import MCPServerLite
from src.safeguards_evaluation import cohort_table, parity_summary

WORKFLOW_STATES = [
    "01_Consent_Verification", "02_MCP_FHIR_Signal_Ingest", "03_Conformal_Risk_Scoring",
    "04_Foundry_IQ_Guideline_RAG", "05_Responsible_AI_Safeguard_Gate", "06_CulturaVoice_Outreach_Dispatch",
]
BOUNDARY = ("POPULATION PRIORITIZATION VS. INDIVIDUAL CLINICAL DECISION-MAKING: This platform prioritizes preventive outreach and recall "
            "timing within approved DoH guidelines; final clinical decisions rest exclusively with licensed physicians.")


# --------------------------------------------------------------------------------------
# 1. TRACING (OpenTelemetry-compatible span records)
# --------------------------------------------------------------------------------------
class TraceLogger:
    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex
        self.spans: list[dict] = []
        self._stack: list[str] = []
        self.stdout = os.getenv("VELOTECH_TRACE_STDOUT") == "1"

    @contextlib.contextmanager
    def span(self, name: str, kind: str = "INTERNAL", **attrs):
        sid = uuid.uuid4().hex[:16]
        rec = {"trace_id": self.trace_id, "span_id": sid, "parent_span_id": self._stack[-1] if self._stack else None,
               "name": name, "kind": kind, "start_time_unix_nano": time.time_ns(), "attributes": dict(attrs), "status": "OK"}
        self._stack.append(sid)
        t0 = time.perf_counter()
        try:
            yield rec
        except Exception as exc:
            rec["status"] = "ERROR"
            rec["attributes"]["exception"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._stack.pop()
            rec["end_time_unix_nano"] = time.time_ns()
            rec["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            self.spans.append(rec)
            if self.stdout:
                print(json.dumps({k: rec[k] for k in ("trace_id", "span_id", "parent_span_id", "name", "kind", "duration_ms", "status")}), file=sys.stderr)

    def to_frame(self) -> pd.DataFrame:
        rows = [{"span": s["name"], "kind": s["kind"], "parent": s["parent_span_id"], "span_id": s["span_id"], "duration_ms": s["duration_ms"],
                 "status": s["status"], "attributes": json.dumps(s["attributes"], default=str)[:240]} for s in self.spans]
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 2. @ai_function (Agent Framework style typed tools)
# --------------------------------------------------------------------------------------
def ai_function(name: str | None = None, description: str | None = None) -> Callable:
    """Mark a callable as an agent tool and attach a JSON schema built from its type hints."""
    def deco(fn: Callable) -> Callable:
        props, required = {}, []
        for pname, param in inspect.signature(fn).parameters.items():
            if pname == "self":
                continue
            ann = str(param.annotation)
            jtype = "array" if "list" in ann else ("integer" if "int" in ann else ("number" if "float" in ann else ("boolean" if "bool" in ann else "string")))
            props[pname] = {"type": jtype}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        fn.ai_schema = {"name": name or fn.__name__, "description": description or (inspect.getdoc(fn) or "").split("\n")[0],
                        "parameters": {"type": "object", "properties": props, "required": required}}
        return fn
    return deco


class MCPClient:
    """Thin MCP client: every call is a JSON-RPC `tools/call` and a CLIENT trace span."""

    def __init__(self, server: MCPServerLite, tracer: TraceLogger):
        self.server, self.tracer, self.calls, self._id = server, tracer, [], 0

    def call(self, tool: str, **arguments) -> Any:
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}
        t0 = time.perf_counter()
        with self.tracer.span(f"mcp.tools/call {tool}", kind="CLIENT", **{"mcp.tool": tool, "mcp.args": json.dumps(arguments, default=str)[:160]}) as sp:
            reply = self.server.handle_jsonrpc(msg)["result"]
            sp["attributes"]["mcp.is_error"] = reply["isError"]
        ms = round((time.perf_counter() - t0) * 1000, 2)
        self.calls.append({"tool": tool, "arguments": arguments, "ms": ms, "is_error": reply["isError"], "request": msg,
                           "response_preview": reply["content"][0]["text"][:400]})
        if reply["isError"]:
            raise RuntimeError(reply["content"][0]["text"])
        return reply["structuredContent"]


# --------------------------------------------------------------------------------------
# 3. CUSTOM TOOLS (typed @ai_function)
# --------------------------------------------------------------------------------------
class VeloTechTools:
    def __init__(self, engine: RiskEngine, tracer: TraceLogger):
        self.engine, self.tracer, self.calls = engine, tracer, []

    def _log(self, name: str, args: dict, summary: str) -> None:
        self.calls.append({"tool": name, "arguments": args, "summary": summary})

    @ai_function(name="score_risk_conformal", description="Scikit-Learn + SHAP + conformal 95% interval risk scorer")
    def score_risk_conformal(self, person_id: str, consent_domains: list[str]) -> dict:
        """Calibrated risk with SHAP attribution and conformal interval (blocked below 40% data completeness)."""
        with self.tracer.span("tool.score_risk_conformal", kind="INTERNAL", person_id=person_id):
            res = self.engine.score_person(get_persona(person_id), set(consent_domains))
        self._log("score_risk_conformal", {"person_id": person_id, "consent_domains": consent_domains}, res["summary"])
        return res

    @ai_function(name="analyze_household_graph", description="NetworkX household kinship graph analyzer")
    def analyze_household_graph(self, household_id: str) -> dict:
        """Detect overdue adult plus overdue child and propose a synchronized family appointment."""
        with self.tracer.span("tool.analyze_household_graph", kind="INTERNAL", household_id=household_id):
            g = persona_household_graph(household_id)
            bundle = household_sync_bundle(g)
        self._log("analyze_household_graph", {"household_id": household_id}, bundle["bundle_name"] if bundle else "no overdue adult+child pair")
        return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "bundle": bundle}

    @ai_function(name="extract_rppg_biosignal", description="OpenCV green-channel rPPG heart-rate / vascular stress proxy extractor")
    def extract_rppg_biosignal(self, hr_bpm: float, seed: int) -> dict:
        """Contactless pulse from a synthetic face video (wellness indicator only)."""
        with self.tracer.span("tool.extract_rppg_biosignal", kind="INTERNAL", hr_bpm=hr_bpm):
            r = bio.extract_rppg(bio.generate_session(hr_bpm=hr_bpm, seed=seed), 30)
        out = {k: r[k] for k in ("hr_bpm", "snr_db", "quality", "rmssd_ms", "stress_index", "disclaimer")}
        self._log("extract_rppg_biosignal", {"hr_bpm": hr_bpm, "seed": seed}, f'{out["hr_bpm"]} bpm, {out["quality"]}')
        return out

    @ai_function(name="retrieve_guidelines", description="Foundry IQ hybrid RAG over DoH Well-Child and IFHAS guidance")
    def retrieve_guidelines(self, query: str, person_id: str) -> dict:
        """Grounded guideline passages with citations and a groundedness score."""
        with self.tracer.span("tool.retrieve_guidelines", kind="INTERNAL", query=query[:80]):
            res = get_kb().answer(query, get_persona(person_id))
        self._log("retrieve_guidelines", {"query": query, "person_id": person_id}, f'{len(res["citations"])} citations, groundedness {res["groundedness"]["score"]:.2f}')
        return res

    def schemas(self) -> list[dict]:
        return [getattr(self, n).ai_schema for n in dir(self) if hasattr(getattr(self, n, None), "ai_schema")]


# --------------------------------------------------------------------------------------
# 4. CULTURAVOICE OUTREACH STUDIO (Arabic Emirati / Standard, English, Urdu)
# --------------------------------------------------------------------------------------
ITEM_LABELS = {
    "IF-01": ("adult health check", "الفحص الصحي الشامل للبالغين", "بالغوں کا مکمل ہیلتھ چیک اپ"),
    "IF-02": ("adult health check", "الفحص الصحي الشامل للبالغين", "بالغوں کا مکمل ہیلتھ چیک اپ"),
    "IF-03": ("cholesterol review", "مراجعة الكوليسترول", "کولیسٹرول کا جائزہ"),
    "IF-04": ("blood sugar (HbA1c) check", "فحص سكر الدم (HbA1c)", "بلڈ شوگر (HbA1c) ٹیسٹ"),
    "IF-05": ("kidney and heat-stress check", "فحص الكلى والإجهاد الحراري", "گردوں اور گرمی کے دباؤ کا ٹیسٹ"),
    "IF-06": ("bowel screening (FIT)", "فحص القولون (FIT)", "آنت کی اسکریننگ (FIT)"),
    "IF-07": ("mammogram", "تصوير الثدي (الماموغرام)", "میموگرام"),
    "IF-08": ("cervical screening", "فحص عنق الرحم", "سروائیکل اسکریننگ"),
    "IF-09": ("bone density scan", "قياس كثافة العظام", "ہڈیوں کی کثافت کا اسکین"),
    "IF-10": ("wellbeing check", "تقييم الصحة النفسية", "ذہنی صحت کا جائزہ"),
    "IF-11": ("memory baseline", "اختبار الذاكرة الأساسي", "یادداشت کا بنیادی ٹیسٹ"),
    "WC-01": ("Well-Child Visit", "زيارة صحة الطفل", "بچے کا ویل چائلڈ وزٹ"),
    "WC-02": ("developmental check", "فحص النمو والتطور", "نشوونما کا جائزہ"),
    "WC-03": ("growth check", "قياس النمو", "نشوونما کی پیمائش"),
    "WC-04": ("iron (anemia) blood test", "فحص فقر الدم", "خون کی کمی کا ٹیسٹ"),
    "WC-05": ("vision check", "فحص النظر", "نظر کا ٹیسٹ"),
    "WC-06": ("hearing check", "فحص السمع", "سماعت کا ٹیسٹ"),
    "WC-07": ("vaccination catch-up", "استكمال التطعيمات", "ویکسینیشن مکمل کرنا"),
    "WC-08": ("family genetic counselling", "استشارة وراثية للعائلة", "خاندانی جینیاتی مشاورت"),
}
SKIP_IN_MESSAGES = {"IF-12", "IF-13", "IF-14", "IF-15", "WC-09", "WC-10", "WC-11"}
WINDOW = {"Urgent": ("within 14 days", "خلال 14 يوماً", "14 دن کے اندر"),
          "Priority": ("within 30 days", "خلال 30 يوماً", "30 دن کے اندر"),
          "Routine": ("at your next scheduled visit", "في موعدكم الدوري القادم", "اپنے اگلے طے شدہ وزٹ پر")}
LANG_INDEX = {"English": 0, "Arabic (Standard)": 1, "Arabic (Emirati)": 1, "Urdu": 2}
FASTING_IDS = {"IF-01", "IF-02", "IF-03", "IF-04", "IF-13"}
TXT = {
    "en": dict(
        intro=" This is the Abu Dhabi preventive health team. Based on your records and the data you chose to share,",
        adult=" we would like to invite you for: {items}.", child=" we would like to invite your child {child} for: {items}.",
        fallback=" we would like to invite you for the standard age-based check-up: {items}. You can choose to share more data to personalise this.",
        book=" Please book {window} at {clinic}.", fasting=" Please fast 9 to 12 hours beforehand (water is fine).",
        sync=" We can book {names} together in one visit so nobody travels twice.", female=" A female clinician is available on request.",
        prayer=" We avoid prayer times when booking.", shift=" Early-morning and evening slots are available around your shift.",
        notdx=" This is an invitation to help you stay well, not a diagnosis.", replies=" Reply 1 to book, 2 to change the time, or STOP to opt out.",
        parent_name="Parent/Guardian", title="Your preventive health visit", buttons="Book now | Change time | Opt out",
        voice_end=" To book, press 1. To hear this again, press 2. To stop calls, press 9.", short_notdx="Not a diagnosis.",
        sms=" Abu Dhabi preventive health invites you: {items} {window}. Reply 1 to book, STOP to opt out."),
    "ar": dict(
        intro=" نتواصل معكم من فريق الرعاية الوقائية في أبوظبي. بناءً على سجلاتكم والبيانات التي اخترتم مشاركتها،",
        adult=" ندعوكم إلى: {items}.", child=" ندعوكم لإحضار طفلكم {child} إلى: {items}.",
        fallback=" ندعوكم إلى الفحص الدوري المعتاد حسب العمر: {items}. يمكنكم اختيار مشاركة المزيد من البيانات لتخصيص الدعوة.",
        book=" يُرجى الحجز {window} في {clinic}.", fasting=" يُرجى الصيام من 9 إلى 12 ساعة قبل الزيارة (يُسمح بالماء).",
        sync=" يمكننا حجز موعد واحد يجمع {names} حتى لا تتنقلوا مرتين.", female=" تتوفر طبيبة عند الطلب.",
        prayer=" نراعي أوقات الصلاة عند الحجز.", shift=" تتوفر مواعيد صباحية ومسائية تناسب ساعات عملكم.",
        notdx=" هذه دعوة للحفاظ على صحتكم وليست تشخيصاً.", replies=" أرسلوا 1 للحجز، 2 لتغيير الموعد، أو STOP للإيقاف.",
        parent_name="ولي الأمر الكريم", title="زيارتكم الوقائية", buttons="احجز الآن | غيّر الموعد | إيقاف",
        voice_end=" للحجز اضغط 1. لإعادة الاستماع اضغط 2. لإيقاف الاتصالات اضغط 9.", short_notdx="ليست تشخيصاً.",
        sms=" الرعاية الوقائية في أبوظبي تدعوكم: {items} {window}. أرسلوا 1 للحجز، STOP للإيقاف."),
    "ur": dict(
        intro=" ہم ابوظہبی کی احتیاطی صحت ٹیم کی طرف سے رابطہ کر رہے ہیں۔ آپ کے ریکارڈ اور آپ کے شیئر کردہ ڈیٹا کی بنیاد پر،",
        adult=" ہم آپ کو دعوت دیتے ہیں: {items}۔", child=" ہم آپ کے بچے {child} کے لیے دعوت دیتے ہیں: {items}۔",
        fallback=" ہم آپ کو عمر کے مطابق معیاری معائنے کی دعوت دیتے ہیں: {items}۔ آپ مزید ڈیٹا شیئر کر کے دعوت کو ذاتی بنا سکتے ہیں۔",
        book=" براہِ کرم {window} {clinic} میں بُک کریں۔", fasting=" براہِ کرم وزٹ سے 9 تا 12 گھنٹے پہلے کچھ نہ کھائیں (پانی پی سکتے ہیں)۔",
        sync=" ہم {names} کے لیے ایک ہی وزٹ بُک کر سکتے ہیں تاکہ دوبارہ سفر نہ کرنا پڑے۔", female=" درخواست پر خاتون ڈاکٹر دستیاب ہے۔",
        prayer=" بکنگ میں نماز کے اوقات کا خیال رکھا جاتا ہے۔", shift=" آپ کی شفٹ کے مطابق صبح اور شام کے اوقات دستیاب ہیں۔",
        notdx=" یہ آپ کی صحت برقرار رکھنے کی دعوت ہے، تشخیص نہیں۔", replies=" بُک کرنے کے لیے 1، وقت بدلنے کے لیے 2 بھیجیں، یا بند کرنے کے لیے STOP۔",
        parent_name="محترم والدین", title="آپ کا احتیاطی صحت وزٹ", buttons="ابھی بُک کریں | وقت بدلیں | بند کریں",
        voice_end=" بُک کرنے کے لیے 1 دبائیں۔ دوبارہ سننے کے لیے 2۔ کالز بند کرنے کے لیے 9۔", short_notdx="یہ تشخیص نہیں۔",
        sms=" ابوظہبی احتیاطی صحت کی دعوت: {items} {window}۔ بُک کرنے کے لیے 1، بند کرنے کے لیے STOP۔"),
}
LATIN, ARABIC_RE = re.compile(r"[A-Za-z]"), re.compile(r"[\u0600-\u06FF]")
CLINIC_LOCAL = {
    "Abu Dhabi City": ("Preventive Care Centre, Abu Dhabi City", "مركز الرعاية الوقائية في مدينة أبوظبي", "پریوینٹو کیئر سینٹر، ابوظہبی سٹی"),
    "Al Ain": ("Preventive Care Centre, Al Ain", "مركز الرعاية الوقائية في العين", "پریوینٹو کیئر سینٹر، العین"),
    "Al Dhafra": ("Preventive Care Centre, Madinat Zayed", "مركز الرعاية الوقائية في مدينة زايد", "پریوینٹو کیئر سینٹر، مدینہ زاید"),
}
LOCAL_NAMES = {  # first names in English, Arabic and Urdu script
    "Fatima": ("Fatima", "فاطمة", "فاطمہ"), "Omar": ("Omar", "عمر", "عمر"), "Khalid": ("Khalid", "خالد", "خالد"),
    "Imran": ("Imran", "عمران", "عمران"), "Mariam": ("Mariam", "مريم", "مریم"), "Aisha": ("Aisha", "عائشة", "عائشہ"),
    "Yousef": ("Yousef", "يوسف", "یوسف"), "Daniel": ("Daniel", "دانيال", "ڈینیئل"),
}


def _local_name(full_name: str, li: int) -> str:
    first = full_name.split()[0]
    return LOCAL_NAMES.get(first, (first, first, first))[li]


def _join(parts: list[str], lang: str) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    sep, last = {"en": (", ", " and "), "ar": ("، ", " و"), "ur": ("، ", " اور ")}[lang]
    return sep.join(parts[:-1]) + last + parts[-1]


def _lk(language: str) -> str:
    return {"English": "en", "Urdu": "ur"}.get(language, "ar")


def compose_outreach(person: dict, items: list[dict], language: str, channel: str, bundle: dict | None = None, fallback: bool = False) -> dict:
    """Culturally adapted invitation: language, channel, tone and audience (parent vs adult/worker)."""
    lk, li = _lk(language), LANG_INDEX[language]
    t = TXT[lk]
    child = person["segment"] == "Well-Child"
    chosen = [i for i in items if i["id"] in ITEM_LABELS and i["id"] not in SKIP_IN_MESSAGES]
    seen, labels = set(), []
    for i in chosen:
        lab = ITEM_LABELS[i["id"]][li]
        if lab not in seen:
            seen.add(lab)
            labels.append(lab)
        if len(labels) == 3:
            break
    top_prio = min((i["priority"] for i in chosen[:3]), key=lambda p: PRIORITY_ORDER[p]) if chosen else "Routine"
    window, clinic = WINDOW[top_prio][li], CLINIC_LOCAL[person["region"]][li]
    items_txt = _join(labels, lk)
    # greeting + audience
    if child:
        contact = next((p for p in get_personas() if p["household_id"] == person["household_id"] and p["segment"] != "Well-Child"), None)
        name = _local_name(contact["name"], li) if contact else t["parent_name"]
        sex = contact["sex"] if contact else "F"
        child_name = _local_name(person["name"], li)
    else:
        name, sex, child_name = _local_name(person["name"], li), person["sex"], ""
    if language == "Arabic (Emirati)":
        greet = f"هلا والله {name}،" if not child or contact else f"هلا والله {t['parent_name']}،"
    elif lk == "ar":
        greet = (f"السلام عليكم ورحمة الله وبركاته {'الأخ' if sex == 'M' else 'الأخت'} {name}،" if (not child or contact)
                 else f"السلام عليكم ورحمة الله وبركاته {t['parent_name']}،")
    elif lk == "ur":
        greet = f"السلام علیکم {name} {'صاحب' if sex == 'M' else 'صاحبہ'}،" if (not child or contact) else f"السلام علیکم {t['parent_name']}،"
    else:
        greet = f"Hello {name},"
    invite = (t["fallback"] if fallback else (t["child"] if child else t["adult"])).format(items=items_txt, child=child_name)
    book = t["book"].format(window=window, clinic=clinic)
    fasting = t["fasting"] if (not fallback and any(i["id"] in FASTING_IDS for i in chosen[:3]) and not child) else ""
    names = ""
    if bundle and person["id"] in [m["id"] for m in bundle["members"]] and len(bundle["members"]) > 1:
        names = _join([_local_name(m["name"], li) for m in bundle["members"]], lk)
    sync = t["sync"].format(names=names) if names else ""
    female = t["female"] if (sex == "F" and not (child and not contact)) else ""
    shift = t["shift"] if person["occupation_group"] == "Outdoor / heat-exposed" else ""
    notes = [f"Language: {language}", f"Channel: {channel}", "Audience: " + ("parent/caregiver of a Well-Child patient" if child else "adult / IFHAS" + (" (outdoor worker)" if shift else ""))]
    if sync:
        notes.append("Household synchronization offered")
    if female:
        notes.append("Female-clinician option included")
    notes.append("Prayer-time-aware scheduling")
    title = ""
    if channel == "WhatsApp":
        body = f"{greet}{t['intro']}{invite}{book}{fasting}{sync}{shift}{female}{t['prayer']}{t['notdx']}{t['replies']}"
        tone = "Warm, reassuring, child-focused" if child else ("Respectful, plain, shift-friendly" if shift else "Respectful and practical")
    elif channel == "SMS":
        body = f"{greet}{t['sms'].format(items=_join(labels[:2], lk), window=window)} {t['short_notdx']}"
        tone = "Short, direct, one action"
    elif channel == "Mobile App":
        title = t["title"]
        body = f"{greet}{invite}{book}{sync}{t['notdx']}\n[{t['buttons']}]"
        tone = "Clear, scannable, actionable buttons"
    else:
        pause = " [pause] "
        body = f"{greet}{pause}{t['intro']}{invite}{pause}{book}{fasting}{sync}{pause}{t['notdx']}{pause}{t['voice_end']}"
        tone = "Slow, calm, repeated key facts, spoken numerals"
    body = re.sub(r"\s+", " ", body).replace(" \n", "\n").strip()
    return {"language": language, "channel": channel, "title": title, "body": body, "rtl": lk in ("ar", "ur"), "chars": len(body),
            "tone": tone, "culture_notes": notes, "items_used": [i["id"] for i in chosen[:3]], "fallback": fallback, "window": window}


def script_ratio(text: str, language: str) -> float:
    letters = len(LATIN.findall(text)) + len(ARABIC_RE.findall(text))
    if letters == 0:
        return 0.0
    return (len(LATIN.findall(text)) if language == "English" else len(ARABIC_RE.findall(text))) / letters


def cultural_score(msg: str, language: str, channel: str, name_token: str = "") -> float:
    """Heuristic cultural-appropriateness score (0-1): script, greeting, opt-out, non-diagnostic, length, name."""
    lk = _lk(language)
    checks = [
        script_ratio(msg, language) >= (0.75 if language == "English" else 0.55),
        bool(re.search(r"Hello|السلام|هلا|مرحبا", msg)),
        "STOP" in msg or channel == "Mobile App",
        any(s.strip(" .،।۔") in msg for s in (TXT[lk]["notdx"], TXT[lk]["short_notdx"])),
        len(msg) <= (320 if channel == "SMS" else 1400),
        (name_token in msg) if name_token else True,
    ]
    return sum(checks) / len(checks)


# --------------------------------------------------------------------------------------
# 5. MODEL ROUTER (AI-103 #1)
# --------------------------------------------------------------------------------------
@dataclass
class ModelProfile:
    id: str
    label: str
    provider: str
    mode: str                       # offline | github | foundry | ollama | jais
    reference: dict = field(default_factory=dict)   # illustrative prior, shown only when not measured live


PROFILES = [
    ModelProfile("deterministic", "Foundry-Local / Deterministic Engine", "Local (built in)", "offline"),
    ModelProfile("phi-4", "Phi-4", "Microsoft Foundry Local", "foundry", {"groundedness": 0.86, "relevance": 0.84, "cultural": 0.66, "latency_ms": 900}),
    ModelProfile("gpt-4o-mini", "GPT-4o-mini", "GitHub Models (free tier)", "github", {"groundedness": 0.90, "relevance": 0.88, "cultural": 0.74, "latency_ms": 650}),
    ModelProfile("jais-clinical-arabic", "Jais-Clinical-Arabic profile", "Inception42 / M42 (Jais)", "jais", {"groundedness": 0.84, "relevance": 0.82, "cultural": 0.93, "latency_ms": 1100}),
]


class ModelRouter:
    def __init__(self, tracer: TraceLogger):
        self.tracer = tracer
        self.mode = os.getenv("VELOTECH_LLM_MODE", "offline").lower()

    def available(self, p: ModelProfile) -> bool:
        if p.mode == "offline":
            return True
        if p.mode == "github":
            return self.mode == "github" and bool(os.getenv("GITHUB_TOKEN"))
        if p.mode == "foundry":
            return self.mode == "foundry"
        if p.mode == "jais":
            return bool(os.getenv("JAIS_ENDPOINT"))
        return False

    def chat(self, p: ModelProfile, system: str, user: str, timeout: int = 20) -> tuple[str, float]:
        """OpenAI-compatible chat call (stdlib only). Raises on any failure."""
        if p.mode == "github":
            url, model, headers = "https://models.inference.ai.azure.com/chat/completions", "gpt-4o-mini", {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}
        elif p.mode == "foundry":
            url, model, headers = os.getenv("FOUNDRY_LOCAL_ENDPOINT", "http://localhost:5272/v1") + "/chat/completions", os.getenv("FOUNDRY_LOCAL_MODEL", "phi-4"), {}
        elif p.mode == "jais":
            url, model, headers = os.environ["JAIS_ENDPOINT"], "jais", {}
        else:
            raise RuntimeError("no live endpoint for this profile")
        body = json.dumps({"model": model, "temperature": 0.2, "max_tokens": 400,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"], (time.perf_counter() - t0) * 1000

    def route(self, language: str) -> tuple[ModelProfile, str]:
        """Pick a profile: Arabic prefers Jais, otherwise the configured live model, else deterministic."""
        live = [p for p in PROFILES if p.mode != "offline" and self.available(p)]
        pick, why = PROFILES[0], "No live model configured: using the built-in deterministic engine (offline, no cost)."
        if language.startswith("Arabic") and any(p.mode == "jais" for p in live):
            pick, why = next(p for p in live if p.mode == "jais"), "Arabic request routed to the Jais clinical profile."
        elif live:
            pick, why = live[0], f"Configured live model ({live[0].label})."
        with self.tracer.span("model_router.route", kind="INTERNAL", language=language, selected=pick.id):
            pass
        return pick, why

    def narrate(self, prompt: str, language: str) -> tuple[str | None, str]:
        """Optional LLM narrative; returns (text or None, model id). Never required for the workflow."""
        p, _ = self.route(language)
        if p.mode == "offline":
            return None, p.id
        try:
            with self.tracer.span("llm.chat", kind="CLIENT", model=p.id):
                text, _ms = self.chat(p, "You are a careful preventive-health assistant. Use only the facts given. Never diagnose.", prompt)
            return text, p.id
        except Exception as exc:
            with self.tracer.span("llm.fallback", kind="INTERNAL", reason=str(exc)[:100]):
                pass
            return None, "deterministic"

    def evaluate_all(self) -> pd.DataFrame:
        """Side-by-side evaluation. Deterministic engine is measured live; others are measured only when reachable."""
        kb = get_kb()
        tasks = [("P-001", "English", "WhatsApp"), ("P-004", "Urdu", "SMS"), ("P-002", "Arabic (Emirati)", "WhatsApp")]
        rows = []
        for p in PROFILES:
            measured, g, r, c, lat, note = False, [], [], [], [], ""
            if self.available(p):
                for pid, lang, ch in tasks:
                    person = get_persona(pid)
                    items = recommend_pathway(person, set(person["consent_default"]))
                    q = f'{items[0]["title"]} timing fasting follow-up'
                    hits = kb.search(q, top_k=3, program="Well-Child" if person["segment"] == "Well-Child" else "IFHAS", age_months=person["age_years"] * 12)
                    try:
                        t0 = time.perf_counter()
                        if p.mode == "offline":
                            expl = kb.answer(q, person)["answer"]
                            out = compose_outreach(person, items, lang, ch)["body"]
                            ms = (time.perf_counter() - t0) * 1000
                        else:
                            ctx = " ".join(h.snippet for h in hits)
                            expl, ms1 = self.chat(p, "Explain in English, two sentences, using only these passages.", f"Passages: {ctx}\nQuestion: why and when is '{items[0]['title']}' recommended?")
                            out, ms2 = self.chat(p, f"Write a {ch} invitation in {lang} for {person['name'].split()[0]}. No diagnosis. Include STOP opt-out.", f"Offer: {items[0]['title']} {items[0]['window']}.")
                            ms = ms1 + ms2
                        g.append(kb.groundedness(expl, hits)["score"])
                        word = re.findall(r"[a-z]{4,}", items[0]["title"].lower())[:2]
                        num = re.findall(r"\d+", items[0]["window"])[:1]
                        r.append(sum(w in (expl + out).lower() for w in word) / max(len(word), 1) * 0.5 + (0.5 if (not num or num[0] in (expl + out)) else 0.0))
                        c.append(cultural_score(out, lang, ch, person["name"].split()[0] if lang == "English" else ""))
                        lat.append(ms)
                        measured = True
                    except Exception as exc:
                        note = f"endpoint error: {str(exc)[:60]}"
                        break
            if measured:
                rows.append({"Model": p.label, "Provider": p.provider, "Groundedness": round(sum(g) / len(g), 2), "Clinical relevance": round(sum(r) / len(r), 2),
                             "Cultural appropriateness": round(sum(c) / len(c), 2), "Latency (ms)": round(sum(lat) / len(lat)), "Basis": "Measured live in this session"})
            else:
                ref = p.reference
                rows.append({"Model": p.label, "Provider": p.provider, "Groundedness": ref.get("groundedness"), "Clinical relevance": ref.get("relevance"),
                             "Cultural appropriateness": ref.get("cultural"), "Latency (ms)": ref.get("latency_ms"),
                             "Basis": "Illustrative prior, not measured (" + (note or "set VELOTECH_LLM_MODE to enable") + ")"})
        with self.tracer.span("model_router.evaluate_all", kind="INTERNAL", profiles=len(rows)):
            pass
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 6. SPECIALIST AGENTS (Microsoft Agent Framework pattern)
# --------------------------------------------------------------------------------------
@dataclass
class Runtime:
    engine: RiskEngine
    tracer: TraceLogger
    mcp: MCPClient
    tools: VeloTechTools
    router: ModelRouter

    @classmethod
    def create(cls, engine: RiskEngine | None = None) -> "Runtime":
        tracer = TraceLogger()
        eng = engine or get_engine()
        return cls(eng, tracer, MCPClient(MCPServerLite(), tracer), VeloTechTools(eng, tracer), ModelRouter(tracer))


class BaseAgent:
    name = "BaseAgent"
    role = ""
    instructions = ""

    def __init__(self, rt: Runtime):
        self.rt = rt

    def describe(self) -> dict:
        return {"name": self.name, "role": self.role, "instructions": self.instructions}


class SignalConformalRiskAnalystAgent(BaseAgent):
    name = "Agent 1: Signal & Conformal Risk Analyst Agent"
    role = "Ingests consented FHIR signals through MCP and produces a calibrated, explainable, uncertainty-aware risk score."
    instructions = "Call MCP for Patient, Observation and EGP data. Never use a domain the person has not authorized. Block extrapolation below 40% completeness."

    def ingest(self, ctx: dict) -> dict:
        pid, consent = ctx["person"]["id"], sorted(ctx["consent"])
        patient = self.rt.mcp.call("get_fhir_patient", person_id=pid)
        obs = self.rt.mcp.call("get_fhir_observations", person_id=pid, consent_domains=consent)
        egp = self.rt.mcp.call("lookup_egp_prs", person_id=pid, consent_domains=consent)
        return {"patient": patient, "observations": obs, "egp": egp}

    def score(self, ctx: dict) -> dict:
        return self.rt.tools.score_risk_conformal(ctx["person"]["id"], sorted(ctx["consent"]))


class FoundryIQGuidelineFamilySyncAgent(BaseAgent):
    name = "Agent 2: Foundry IQ Guideline & Family-Sync Pathway Agent"
    role = "Grounds the pathway in DoH guidance via hybrid RAG and bundles overdue household members into one visit."
    instructions = "Cite approved guideline passages only. Fall back to the baseline age and gender schedule when the risk score is withheld."

    def run(self, ctx: dict) -> dict:
        person, risk, kb = ctx["person"], ctx["risk"], get_kb()
        household = self.rt.mcp.call("get_household_graph", household_id=person["household_id"])
        bundle = household["sync_bundle"]
        in_bundle = bool(bundle and person["id"] in [m["id"] for m in bundle["members"]])
        if risk["fallback"]:
            items = baseline_schedule(person)
        else:
            items = recommend_pathway(person, ctx["consent"], household_overdue=in_bundle)
        careplan = self.rt.mcp.call("get_fhir_careplan", person_id=person["id"], consent_domains=sorted(ctx["consent"]))
        query = f'{items[0]["title"]} timing fasting recall follow-up'
        rag = self.rt.tools.retrieve_guidelines(query, person["id"])
        program = "Well-Child" if person["segment"] == "Well-Child" else "IFHAS"
        agree = 0
        for it in items:
            hits = kb.search(f'{it["title"]} {it["rationale"]}', top_k=4, program=program, age_months=person["age_years"] * 12, sex=person["sex"])
            agree += any(h.id == it["id"] for h in hits)
        guideline_text = " ".join(f'{i["window"]}. {i["followup"]}' for i in items)
        cited = [kb.get(i["id"]) for i in items]
        support = {t for c in cited for t in tokenize(" ".join([c["title"], c["text"], c["timing"], c["followup"], c["recall"], c["fasting"]]))}
        toks = [t for t in tokenize(guideline_text) if not t.isdigit()]
        grounded = sum(t in support for t in toks) / max(len(toks), 1)
        return {"items": items, "previsit": consolidated_previsit_rules(items), "recall": recall_summary(items), "household": household, "bundle": bundle,
                "careplan": careplan, "rag": rag, "retrieval_agreement": agree / max(len(items), 1), "pathway_groundedness": grounded, "baseline_only": risk["fallback"]}


class CulturaVoiceEngagementAgent(BaseAgent):
    name = "Agent 3: CulturaVoice Multilingual Engagement Agent"
    role = "Writes culturally appropriate invitations in Arabic, English or Urdu for the right channel and audience."
    instructions = "Never state a diagnosis or a risk number. Always include opt-out. Respect language, channel and caregiver audience."

    def run(self, ctx: dict) -> dict:
        person, pathway = ctx["person"], ctx["pathway"]
        language, channel = ctx["language"], ctx["channel"]
        model, why = self.rt.router.route(language)
        msg = compose_outreach(person, pathway["items"], language, channel, pathway["bundle"], fallback=pathway["baseline_only"])
        nudge = None
        if ctx.get("biosignal"):
            nudge = bio.motivation_message(ctx["biosignal"]["quality"], language)
        narrative, used = self.rt.router.narrate(
            f'Write one clinician-facing sentence summarising why {person["name"]} is prioritised: {ctx["risk"]["summary"]}; items {msg["items_used"]}.', language)
        return {"message": msg, "model": model.label, "model_reason": why, "biosignal_nudge": nudge, "narrative": narrative, "narrative_model": used}


class ResponsibleAIAuditorAgent(BaseAgent):
    name = "Agent 4: Responsible AI, Fairness & Calibration Auditor Agent"
    role = "Verifies consent, enforces the data-completeness fallback and clinical boundary, and audits calibration, parity, groundedness and privacy."
    instructions = "Block or downgrade any output that violates consent, groundedness, privacy or the population-vs-clinician boundary."

    def verify_consent(self, ctx: dict) -> dict:
        consent, avail = ctx["consent"], set(ctx["person"]["available_domains"])
        comp = compute_completeness(avail, consent)
        used = sorted(consent & avail)
        return {"consented": sorted(consent), "used_domains": used, "completeness": comp, "ok": bool(used), "fallback": comp < FALLBACK_THRESHOLD}

    def gate(self, ctx: dict) -> dict:
        m, k = self.rt.engine.metrics, ctx["k"]
        risk, pathway, consent = ctx["risk"], ctx["pathway"], ctx["consent_check"]
        fair = fairness_snapshot(self.rt.engine, k)
        checks = [
            ("Clinical boundary attached", "ok", "Population prioritization vs individual clinical decision-making notice is attached to every output."),
            ("Consent verified", "ok" if consent["ok"] else "fail", f'{len(consent["used_domains"])} authorized data domain(s) in use.'),
            ("Data completeness >= 40%", "warn" if consent["fallback"] else "ok",
             f'Completeness {consent["completeness"]:.0%}. ' + ("Risk extrapolation blocked; baseline DoH schedule applied." if consent["fallback"] else "AI scoring permitted.")),
            ("Calibration (ECE <= 0.05)", "ok" if m["ece_calibrated"] <= 0.05 else "warn", f'ECE {m["ece_calibrated"]:.3f}, Brier {m["brier_calibrated"]:.3f}.'),
            ("Conformal coverage >= 93%", "ok" if m["interval_coverage"] >= 0.93 else "warn", f'Empirical interval coverage {m["interval_coverage"]:.1%}.'),
            ("Guideline retrieval agreement >= 80%", "ok" if pathway["retrieval_agreement"] >= 0.8 else "warn", f'{pathway["retrieval_agreement"]:.0%} of pathway items re-retrieved from their own rationale.'),
            ("Pathway groundedness >= 80%", "ok" if pathway["pathway_groundedness"] >= 0.8 else "warn", f'{pathway["pathway_groundedness"]:.0%} of timing and follow-up wording supported by cited guidance.'),
            ("Fairness parity (FPR/FNR gap)", fair["status"], fair["detail"]),
            ("k-anonymity small-cell suppression", "ok", f'Cohort outputs suppress cells with n < {k}.'),
        ]
        if risk["fallback"]:
            checks.append(("No AI risk shown to user", "ok", "Risk percentage withheld because completeness is below the safety threshold."))
        verdict = "blocked" if any(c[1] == "fail" for c in checks) else ("passed with fallback" if risk["fallback"] else "passed")
        return {"checks": [{"check": a, "status": b, "detail": c} for a, b, c in checks], "verdict": verdict,
                "passed": sum(c[1] == "ok" for c in checks), "total": len(checks), "boundary": BOUNDARY}

    def verify_message(self, msg: dict) -> dict:
        body, lk = msg["body"], _lk(msg["language"])
        checks = {
            "opt-out present": ("STOP" in body) or msg["channel"] == "Mobile App",
            "non-diagnostic wording": any(s.strip(" .،۔") in body for s in (TXT[lk]["notdx"], TXT[lk]["short_notdx"])),
            "no risk percentage shared": "%" not in body,
            "language script matches": script_ratio(body, msg["language"]) >= (0.7 if msg["language"] == "English" else 0.5),
            "channel length ok": msg["channel"] != "SMS" or msg["chars"] <= 320,
        }
        return {"checks": checks, "ok": all(checks.values())}


@lru_cache(maxsize=8)
def _fairness(engine_id: int, k: int, engine: RiskEngine) -> dict:
    scored = engine.score_population(engine.pop)
    table = cohort_table(scored, ["region"], k=k)
    ps = parity_summary(table)
    eo = ps["equalized_odds_difference"]
    status = "ok" if (eo is not None and eo <= 0.10) else ("warn" if eo is not None and eo <= 0.20 else "fail")
    return {"status": status, "eo": eo, "detail": f"Equalised-odds gap across regions: {eo:.1%} (green <= 10%, amber <= 20%)." if eo is not None else "n/a"}


def fairness_snapshot(engine: RiskEngine, k: int = 10) -> dict:
    return _fairness(id(engine), k, engine)


# --------------------------------------------------------------------------------------
# 7. SIX-STATE WORKFLOW + 4-AGENT SWARM
# --------------------------------------------------------------------------------------
@dataclass
class WorkflowResult:
    person: dict
    consent: list[str]
    state_log: list[dict]
    consent_check: dict
    ingest: dict
    risk: dict
    pathway: dict
    audit: dict
    outreach: dict
    message_check: dict
    spans: list[dict]
    span_frame: pd.DataFrame
    mcp_calls: list[dict]
    tool_calls: list[dict]
    agents: list[dict]
    dispatched: bool
    trace_id: str
    tool_schemas: list[dict]
    boundary: str = BOUNDARY


def run_workflow(person_id: str, consent: list[str] | set[str] | None = None, language: str | None = None, channel: str | None = None,
                 k: int = 10, biosignal: dict | None = None, runtime: Runtime | None = None) -> WorkflowResult:
    """Run the full 6-state workflow with the 4-agent swarm and return a traceable result."""
    rt = runtime or Runtime.create()
    person = get_persona(person_id)
    consent_set = set(person["consent_default"] if consent is None else consent)
    ctx: dict[str, Any] = {"person": person, "consent": consent_set, "language": language or person["language"],
                           "channel": channel or person["channel"], "k": k, "biosignal": biosignal}
    a1, a2, a3, a4 = (SignalConformalRiskAnalystAgent(rt), FoundryIQGuidelineFamilySyncAgent(rt), CulturaVoiceEngagementAgent(rt), ResponsibleAIAuditorAgent(rt))
    log: list[dict] = []
    held = False

    def step(state: str, agent: BaseAgent, fn: Callable[[], Any], summarize: Callable[[Any], str]) -> Any:
        t0 = time.perf_counter()
        with rt.tracer.span(f"state.{state}", kind="INTERNAL", agent=agent.name, state=state):
            out = fn()
        log.append({"state": state, "agent": agent.name, "status": "completed", "ms": round((time.perf_counter() - t0) * 1000, 1), "summary": summarize(out)})
        return out

    with rt.tracer.span("workflow.run", kind="SERVER", person_id=person_id, language=ctx["language"], channel=ctx["channel"]):
        ctx["consent_check"] = step(WORKFLOW_STATES[0], a4, lambda: a4.verify_consent(ctx),
                                    lambda o: f'{len(o["used_domains"])} domains authorized, completeness {o["completeness"]:.0%}' + (" (fallback)" if o["fallback"] else ""))
        ctx["ingest"] = step(WORKFLOW_STATES[1], a1, lambda: a1.ingest(ctx),
                             lambda o: f'FHIR Patient + {o["observations"]["total"]} Observation(s); EGP {"enrolled" if o["egp"].get("enrolled") else "no record / not authorized"}')
        ctx["risk"] = step(WORKFLOW_STATES[2], a1, lambda: a1.score(ctx), lambda o: o["summary"])
        ctx["pathway"] = step(WORKFLOW_STATES[3], a2, lambda: a2.run(ctx),
                              lambda o: f'{len(o["items"])} pathway items, {len(o["rag"]["citations"])} citations' + (", family bundle proposed" if o["bundle"] else "") + (", baseline schedule only" if o["baseline_only"] else ""))
        ctx["audit"] = step(WORKFLOW_STATES[4], a4, lambda: a4.gate(ctx), lambda o: f'{o["verdict"]} ({o["passed"]}/{o["total"]} checks green)')
        held = ctx["audit"]["verdict"] == "blocked"
        if held:
            ctx["outreach"] = {"message": None, "model": "n/a", "model_reason": "Gate blocked dispatch; held for human review.", "biosignal_nudge": None, "narrative": None}
            log.append({"state": WORKFLOW_STATES[5], "agent": a3.name, "status": "held", "ms": 0.0, "summary": "Dispatch held for human review"})
            ctx["message_check"] = {"checks": {}, "ok": False}
        else:
            ctx["outreach"] = step(WORKFLOW_STATES[5], a3, lambda: a3.run(ctx),
                                   lambda o: f'{o["message"]["language"]} / {o["message"]["channel"]} message, model: {o["model"]}')
            with rt.tracer.span("agent4.verify_message", kind="INTERNAL"):
                ctx["message_check"] = a4.verify_message(ctx["outreach"]["message"])
    return WorkflowResult(person=person, consent=sorted(consent_set), state_log=log, consent_check=ctx["consent_check"], ingest=ctx["ingest"], risk=ctx["risk"],
                          pathway=ctx["pathway"], audit=ctx["audit"], outreach=ctx["outreach"], message_check=ctx["message_check"], spans=rt.tracer.spans,
                          span_frame=rt.tracer.to_frame(), mcp_calls=rt.mcp.calls, tool_calls=rt.tools.calls, agents=[a.describe() for a in (a1, a2, a3, a4)],
                          dispatched=not held, trace_id=rt.tracer.trace_id, tool_schemas=rt.tools.schemas())


# --------------------------------------------------------------------------------------
# 8. AI-103 COMPLIANCE MATRIX (live evidence)
# --------------------------------------------------------------------------------------
def compliance_matrix(res: WorkflowResult, router_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    n_models = len(router_frame) if router_frame is not None else len(PROFILES)
    measured = int((router_frame["Basis"].str.startswith("Measured")).sum()) if router_frame is not None else 0
    rows = [
        ("1. Select, deploy and evaluate Foundry models", "ModelRouter + side-by-side evaluation", "src/foundry_agents.py", f"{n_models} profiles compared, {measured} measured live"),
        ("2. Generative apps that use tools", "Streamlit UI invokes typed tool pipelines", "app.py", f"{len(res.tool_calls)} custom tool call(s) this run"),
        ("3. Agents with Foundry and VS Code", "launch.json, settings.json, OTel-style spans", ".vscode/", f"{len(res.spans)} spans, trace {res.trace_id[:8]}"),
        ("4. Custom tools in an agent", "@ai_function typed tools", "src/foundry_agents.py", f"{len(res.tool_schemas)} tool schemas registered"),
        ("5. MCP tools with agents", "MCP JSON-RPC server + client (FHIR R4, EGP, k-anon)", "src/mcp_server.py", f"{len(res.mcp_calls)} MCP tools/call this run"),
        ("6. Knowledge-enhanced agents (Foundry IQ RAG)", "Hybrid BM25 + TF-IDF with citations", "src/doh_guidelines_kb.py", f'{len(res.pathway["rag"]["citations"])} citations, retrieval agreement {res.pathway["retrieval_agreement"]:.0%}'),
        ("7. Agent-driven workflows", "6-state machine", "src/foundry_agents.py", f'{sum(s["status"] == "completed" for s in res.state_log)}/6 states completed'),
        ("8. Agent with Microsoft Agent Framework", "Specialist agent classes, @ai_function pattern", "src/foundry_agents.py", f"{len(res.agents)} agent classes"),
        ("9. Multi-agent orchestration", "4-agent swarm with handoffs", "src/foundry_agents.py", "Agents 1-4 invoked in sequence"),
        ("10. Responsible generative AI", "Gate: boundary, fallback, calibration, parity, groundedness, k-anon", "src/foundry_agents.py", f'{res.audit["passed"]}/{res.audit["total"]} checks green, verdict: {res.audit["verdict"]}'),
    ]
    return pd.DataFrame(rows, columns=["AI-103 capability", "Implementation", "Where", "Live evidence"])


if __name__ == "__main__":  # python -m src.foundry_agents
    for pid in ("P-001", "P-008"):
        r = run_workflow(pid)
        print(f"\n== {r.person['name']} == dispatched={r.dispatched} verdict={r.audit['verdict']}")
        for s in r.state_log:
            print(f'  {s["state"]:<36} {s["status"]:<9} {s["ms"]:>7} ms  {s["summary"]}')
        print("  message:", r.outreach["message"]["body"][:160] if r.outreach["message"] else None)
    print(compliance_matrix(r).to_string(index=False)[:600])
