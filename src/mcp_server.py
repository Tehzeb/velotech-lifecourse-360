"""
VeloTech LifeCourse-360 - Model Context Protocol (MCP) server.

Exposes standardised, consent-gated tools over the Model Context Protocol:

  get_fhir_patient        HL7 FHIR R4 Patient
  get_fhir_observations   HL7 FHIR R4 Observation bundle (only authorized data domains)
  get_fhir_careplan       HL7 FHIR R4 CarePlan (draft proposal, grounded in DoH guidance)
  lookup_egp_prs          Emirati Genome Program polygenic-risk lookup (synthetic)
  get_household_graph     NetworkX kinship graph + Synchronized Family Prevention Appointment
  kanonymity_filter       k-anonymity small-cell privacy filter for cohort counts

Two transports, one implementation:
  * `MCPServerLite` - dependency-free JSON-RPC 2.0 (`tools/list`, `tools/call`) used in-process by
    the agents, so the whole demo works offline.
  * `build_fastmcp()` - registers the same functions with the official `mcp` SDK (FastMCP in 1.x,
    MCPServer in 2.x) when it is installed (`pip install mcp`). Run `python src/mcp_server.py` to serve over stdio.

ALL DATA IS SYNTHETIC. Resource ids use the example.org namespace; no real Malaffi endpoint is called.
"""
from __future__ import annotations

import datetime as dt
import inspect
import json
import os
import sys
from functools import lru_cache
from typing import Any, Callable

# Allow `python src/mcp_server.py` from the project root (no PYTHONPATH needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_generator import (
    DOMAIN_KEYS, DOMAIN_LABELS, egp_record, generate_population, get_persona, get_personas,
    household_sync_bundle, persona_household_graph,
)
from src.doh_guidelines_kb import recommend_pathway

TODAY = dt.date(2026, 10, 26)
ID_SYSTEM = "https://example.org/velotech/synthetic-id"
LOCAL_CODES = "https://example.org/velotech/codes"
BOUNDARY_NOTE = ("Population prioritization and outreach proposal within approved DoH guidelines. "
                 "Final clinical decisions rest exclusively with licensed physicians.")

# feature -> (system, code, display, unit, ucum, category, segments)
OBS_MAP = {
    "ldl": ("http://loinc.org", "18262-6", "Cholesterol in LDL [Moles/volume] in Serum or Plasma by Direct assay", "mmol/L", "mmol/L", "laboratory", "IFHAS"),
    "hba1c": ("http://loinc.org", "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%", "%", "laboratory", "IFHAS"),
    "egfr": ("http://loinc.org", "62238-1", "Glomerular filtration rate/1.73 sq M predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)", "mL/min/1.73m2", "mL/min/{1.73_m2}", "laboratory", "IFHAS"),
    "uacr": ("http://loinc.org", "9318-7", "Albumin/Creatinine [Mass Ratio] in Urine", "mg/g", "mg/g", "laboratory", "IFHAS"),
    "hemoglobin": ("http://loinc.org", "718-7", "Hemoglobin [Mass/volume] in Blood", "g/dL", "g/dL", "laboratory", "both"),
    "sbp": ("http://loinc.org", "8480-6", "Systolic blood pressure", "mmHg", "mm[Hg]", "vital-signs", "IFHAS"),
    "phq9": ("http://loinc.org", "44261-6", "Patient Health Questionnaire 9 item total score", "{score}", "{score}", "survey", "IFHAS"),
    "bmi_or_wfh_z": ("http://loinc.org", "39156-5", "Body mass index (BMI) [Ratio]", "kg/m2", "kg/m2", "vital-signs", "IFHAS"),
    "mchat_score": (LOCAL_CODES, "mchat-r-f-total", "M-CHAT-R/F total score", "{score}", "{score}", "survey", "Well-Child"),
}
CHILD_OBS_Z = ("bmi_or_wfh_z", (LOCAL_CODES, "wfh-z", "Weight-for-height / BMI-for-age z-score (WHO)", "{z-score}", "{z-score}", "vital-signs", "Well-Child"))
OBS_DOMAIN = {"ldl": "labs", "hba1c": "labs", "egfr": "labs", "uacr": "labs", "hemoglobin": "labs", "sbp": "ehr_history",
              "phq9": "developmental_behavioral", "bmi_or_wfh_z": "ehr_history", "mchat_score": "developmental_behavioral"}


def _consent(person: dict, consent_domains: list[str] | None) -> set[str]:
    return set(person["consent_default"] if consent_domains is None else consent_domains)


def _birth_date(person: dict) -> str:
    return (TODAY - dt.timedelta(days=int(person["age_years"] * 365.25))).isoformat()


# --------------------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS (plain functions, typed, documented)
# --------------------------------------------------------------------------------------
def get_fhir_patient(person_id: str) -> dict:
    """Return an HL7 FHIR R4 Patient resource for a synthetic person."""
    p = get_persona(person_id)
    lang = {"Arabic (Emirati)": "ar", "Arabic (Standard)": "ar", "English": "en", "Urdu": "ur"}[p["language"]]
    return {
        "resourceType": "Patient", "id": p["id"].lower(),
        "meta": {"tag": [{"system": LOCAL_CODES, "code": "synthetic-data", "display": "Synthetic demonstration record"}]},
        "identifier": [{"system": ID_SYSTEM, "value": p["id"]}],
        "active": True, "name": [{"use": "official", "text": p["name"]}],
        "gender": "female" if p["sex"] == "F" else "male", "birthDate": _birth_date(p),
        "address": [{"state": "Abu Dhabi", "district": p["region"], "country": "AE"}],
        "communication": [{"language": {"coding": [{"system": "urn:ietf:bcp:47", "code": lang}], "text": p["language"]}, "preferred": True}],
        "extension": [
            {"url": f"{LOCAL_CODES}/pathway", "valueString": "Well-Child Visit" if p["segment"] == "Well-Child" else "IFHAS"},
            {"url": f"{LOCAL_CODES}/household", "valueString": p["household_id"]},
            {"url": f"{LOCAL_CODES}/preferred-channel", "valueString": p["channel"]},
        ],
    }


def get_fhir_observations(person_id: str, consent_domains: list[str] | None = None) -> dict:
    """Return a FHIR R4 Observation Bundle limited to authorized data domains."""
    p = get_persona(person_id)
    consent = _consent(p, consent_domains)
    seg = p["segment"]
    last = (TODAY - dt.timedelta(days=int((p["last_visit_months_ago"] or 0) * 30.4))).isoformat() if p["last_visit_months_ago"] is not None else TODAY.isoformat()
    entries, withheld = [], []
    for feat, spec in OBS_MAP.items():
        if spec[6] not in ("both", seg):
            continue
        if seg == "Well-Child" and feat == "bmi_or_wfh_z":
            spec = CHILD_OBS_Z[1]
        dom = OBS_DOMAIN[feat]
        val = p["features"].get(feat)
        if dom not in consent or dom not in p["available_domains"]:
            withheld.append({"feature": feat, "domain": dom, "reason": "not authorized" if dom not in consent else "no record"})
            continue
        if val is None:
            continue
        system, code, display, unit, ucum, cat = spec[:6]
        entries.append({"resource": {
            "resourceType": "Observation", "id": f"{p['id'].lower()}-{feat}", "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": cat}]}],
            "code": {"coding": [{"system": system, "code": code, "display": display}], "text": display.split(" [")[0]},
            "subject": {"reference": f"Patient/{p['id'].lower()}"}, "effectiveDateTime": last,
            "valueQuantity": {"value": round(float(val), 2), "unit": unit, "system": "http://unitsofmeasure.org", "code": ucum},
        }})
    return {"resourceType": "Bundle", "type": "collection", "total": len(entries), "entry": entries,
            "meta": {"tag": [{"system": LOCAL_CODES, "code": "consent-filtered"}]},
            "extension": [{"url": f"{LOCAL_CODES}/withheld-by-consent", "valueString": json.dumps(withheld)}]}


def get_fhir_careplan(person_id: str, consent_domains: list[str] | None = None) -> dict:
    """Return a draft FHIR R4 CarePlan (intent=proposal) built from the DoH-grounded pathway."""
    p = get_persona(person_id)
    consent = _consent(p, consent_domains)
    bundle = household_sync_bundle(persona_household_graph(p["household_id"]))
    items = recommend_pathway(p, consent, household_overdue=bool(bundle and p["id"] in [m["id"] for m in bundle["members"]]))
    return {
        "resourceType": "CarePlan", "id": f"{p['id'].lower()}-preventive-proposal", "status": "draft", "intent": "proposal",
        "category": [{"coding": [{"system": "http://snomed.info/sct", "code": "409073007", "display": "Education"}],
                      "text": "Preventive screening outreach proposal"}],
        "title": ("Well-Child Visit" if p["segment"] == "Well-Child" else "IFHAS") + " preventive pathway (proposal)",
        "description": BOUNDARY_NOTE, "subject": {"reference": f"Patient/{p['id'].lower()}"},
        "period": {"start": TODAY.isoformat()},
        "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">{BOUNDARY_NOTE}</div>"},
        "activity": [{
            "detail": {"kind": "ServiceRequest", "code": {"text": i["title"]}, "status": "not-started",
                       "description": i["rationale"], "scheduledString": i["window"],
                       "extension": [{"url": f"{LOCAL_CODES}/priority", "valueString": i["priority"]},
                                     {"url": f"{LOCAL_CODES}/guideline", "valueString": f'{i["citation"]["id"]}: {i["citation"]["source"]}'},
                                     {"url": f"{LOCAL_CODES}/recall", "valueString": i["recall"]},
                                     {"url": f"{LOCAL_CODES}/followup", "valueString": i["followup"]}]}} for i in items],
    }


def lookup_egp_prs(person_id: str, consent_domains: list[str] | None = None) -> dict:
    """Emirati Genome Program polygenic risk and hereditary flags (synthetic). Requires genomic consent."""
    p = get_persona(person_id)
    if "genomic_egp" not in _consent(p, consent_domains):
        return {"person_id": person_id, "enrolled": None, "note": "Genomic data domain not authorized by the individual or family."}
    return egp_record(p)


def get_household_graph(household_id: str) -> dict:
    """Return household kinship graph nodes/edges and any Synchronized Family Prevention Appointment."""
    g = persona_household_graph(household_id)
    return {"household_id": household_id,
            "nodes": [{"id": n, **{k: v for k, v in d.items()}} for n, d in g.nodes(data=True)],
            "edges": [{"source": a, "target": b, **d} for a, b, d in g.edges(data=True)],
            "sync_bundle": household_sync_bundle(g)}


@lru_cache(maxsize=1)
def _population():
    return generate_population(4000, seed=42)


def kanonymity_filter(group_by: list[str], k: int = 10) -> dict:
    """Count cohort sizes and suppress every cell with n < k (small-cell k-anonymity safeguard)."""
    pop = _population()
    cols = [c for c in group_by if c in pop.columns and not c.startswith(("avail_", "consent_"))]
    if not cols:
        return {"error": "No valid grouping columns.", "allowed": ["region", "nationality_group", "occupation_group", "age_band", "segment", "sex"]}
    counts = pop.groupby(cols, observed=True).size().reset_index(name="n")
    rows = []
    for _, r in counts.iterrows():
        row = {c: str(r[c]) for c in cols}
        row["n"] = int(r["n"]) if r["n"] >= k else "[SUPPRESSED: n < k Privacy Safeguard]"
        rows.append(row)
    return {"k": k, "cells": len(rows), "suppressed_cells": int((counts["n"] < k).sum()), "rows": rows}


TOOLS: list[Callable[..., Any]] = [get_fhir_patient, get_fhir_observations, get_fhir_careplan, lookup_egp_prs, get_household_graph, kanonymity_filter]

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _schema(fn: Callable) -> dict:
    props, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        ann = param.annotation
        origin = getattr(ann, "__origin__", ann)
        jtype = "array" if (origin is list or "list" in str(ann)) else _JSON_TYPES.get(ann, "string")
        props[name] = {"type": jtype}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


class MCPServerLite:
    """Minimal MCP-compatible JSON-RPC 2.0 server (tools/list, tools/call, initialize)."""

    name = "velotech-lifecourse-360"
    version = "1.0.0"

    def __init__(self, tools: list[Callable] | None = None):
        self.tools = {t.__name__: t for t in (tools or TOOLS)}

    def list_tools(self) -> list[dict]:
        return [{"name": n, "description": (inspect.getdoc(f) or "").split("\n")[0], "inputSchema": _schema(f)} for n, f in self.tools.items()]

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        if name not in self.tools:
            return {"isError": True, "content": [{"type": "text", "text": f"Unknown tool: {name}"}]}
        try:
            result = self.tools[name](**(arguments or {}))
            return {"isError": False, "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}], "structuredContent": result}
        except Exception as exc:  # surfaced to the agent, never crashes the server
            return {"isError": True, "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}]}

    def handle_jsonrpc(self, msg: dict) -> dict | None:
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if method == "initialize":
            res = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": self.name, "version": self.version}}
        elif method == "tools/list":
            res = {"tools": self.list_tools()}
        elif method == "tools/call":
            res = self.call_tool(params.get("name", ""), params.get("arguments"))
        elif method and method.startswith("notifications/"):
            return None
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": res}


def build_fastmcp():
    """Register the same tools with the official MCP SDK. Returns None if `mcp` is not installed.

    Works with both SDK generations: `FastMCP` (mcp 1.x) and `MCPServer` (mcp 2.x, FastMCP was renamed)."""
    server_cls = None
    try:
        from mcp.server.fastmcp import FastMCP as server_cls  # type: ignore  # mcp 1.x
    except Exception:
        try:
            from mcp.server.mcpserver import MCPServer as server_cls  # type: ignore  # mcp 2.x
        except Exception:
            return None
    server = server_cls("velotech-lifecourse-360")
    for fn in TOOLS:
        server.tool()(fn)
    return server


def _stdio_loop() -> None:
    srv = MCPServerLite()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        reply = srv.handle_jsonrpc(json.loads(line))
        if reply is not None:
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    fast = build_fastmcp()
    if fast is not None:
        fast.run()  # stdio transport via the official SDK
    else:
        print("mcp SDK not installed: serving JSON-RPC 2.0 over stdio (MCPServerLite). Try:", file=sys.stderr)
        print('  {"jsonrpc":"2.0","id":1,"method":"tools/list"}', file=sys.stderr)
        _stdio_loop()
