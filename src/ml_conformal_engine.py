"""
VeloTech LifeCourse-360 - Calibrated risk engine with SHAP attribution and Conformal Prediction.

Design
------
* Two segment models (Well-Child, IFHAS adult): standardised logistic regression, then Platt (sigmoid)
  recalibration on a held-out recalibration split.
* SHAP: for a linear model in logit space the Shapley values are exact,
  phi_j = coef_j * (x_j - E[x_j]) / sd_j, so no approximation or extra dependency is needed.
  Contributions are grouped by clinical feature and shown in percentage points versus baseline.
* Conformal Prediction (split conformal, Mondrian by data-completeness band):
    - Interval around the calibrated risk using absolute residuals against the synthetic
      "adjudicated risk" proxy on a held-out conformal split. Intervals widen automatically when
      authorized data completeness is low. (In a real deployment the proxy would be replaced by
      adjudicated clinical risk or Venn-Abers intervals - see README assumptions.)
    - Class-conditional conformal prediction SETS (LAC score) for the binary decision.
* Insufficient-data guardrail: authorized completeness < 40% blocks risk extrapolation entirely.
* Calibration suite: Brier, ECE, reliability bins, AUC, empirical coverage, Decision Curve Analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from data.synthetic_generator import (
    DATA_DOMAINS, DOMAIN_KEYS, DOMAIN_WEIGHTS, FEATURE_LABELS, FEATURE_TO_DOMAIN, compute_completeness,
    generate_population,
)

FALLBACK_THRESHOLD = 0.40
ALPHA = 0.05  # 95% intervals

ADULT_BASE = ["age_years", "bmi_or_wfh_z", "sbp", "ldl", "hba1c", "egfr", "uacr", "phq9", "fam_hx_cvd", "fam_hx_dm",
              "smoker", "months_overdue", "missed_visits_2y", "heat_exposure", "pm25", "prs_cvd", "prs_t2d",
              "fh_variant", "distance_km"]
CHILD_BASE = ["age_years", "bmi_or_wfh_z", "mchat_score", "hemoglobin", "fam_hx_dm", "months_overdue",
              "missed_visits_2y", "immunization_gaps", "pm25", "metabolic_carrier", "distance_km"]
DERIVED_PARENT = {"egfr_low": "egfr", "uacr_high": "uacr", "anemia": "hemoglobin", "mchat_pos": "mchat_score",
                  "wfh_extreme": "bmi_or_wfh_z"}
DERIVED_LABELS = {"egfr_low": "eGFR below 75", "uacr_high": "uACR 30 mg/g or more", "anemia": "Hemoglobin below 11 g/dL",
                  "mchat_pos": "M-CHAT-R/F score 3 or more", "wfh_extreme": "Weight-for-height z beyond plus/minus 2"}


def _design(raw: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Build the model design matrix (NaN-propagating derived threshold features)."""
    base = ADULT_BASE if segment == "IFHAS" else CHILD_BASE
    d = raw[base].astype(float).copy()
    if segment == "IFHAS":
        d["egfr_low"] = np.where(raw["egfr"].isna(), np.nan, (raw["egfr"] < 75).astype(float))
        d["uacr_high"] = np.where(raw["uacr"].isna(), np.nan, (raw["uacr"] > 30).astype(float))
    else:
        d["anemia"] = np.where(raw["hemoglobin"].isna(), np.nan, (raw["hemoglobin"] < 11).astype(float))
        d["mchat_pos"] = np.where(raw["mchat_score"].isna(), np.nan, (raw["mchat_score"] >= 3).astype(float))
        d["wfh_extreme"] = np.where(raw["bmi_or_wfh_z"].isna(), np.nan, (raw["bmi_or_wfh_z"].abs() > 2).astype(float))
    return d


def mask_by_consent(df: pd.DataFrame, consent_override: dict[str, bool] | None = None) -> pd.DataFrame:
    """Blank (NaN) every feature whose data domain is unavailable or not consented, per row."""
    out = df.copy()
    for dom in DATA_DOMAINS:
        key = dom["key"]
        ok = df[f"avail_{key}"].to_numpy(bool)
        if consent_override is not None and key in consent_override:
            ok = ok & bool(consent_override[key])
        else:
            ok = ok & df[f"consent_{key}"].to_numpy(bool)
        for f in dom["features"]:
            out.loc[~ok, f] = np.nan
    return out


def _completeness_band(c: float) -> str:
    return "low" if c < 0.55 else ("mid" if c < 0.80 else "high")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SegmentModel:
    segment: str
    cols: list[str] = field(default_factory=list)
    medians: pd.Series | None = None
    scaler: StandardScaler | None = None
    lr: LogisticRegression | None = None
    platt: LogisticRegression | None = None

    def fit(self, raw_masked: pd.DataFrame, y: np.ndarray) -> None:
        X = _design(raw_masked, self.segment)
        self.cols = list(X.columns)
        self.medians = X.median(numeric_only=True).fillna(0.0)
        Xi = X.fillna(self.medians)
        self.scaler = StandardScaler().fit(Xi)
        self.lr = LogisticRegression(C=1.0, max_iter=1000).fit(self.scaler.transform(Xi), y)

    def _z(self, raw_masked: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        X = _design(raw_masked, self.segment)
        was_missing = X.isna()
        return self.scaler.transform(X.fillna(self.medians)), was_missing

    def raw_proba(self, raw_masked: pd.DataFrame) -> np.ndarray:
        Z, _ = self._z(raw_masked)
        return self.lr.predict_proba(Z)[:, 1]

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit_platt(self, raw_masked: pd.DataFrame, y: np.ndarray) -> None:
        """Smooth Platt recalibration on a held-out split (monotone, no step artefacts at the extremes)."""
        self.platt = LogisticRegression(C=1e3, max_iter=500).fit(self._logit(self.raw_proba(raw_masked)), y)

    def proba(self, raw_masked: pd.DataFrame) -> np.ndarray:
        p = self.raw_proba(raw_masked)
        return self.platt.predict_proba(self._logit(p))[:, 1] if self.platt is not None else p

    def shap(self, raw_masked_row: pd.DataFrame) -> list[dict]:
        """Exact linear SHAP in logit space, grouped by clinical feature, shown in percentage points."""
        Z, miss = self._z(raw_masked_row)
        phi = self.lr.coef_[0] * Z[0]
        base = float(self.lr.intercept_[0])
        groups: dict[str, dict] = {}
        for j, col in enumerate(self.cols):
            parent = DERIVED_PARENT.get(col, col)
            g = groups.setdefault(parent, {"phi": 0.0, "missing": True, "cols": []})
            g["phi"] += float(phi[j])
            g["missing"] = g["missing"] and bool(miss.iloc[0, j])
            g["cols"].append(col)
        rows = []
        p0 = float(_sigmoid(base))
        for parent, g in groups.items():
            pp = (float(_sigmoid(base + g["phi"])) - p0) * 100.0
            val = raw_masked_row.iloc[0].get(parent, np.nan)
            rows.append({"feature": parent, "label": FEATURE_LABELS.get(parent, parent), "value": None if pd.isna(val) else float(val),
                         "logit_phi": g["phi"], "contribution_pp": pp, "imputed": g["missing"],
                         "direction": "raises" if pp > 0 else ("lowers" if pp < 0 else "neutral")})
        rows.sort(key=lambda r: abs(r["contribution_pp"]), reverse=True)
        return rows


class RiskEngine:
    """Train once, score many. All training data is synthetic."""

    def __init__(self, n: int = 10000, seed: int = 7):
        self.seed = seed
        self.pop = generate_population(n, seed)
        self.models: dict[str, SegmentModel] = {}
        self.q_resid: dict[tuple[str, str], float] = {}
        self.q_class: dict[tuple[str, int], float] = {}
        self.test_df: pd.DataFrame | None = None
        self.metrics: dict = {}
        self._fit()

    # ---------------------------------------------------------------- training
    def _fit(self) -> None:
        rng = np.random.default_rng(self.seed)
        pop = self.pop
        masked = mask_by_consent(pop)
        split = np.empty(len(pop), dtype=object)
        for seg in ("Well-Child", "IFHAS"):
            idx = np.where(pop["segment"].to_numpy() == seg)[0]
            rng.shuffle(idx)
            a, b, c = int(len(idx) * .45), int(len(idx) * .60), int(len(idx) * .75)
            split[idx[:a]], split[idx[a:b]], split[idx[b:c]], split[idx[c:]] = "train", "recal", "conf", "test"
        pop["split"] = split
        for seg in ("Well-Child", "IFHAS"):
            m = SegmentModel(seg)
            tr = (pop["segment"] == seg) & (pop["split"] == "train")
            m.fit(masked[tr], pop.loc[tr, "unmet_need"].to_numpy())
            rc = (pop["segment"] == seg) & (pop["split"] == "recal")
            m.fit_platt(masked[rc], pop.loc[rc, "unmet_need"].to_numpy())
            self.models[seg] = m
        p_all = self.predict_population(pop, masked=masked)
        pop["p"] = p_all
        pop["band"] = [_completeness_band(c) for c in pop["data_completeness"]]
        conf = pop[pop["split"] == "conf"]
        resid = (conf["risk_proxy"] - conf["p"]).abs()
        glob = self._conf_quantile(resid.to_numpy())
        for band in ("low", "mid", "high"):
            r = resid[conf["band"] == band].to_numpy()
            self.q_resid[("all", band)] = self._conf_quantile(r) if len(r) >= 40 else glob
        # class-conditional LAC thresholds
        for y in (0, 1):
            sel = conf[conf["unmet_need"] == y]
            s = (1 - sel["p"]).to_numpy() if y == 1 else sel["p"].to_numpy()  # LAC score = 1 - p(true class)
            self.q_class[("lac", y)] = self._conf_quantile(s)
        self.test_df = pop[pop["split"] == "test"].copy()
        self._compute_metrics()

    @staticmethod
    def _conf_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
        n = len(scores)
        if n == 0:
            return 0.5
        level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(scores, level, method="higher"))

    # ---------------------------------------------------------------- prediction
    def predict_population(self, df: pd.DataFrame, masked: pd.DataFrame | None = None,
                           consent_override: dict[str, bool] | None = None) -> np.ndarray:
        masked = mask_by_consent(df, consent_override) if masked is None or consent_override else masked
        out = np.zeros(len(df))
        for seg, m in self.models.items():
            sel = (df["segment"] == seg).to_numpy()
            if sel.any():
                out[sel] = m.proba(masked[sel])
        return out

    def score_population(self, df: pd.DataFrame, consent_override: dict[str, bool] | None = None) -> pd.DataFrame:
        """Return df with p, ci_low, ci_high, completeness and fallback columns."""
        out = df.copy()
        masked = mask_by_consent(df, consent_override)
        out["p"] = self.predict_population(df, masked=masked, consent_override=None)
        comp = np.zeros(len(df))
        for k in DOMAIN_KEYS:
            ok = df[f"avail_{k}"].to_numpy(bool) & (bool(consent_override[k]) if consent_override and k in consent_override else df[f"consent_{k}"].to_numpy(bool))
            comp += DOMAIN_WEIGHTS[k] * ok
        out["data_completeness"] = comp
        out["fallback"] = comp < FALLBACK_THRESHOLD
        q = np.array([self.q_resid[("all", _completeness_band(c))] for c in comp])
        out["ci_low"] = np.clip(out["p"] - q, 0, 1)
        out["ci_high"] = np.clip(out["p"] + q, 0, 1)
        out.loc[out["fallback"], ["p", "ci_low", "ci_high"]] = np.nan
        return out

    def _person_frame(self, person: dict, consent: set[str]) -> pd.DataFrame:
        row = {}
        for col, v in person["features"].items():
            dom = FEATURE_TO_DOMAIN.get(col)
            blocked = dom is not None and (dom not in consent or dom not in person["available_domains"])
            row[col] = np.nan if (v is None or blocked) else float(v)
        return pd.DataFrame([row])

    def score_person(self, person: dict, consent: set[str] | None = None) -> dict:
        """Full explainable score for one persona under the given consent set."""
        consent = set(person["consent_default"] if consent is None else consent)
        completeness = compute_completeness(person["available_domains"], consent)
        base = {"person_id": person["id"], "completeness": completeness, "consent": sorted(consent),
                "fallback": completeness < FALLBACK_THRESHOLD, "threshold": FALLBACK_THRESHOLD}
        if base["fallback"]:
            base.update({"risk": None, "ci_low": None, "ci_high": None, "half_width": None, "shap": [], "conformal_set": [],
                         "care_gap_score": None, "tier": "Baseline schedule",
                         "summary": f"Risk estimate withheld: authorized data completeness {completeness:.0%} is below {FALLBACK_THRESHOLD:.0%}."})
            return base
        frame = self._person_frame(person, consent)
        seg = person["segment"]
        m = self.models[seg]
        p = float(m.proba(frame)[0])
        band = _completeness_band(completeness)
        q = self.q_resid[("all", band)]
        lo, hi = max(0.0, p - q), min(1.0, p + q)
        sets = []
        if 1 - p <= self.q_class[("lac", 1)]:
            sets.append("Elevated unmet need")
        if p <= self.q_class[("lac", 0)]:
            sets.append("Lower unmet need")
        overdue = float(person.get("months_overdue") or 0.0)
        care_gap = 100.0 * (0.6 * p + 0.4 * min(overdue / 24.0, 1.0))
        if p >= 0.60 or (p >= 0.45 and overdue >= 12):
            tier = "Urgent"
        elif p >= 0.30 or overdue > 0:
            tier = "Priority"
        else:
            tier = "Routine"
        red_flags = self._red_flags(frame.iloc[0])
        if red_flags and tier != "Urgent":
            tier = "Urgent"
        base.update({
            "red_flags": red_flags,
            "risk": p, "ci_low": lo, "ci_high": hi, "half_width": q, "band": band, "conformal_set": sets,
            "care_gap_score": care_gap, "tier": tier, "shap": m.shap(frame),
            "baseline_rate": float(_sigmoid(m.lr.intercept_[0])),
            "summary": f"{p:.1%} ± {q:.1%} [95% CI: {lo:.1%} – {hi:.1%}]",
        })
        return base

    @staticmethod
    def _red_flags(r: pd.Series) -> list[str]:
        """Hard clinical rules that escalate outreach urgency regardless of the statistical score."""
        flags = []
        if r.get("fh_variant") == 1:
            flags.append("EGP familial hypercholesterolemia variant")
        if (r.get("ldl") or 0) >= 4.9:
            flags.append("LDL 4.9 mmol/L or more")
        if (r.get("hba1c") or 0) >= 6.5:
            flags.append("HbA1c in the diabetes range")
        if r.get("egfr") is not None and not pd.isna(r.get("egfr")) and r["egfr"] < 60 and (r.get("uacr") or 0) >= 30:
            flags.append("eGFR under 60 with raised uACR")
        if (r.get("mchat_score") or 0) >= 3:
            flags.append("M-CHAT-R/F score in medium or high band")
        return flags

    # ---------------------------------------------------------------- metrics
    @staticmethod
    def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
        edges = np.linspace(0, 1, bins + 1)
        ece = 0.0
        for i in range(bins):
            sel = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
            if sel.any():
                ece += sel.mean() * abs(y[sel].mean() - p[sel].mean())
        return float(ece)

    @staticmethod
    def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
        edges = np.linspace(0, 1, bins + 1)
        rows = []
        for i in range(bins):
            sel = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
            if sel.sum() >= 5:
                rows.append({"mean_predicted": float(p[sel].mean()), "observed_rate": float(y[sel].mean()), "n": int(sel.sum())})
        return pd.DataFrame(rows)

    @staticmethod
    def decision_curve(y: np.ndarray, p: np.ndarray, thresholds=None) -> pd.DataFrame:
        thresholds = np.arange(0.05, 0.61, 0.025) if thresholds is None else thresholds
        n, prev = len(y), y.mean()
        rows = []
        for t in thresholds:
            pred = p >= t
            tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
            w = t / (1 - t)
            rows.append({"threshold": float(t), "model": tp / n - fp / n * w, "treat_all": prev - (1 - prev) * w, "treat_none": 0.0})
        return pd.DataFrame(rows)

    def _compute_metrics(self) -> None:
        t = self.test_df
        y, p = t["unmet_need"].to_numpy(), t["p"].to_numpy()
        masked = mask_by_consent(t)
        p_raw = np.zeros(len(t))
        for seg, m in self.models.items():
            sel = (t["segment"] == seg).to_numpy()
            p_raw[sel] = m.raw_proba(masked[sel])
        covered = ((t["risk_proxy"] >= (t["p"] - t["band"].map(lambda b: self.q_resid[("all", b)]))) &
                   (t["risk_proxy"] <= (t["p"] + t["band"].map(lambda b: self.q_resid[("all", b)])))).mean()
        set_cov, set_size = [], []
        for pi, yi in zip(p, y):
            s = {1} if (1 - pi) <= self.q_class[("lac", 1)] else set()
            if pi <= self.q_class[("lac", 0)]:
                s.add(0)
            set_cov.append(yi in s)
            set_size.append(len(s))
        self.metrics = {
            "n_test": int(len(t)), "prevalence": float(y.mean()), "auc": float(roc_auc_score(y, p)),
            "brier_calibrated": float(brier_score_loss(y, p)), "brier_uncalibrated": float(brier_score_loss(y, p_raw)),
            "ece_calibrated": self.expected_calibration_error(y, p), "ece_uncalibrated": self.expected_calibration_error(y, p_raw),
            "reliability_calibrated": self.reliability_table(y, p), "reliability_uncalibrated": self.reliability_table(y, p_raw),
            "decision_curve": self.decision_curve(y, p), "interval_coverage": float(covered),
            "set_coverage": float(np.mean(set_cov)), "mean_set_size": float(np.mean(set_size)),
            "interval_halfwidth": {b: float(self.q_resid[("all", b)]) for b in ("low", "mid", "high")},
        }


@lru_cache(maxsize=2)
def get_engine(n: int = 10000, seed: int = 7) -> RiskEngine:
    return RiskEngine(n, seed)


if __name__ == "__main__":  # smoke test: python -m src.ml_conformal_engine
    from data.synthetic_generator import get_personas

    eng = get_engine()
    m = eng.metrics
    print(f'AUC {m["auc"]:.3f} | Brier {m["brier_calibrated"]:.3f} | ECE {m["ece_calibrated"]:.3f} (raw {m["ece_uncalibrated"]:.3f}) | '
          f'interval coverage {m["interval_coverage"]:.1%} | set coverage {m["set_coverage"]:.1%}')
    print("half-widths by completeness band:", {k: round(v, 3) for k, v in m["interval_halfwidth"].items()})
    for p in get_personas():
        s = eng.score_person(p)
        top = ", ".join(f'{r["label"]} {r["contribution_pp"]:+.1f}pp' for r in s["shap"][:3])
        print(f'{p["id"]} {p["name"]:<20} {s["tier"]:<18} {s["summary"]}  | {top}')
