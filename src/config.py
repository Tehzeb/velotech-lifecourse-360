"""
VeloTech LifeCourse-360 - Design system (Future Health "Warm Editorial MedTech Minimalist").

Two complete palettes (LIGHT and DARK) share one set of CSS variables. `set_theme()` swaps the
palette in place, so every builder, inline style and Plotly chart picks up the active theme.
Native Streamlit widgets are styled explicitly in both themes so text is always readable.

Streamlit markdown quirk: lines indented 4+ spaces, or separated by blank lines, become code
blocks. HTML builders therefore pass through `_c()`, which strips indentation and blank lines.
"""
from __future__ import annotations

import colorsys
import json
import math
from functools import lru_cache
from string import Template
from typing import Iterable, Sequence

# --------------------------------------------------------------------------------------
# 1. DEVELOPER / CONTACT DETAILS
# --------------------------------------------------------------------------------------
TEAM = [
    {
        "name": "Muhammad Tehzeeb ul Hassan",
        "short": "Tehzeeb",
        "initials": "MT",
        "role": "Team Lead, AI & ML Engineer",
        "team": "VeloTech Health AI",
        "email": "tehzeeb.x51214@gmail.com",
        "github": "https://github.com/Tehzeb",
        "github_label": "github.com/Tehzeb",
        "linkedin": "https://www.linkedin.com/in/muhammad-tehzeeb-ul-hassan/",
        "linkedin_label": "linkedin.com/in/muhammad-tehzeeb-ul-hassan",
    },
    {
        "name": "Hajra Bibi",
        "short": "Hajra",
        "initials": "HB",
        "role": "Team Member, AI Engineer",
        "team": "VeloTech Health AI",
        "email": "hajramurtaza260@gmail.com",
        "github": "https://github.com/HajraBibi73",
        "github_label": "github.com/HajraBibi73",
        "linkedin": "https://www.linkedin.com/in/engr-hajra-bibi-4103042a5",
        "linkedin_label": "linkedin.com/in/engr-hajra-bibi-4103042a5",
    },
]
PROFILE = TEAM[0]  # team lead; used for the primary contact (CTA email, sidebar brand)

ICON_GITHUB = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.921.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>')
ICON_LINKEDIN = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>')
ICON_MAIL = ('<svg viewBox="0 0 24 24" aria-hidden="true" style="fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round">'
             '<rect x="3" y="5" width="18" height="14" rx="2.5"/><path d="M3.5 7.5l8.5 6 8.5-6"/></svg>')

# --------------------------------------------------------------------------------------
# 2. PALETTES
# --------------------------------------------------------------------------------------
# --- Colour themes: 4 palettes x (light, dark). Tuple order is given by _KEYS. ---
_KEYS = ("page", "sidebar", "sand", "sand_alt", "white", "gallery", "callout", "proof_bg", "taupe", "taupe_card", "taupe_text",
         "cream", "cream_text", "heading", "ink", "slate", "accent", "accent_hover", "accent_text", "hairline", "border_strong",
         "dots", "input_bg")
_PALETTES: dict[str, dict[str, tuple]] = {
    "Ocean": {
        "light": ("#F4F8FA", "#E8F0F4", "#E3EEF3", "#EAF2F6", "#FFFFFF", "#F8FBFC", "#EEF4F7", "#F0F6F9", "#0F3D4C", "#1B5163", "#FFFFFF",
                  "#DDEBF1", "#0B2530", "#17506A", "#0B1F29", "#4A5F6B", "#0F766E", "#0B5F58", "#FFFFFF", "#D3E0E7", "#AFC3CE", "#A9C2CE", "#FFFFFF"),
        "dark": ("#0A1317", "#0E1B21", "#0F1D24", "#0C171D", "#13232C", "#162A34", "#1A303B", "#0D1A20", "#12303B", "#1B4452", "#F2FAFC",
                 "#16303B", "#E9F5F8", "#CFE9F0", "#EAF5F8", "#A9C1CB", "#2DD4BF", "#5EEAD4", "#052E2B", "#26414D", "#3F5E6C", "#35525F", "#0F1E25"),
    },
    "Indigo": {
        "light": ("#F6F6FB", "#ECECF7", "#E6E7F6", "#ECEDF8", "#FFFFFF", "#FAFAFE", "#F0F0FA", "#F3F3FA", "#26255E", "#37367F", "#FFFFFF",
                  "#E1E2F5", "#14143A", "#2E2C7E", "#12122E", "#56566F", "#4F46E5", "#4338CA", "#FFFFFF", "#DADBEE", "#B6B8DA", "#B4B7DF", "#FFFFFF"),
        "dark": ("#0D0D1C", "#121226", "#15152E", "#11112A", "#1A1A36", "#1E1E3E", "#24244A", "#101024", "#23235A", "#31317A", "#F4F4FF",
                 "#22224C", "#ECECFF", "#D8D9FF", "#F0F0FF", "#B4B6D8", "#A5B4FC", "#C7D2FE", "#14143A", "#2F2F5C", "#4B4B86", "#45457A", "#141430"),
    },
    "Forest": {
        "light": ("#F5F8F4", "#E9F0E8", "#E4EDE1", "#EBF2E9", "#FFFFFF", "#FAFCF9", "#EFF4EC", "#F1F6EF", "#1E4630", "#2D5E41", "#FFFFFF",
                  "#DDEADB", "#112619", "#24523A", "#10201A", "#52625A", "#1F7A4D", "#176340", "#FFFFFF", "#D4E1D2", "#AEC3AB", "#A9C2A6", "#FFFFFF"),
        "dark": ("#0A130E", "#0E1B13", "#102017", "#0C1911", "#14261B", "#182C20", "#1C3326", "#0D1A12", "#18402A", "#235A3B", "#F1FBF4",
                 "#173324", "#E8F7EC", "#CDEBD7", "#EAF7EE", "#A9C4B3", "#4ADE80", "#86EFAC", "#052E16", "#25422F", "#3D6149", "#355843", "#0F1E15"),
    },
    "Sand": {
        "light": ("#F5F2EB", "#EAE4D8", "#E6DFD3", "#EAE4D8", "#FFFFFF", "#FAF8F5", "#F3EFEA", "#F5F2EB", "#6E5D53", "#7D6C62", "#FFFFFF",
                  "#EAE3D5", "#1C1B19", "#6D594D", "#1C1B19", "#5C5650", "#736054", "#5E4D42", "#FFFFFF", "#D5CDC0", "#B3AA9C", "#B8B0A4", "#FFFFFF"),
        "dark": ("#14110E", "#1A1613", "#221D18", "#1B1714", "#27211C", "#2B251F", "#322B25", "#191512", "#3B312A", "#4A3D34", "#F7F2EA",
                 "#322B25", "#F1EBE2", "#E7D5C6", "#F4EFE8", "#CBC2B8", "#D4B8A0", "#E3CBB7", "#1E1814", "#4B423A", "#6C6056", "#6A5F55", "#1F1A16"),
    },
}
PALETTE_NAMES = list(_PALETTES)
BASE_PRIMARY = "#0F766E"   # Streamlit primaryColor in .streamlit/config.toml; native slider fills are hue-shifted from it


def _sem(mode: str) -> dict:
    if mode == "light":
        return dict(azure="#2563EB", emerald="#1F7A5C", amber="#B45309", ok_bg="#E8F3EE", warn_bg="#FBF0E4", info_bg="#E9F0FD",
                    shadow="0 1px 2px rgba(20,40,50,.05), 0 8px 24px rgba(20,40,50,.07)", overlay="rgba(20,30,38,0.42)")
    return dict(azure="#7DA8FF", emerald="#4CC59A", amber="#F2A55A", ok_bg="rgba(76,197,154,0.14)", warn_bg="rgba(242,165,90,0.16)",
                info_bg="rgba(125,168,255,0.16)", shadow="0 1px 2px rgba(0,0,0,.35), 0 10px 28px rgba(0,0,0,.30)", overlay="rgba(0,0,0,0.66)")


def _hls(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)


def _primary_filter(accent: str, mode: str = "light") -> str:
    """CSS filter that turns Streamlit's fixed primary colour (BASE_PRIMARY) into the active accent colour."""
    hb, lb, sb = _hls(BASE_PRIMARY)
    ha, la, sa = _hls(accent)
    shift = ((ha - hb) * 360) % 360
    bright = max(0.7, min(2.4 if mode == "dark" else 1.5, la / lb))
    sat = max(0.2, min(1.4, sa / sb))
    return f"hue-rotate({shift:.0f}deg) saturate({sat:.2f}) brightness({bright:.2f})"


def build_palette(name: str, mode: str) -> dict:
    vals = dict(zip(_KEYS, _PALETTES[name][mode]))
    t = vals["taupe_text"].lstrip("#")
    out = {"mode": mode, **_sem(mode), **vals}
    out["taupe_sub"] = f"rgba({int(t[0:2], 16)},{int(t[2:4], 16)},{int(t[4:6], 16)},0.80)"
    out.update(bronze=out["accent"], bronze_hover=out["accent_hover"], ghost_text=out["ink"], series=out["accent"], pf=_primary_filter(out["accent"], mode))
    return out


COLORS: dict[str, str] = build_palette("Ocean", "light")  # mutated in place by set_theme()
_ACTIVE = {"palette": "Ocean"}
FONT_STACK = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"


def set_theme(mode: str, palette: str | None = None) -> str:
    """Switch the active palette and mode ('light' or 'dark'). Mutates COLORS in place."""
    name = palette if palette in _PALETTES else _ACTIVE["palette"]
    _ACTIVE["palette"] = name
    COLORS.clear()
    COLORS.update(build_palette(name, "dark" if mode == "dark" else "light"))
    return COLORS["mode"]


def current_theme() -> str:
    return COLORS["mode"]


def current_palette() -> str:
    return _ACTIVE["palette"]


def palette_swatch(name: str) -> str:
    return _PALETTES[name][COLORS["mode"]][_KEYS.index("accent")]


def plot_colorway() -> list[str]:
    return [COLORS["series"], COLORS["azure"], COLORS["emerald"], COLORS["amber"], COLORS["border_strong"], COLORS["hairline"]]


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------------------
# 3. HELPERS
# --------------------------------------------------------------------------------------
def _c(html: str) -> str:
    """Compact HTML: strip indentation and blank lines so Streamlit never renders a code block."""
    return "\n".join(line.strip() for line in html.splitlines() if line.strip())


def _esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# --------------------------------------------------------------------------------------
# 4. GLOBAL CSS
# --------------------------------------------------------------------------------------
_CSS = Template(
    """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root{
  --page:${page}; --sidebar:${sidebar}; --sand:${sand}; --sand-alt:${sand_alt}; --white:${white}; --gallery:${gallery};
  --callout:${callout}; --proof:${proof_bg}; --taupe:${taupe}; --taupe-card:${taupe_card}; --taupe-text:${taupe_text};
  --taupe-sub:${taupe_sub}; --cream:${cream}; --cream-text:${cream_text}; --heading:${heading}; --ink:${ink};
  --slate:${slate}; --accent:${accent}; --accent-h:${accent_hover}; --accent-t:${accent_text}; --ghost:${ghost_text};
  --azure:${azure}; --emerald:${emerald}; --amber:${amber}; --hair:${hairline}; --strong:${border_strong};
  --input:${input_bg}; --ok-bg:${ok_bg}; --warn-bg:${warn_bg}; --info-bg:${info_bg}; --shadow:${shadow}; --overlay:${overlay}; --dots:${dots}; --pf:${pf};
  color-scheme:${mode};
}
html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(svg):not(path):not(code):not(pre):not(.vt-code){font-family:${font};}
html, body{background:var(--page) !important;}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]{background:var(--page) !important;color:var(--slate);}
[data-testid="stHeader"]{background:transparent !important;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{visibility:hidden;height:0;}
a{color:var(--accent);}
hr{border:none !important;height:1px;background:var(--hair) !important;}
h1,h2,h3,h4,h5{font-family:${font};letter-spacing:-0.015em;}
[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,[data-testid="stMarkdownContainer"] h5{color:var(--heading);font-weight:500;}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{color:var(--slate);}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *{color:var(--slate) !important;}
[data-testid="stSpinner"] *{color:var(--slate) !important;}
:focus-visible{outline:2px solid var(--azure);outline-offset:2px;}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"], [data-testid="stSidebarContent"]{background:var(--sidebar) !important;border-right:1px solid var(--hair);}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:var(--ink);}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *{color:var(--slate) !important;}
[data-testid="stSidebarCollapseButton"] svg, [data-testid="stExpandSidebarButton"] svg{color:var(--slate) !important;}
.vt-sb-brand{display:flex;align-items:center;gap:10px;margin:2px 0 14px 0;}
.vt-sb-logo{width:34px;height:34px;border-radius:10px;background:var(--accent);color:var(--accent-t);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;}
.vt-sb-name{font-size:15px;font-weight:600;color:var(--ink);line-height:1.2;}
.vt-sb-sub{font-size:11.5px;color:var(--slate);}
.vt-sb-label{font-size:11.5px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--slate);margin:14px 0 4px 0;}
.vt-sb-card{background:var(--white);border:1px solid var(--hair);border-radius:14px;padding:12px 14px;}
.vt-sb-card .nm{font-size:13px;font-weight:600;color:var(--ink);}
.vt-sb-card .rl{font-size:11.5px;color:var(--slate);margin-bottom:8px;}

/* ---------- Native widgets (explicit colours so both themes stay readable) ---------- */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *, .stCheckbox label, .stCheckbox label *, .stRadio label, .stRadio label *,
.stToggle label, .stToggle label *{color:var(--ink) !important;}
[data-testid="stSelectbox"] > div > div, [data-testid="stMultiSelect"] > div > div, [data-testid="stTextInput"] > div > div,
[data-testid="stNumberInput"] > div > div, div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"]{
  background:var(--input) !important;border-color:var(--hair) !important;border-radius:10px !important;}
[data-testid="stSelectbox"] > div > div *, [data-testid="stMultiSelect"] > div > div *, [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
div[data-baseweb="select"] *{color:var(--ink) !important;-webkit-text-fill-color:var(--ink);}
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg, div[data-baseweb="select"] svg{fill:var(--slate) !important;color:var(--slate) !important;}
input::placeholder{color:var(--slate) !important;opacity:.8;}
div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul, ul[role="listbox"], div[data-baseweb="menu"], [data-testid="stSelectboxVirtualDropdown"]{background:var(--white) !important;border:1px solid var(--hair);border-radius:12px;}
li[role="option"], li[role="option"] *, [role="option"], [role="option"] *{color:var(--ink) !important;background:transparent !important;}
li[role="option"]:hover, li[role="option"][aria-selected="true"], [role="option"]:hover, [role="option"][aria-selected="true"]{background:var(--callout) !important;}
span[data-baseweb="tag"]{background:var(--accent) !important;border-radius:8px !important;}
span[data-baseweb="tag"] *{color:var(--accent-t) !important;}
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"]{color:var(--slate) !important;}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"]{color:var(--ink) !important;}
[data-testid="stCheckbox"] label:has(input:not(:checked)) > div:first-of-type{background:var(--input) !important;border-color:var(--strong) !important;}
[data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type{background:var(--accent) !important;border-color:var(--accent) !important;}
[data-testid="stCheckbox"] label:has(input:checked) svg polyline, [data-testid="stCheckbox"] label:has(input:checked) svg path{stroke:var(--accent-t) !important;}
[data-testid="stCheckbox"] label:has(input[role="switch"]:checked) > div:first-of-type{background:var(--accent) !important;border-color:var(--accent) !important;}
[data-testid="stCheckbox"] label:has(input[role="switch"]:not(:checked)) > div:first-of-type{background:var(--strong) !important;border-color:var(--strong) !important;}
[data-testid="stCheckbox"] label:has(input[role="switch"]) > div:first-of-type > div{background:#FFFFFF !important;box-shadow:0 1px 3px rgba(0,0,0,.35);}
[data-testid="stSlider"] [role="group"] > div > div:first-child{filter:var(--pf);}
[data-testid="stSlider"] [role="group"] > div > div:nth-child(2){background:var(--accent) !important;}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"]{color:var(--ink) !important;}
[data-testid="stRadio"] label > div:first-child{filter:var(--pf);}
[data-testid="stCameraInput"] *{color:var(--ink);}

.stButton button, .stDownloadButton button{border-radius:9999px;padding:.45rem 1.25rem;font-weight:500;font-family:${font};transition:background .15s ease,border-color .15s ease;}
.stButton button[kind="primary"], .stDownloadButton button{background:var(--accent) !important;border:1px solid var(--accent) !important;}
.stButton button[kind="primary"] *, .stDownloadButton button *{color:var(--accent-t) !important;}
.stButton button[kind="primary"]:hover{background:var(--accent-h) !important;border-color:var(--accent-h) !important;}
.stButton button[kind="secondary"]{background:transparent !important;border:1px solid var(--strong) !important;}
.stButton button[kind="secondary"] *{color:var(--ghost) !important;}
.stButton button[kind="secondary"]:hover{border-color:var(--accent) !important;background:var(--callout) !important;}

.stTabs [data-baseweb="tab-list"]{gap:.15rem;flex-wrap:wrap;}
.stTabs [data-baseweb="tab-border"]{background:var(--hair) !important;}
.stTabs [data-baseweb="tab"]{background:transparent !important;padding:.75rem 1rem;}
.stTabs [data-baseweb="tab"] p{color:var(--slate) !important;font-weight:500;font-family:${font};}
.stTabs [aria-selected="true"] p{color:var(--heading) !important;font-weight:600;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--accent) !important;height:2px;}
.stTabs [data-baseweb="tab-panel"], [data-testid="stTabs"] [role="tabpanel"]{background:var(--white);border:1px solid var(--hair);border-radius:0 0 16px 16px;padding:1.4rem 1.4rem 1.8rem 1.4rem;box-shadow:var(--shadow);}
div[data-testid="stMetric"]{background:var(--callout);border-radius:12px;padding:12px 16px;border:1px solid var(--hair);}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] *{color:var(--heading) !important;font-weight:400;font-size:1.55rem !important;
  line-height:1.2 !important;white-space:normal !important;overflow:visible !important;text-overflow:clip !important;}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] *{color:var(--slate) !important;}
[data-testid="stExpander"]{background:var(--white);border:1px solid var(--hair) !important;border-radius:12px;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary *{color:var(--ink) !important;}
[data-testid="stImageCaption"]{color:var(--slate) !important;}
[data-testid="stDialog"]{background:var(--overlay) !important;}
[data-testid="stDialog"] > div{background:var(--white) !important;border:1px solid var(--hair);border-radius:20px;box-shadow:var(--shadow);}
section[role="dialog"]{background:transparent !important;color:var(--slate);}
section[role="dialog"] h1, section[role="dialog"] h2, section[role="dialog"] h3{color:var(--heading) !important;font-weight:500;}
section[role="dialog"] button svg{color:var(--slate) !important;}

/* ---------- Page shell: full-bleed sections ---------- */
[data-testid="stMainBlockContainer"]{max-width:none !important;padding:0 0 2rem 0 !important;}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]{gap:0 !important;}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *{width:100%;max-width:1240px;margin:0 auto .85rem auto;padding-left:1.6rem;padding-right:1.6rem;box-sizing:border-box;}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:has(.vt-full){max-width:none;margin:0;padding:0;}
.vt-full{width:100%;}
.vt-inner{max-width:1240px;margin:0 auto;padding:0 1.6rem;position:relative;}
.vt-sec{padding-top:84px;padding-bottom:84px;}
.vt-bg-sand{background:var(--sand);} .vt-bg-white{background:var(--white);} .vt-bg-alt{background:var(--sand-alt);}
.vt-bg-page{background:var(--page);} .vt-bg-proof{background:var(--proof);} .vt-bg-taupe{background:var(--taupe);color:var(--taupe-text);}

/* ---------- Typography ---------- */
.vt-eyebrow{font-size:11.5px;font-weight:500;letter-spacing:.14em;color:var(--slate);margin-bottom:14px;}
.vt-eyebrow.dot::before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ink);margin-right:10px;vertical-align:1px;}
h2.vt-h2, .vt-h2{font-size:clamp(30px,3.7vw,46px) !important;font-weight:300 !important;color:var(--heading) !important;line-height:1.12 !important;margin:0 0 16px 0;letter-spacing:-0.022em;padding:0 !important;}
h2.vt-h2.sm, .vt-h2.sm{font-size:clamp(24px,2.5vw,32px) !important;margin-bottom:10px;}
.vt-h3{font-size:19px;font-weight:500;color:var(--ink);margin:0 0 10px 0;}
.vt-body{font-size:14.5px;line-height:1.7;color:var(--slate);}
.vt-two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:40px;}
.vt-btn{display:inline-block;border-radius:9999px;padding:14px 28px;font-size:14.5px;font-weight:500;text-decoration:none !important;transition:background .15s ease,border-color .15s ease;}
.vt-btn.primary{background:linear-gradient(135deg,var(--accent),var(--accent-h));color:var(--accent-t) !important;box-shadow:0 6px 18px -8px var(--accent);}
.vt-btn.primary:hover{background:var(--accent-h);transform:translateY(-1px);}
.vt-btn.ghost{background:transparent;color:var(--ghost) !important;border:1px solid var(--strong);}
.vt-btn.ghost:hover{border-color:var(--accent);}

/* ---------- Utility row (developer + social) ---------- */
.vt-profilebar{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;padding:6px 0 10px 0;border-bottom:1px solid var(--hair);}
.vt-who{display:flex;align-items:center;gap:12px;}
.vt-avatar{width:36px;height:36px;border-radius:50%;background:var(--accent);color:var(--accent-t);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;letter-spacing:.02em;}
.vt-who .nm{font-size:13.5px;font-weight:600;color:var(--ink);line-height:1.2;}
.vt-who .rl{font-size:12px;color:var(--slate);}
.vt-social{display:flex;gap:8px;flex-wrap:wrap;align-items:center;}
.vt-social a{display:inline-flex;align-items:center;gap:8px;padding:7px 13px;border:1px solid var(--hair);border-radius:9999px;color:var(--ink) !important;font-size:12.5px;font-weight:500;background:var(--callout);text-decoration:none !important;transition:border-color .15s ease,background .15s ease;}
.vt-social a:hover{border-color:var(--accent);background:var(--white);}
.vt-social svg{width:16px;height:16px;fill:currentColor;flex:0 0 16px;}
.vt-social.col{flex-direction:column;align-items:flex-start;}
.vt-social.icons{gap:6px;flex-wrap:nowrap;}
.vt-social.icons a{padding:0;width:32px;height:32px;justify-content:center;border-radius:50%;gap:0;}
.vt-social.icons a span{display:none;}
.vt-social.icons svg{width:15px;height:15px;flex:0 0 15px;}
.vt-social.row a{padding:7px 14px;}
.vt-team-bar{display:flex;flex-wrap:wrap;align-items:center;gap:10px 28px;padding:8px 0 12px 0;border-bottom:1px solid var(--hair);}
.vt-member{display:flex;align-items:center;gap:11px;}
.vt-member .vt-avatar{width:34px;height:34px;flex:0 0 34px;}
.vt-member .nm{font-size:13px;font-weight:600;color:var(--ink);line-height:1.2;white-space:nowrap;}
.vt-member .rl{font-size:11.5px;color:var(--slate);line-height:1.3;white-space:nowrap;}
.vt-member .vt-social{margin-left:8px;padding-left:12px;border-left:1px solid var(--hair);}
.vt-team-bar .div{width:1px;align-self:stretch;background:var(--hair);}
.vt-sbm{display:flex;align-items:center;gap:10px;margin-bottom:6px;}
.vt-sbm .vt-avatar{width:30px;height:30px;flex:0 0 30px;font-size:11.5px;}
.vt-sb-card .vt-sbm .rl{margin-bottom:0;}
.vt-sb-card .vt-social.icons{margin:0 0 14px 40px;}
.vt-sb-card .vt-social.icons.last{margin-bottom:0;}
.vt-team-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin-top:34px;}
.vt-tcard{background:var(--callout);border:1px solid var(--hair);border-radius:22px;padding:26px 28px;display:flex;gap:20px;align-items:flex-start;}
.vt-tcard .vt-avatar{width:60px;height:60px;flex:0 0 60px;font-size:20px;}
.vt-tcard .nm{font-size:20px;font-weight:400;color:var(--heading);line-height:1.2;}
.vt-tcard .rl{font-size:13.5px;color:var(--slate);margin:4px 0 16px 0;}
.vt-ftteam{display:flex;gap:30px;flex-wrap:wrap;}
.vt-ftteam > div{flex:1 1 190px;}
@media (max-width:760px){.vt-team-grid{grid-template-columns:1fr;}}
.vt-sb-brand{display:flex;align-items:center;gap:10px;margin:2px 0 14px 0;}
.vt-sb-logo{width:34px;height:34px;border-radius:10px;background:var(--accent);color:var(--accent-t);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;}
.vt-sb-name{font-size:15px;font-weight:600;color:var(--ink);line-height:1.2;}
.vt-sb-sub{font-size:11.5px;color:var(--slate);}
.vt-sb-label{font-size:11.5px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--slate);margin:14px 0 4px 0;}
.vt-sb-card{background:var(--white);border:1px solid var(--hair);border-radius:14px;padding:12px 14px;}
.vt-sb-card .nm{font-size:13px;font-weight:600;color:var(--ink);}
.vt-sb-card .rl{font-size:11.5px;color:var(--slate);margin-bottom:8px;}

/* ---------- Nav + hero ---------- */
.vt-nav{display:flex;align-items:center;justify-content:space-between;gap:24px;flex-wrap:wrap;padding:20px 0;border-bottom:1px solid var(--strong);}
.vt-logo{display:flex;align-items:center;gap:14px;flex-wrap:wrap;}
.vt-logo .fh{font-size:13px;letter-spacing:.2em;line-height:1.15;color:var(--slate);font-weight:400;}
.vt-logo .gi{font-size:8.5px;letter-spacing:.14em;line-height:1.35;color:var(--slate);max-width:90px;}
.vt-logo .sep{width:1px;height:30px;background:var(--strong);}
.vt-logo .h2s{font-weight:800;letter-spacing:.03em;color:var(--ink);font-size:21px;}
.vt-logo .vt{font-size:12px;font-weight:600;color:var(--ink);letter-spacing:.1em;}
.vt-links{display:flex;gap:30px;flex-wrap:wrap;}
.vt-links a{font-size:14px;color:var(--ink) !important;text-decoration:none !important;padding:4px 0;border-bottom:1px solid transparent;}
.vt-links a:hover{border-bottom-color:var(--accent);}
.vt-hero-body{position:relative;padding:54px 0 8px 0;}
.vt-hero-body h1{font-size:clamp(40px,5.8vw,70px) !important;line-height:1.06 !important;font-weight:300 !important;color:var(--heading) !important;margin:0 0 30px 0 !important;letter-spacing:-0.026em;padding:0 !important;}
.vt-hero-body h1 .sub{display:block;font-size:.5em;color:var(--ink);letter-spacing:-0.01em;margin-top:14px;font-weight:300;}
[data-testid="stHeaderActionElements"]{display:none !important;}
.vt-tagline{font-size:22px;font-weight:500;color:var(--ink);margin:0 0 14px 0;}
.vt-lede{font-size:17px;line-height:1.7;color:var(--slate);max-width:560px;margin:0 0 30px 0;}
.vt-cta-row{display:flex;gap:12px;flex-wrap:wrap;}
.vt-hero-art{position:absolute;right:0;top:215px;width:44%;max-width:540px;pointer-events:none;}
.vt-wave{width:100%;height:auto;}
.vt-together{margin-top:52px;}
.vt-together .lead{font-size:11.5px;letter-spacing:.14em;color:var(--slate);margin-bottom:20px;}
.vt-co-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:24px;}
.vt-co .role{font-size:12px;color:var(--slate);margin-bottom:12px;}
.vt-co .name{font-size:21px;font-weight:600;color:var(--ink);letter-spacing:-0.01em;line-height:1.2;}
.vt-co .name small{display:block;font-size:11.5px;font-weight:400;color:var(--slate);letter-spacing:0;margin-top:3px;}
.vt-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-top:1px solid var(--strong);margin-top:46px;}
.vt-stats > div{padding:22px 24px 30px 24px;border-left:1px solid var(--strong);}
.vt-stats > div:first-child{border-left:none;padding-left:0;}
.vt-stats .k{font-size:11px;letter-spacing:.14em;color:var(--slate);margin-bottom:10px;}
.vt-stats .v{font-size:20px;font-weight:400;color:var(--ink);line-height:1.3;}

/* ---------- Intro / how it works ---------- */
.vt-intro-grid{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:56px;align-items:start;}
.vt-intro-grid .vt-body{font-size:15.5px;margin-bottom:18px;}
.vt-steps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-top:1px solid var(--hair);margin-top:56px;}
.vt-step{padding:24px 24px 6px 0;border-right:1px solid var(--hair);margin-right:24px;}
.vt-step:last-child{border-right:none;margin-right:0;}
.vt-step .n{font-size:13px;letter-spacing:.1em;color:var(--slate);}
.vt-step .t{font-size:20px;font-weight:400;color:var(--ink);margin:22px 0 10px 0;}
.vt-step .d{font-size:13.5px;line-height:1.65;color:var(--slate);}
.vt-note{font-size:12.5px;color:var(--slate);margin-top:34px;}

/* ---------- Focus explorer ---------- */
.vt-focus-head{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;flex-wrap:wrap;margin-bottom:40px;}
.vt-focus-head .aside{font-size:13.5px;color:var(--slate);max-width:300px;text-align:right;}
.vt-focus-grid{display:grid;grid-template-columns:minmax(0,.78fr) minmax(0,2.3fr);gap:24px;align-items:start;}
.vt-fa{border-radius:20px;background:var(--callout);padding:20px 24px;font-size:16px;color:var(--ink);margin-bottom:12px;}
.vt-fa .n{color:var(--slate);font-size:13px;margin-right:12px;}
.vt-fa.open{background:var(--white);border:1px solid var(--hair);padding:22px 12px 14px 12px;box-shadow:var(--shadow);}
.vt-fa.open .head{padding:0 12px 14px 12px;font-weight:500;display:flex;justify-content:space-between;align-items:center;}
.vt-fa.open .head i{width:7px;height:7px;border-radius:50%;background:var(--ink);}
.vt-fa .sub{display:block;font-size:13.5px;line-height:1.4;color:var(--slate);padding:11px 14px;border-radius:12px;}
.vt-fa .sub.on{background:var(--callout);color:var(--ink);font-weight:500;}
.vt-card{background:var(--white);border-radius:26px;padding:34px 38px 40px 38px;border:1px solid var(--hair);box-shadow:var(--shadow);}
.vt-card .title{font-size:30px;font-weight:400;color:var(--ink);margin:0 0 20px 0;letter-spacing:-0.015em;}
.vt-callout{background:var(--callout);border-radius:14px;padding:18px 20px;margin:14px 0 18px 0;border:1px solid var(--hair);}
.vt-callout .lbl{font-size:12px;font-weight:600;color:var(--ink);margin-bottom:6px;}
.vt-callout .txt{font-size:14px;line-height:1.65;color:var(--slate);}
.vt-card .vt-callout{margin:0 0 30px 0;padding:24px 26px;border:none;}
.vt-card .vt-callout .txt{font-size:17px;color:var(--ink);line-height:1.55;}
.vt-req{list-style:none;margin:0;padding:0;}
.vt-req li{display:flex;gap:14px;padding:0 0 18px 0;font-size:13.5px;line-height:1.6;color:var(--slate);}
.vt-req .dot{flex:0 0 26px;height:26px;border-radius:50%;border:1px solid var(--azure);color:var(--azure);font-size:12px;font-weight:600;display:flex;align-items:center;justify-content:center;margin-top:1px;}
.vt-req .ttl{font-weight:600;color:var(--ink);}
.vt-req .st{display:inline-block;margin-left:8px;font-size:11px;font-weight:600;border-radius:9999px;padding:1px 9px;}
.vt-req .st.verified{color:var(--emerald);border:1px solid var(--emerald);}
.vt-req .st.warning{color:var(--amber);border:1px solid var(--amber);}
.vt-req .st.pending{color:var(--slate);border:1px solid var(--strong);}

/* ---------- Guardrails, chips, messages ---------- */
.vt-boundary{background:var(--cream);border:1px solid var(--strong);border-left:6px solid var(--amber);border-radius:12px;padding:14px 18px;margin:6px 0 16px 0;font-size:13.5px;line-height:1.55;color:var(--cream-text);}
.vt-boundary b{color:var(--heading);}
.vt-chip{display:inline-block;font-size:11.5px;font-weight:600;border-radius:9999px;padding:3px 11px;margin:2px 6px 2px 0;}
.vt-chip.ok{color:var(--emerald);background:var(--ok-bg);border:1px solid var(--emerald);}
.vt-chip.warn{color:var(--amber);background:var(--warn-bg);border:1px solid var(--amber);}
.vt-chip.info{color:var(--azure);background:var(--info-bg);border:1px solid var(--azure);}
.vt-chip.off{color:var(--slate);background:transparent;border:1px dashed var(--strong);}
.vt-prov{background:var(--callout);border-radius:12px;padding:14px 16px;border:1px solid var(--hair);}
.vt-prov .lbl{font-size:12px;font-weight:600;color:var(--ink);margin-bottom:6px;}
.vt-fallback{background:var(--warn-bg);border:1px solid var(--amber);border-radius:12px;padding:16px 18px;color:var(--ink);font-size:14px;line-height:1.55;}
.vt-fallback b{color:var(--amber);}
.vt-msg{background:var(--callout);border-radius:14px;padding:16px 18px;border:1px solid var(--hair);font-size:14.5px;line-height:1.7;color:var(--ink);white-space:pre-wrap;}
.vt-rtl{direction:rtl;text-align:right;}
.vt-pathrow{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr) minmax(0,1fr);gap:16px;padding:14px 0;border-bottom:1px solid var(--hair);font-size:13.5px;}
.vt-pathrow .h{font-weight:600;color:var(--ink);}
.vt-pathrow .m{color:var(--slate);font-size:12.5px;line-height:1.5;}
.vt-pathrow .m b{color:var(--ink);}

/* ---------- Stat row ---------- */
.vt-statrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin:6px 0 12px 0;}
.vt-stat{background:var(--callout);border:1px solid var(--hair);border-radius:14px;padding:12px 14px;}
.vt-stat .k{font-size:12px;color:var(--slate);margin-bottom:4px;}
.vt-stat .v{font-size:22px;font-weight:400;color:var(--heading);line-height:1.2;}

/* ---------- Tables & code ---------- */
.vt-tablewrap{overflow:auto;border:1px solid var(--hair);border-radius:12px;background:var(--white);margin:6px 0 12px 0;}
.vt-table{border-collapse:collapse;width:100%;font-size:13px;}
.vt-table th{position:sticky;top:0;background:var(--callout);color:var(--ink);text-align:left;padding:9px 12px;font-weight:600;border-bottom:1px solid var(--hair);white-space:nowrap;}
.vt-table td{padding:8px 12px;border-bottom:1px solid var(--hair);color:var(--slate);vertical-align:top;}
.vt-table tr:last-child td{border-bottom:none;}
.vt-table tbody tr:hover td{background:var(--callout);}
.vt-table td.sup{color:var(--amber);font-style:italic;}
.vt-code{background:var(--input);color:var(--ink);border:1px solid var(--hair);border-radius:12px;padding:14px 16px;font-size:12px;line-height:1.6;overflow:auto;max-height:380px;margin:6px 0 12px 0;overflow-wrap:anywhere;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}

/* ---------- Summary band (taupe) ---------- */
.vt-sum{position:relative;overflow:hidden;}
.vt-sum .art{position:absolute;right:-4%;top:-10px;width:46%;opacity:.55;pointer-events:none;}
.vt-sum .eyebrow{font-size:11.5px;letter-spacing:.14em;color:var(--taupe-sub);margin-bottom:22px;}
.vt-bigrow{display:flex;align-items:flex-end;justify-content:space-between;gap:30px;flex-wrap:wrap;margin-bottom:44px;position:relative;}
.vt-big{font-size:clamp(44px,6vw,76px);font-weight:300;color:var(--taupe-text);line-height:1;letter-spacing:-0.025em;}
.vt-bigcap{font-size:15px;color:var(--taupe-text);max-width:340px;line-height:1.5;}
.vt-kcards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;margin-bottom:26px;position:relative;}
.vt-kcard{background:var(--taupe-card);border:1px solid rgba(255,255,255,0.2);border-radius:22px;padding:28px 28px 30px 28px;min-height:210px;}
.vt-kcard svg{width:30px;height:30px;stroke:var(--taupe-text);fill:none;stroke-width:1.4;stroke-linecap:round;stroke-linejoin:round;margin-bottom:30px;}
.vt-kcard .v{font-size:30px;font-weight:300;color:var(--taupe-text);line-height:1.1;margin-bottom:10px;}
.vt-kcard .k{font-size:14px;color:var(--taupe-text);margin-bottom:6px;}
.vt-kcard .s{font-size:12.5px;color:var(--taupe-sub);line-height:1.5;}
.vt-stage{display:flex;flex-wrap:wrap;align-items:center;gap:34px;background:var(--cream);border-radius:26px;padding:34px 40px;position:relative;overflow:hidden;}
.vt-stage .ring{flex:0 0 96px;height:96px;border-radius:50%;border:1px solid var(--strong);display:flex;align-items:center;justify-content:center;}
.vt-stage .ring svg{width:38px;height:38px;stroke:var(--heading);fill:none;stroke-width:1.3;stroke-linecap:round;stroke-linejoin:round;}
.vt-stage .body{flex:1 1 320px;min-width:0;}
.vt-stage .eb{font-size:11.5px;letter-spacing:.14em;color:var(--cream-text);margin-bottom:10px;}
.vt-stage .big{font-size:clamp(24px,3vw,38px);font-weight:300;color:var(--heading);line-height:1.15;margin-bottom:12px;letter-spacing:-0.015em;}
.vt-stage .txt{font-size:13.5px;line-height:1.65;color:var(--cream-text);max-width:760px;}
.vt-stage .txt b{color:var(--heading);}
.vt-foot{font-size:12px;color:var(--taupe-sub);margin-top:22px;}

/* ---------- Who it serves ---------- */
.vt-who-grid{display:grid;grid-template-columns:minmax(0,.9fr) minmax(0,1.1fr);gap:64px;align-items:start;}
.vt-art{border-radius:28px;overflow:hidden;border:1px solid var(--hair);background:var(--sand);}
.vt-art svg{display:block;width:100%;height:auto;}
.vt-caption{font-size:13px;color:var(--slate);margin-top:14px;max-width:440px;line-height:1.6;}
.vt-rows{margin-top:30px;border-top:1px solid var(--hair);}
.vt-row{padding:24px 0;border-bottom:1px solid var(--hair);}
.vt-row .t{font-size:21px;font-weight:400;color:var(--ink);margin-bottom:8px;}
.vt-row .d{font-size:14.5px;line-height:1.7;color:var(--slate);}

/* ---------- Timeline ---------- */
.vt-tl{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));margin-top:56px;position:relative;}
.vt-tl::before{content:"";position:absolute;left:6px;right:0;top:6px;height:1px;background:var(--strong);}
.vt-tl > div{position:relative;padding:34px 24px 0 0;}
.vt-tl .pt{position:absolute;left:0;top:0;width:13px;height:13px;border-radius:50%;border:1px solid var(--strong);background:var(--sand);}
.vt-tl .pt.on{background:var(--ink);border-color:var(--ink);}
.vt-tl .d{font-size:22px;font-weight:300;color:var(--heading);margin-bottom:8px;}
.vt-tl .t{font-size:14px;font-weight:500;color:var(--ink);margin-bottom:6px;}
.vt-tl .x{font-size:13px;line-height:1.6;color:var(--slate);}

/* ---------- Assessment grid ---------- */
.vt-assess{display:grid;grid-template-columns:minmax(0,.85fr) minmax(0,1.5fr);gap:56px;align-items:start;}
.vt-proofgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px;}
.vt-proofcard{background:var(--white);border-radius:22px;padding:26px 28px;border:1px solid var(--hair);box-shadow:var(--shadow);}
.vt-proofcard .t{font-size:15px;font-weight:500;color:var(--ink);margin-bottom:14px;line-height:1.35;}
.vt-proofcard .v{font-size:clamp(24px,2.4vw,34px);font-weight:300;color:var(--heading);line-height:1.1;letter-spacing:-0.02em;overflow-wrap:normal;word-break:normal;}
.vt-proofcard, .vt-kcard, .vt-card{transition:transform .15s ease, box-shadow .15s ease;}
.vt-proofcard:hover{transform:translateY(-2px);}
.vt-proofcard .c{font-size:12.5px;color:var(--slate);margin:8px 0 14px 0;line-height:1.5;}
.vt-proofcard ul{margin:0;padding:0;list-style:none;font-size:13px;color:var(--slate);line-height:1.5;}
.vt-proofcard li{display:flex;gap:10px;padding:3px 0;}
.vt-proofcard li::before{content:"";flex:0 0 6px;height:6px;border-radius:50%;background:var(--accent);margin-top:7px;}

/* ---------- FAQ ---------- */
.vt-faq2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 56px;margin-top:44px;}
.vt-acc details{border-top:1px solid var(--hair);}
.vt-acc details:last-child{border-bottom:1px solid var(--hair);}
.vt-acc summary{list-style:none;cursor:pointer;display:flex;justify-content:space-between;align-items:center;gap:18px;padding:20px 0;font-size:16.5px;font-weight:400;color:var(--ink);}
.vt-acc summary::-webkit-details-marker{display:none;}
.vt-acc summary::after{content:"+";flex:0 0 36px;height:36px;border:1px solid var(--hair);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:300;color:var(--slate);}
.vt-acc details[open] summary::after{content:"\\00d7";}
.vt-acc .ans{padding:0 0 24px 0;font-size:14px;line-height:1.75;color:var(--slate);}
.vt-acc .ans p, .vt-acc .ans li{color:var(--slate);}
.vt-acc .ans ul, .vt-acc .ans ol{margin:6px 0 6px 18px;padding:0;}
.vt-acc .ans code{background:var(--callout);padding:1px 6px;border-radius:6px;color:var(--ink);font-size:12.5px;}
.vt-acc .ans b{color:var(--ink);}

/* ---------- CTA + footer ---------- */
.vt-cta{position:relative;overflow:hidden;}
.vt-cta .art{position:absolute;right:0;bottom:-30px;width:42%;pointer-events:none;}
.vt-cta .vt-h2{max-width:640px;font-size:clamp(34px,4.6vw,58px);}
.vt-cta .contact{display:flex;align-items:center;gap:10px;margin-top:34px;font-size:14px;color:var(--slate);}
.vt-cta .contact a{color:var(--ink) !important;text-decoration:underline !important;text-underline-offset:3px;}
.vt-cta .contact svg{width:18px;height:18px;stroke:var(--slate);fill:none;stroke-width:1.6;}
.vt-foot2{padding:48px 0 30px 0;border-top:1px solid var(--hair);}
.vt-footgrid{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,.85fr) minmax(0,.85fr) minmax(0,2fr);gap:36px;align-items:start;}
.vt-footgrid .brand{font-size:17px;font-weight:600;color:var(--ink);margin:14px 0 8px 0;}
.vt-footgrid h4{font-size:11px;letter-spacing:.14em;color:var(--slate);margin:0 0 14px 0;font-weight:500;}
.vt-footgrid a.l{display:block;font-size:14px;color:var(--ink) !important;text-decoration:none !important;padding:5px 0;}
.vt-footgrid a.l:hover{color:var(--accent) !important;}
.vt-footgrid .nm{font-size:14px;font-weight:600;color:var(--ink);}
.vt-footgrid .rl{font-size:12.5px;color:var(--slate);margin-bottom:6px;}
.vt-legal{border-top:1px solid var(--hair);margin-top:34px;padding-top:16px;font-size:12px;color:var(--slate);}

/* ---------- Guided tour ---------- */
.vt-tour-progress{display:flex;gap:5px;margin:2px 0 16px 0;}
.vt-tour-progress span{flex:1;height:5px;border-radius:9999px;background:var(--hair);}
.vt-tour-progress span.done{background:var(--accent);}
.vt-tour-where{display:inline-block;font-size:11.5px;font-weight:600;color:var(--azure);background:var(--info-bg);border:1px solid var(--azure);border-radius:9999px;padding:3px 12px;margin-bottom:10px;}
.vt-tour-title{font-size:24px;font-weight:400;color:var(--heading);margin:0 0 8px 0;line-height:1.2;}
.vt-tour-intro{font-size:14.5px;line-height:1.65;color:var(--slate);margin-bottom:12px;}
.vt-tour-do{margin:0 0 14px 0;padding:0;list-style:none;counter-reset:s;}
.vt-tour-do li{counter-increment:s;display:flex;gap:12px;padding:9px 0;border-bottom:1px solid var(--hair);font-size:14px;line-height:1.55;color:var(--ink);}
.vt-tour-do li::before{content:counter(s);flex:0 0 24px;height:24px;border-radius:50%;background:var(--accent);color:var(--accent-t);font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center;margin-top:1px;}
.vt-tour-do li:last-child{border-bottom:none;}
.vt-tour-see{background:var(--ok-bg);border:1px solid var(--emerald);border-radius:12px;padding:12px 16px;font-size:13.5px;line-height:1.55;color:var(--ink);}
.vt-tour-see b{color:var(--emerald);}
.vt-tour-credit{margin-top:16px;padding-top:14px;border-top:1px solid var(--hair);}
.vt-tour-credit .lbl{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--slate);font-weight:600;margin-bottom:10px;}
.vt-tour-credit .ppl{display:flex;gap:26px;flex-wrap:wrap;}
.vt-tour-person{display:flex;align-items:center;gap:10px;}
.vt-tour-person .vt-avatar{width:34px;height:34px;font-size:12.5px;}
.vt-tour-person .nm{font-size:13.5px;font-weight:600;color:var(--ink);line-height:1.2;}
.vt-tour-person .rl{font-size:12px;color:var(--slate);}

/* ---------- Responsive (breakpoints account for the open sidebar) ---------- */
[data-testid="stHorizontalBlock"]{flex-wrap:wrap !important;}
[data-testid="stColumn"]{min-width:min(100%,170px) !important;}
[data-testid="stHorizontalBlock"]:has(.vt-wide2) > [data-testid="stColumn"]{min-width:min(100%,380px) !important;}
[data-testid="stElementContainer"]:has(.vt-wide2){display:none !important;}
@media (max-width: 1280px){
  .vt-hero-art{display:none;}
  .vt-intro-grid,.vt-two,.vt-focus-grid,.vt-who-grid,.vt-assess,.vt-faq2{grid-template-columns:1fr;gap:32px;}
  .vt-co-grid,.vt-kcards,.vt-steps,.vt-stats{grid-template-columns:repeat(2,minmax(0,1fr));}
  .vt-stats > div, .vt-stats > div:first-child{border-left:none;padding-left:0;}
  .vt-stats > div:nth-child(even){border-left:1px solid var(--strong);padding-left:24px;}
  .vt-steps{gap:0 28px;border-top:none;}
  .vt-step{border-right:none;margin-right:0;border-top:1px solid var(--hair);}
  .vt-tl{grid-template-columns:1fr;border-left:1px solid var(--strong);margin-left:6px;}
  .vt-tl::before{display:none;}
  .vt-tl > div{padding:0 0 28px 28px;}
  .vt-tl .pt{left:-7px;top:4px;}
  .vt-footgrid{grid-template-columns:repeat(2,minmax(0,1fr));}
  .vt-footgrid > div:first-child{grid-column:1/-1;}
  .vt-sec{padding-top:56px;padding-bottom:56px;}
  .vt-focus-head .aside{text-align:left;}
}
@media (max-width: 640px){
  .vt-co-grid,.vt-kcards,.vt-steps,.vt-stats,.vt-footgrid{grid-template-columns:1fr;}
  .vt-stats > div, .vt-stats > div:nth-child(even){border-left:none;padding-left:0;}
  .vt-stage{flex-direction:column;align-items:flex-start;padding:26px;}
  .vt-pathrow{grid-template-columns:1fr;}
  .vt-card{padding:24px 20px;}
}
@media (prefers-reduced-motion: reduce){*{transition:none !important;animation:none !important;}}
"""
)


def global_css() -> str:
    """Full <style> block for the active theme. Inject once per run with unsafe_allow_html=True."""
    return f"<style>{_CSS.safe_substitute(font=FONT_STACK, **COLORS)}</style>"


# --------------------------------------------------------------------------------------
# 5. STIPPLED PARTICLE-WAVE
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _wave(color: str, width: int, height: int, ribbons: int, points: int) -> str:
    circles: list[str] = []
    cy = height * 0.52
    for r in range(ribbons):
        depth = r / (ribbons - 1)
        amp = 34 + 58 * math.sin(math.pi * depth) ** 1.2
        phase, y_off = r * 0.34, (r - ribbons / 2) * 15.5
        for i in range(points):
            t = i / (points - 1)
            x = 12 + t * (width - 24)
            swell = 0.55 + 0.45 * math.sin(t * math.pi + 0.4)
            y = cy + y_off + amp * swell * math.sin(t * math.pi * 3.2 + phase) + 14 * math.sin(t * math.pi * 7.0 + r * 0.6)
            radius = 0.9 + 1.7 * (0.5 + 0.5 * math.cos(t * math.pi * 3.2 + phase)) * (0.45 + 0.55 * depth)
            opacity = 0.28 + 0.62 * (0.5 + 0.5 * math.sin(t * math.pi * 3.2 + phase)) * (0.5 + 0.5 * depth)
            circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.2f}" fill-opacity="{opacity:.2f}"/>')
    return (f'<svg class="vt-wave" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="Stippled dot-matrix wave illustration"><g style="fill:{color}">{"".join(circles)}</g></svg>')


def svg_particle_wave(width: int = 640, height: int = 360, ribbons: int = 15, points: int = 74, color: str | None = None) -> str:
    """Stippled dot-matrix wave. `color` may be a hex value or a CSS variable such as var(--dots)."""
    return _wave(color or "var(--dots)", width, height, ribbons, points)


# --------------------------------------------------------------------------------------
# 6. PROFILE BAR, SOCIAL LINKS, FOOTER
# --------------------------------------------------------------------------------------
def _social_links(labels: bool = True, p: dict | None = None, short_mail: bool = False) -> str:
    p = p or PROFILE
    gh = f'<a href="{p["github"]}" target="_blank" rel="noopener noreferrer" aria-label="GitHub profile of {_esc(p["name"])}" title="GitHub · {_esc(p["github_label"])}">{ICON_GITHUB}<span>{_esc(p["github_label"]) if labels else "GitHub"}</span></a>'
    li = f'<a href="{p["linkedin"]}" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn profile of {_esc(p["name"])}" title="LinkedIn · {_esc(p["linkedin_label"])}">{ICON_LINKEDIN}<span>{_esc(p["linkedin_label"]) if labels else "LinkedIn"}</span></a>'
    ml = f'<a href="mailto:{p["email"]}" aria-label="Email {_esc(p["name"])}" title="Email · {_esc(p["email"])}">{ICON_MAIL}<span>{"Email" if short_mail else _esc(p["email"])}</span></a>'
    return gh + li + ml


def build_profile_bar() -> str:
    """Slim top bar: the whole team side by side, each with icon links to GitHub, LinkedIn and email."""
    members = []
    for i, m in enumerate(TEAM):
        if i:
            members.append('<div class="div"></div>')
        members.append(
            f'''<div class="vt-member"><div class="vt-avatar">{m['initials']}</div>
<div><div class="nm">{_esc(m['name'])}</div><div class="rl">{_esc(m['role'])}</div></div>
<div class="vt-social icons">{_social_links(p=m)}</div></div>''')
    return _c(f'<div class="vt-team-bar">{"".join(members)}</div>')


def build_sidebar_brand() -> str:
    return _c(f"""
<div class="vt-sb-brand"><div class="vt-sb-logo">V</div><div><div class="vt-sb-name">LifeCourse-360</div><div class="vt-sb-sub">{_esc(PROFILE['team'])}</div></div></div>
""")


def build_sidebar_contact() -> str:
    rows = []
    for i, m in enumerate(TEAM):
        last = " last" if i == len(TEAM) - 1 else ""
        rows.append(
            f'''<div class="vt-sbm"><div class="vt-avatar">{m['initials']}</div><div><div class="nm">{_esc(m['name'])}</div><div class="rl">{_esc(m['role'])}</div></div></div>
<div class="vt-social icons{last}">{_social_links(p=m)}</div>''')
    return _c(f'<div class="vt-sb-card">{"".join(rows)}</div>')


# --------------------------------------------------------------------------------------
# 7. SECTION BUILDERS
# --------------------------------------------------------------------------------------
NAV_ITEMS = [("The Challenge", "vt-challenge"), ("Focus Areas", "vt-focus"), ("Workspace", "vt-workspace"),
             ("Safeguards", "vt-safeguards"), ("Timeline", "vt-timeline"), ("Evaluation", "vt-evaluation"), ("FAQ", "vt-faq"), ("Team", "vt-team")]

_ICON = lambda body: f'<svg viewBox="0 0 32 32" aria-hidden="true">{body}</svg>'
ICON_SHIELD = _ICON('<path d="M16 3l10 4v8c0 7-4.5 11.5-10 14C10.500 26.500 6 22 6 15V7z"/><path d="M11.500 16l3.200 3.200L21 12.800"/>')
ICON_GAUGE = _ICON('<path d="M5 22a11 11 0 1 1 22 0"/><path d="M16 22l5-8"/><circle cx="16" cy="22" r="1.600"/>')
ICON_TREND = _ICON('<path d="M4 23l8-8 5 5 11-11"/><path d="M21 9h7v7"/>')
ICON_LOCK = _ICON('<rect x="6" y="14" width="20" height="14" rx="3"/><path d="M10.500 14V10a5.500 5.500 0 0 1 11 0v4"/><circle cx="16" cy="21" r="1.500"/>')
ICON_STAGE = _ICON('<path d="M7 27h18"/><path d="M9 27v-5h14v5"/><path d="M16 10v7"/><rect x="13.500" y="4.500" width="5" height="9" rx="2.500"/><path d="M6 8l2 1.500M26 8l-2 1.500M4 14h2.500M25.500 14H28"/>')
ICON_MAIL_L = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2.500"/><path d="M3.500 7.500l8.500 6 8.500-6"/></svg>'


def build_hero(stats: Sequence[tuple[str, str]] | None = None) -> str:
    """Full-width sand hero: plain nav, light display heading, 'Together with' strip and 4-cell stat strip."""
    stats = stats or [
        ("FOCUS AREA", "Population Health · Challenge 04"),
        ("STACK", "Foundry, MCP, conformal ML and OpenCV"),
        ("SAFEGUARDS", "k-anonymity, parity audit, 95% intervals"),
        ("TRY IT", "Sample people or your own data"),
    ]
    links = "".join(f'<a href="#{a}">{_esc(l)}</a>' for l, a in NAV_ITEMS)
    stat_html = "".join(f'<div><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>' for k, v in stats)
    cos = [("Co-creator", "Microsoft", ""), ("Co-creator", "inception42", ""), ("Co-creator", "MBZUAI", "Mohamed bin Zayed University of Artificial Intelligence"), ("Collaborator", "H2S", "")]
    co_html = "".join(f'<div class="vt-co"><div class="role">{r}</div><div class="name">{_esc(n)}{f"<small>{_esc(sub)}</small>" if sub else ""}</div></div>' for r, n, sub in cos)
    return _c(f"""
<div class="vt-full vt-bg-sand" id="vt-challenge"><div class="vt-inner">
<div class="vt-nav">
<div class="vt-logo"><div class="fh">FUTURE<br>HEALTH</div><div class="gi">A GLOBAL INITIATIVE BY ABU DHABI</div><div class="sep"></div><div class="h2s">H2S</div><div class="sep"></div><div class="vt">VELOTECH</div></div>
<nav class="vt-links" aria-label="Sections">{links}</nav>
</div>
<div class="vt-hero-body">
<div class="vt-hero-art">{svg_particle_wave(720, 440)}</div>
<h1>Future Health<br>Digital Innovation Challenge<span class="sub">VeloTech LifeCourse-360</span></h1>
<p class="vt-tagline">From signals to solutions.</p>
<p class="vt-lede">Credible, clinically relevant AI, multi-omic, and connected bio-sensing solutions supporting earlier, safer, and more scalable preventive action across Abu Dhabi.</p>
<div class="vt-cta-row"><a class="vt-btn primary" href="#vt-workspace">Open the live workspace →</a><a class="vt-btn ghost" href="#vt-focus">Explore the focus area</a></div>
</div>
<div class="vt-together"><div class="lead">Challenge organized by</div><div class="vt-co-grid">{co_html}</div></div>
<div class="vt-stats">{stat_html}</div>
<div style="height:56px"></div>
</div></div>
""")


def build_intro() -> str:
    """'Credible, clinically relevant solutions' + How it works 01-04 (white band)."""
    steps = [
        ("01", "Sense & Authorize", "Consent-gated ingestion of clinical, genomic (EGP), developmental, family-history and environmental signals through MCP and HL7 FHIR R4."),
        ("02", "Stratify & Explain", "Calibrated risk and care-gap scoring with SHAP attribution and a conformal 95% interval, so uncertainty is always visible."),
        ("03", "Route & Synchronize", "Foundry IQ grounds the pathway in DoH Well-Child and IFHAS guidance, then bundles overdue family members into one visit."),
        ("04", "Engage, Safeguard & Evaluate", "Culturally adapted outreach in Arabic, English or Urdu, with k-anonymity, fairness audits and outcome tracking."),
    ]
    html = "".join(f'<div class="vt-step"><div class="n">{n}</div><div class="t">{_esc(t)}</div><div class="d">{_esc(d)}</div></div>' for n, t, d in steps)
    return _c(f"""
<div class="vt-full vt-bg-white"><div class="vt-inner vt-sec">
<div class="vt-intro-grid">
<div><div class="vt-eyebrow">The approach</div><h2 class="vt-h2">Credible, clinically relevant solutions to prevention gaps.</h2></div>
<div><p class="vt-body">Eligibility and outreach for the Well-Child Visit programme and adult comprehensive screening (IFHAS) still lean on broad age and gender rules, while real risk sits in family history, genomics, chronic conditions, development, visit history and place.</p>
<p class="vt-body">VeloTech LifeCourse-360 reads those signals, shows how sure it is, and turns them into the right visit at the right time for the whole household, in the language and channel each family prefers.</p></div>
</div>
<div style="height:48px"></div><div class="vt-eyebrow">How it works</div>
<div class="vt-steps" style="margin-top:14px">{html}</div>
<div class="vt-note">Note: the sample people are synthetic. If you enter your own details, they are used only in your current session and are not saved. Final clinical decisions rest with licensed physicians.</div>
</div></div>
""")


FOCUS_OUTCOMES = [
    ("Transparent risk and care-gap identification", "Calibrated score, SHAP attribution, conformal 95% interval and a data provenance badge."),
    ("Personalized preventive pathways", "Foundry IQ grounded in DoH Well-Child and IFHAS guidance, with household synchronization."),
    ("Culturally appropriate engagement and consent choice", "Arabic, English and Urdu outreach, granular consent, optional OpenCV pre-screener."),
    ("Local population safeguards and low-data fallbacks", "k-anonymity suppression, FPR/FNR parity audit and a below-40% completeness fallback."),
    ("Outcome-based evaluation", "Calibration, decision curve analysis, loop closure and estimated QALYs."),
]


def build_focus_explorer(outcome_status: Sequence[str] | None = None) -> str:
    """Master-detail focus-area explorer: accordion list on the left, white detail card on the right."""
    outcome_status = list(outcome_status or ["verified"] * 5)
    labels = {"verified": "Verified live", "warning": "Needs attention", "pending": "Pending"}
    req = "".join(
        f'<li><span class="dot">{i + 1}</span><span><span class="ttl">{_esc(t)}</span><span class="st {s}">{labels.get(s, s)}</span><br>{_esc(d)}</span></li>'
        for i, ((t, d), s) in enumerate(zip(FOCUS_OUTCOMES, outcome_status)))
    subs = [("Population Health · Challenge 04", True), ("Well-Child Visit pathway · children 0-5", False),
            ("IFHAS adult screening pathway · 18+", False), ("Outreach in Arabic, English and Urdu", False)]
    sub_html = "".join(f'<span class="sub{" on" if on else ""}">{_esc(t)}</span>' for t, on in subs)
    return _c(f"""
<div class="vt-full vt-bg-alt" id="vt-focus"><div class="vt-inner vt-sec">
<div class="vt-focus-head"><div><div class="vt-eyebrow">Focus areas</div><h2 class="vt-h2" style="margin:0">Digital Innovation Focus Area</h2></div>
<div class="aside">One life-course engine for children, adults and whole households.</div></div>
<div class="vt-focus-grid">
<div>
<div class="vt-fa open"><div class="head"><span>Scope of this prototype</span><i></i></div>{sub_html}</div>
</div>
<div class="vt-card">
<div class="vt-eyebrow dot">Population Health · Challenge 04</div>
<div class="title">Precision Prevention and Screening</div>
<div class="vt-callout"><div class="lbl">The challenge</div><div class="txt">How might we identify children, adults and communities with unmet or elevated health risk and connect them to the right preventive intervention, Well-Child Visit or Adult comprehensive screening pathway, at the right time, through approaches that are personalized and culturally appropriate?</div></div>
<div class="vt-two">
<div><div class="vt-h3">The challenge</div><p class="vt-body">Abu Dhabi is moving from reactive care to predictive prevention across the life course. Outreach for Well-Child Visits and IFHAS still depends heavily on age and gender, so families with the highest unmet need can be missed, invited late, or sent to a pathway that does not fit.</p>
<div class="vt-h3" style="margin-top:22px">Objective</div><p class="vt-body">Use relevant health and genomic insight to connect each child, adult and community to the right visit at the right time, while keeping uncertainty and authorized data use visible.</p></div>
<div><div class="vt-h3">Your solution should</div><ul class="vt-req">{req}</ul></div>
</div>
</div>
</div>
</div></div>
""")


def build_summary_band(kpis: Sequence[tuple[str, str, str]], headline: tuple[str, str] = ("", "")) -> str:
    """Taupe live-summary band: big number, four frosted cards, cream clinical-boundary banner."""
    icons = [ICON_SHIELD, ICON_GAUGE, ICON_TREND, ICON_LOCK]
    cards = "".join(
        f'<div class="vt-kcard">{icons[i % 4]}<div class="v">{_esc(v)}</div><div class="k">{_esc(k)}</div><div class="s">{_esc(sub)}</div></div>'
        for i, (k, v, sub) in enumerate(list(kpis)[:4]))
    big, cap = headline
    big_html = f'<div class="vt-bigrow"><div class="vt-big">{_esc(big)}</div><div class="vt-bigcap">{_esc(cap)}</div></div>' if big else ""
    return _c(f"""
<div class="vt-full vt-bg-taupe vt-sum" id="vt-safeguards"><div class="art">{svg_particle_wave(640, 380, color="var(--taupe-text)")}</div><div class="vt-inner vt-sec">
<div class="eyebrow">Live summary</div>{big_html}
<div class="vt-kcards">{cards}</div>
<div class="vt-stage"><div class="ring">{ICON_STAGE}</div><div class="body">
<div class="eb">The clinical boundary · Population prioritization vs. individual decision-making</div>
<div class="big">Prioritizing outreach, never replacing the physician.</div>
<div class="txt"><b>POPULATION PRIORITIZATION VS. INDIVIDUAL CLINICAL DECISION-MAKING:</b> This platform prioritizes preventive outreach and recall timing within approved DoH guidelines; final clinical decisions rest exclusively with licensed physicians.</div></div></div>
<div class="vt-foot">Figures update with the selected person, consent choices and the safeguards in the workspace above. Outcome numbers are modelled on synthetic data.</div>
</div></div>
""")


def build_exec_strip(kpis: Sequence[tuple[str, str, str]]) -> str:  # backward-compatible alias
    return build_summary_band(kpis)


def _household_art() -> str:
    """Original illustration: one household, one visit (no photography needed)."""
    cx, cy = 280, 300
    nodes = [(cx - 150, cy - 160, "A", "Parent · overdue"), (cx + 150, cy - 140, "C", "Child · overdue"),
             (cx - 160, cy + 150, "G", "Grandparent"), (cx + 150, cy + 165, "S", "Sibling")]
    halo = "paint-order:stroke;stroke:var(--sand);stroke-width:6px;stroke-linejoin:round"
    edges = "".join(f'<line x1="{x}" y1="{y}" x2="{cx}" y2="{cy}" style="stroke:var(--accent);stroke-width:1.5;opacity:.55"/>' for x, y, _, _ in nodes)
    circ = "".join(
        f'<circle cx="{x}" cy="{y}" r="38" style="fill:var(--white);stroke:var(--accent);stroke-width:1.5"/>'
        f'<text x="{x}" y="{y + 8}" text-anchor="middle" style="fill:var(--heading);font-size:24px;font-weight:300">{ini}</text>'
        f'<text x="{x}" y="{y + 66}" text-anchor="middle" style="fill:var(--ink);font-size:13.5px;{halo}">{_esc(lab)}</text>' for x, y, ini, lab in nodes)
    return (f'<svg viewBox="0 0 560 600" role="img" aria-label="Illustration: four household members linked to one combined visit">'
            f'<defs><pattern id="vtdg" width="28" height="28" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.6" style="fill:var(--dots)"/></pattern></defs>'
            f'<rect width="560" height="600" style="fill:var(--sand)"/><rect width="560" height="600" fill="url(#vtdg)" opacity=".7"/>'
            f'<circle cx="{cx}" cy="{cy}" r="120" style="fill:none;stroke:var(--strong);stroke-width:1;stroke-dasharray:3 7"/>'
            f'<circle cx="{cx}" cy="{cy}" r="205" style="fill:none;stroke:var(--strong);stroke-width:1;stroke-dasharray:3 9"/>'
            f'{edges}<circle cx="{cx}" cy="{cy}" r="58" style="fill:var(--accent)"/>'
            f'<text x="{cx}" y="{cy + 2}" text-anchor="middle" style="fill:var(--accent-t);font-size:30px;font-weight:300">1</text>'
            f'<text x="{cx}" y="{cy + 24}" text-anchor="middle" style="fill:var(--accent-t);font-size:12px;letter-spacing:.08em">VISIT</text>{circ}</svg>')


def build_who() -> str:
    rows = [("Families and households", "Parents, grandparents and children get one combined, culturally adapted invitation instead of separate reminders."),
            ("Clinicians and screening teams", "Every recommendation shows its reasons, its uncertainty and the guideline it comes from, and stays a proposal for the physician to decide."),
            ("Population health planners", "See reach, error rates and uptake by region and community, with small groups protected by k-anonymity."),
            ("Workers in high-heat settings", "Heat exposure and kidney markers bring the right panel forward, with early and late slots around shifts.")]
    html = "".join(f'<div class="vt-row"><div class="t">{_esc(t)}</div><div class="d">{_esc(d)}</div></div>' for t, d in rows)
    return _c(f"""
<div class="vt-full vt-bg-page"><div class="vt-inner vt-sec"><div class="vt-who-grid">
<div><div class="vt-art">{_household_art()}</div><div class="vt-caption">One household, one visit: overdue adults and children are bundled into a single appointment.</div></div>
<div><div class="vt-eyebrow">Who it serves</div><h2 class="vt-h2">Built for the people who turn health signals into earlier action.</h2><div class="vt-rows">{html}</div></div>
</div></div></div>
""")


def build_timeline() -> str:
    items = [("15 Oct", "Registration opens", "The challenge opens and the build begins.", True),
             ("10 Nov", "Registration closes", "Teams register and start building.", False),
             ("20 Nov", "Working prototype due", "Prototype, technical documentation, assumptions and limitations.", False),
             ("20–30 Nov", "Evaluation", "Entries are assessed by an expert panel.", False),
             ("7–11 Dec", "Awards at GITEX 2026", "The top teams present on stage in Abu Dhabi.", True)]
    html = "".join(f'<div><span class="pt{" on" if on else ""}"></span><div class="d">{_esc(d)}</div><div class="t">{_esc(t)}</div><div class="x">{_esc(x)}</div></div>' for d, t, x, on in items)
    return _c(f"""
<div class="vt-full vt-bg-sand" id="vt-timeline"><div class="vt-inner vt-sec">
<div class="vt-eyebrow">Timeline</div><h2 class="vt-h2" style="margin:0">From prototype to the GITEX stage.</h2>
<div class="vt-tl">{html}</div><div class="vt-note">Dates may be subject to change.</div>
</div></div>
""")


def default_proof_metrics() -> list[dict]:
    return [
        {"title": "Health impact and clinical relevance", "value": "n/a", "caption": "Overdue-care capture", "bullets": ["Well-Child and IFHAS gaps", "Estimated QALYs gained"]},
        {"title": "Technical depth and architecture", "value": "4 agents", "caption": "Orchestrated swarm", "bullets": ["MCP FHIR R4 tools", "Hybrid RAG with citations"]},
        {"title": "Working solution and demonstration quality", "value": "8 personas", "caption": "Runs offline at zero cost", "bullets": ["Deterministic engine", "Live consent recalculation"]},
        {"title": "Responsible AI, safety and data governance", "value": "k >= 10", "caption": "Small-cell suppression", "bullets": ["Fairlearn parity audit", "Below-40% fallback"]},
        {"title": "Scalability and deployment readiness", "value": "Local-first", "caption": "Foundry Local ready", "bullets": ["Stateless scoring", "FHIR-native interfaces"]},
        {"title": "Commercial viability and path to pilot", "value": "5 partners", "caption": "SEHA · PureHealth · M42 · Mediclinic · Burjeel", "bullets": ["Outreach-cost savings", "Pilot readiness checklist"]},
    ]


def build_proof_grid(metrics: Sequence[dict] | None = None) -> str:
    """Heading on the left, 2-column by 3-row assessment grid on the right."""
    metrics = list(metrics or default_proof_metrics())[:6]
    cards = []
    for m in metrics:
        bullets = "".join(f"<li>{_esc(b)}</li>" for b in m.get("bullets", []))
        cards.append(f'<div class="vt-proofcard"><div class="t">{_esc(m["title"])}</div><div class="v">{_esc(m["value"])}</div><div class="c">{_esc(m.get("caption", ""))}</div><ul>{bullets}</ul></div>')
    return _c(f"""
<div class="vt-full vt-bg-proof" id="vt-evaluation"><div class="vt-inner vt-sec"><div class="vt-assess">
<div><div class="vt-eyebrow">How the assessment works</div><h2 class="vt-h2">Impact first. Then proof that it works.</h2>
<p class="vt-body">Each card is computed from the running prototype against the six official assessment pillars, using the person, consent and safeguards you have selected.</p></div>
<div class="vt-proofgrid">{"".join(cards)}</div>
</div></div></div>
""")


def build_faq_accordion(items: Iterable[tuple[str, str]]) -> str:
    """Two-column FAQ with outlined circle +/× icons."""
    items = list(items)
    half = (len(items) + 1) // 2
    def col(sub):
        return '<div class="vt-acc">' + "".join(f'<details><summary>{_esc(q)}</summary><div class="ans">{a}</div></details>' for q, a in sub) + "</div>"
    return _c(f"""
<div class="vt-full vt-bg-white" id="vt-faq"><div class="vt-inner vt-sec">
<div class="vt-eyebrow">FAQ</div><h2 class="vt-h2" style="margin:0">Your questions, answered.</h2>
<div class="vt-body" style="margin-top:12px;max-width:640px">How to test it, architecture, modelling assumptions, known limitations, data governance, hardware extensibility and pilot readiness.</div>
<div class="vt-faq2">{col(items[:half])}{col(items[half:])}</div>
</div></div>
""")


def build_team() -> str:
    """'The team' band: one card per member with GitHub, LinkedIn and email."""
    cards = "".join(
        f'''<div class="vt-tcard"><div class="vt-avatar">{m['initials']}</div><div>
<div class="nm">{_esc(m['name'])}</div><div class="rl">{_esc(m['role'])} · {_esc(m['team'])}</div>
<div class="vt-social row">{_social_links(labels=False, p=m, short_mail=True)}</div></div></div>''' for m in TEAM)
    return _c(f"""
<div class="vt-full vt-bg-white" id="vt-team"><div class="vt-inner vt-sec">
<div class="vt-eyebrow">The team</div><h2 class="vt-h2">Built by VeloTech Health AI.</h2>
<p class="vt-body" style="max-width:620px">Questions, feedback or collaboration ideas are welcome. Reach any member of the team directly.</p>
<div class="vt-team-grid">{cards}</div>
</div></div>
""")


def build_cta() -> str:
    p = PROFILE
    contact_links = " or ".join(f'<a href="mailto:{m["email"]}">{_esc(m["short"])} ({m["email"]})</a>' for m in TEAM)
    return _c(f"""
<div class="vt-full vt-bg-sand vt-cta"><div class="art">{svg_particle_wave(640, 380)}</div><div class="vt-inner vt-sec">
<div class="vt-eyebrow">Questions or feedback?</div>
<h2 class="vt-h2">Turn a health signal into a working solution.</h2>
<div class="vt-cta-row" style="margin-top:26px"><a class="vt-btn primary" href="#vt-workspace">Open the live workspace →</a><a class="vt-btn ghost" href="mailto:{p['email']}">Email the team lead</a></div>
<div class="contact">{ICON_MAIL_L}<span>Questions? Contact {contact_links}</span></div>
</div></div>
""")


def build_footer() -> str:
    p = PROFILE
    built_by = "".join(
        f'<div><div class="nm">{_esc(m["name"])}</div><div class="rl">{_esc(m["role"])}, {_esc(m["team"])}</div><div class="vt-social icons">{_social_links(p=m)}</div></div>'
        for m in TEAM)
    return _c(f"""
<div class="vt-full vt-bg-page"><div class="vt-inner"><div class="vt-foot2">
<div class="vt-footgrid">
<div><div class="vt-logo"><div class="fh">FUTURE<br>HEALTH</div><div class="gi">A GLOBAL INITIATIVE BY ABU DHABI</div><div class="sep"></div><div class="vt">VELOTECH</div></div>
<div class="brand">VeloTech LifeCourse-360</div><div class="vt-body" style="max-width:360px">Prototype for the Future Health Digital Innovation Challenge, Population Health, Challenge 04: Precision Prevention and Screening.</div></div>
<div><h4>PROJECT</h4><a class="l" href="#vt-challenge">The challenge</a><a class="l" href="#vt-focus">Focus areas</a><a class="l" href="#vt-workspace">Workspace</a><a class="l" href="#vt-timeline">Timeline</a><a class="l" href="#vt-faq">FAQ</a></div>
<div><h4>RESPONSIBLE AI</h4><a class="l" href="#vt-safeguards">Clinical boundary</a><a class="l" href="#vt-evaluation">Evaluation</a><a class="l" href="#vt-faq">Assumptions and limitations</a></div>
<div><h4>TEAM</h4><div class="vt-ftteam">{built_by}</div></div>
</div>
<div class="vt-legal">Synthetic data only. Not a medical device. Population prioritization and outreach support; final clinical decisions rest with licensed physicians.</div>
</div></div></div>
""")


# --------------------------------------------------------------------------------------
# 8. SMALL COMPONENTS
# --------------------------------------------------------------------------------------
def boundary_banner() -> str:
    return _c('<div class="vt-boundary" role="note"><b>Population prioritization and outreach, not a clinical decision.</b> This tool ranks who to invite and when, within approved DoH guidelines. Diagnosis, treatment and final screening decisions rest exclusively with the licensed physician.</div>')


def chip(text: str, kind: str = "info") -> str:
    return f'<span class="vt-chip {kind}">{_esc(text)}</span>'


def provenance_badge(domains: Sequence[tuple[str, bool, bool]]) -> str:
    chips = []
    for label, consented, available in domains:
        if consented and available:
            chips.append(chip(f"{label}: in use", "ok"))
        elif consented and not available:
            chips.append(chip(f"{label}: no record", "warn"))
        else:
            chips.append(chip(f"{label}: not shared", "off"))
    return _c(f'<div class="vt-prov"><div class="lbl">Authorized data provenance</div>{"".join(chips)}</div>')


def callout(title: str, body_html: str) -> str:
    return _c(f'<div class="vt-callout"><div class="lbl">{_esc(title)}</div><div class="txt">{body_html}</div></div>')


def fallback_banner(completeness: float, threshold: float = 0.40) -> str:
    return _c(f'<div class="vt-fallback"><b>Insufficient-data fallback active.</b> Authorized data completeness is {completeness:.0%}, below the {threshold:.0%} safety threshold. AI risk extrapolation is blocked to avoid a misleading result, and the baseline DoH age and gender schedule applies instead.</div>')


def section_header(eyebrow: str, title: str, body: str = "") -> str:
    body_html = f'<div class="vt-body">{_esc(body)}</div>' if body else ""
    return _c(f'<div class="vt-eyebrow">{_esc(eyebrow)}</div><h2 class="vt-h2 sm">{_esc(title)}</h2>{body_html}')


def html_table(df, max_height: int = 440) -> str:
    """Themed HTML table (replaces st.dataframe so both themes are fully readable)."""
    def cell(v) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "<td>n/a</td>"
        if isinstance(v, bool):
            return f"<td>{'Yes' if v else 'No'}</td>"
        if isinstance(v, float):
            return f"<td>{v:,.3f}".rstrip("0").rstrip(".") + "</td>"
        s = str(v)
        if s.startswith("[SUPPRESSED"):
            return f'<td class="sup">{_esc(s)}</td>'
        if s in ("PASS", "WARN", "FAIL"):
            return f'<td>{chip(s, {"PASS": "ok", "WARN": "warn", "FAIL": "warn"}[s])}</td>'
        return f"<td>{_esc(s)}</td>"
    head = "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(cell(v) for v in row) + "</tr>" for row in df.itertuples(index=False))
    return _c(f'<div class="vt-tablewrap" style="max-height:{max_height}px"><table class="vt-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


def json_block(obj, max_height: int = 380) -> str:
    """Themed, scrollable JSON viewer. Built as ONE line of HTML (indent via &nbsp;, newlines via <br>)
    so the Markdown renderer cannot collapse whitespace or turn indentation into a code block."""
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    lines = []
    for line in text.split("\n"):
        stripped = line.lstrip(" ")
        lines.append("&nbsp;" * (len(line) - len(stripped)) + _esc(stripped))
    return f'<div class="vt-code" style="max-height:{max_height}px">{"<br>".join(lines)}</div>'


# --------------------------------------------------------------------------------------
# 9. PLOTLY THEME
# --------------------------------------------------------------------------------------
def style_figure(fig, height: int = 360, legend: bool = True):
    """Brand palette with fully transparent backgrounds, readable in light and dark."""
    C = COLORS
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=plot_colorway(),
        font=dict(family=FONT_STACK, color=C["slate"], size=12), margin=dict(l=10, r=10, t=44, b=10), showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=C["slate"])),
        title=dict(font=dict(color=C["heading"], size=15)),
        hoverlabel=dict(bgcolor=C["white"], font=dict(color=C["ink"]), bordercolor=C["hairline"]),
    )
    fig.update_xaxes(gridcolor=C["hairline"], zerolinecolor=C["hairline"], linecolor=C["hairline"], tickfont=dict(color=C["slate"]), title_font=dict(color=C["slate"]), automargin=True)
    fig.update_yaxes(gridcolor=C["hairline"], zerolinecolor=C["hairline"], linecolor=C["hairline"], tickfont=dict(color=C["slate"]), title_font=dict(color=C["slate"]), automargin=True)
    return fig
