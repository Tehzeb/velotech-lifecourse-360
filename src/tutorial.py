"""
VeloTech LifeCourse-360 - Guided tour.

One source of truth for the step-by-step test guide. It is shown:
  * automatically in a pop-up the first time the app opens,
  * any time from the "Guided tour" button in the sidebar,
  * and as the first question in the Technical Documentation & FAQ accordion.
"""
from __future__ import annotations

from src.config import TEAM, _c, _esc

TOUR: list[dict] = [
    {
        "where": "Welcome", "title": "Welcome to VeloTech LifeCourse-360",
        "intro": ("This app finds children and adults who are overdue for a check-up or at higher risk, explains why, "
                  "and invites the right family members to the right visit. This short tour shows you how to test every part. "
                  "Eight sample people are built in (all made up), and you can also enter your own details. Nothing you enter is stored."),
        "do": ["Click Next to start.",
               "You can reopen this tour at any time with the Guided tour button in the left sidebar."],
        "see": "A left sidebar with a person and consent boxes, and a main page with six tabs.",
        "credit": True,
    },
    {
        "where": "Left sidebar", "title": "Pick a person",
        "intro": "Each person is a made-up example that shows a different situation.",
        "do": ["In the sidebar, keep Fatima Al Mansoori selected. She is 34, a mother, and overdue for her adult check-up.",
               "Scroll to the top of the main page and read her short story and tags."],
        "see": "Her name, age and story above the tabs. Try the others later: Omar, Khalid, Imran, Mariam, Aisha, Yousef and Daniel.",
    },
    {
        "where": "Left sidebar", "title": "Check your own data",
        "intro": "Besides the sample people, you can get a result for yourself or a family member by entering your own details.",
        "do": ["In the sidebar, open Choose a person and pick My own data. A form opens.",
               "Fill in what you know: age, check-up history, lab results, family history and so on. Leave unknown fields blank; they are treated as missing, never as zero.",
               "Watch the completeness message at the bottom of the form. At least 40% is needed for a risk estimate.",
               "Click Save and see my result. Use Edit my data in the sidebar to change anything later."],
        "see": "A risk estimate for your own entry, or the safe age-based schedule if too little was entered. Nothing is saved after you close the session.",
    },
    {
        "where": "Tab 1 · Risk & Uncertainty", "title": "Read the risk and how sure the AI is",
        "intro": "Tab 1 shows the main result and the reasons behind it.",
        "do": ["Open the first tab.",
               "Find the big percentage and the plus-or-minus range. The range shows how sure the AI is.",
               "Look at the bar chart. Amber bars push the risk up, green bars pull it down, grey bars mean there was no data."],
        "see": "A risk percentage with a 95% range, a tier (Urgent, Priority or Routine) and a chart of reasons.",
    },
    {
        "where": "Left sidebar", "title": "Change consent and test the safety fallback",
        "intro": "People choose which kinds of data the AI may use. The result must change when they change their mind.",
        "do": ["Untick Laboratory results in the sidebar. The risk updates straight away and the range gets wider.",
               "Click Withdraw consent (demo). Go back to tab 1: the risk is hidden because less than 40% of data is shared.",
               "Open tab 2 and notice it now shows the standard age-based schedule instead.",
               "Click Restore defaults to bring everything back."],
        "see": "A yellow message saying the AI will not guess when too little data is shared.",
    },
    {
        "where": "Tab 2 · DoH Pathways & Household", "title": "Care plan and the family visit",
        "intro": "Tab 2 turns the result into a plan that is based on the approved guidelines.",
        "do": ["Open the second tab and read the care plan. Each line shows when to go, any fasting rule and what happens after.",
               "Scroll to the family graph. Fatima and her toddler Omar are both overdue, so they are offered one combined visit.",
               "Type a question in the guideline box, for example: when is the developmental screening done?"],
        "see": "A Synchronized Family Prevention Appointment card and an answer with guideline citations.",
    },
    {
        "where": "Tab 3 · Engagement & OpenCV", "title": "Messages in the person's language, and the pulse check",
        "intro": "Tab 3 writes the invitation in the right language and channel, and has an optional camera pre-screener.",
        "do": ["Open the third tab. In the sidebar change Outreach language to Urdu and Outreach channel to SMS. The message rewrites itself.",
               "Switch on Optional OpenCV bio-signal pre-screener in the sidebar, then tick Bio-signal pre-screener / wearable in the consent list.",
               "Move the heart-rate slider and watch the pulse wave and the estimated heart rate follow it.",
               "Select Omar in the sidebar to try the child milestone and eye-tracking check instead."],
        "see": "A message in the chosen language, then a pulse waveform with a heart rate close to your slider value.",
    },
    {
        "where": "Tab 4 · Population Safeguards", "title": "Fairness and privacy",
        "intro": "Tab 4 checks that the system is fair across regions and protects small groups.",
        "do": ["Open the fourth tab.",
               "Drag the k-anonymity slider in the sidebar from 5 up to 25.",
               "Watch more table cells change to the privacy message as k grows.",
               "Scroll down to the fallback simulator to see completeness fall as consent is withdrawn."],
        "see": "Cells marked SUPPRESSED when a group is too small, and bar charts comparing error rates across groups.",
    },
    {
        "where": "Tab 5 · Evaluation & Calibration", "title": "Proof that it works",
        "intro": "Tab 5 checks the numbers: are the predicted risks honest, and does the plan help?",
        "do": ["Open the fifth tab.",
               "On the reliability chart, dots close to the diagonal line mean the risks are honest.",
               "Move the outreach capacity slider and watch detections, estimated QALYs and the follow-up funnel change."],
        "see": "Calibration and decision-curve charts, and a comparison against age-only outreach.",
    },
    {
        "where": "Tab 6 · Foundry Agents & MCP", "title": "Under the hood",
        "intro": "Tab 6 shows the four AI agents, the six workflow steps and every tool call.",
        "do": ["Open the sixth tab and look at the six workflow boxes.",
               "Pick an MCP call from the list to see the exact request and response.",
               "Scroll to the AI-103 table at the bottom to see each capability with live evidence."],
        "see": "A step timeline, an MCP request and response viewer and a compliance table.",
    },
    {
        "where": "Top right", "title": "Light and dark mode, then you are done",
        "intro": "Everything is designed to stay readable in both themes.",
        "do": ["Flip the Dark mode switch and pick a Color theme (Ocean, Indigo, Forest or Sand) at the top of the page, or in the sidebar.",
               "Reopen this tour any time with the Guided tour button.",
               "Questions or feedback? You can reach the team from the top bar, the Team section and the footer."],
        "see": "The same content in your chosen colours. Your choices are remembered if you refresh the page.",
    },
]


def progress_html(step: int) -> str:
    bars = "".join(f'<span class="{"done" if i <= step else ""}"></span>' for i in range(len(TOUR)))
    return f'<div class="vt-tour-progress" aria-label="Step {step + 1} of {len(TOUR)}">{bars}</div>'


def _credit_html() -> str:
    people = "".join(
        f'<div class="vt-tour-person"><div class="vt-avatar">{_esc(m["initials"])}</div><div><div class="nm">{_esc(m["name"])}</div>'
        f'<div class="rl">{_esc(m["role"])}</div></div></div>' for m in TEAM)
    return f'<div class="vt-tour-credit"><div class="lbl">Built by VeloTech Health AI</div><div class="ppl">{people}</div></div>'


def step_html(step: int) -> str:
    s = TOUR[step]
    do = "".join(f"<li><span>{_esc(d)}</span></li>" for d in s["do"])
    credit = _credit_html() if s.get("credit") else ""
    return _c(f"""
<div class="vt-tour-where">Step {step + 1} of {len(TOUR)} · {_esc(s['where'])}</div>
<div class="vt-tour-title">{_esc(s['title'])}</div>
<div class="vt-tour-intro">{_esc(s['intro'])}</div>
<ul class="vt-tour-do">{do}</ul>
<div class="vt-tour-see"><b>What you should see:</b> {_esc(s['see'])}</div>
{credit}
""")


def faq_item() -> tuple[str, str]:
    """First FAQ entry: the whole tour as a numbered checklist."""
    parts = []
    for i, s in enumerate(TOUR[1:], start=1):
        steps = "".join(f"<li>{_esc(d)}</li>" for d in s["do"])
        parts.append(f"<p><b>{i}. {_esc(s['title'])}</b> <i>({_esc(s['where'])})</i></p><ul>{steps}</ul><p>You should see: {_esc(s['see'])}</p>")
    return ("How do I test this prototype, step by step?", "".join(parts))
