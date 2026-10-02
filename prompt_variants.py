"""Catalogue/instruction texts for the prompt-variant experiment (decide_client.py --prompt-variant NAME).

Each variant is the request's top-level `instructions` plus a description mode for the fields:
"full" = decide_client's field descriptions (one line per option / level in the engine's catalogue),
"short" = one-word descriptions, bare queue options and one-word urgency levels.
The engine always renders its own catalogue block from the fields after the instructions.
"""

DEFAULT = "Answer each question about this support ticket from its text."

MERGED = """Answer each question about this support ticket from its text.

queue (which team must act first): billing = Payments, refunds, invoices, charges; technical = Bugs, crashes, outages, errors, performance, login problems; sales = Contracts, plan changes, pricing questions, cancellation threats, procurement; feedback = Praise, feature requests, general non-problem questions
urgency (how urgent): 0 = Can wait: praise, feature requests, idle questions, no problem; 1 = Normal: a real problem for one user without time pressure; 2 = Should be handled today: user is blocked, money wrongly taken, a deadline within days, or repeated contact; 3 = Drop everything: outage affecting many users, security incident, data loss, or enterprise customer about to churn
angry: true if the text itself shows anger (hostile wording, shouting, threats, sarcasm aimed at the company)"""

FRAMING = ("You are triaging customer support tickets for a software company. "
           "Read the ticket and answer each question from the text alone.")

KEYWORDS = """Answer each question about this support ticket from its text.

Queue cues:
- billing: payment, refund, invoice, charge, billed
- technical: bug, crash, outage, error, slow, login
- sales: contract, plan change, pricing, cancel, procurement
- feedback: praise, thanks, feature request, suggestion, general question without a problem
Urgency rules:
- 0: nothing is wrong (praise, a feature request, an idle question).
- 1: one user has a real problem, with no time pressure.
- 2: the user is blocked, money was wrongly taken, a deadline is days away, or they are writing again.
- 3: an outage for many users, a security incident, data loss, or an enterprise customer about to leave.
Anger: true only when the wording itself is angry: hostile words, shouting (ALL CAPS, !!!), threats, or sarcasm aimed at the company."""

VARIANTS = {
    "default": {"instructions": DEFAULT, "description_mode": "full"},
    "merged": {"instructions": MERGED, "description_mode": "short"},
    "framing": {"instructions": FRAMING, "description_mode": "full"},
    "keywords": {"instructions": KEYWORDS, "description_mode": "full"},
}
