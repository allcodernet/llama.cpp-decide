# Support-ticket triage set (synthetic)

`triage_test.jsonl` (80 tickets) and `triage_train.jsonl` (320 tickets): the labelled data behind every triage number in
this repository. Each line is one ticket:

```json
{"text": "Does the annual receipt break out the sales tax separately? ...", "queue": "billing", "urgency": 0, "angry": false}
```

## Provenance

- **Synthetic.** The tickets are not collected from real users or customers and contain no personal data. They were
  written and labelled by a large language model on 2026-09-18, for an earlier experiment of the same author on
  one-pass decision models (logit readout, LoRA and encoder baselines; that experiment is not part of this
  repository). The generator was told to avoid templates, to vary product type, length, tone and writing quality,
  and to include about 15% hard cases (mixed topics, misleading keywords, polite but critical, angry but trivial).
- **Review.** A second model pass inspected the labels and corrected 17 records (ambiguous labels, test tickets too
  close to training scenarios). Checked afterwards: every line parses, every value is valid, and no text occurs in
  both files.
- **Unchanged since.** The files here are byte-identical to the ones every recorded run used (SHA-256 below).

## Labels

The label definitions were given to the generator; the same texts are the option descriptions of the requests
(`decide_client.py`, and the same texts in the baseline runs under `results/`).

| field | values | definition |
|---|---|---|
| `queue` | `billing` | Payments, refunds, invoices, charges |
| | `technical` | Bugs, crashes, outages, errors, performance, login problems |
| | `sales` | Contracts, plan changes, pricing questions, cancellation threats, procurement |
| | `feedback` | Praise, feature requests, general non-problem questions |
| `urgency` | `0` | Can wait: praise, feature requests, idle questions, no problem |
| | `1` | Normal: a real problem for one user without time pressure |
| | `2` | Should be handled today: user is blocked, money wrongly taken, a deadline within days, or repeated contact |
| | `3` | Drop everything: outage affecting many users, security incident, data loss, or enterprise customer about to churn |
| `angry` | `true` / `false` | The text itself shows anger (hostile wording, shouting, threats, sarcasm aimed at the company) |

| file | tickets | queue (each) | urgency 0 / 1 / 2 / 3 | angry |
|---|---|---|---|---|
| `triage_test.jsonl` | 80 | 20 | 24 / 19 / 23 / 14 | 23 |
| `triage_train.jsonl` | 320 | 80 | 94 / 79 / 93 / 54 | 86 |

## Caveats

- Train and test come from the same generator, so results on them are probably optimistic for real tickets.
- Known quirks the generator reported: no `feedback` ticket has urgency 3 (impossible by definition); mass wrong
  charges affecting many customers are labelled urgency 3, although the definition does not list them; the
  `sales` + urgency 3 tickets (14 train, 4 test) share a similar pattern.
- 80 test tickets: 2.5 percentage points are 2 tickets.

## Use

Every script reads this directory by default; `TRIAGE_DATA=DIR` points them at another directory with the same two
file names (relative paths from the project directory). The files are released with the repository under its
licence (`LICENSE`).

```
6d18a4d39c3afdf845dc0a6d00194cf624238295545243a07c296b491b712b51  triage_test.jsonl
e63a75bc10fa9f246c088074b9b93c737271901f6f9e025722742af2d00c3e65  triage_train.jsonl
```
