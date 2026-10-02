"""Control for the round case of checks.py swa (Task 6b diagnostic): the round-1 pair of the distinct texts against each text
alone, on a preset of choice and at sizes of choice, to tell a batching effect from an effect of the full SWA cache.
Outputs go to results-engine/image/swa/control/. A diagnosis, not a check: it records and exits 0 when every request
answered 200, else 1.

  pair --url U --model P --sizes N1,N2 --repeat K --out F.json
      the texts of checks.py round_cases "distinct" states 0 and 1 (ROUND_WORDS[0], ROUND_WORDS[1]) sized to N1 and N2 tokens;
      K times: both texts in one request (the pair), then each text alone; per repeat the statistic (answers_match) of the
      pair against alone, per state and both states; with K > 1 also every repeat's pair and alone answers against repeat 0
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from checks import ROUND_WORDS, SWA_SCHEMA, answers_match, info, post, sized_text, swa_body  # noqa: E402

FIELDS = list(SWA_SCHEMA["fields"])


def probs(results):
    return [r["answers"]["animal"]["probabilities"] for r in results]


def cmd_pair(args):
    client = httpx.Client(base_url=args.url, timeout=900)
    sizes = [int(x) for x in args.sizes.split(",")]
    texts = [sized_text(client, args.model, n, ROUND_WORDS[i]) for i, n in enumerate(sizes)]
    i = info(client, args.model)
    out = {"model": args.model, "sizes": sizes, "text_sha256": [hashlib.sha256(t.encode()).hexdigest() for t in texts],
           "info": {k: i.get(k) for k in ("n_ctx", "decide_seqs", "memory")}, "repeats": []}
    ok = True
    for _ in range(args.repeat):
        pair = post(client, "/v1/decide", swa_body(args.model, texts, 1))
        alone = [post(client, "/v1/decide", swa_body(args.model, [t], 1)) for t in texts]
        rep = {"pair": {"status": pair["status"], "timings": pair["response"].get("timings"), "results": pair["response"].get("results")},
               "alone": [{"status": a["status"], "results": a["response"].get("results")} for a in alone]}
        ok = ok and pair["status"] == 200 and all(a["status"] == 200 for a in alone)
        if ok:
            pr, ar = pair["response"]["results"], [a["response"]["results"][0] for a in alone]
            rep["pair_probs"], rep["alone_probs"] = probs(pr), probs(ar)
            rep["pair_vs_alone"] = answers_match(ar, pr, FIELDS)
            rep["per_state"] = [answers_match([ar[k]], [pr[k]], FIELDS) for k in range(len(texts))]
        out["repeats"].append(rep)
    if ok and args.repeat > 1:
        r0 = out["repeats"][0]
        out["repeat_vs_0"] = [{"pair": answers_match(r0["pair"]["results"], r["pair"]["results"], FIELDS),
                               "alone": answers_match([a["results"][0] for a in r0["alone"]], [a["results"][0] for a in r["alone"]], FIELDS)}
                              for r in out["repeats"][1:]]
    out["all_200"] = ok
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    summary = [{"pair": r.get("pair_probs"), "alone": r.get("alone_probs"),
                "per_state_max_abs_diff": [s["max_abs_diff"] for s in r.get("per_state", [])]} for r in out["repeats"]]
    print(json.dumps({"sizes": sizes, "summary": summary,
                      "repeat_vs_0_max": [(x["pair"]["max_abs_diff"], x["alone"]["max_abs_diff"]) for x in out.get("repeat_vs_0", [])]},
                     indent=1))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--sizes", required=True, help="N1,N2: the token counts of the two texts")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_pair)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
