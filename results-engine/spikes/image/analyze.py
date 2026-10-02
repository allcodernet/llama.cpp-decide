"""Spike (throwaway): tables for notes.md from runs/*.json. usage: uv run analyze.py [RUNS_DIR]"""
import json, pathlib, statistics, sys

RUNS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).parent / "runs")
TRUTH = {"cat03": True, "kitten": True, "coco_39769": True, "wildcat": True, "lynx": True, "fox": False,
         "labrador": False, "coco_776": False, "coco_139": False, "coco_632": False}


def load(name):
    return json.loads((RUNS / name).read_text())


def eng_probs(answers, field, fdef):
    a = answers[field]
    if fdef["type"] == "bool":
        return [a["p_true"], 1 - a["p_true"]]
    return [a["probabilities"][k] for k in fdef["options"]]


def reference_statistic(run):
    """per field: the reference check's statistic (median abs diff <= 0.01, no argmax disagreement where the
    reference's top-2 margin > 0.15), engine vs /completion replay"""
    schema = run["schema"]["fields"]
    out = {}
    for field, fdef in schema.items():
        diffs, agree, bad = [], 0, 0
        for row in run["rows"]:
            ref = next(v for k, v in row["reference"].items() if k.startswith(field + "/"))
            e, r = eng_probs(row["answers"], field, fdef), ref["probs"]
            diffs += [abs(x - y) for x, y in zip(e, r)]
            ea, ra = e.index(max(e)), r.index(max(r))
            agree += ea == ra
            top = sorted(r, reverse=True)
            bad += ea != ra and top[0] - top[1] > 0.15
        med = statistics.median(diffs)
        out[field] = {"median": med, "max": max(diffs), "argmax_agree": agree / len(run["rows"]), "bad": bad,
                      "pass": med <= 0.01 and bad == 0}
    return out


def main():
    single, single_d, multi = load("decide-single.json"), load("decide-single-imgdefault.json"), load("decide-multi.json")
    chat, chat_d = load("ref-chat-cat.json"), load("ref-chat-cat-imgdefault.json")
    chat_by = {r["image"]: r for r in chat["rows"]}
    chat_d_by = {r["image"]: r for r in chat_d["rows"]}
    multi_by = {r["image"]: r for r in multi["rows"]}
    sd_by = {r["image"]: r for r in single_d["rows"]}

    print("## p(cat) per image (image-min-tokens 1024 unless noted)\n")
    print("| image | truth | chat p_yes | decide p_true (cat only) | /completion replay | decide p_true (4 fields) | image cells / positions | "
          "chat p_yes, default tokens | decide p_true, default tokens | cells / positions, default |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for row in single["rows"]:
        n = row["image"]
        rep = row["reference"]["cat/" + next(k.split("/")[1] for k in row["reference"])]["probs"][0]
        sd = sd_by[n]
        print(f"| {n} | {'cat' if TRUTH[n] else 'no cat'} | {chat_by[n]['p_yes']:.4f} | {row['answers']['cat']['p_true']:.4f} | {rep:.4f} | "
              f"{multi_by[n]['answers']['cat']['p_true']:.4f} | {row['usage']['image_cells']} / {row['usage']['image_positions']} | "
              f"{chat_d_by[n]['p_yes']:.4f} | {sd['answers']['cat']['p_true']:.4f} | {sd['usage']['image_cells']} / {sd['usage']['image_positions']} |")

    print("\n## engine vs /completion replay (reference statistic)\n")
    print("| run | field | median abs diff | max abs diff | argmax agreement | disagreements over margin | pass |")
    print("|---|---|---|---|---|---|---|")
    for name, run in (("cat only, 1024", single), ("cat only, default tokens", single_d), ("4 fields, 1024", multi)):
        for f, s in reference_statistic(run).items():
            print(f"| {name} | {f} | {s['median']:.2g} | {s['max']:.4f} | {s['argmax_agree']:.2f} | {s['bad']} | {'PASS' if s['pass'] else 'FAIL'} |")

    print("\n## multi-field answers (4 fields, 1024)\n")
    print("| image | cat | dog | animal | indoors |")
    print("|---|---|---|---|---|")
    for row in multi["rows"]:
        a = row["answers"]
        print(f"| {row['image']} | {a['cat']['p_true']:.4f} | {a['dog']['p_true']:.4f} | {a['animal']['value']} ({a['animal']['confidence']:.3f}) | "
              f"{a['indoors']['p_true']:.4f} |")

    print("\n## timings per request with one image state (ms; medians over the 10 images)\n")
    print("| run | image cells (median) | encode | image decode | prefill (incl. both) | scoring | total |")
    print("|---|---|---|---|---|---|---|")
    for name, run in (("cat only, 1024", single), ("cat only, default tokens", single_d), ("4 fields, 1024", multi)):
        rows = run["rows"]
        med = lambda k: statistics.median(r["timings"][k] for r in rows)
        cells = statistics.median(r["usage"]["image_cells"] for r in rows)
        print(f"| {name} | {cells:.0f} | {med('image_encode_ms'):.0f} | {med('image_decode_ms'):.0f} | {med('prefill_ms'):.0f} | "
              f"{med('scoring_ms'):.1f} | {med('total_ms'):.0f} |")
    chat_wall = statistics.median(r["wall_s"] for r in chat["rows"]) * 1000
    chat_wall_d = statistics.median(r["wall_s"] for r in chat_d["rows"]) * 1000
    print(f"\nchat reference wall per image (one token, includes HTTP + base64): median {chat_wall:.0f} ms (1024), {chat_wall_d:.0f} ms (default)")

    b = load("decide-batch.json")
    t = b["batch_response"]["timings"]
    print(f"\n3 image states in one request (4 fields): total {t['total_ms']:.0f} ms, encode {t['image_encode_ms']:.0f}, image decode "
          f"{t['image_decode_ms']:.0f}, scoring {t['scoring_ms']:.0f}, decodes {t['decodes']}; singles total "
          f"{sum(s['timings']['total_ms'] for s in b['singles']):.0f} ms; batch vs single max abs diff "
          f"{', '.join(f'{d:.2g}' for d in b['batch_vs_single'])}")
    m = load("decide-mixed.json")
    print(f"mixed request (text + image state): text state vs the same text alone max abs diff {m['text_state_diff']:.3g}")
    r = load("decide-repeat.json")
    print("repeat cat03 x5 p_true: " + ", ".join(f"{x['answers']['cat']['p_true']:.6f}" for x in r["rows"]) +
          "; total ms: " + ", ".join(f"{x['timings']['total_ms']:.0f}" for x in r["rows"]))


if __name__ == "__main__":
    main()
