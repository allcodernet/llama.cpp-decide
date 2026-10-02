"""The fit check (run once while the presets were set up): does every integrity reference tag fit a vision preset's decide-seqs? Runs llama-decide --info (the build
in ENGINE_BIN, default engine/build/bin; the schema check needs no projector) with each tag's schema and options; a tag fits when
schema.states_per_round >= 1.
Usage: uv run results-engine/image/fit_check.py PRESET -> one line per tag and results-engine/image/task-0/fit-PRESET.json
({tag: states_per_round}); exit 1 when a tag does not fit.
       uv run results-engine/image/fit_check.py PRESET --unavailable REASON -> fit-PRESET.json {"unavailable": REASON} without
loading anything (a failed download or load; Task 6 turns it into the preset's SKIPPED.txt)."""
import copy
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import reference_check as rc  # noqa: E402
from decide_client import _preset_settings  # noqa: E402

# the integrity reference tags (checks.py REF_TAGS, run.sh ref_args): schema file, options, `after`
TAGS = {"tree-one": ("schema-one.json", {}, {}), "tree-multi": ("schema-multi.json", {}, {}),
        "fp": ("schema-multi.json", {"scoring": "full_path"}, {}), "after": ("schema-multi.json", {}, {"dog": ["cat"]}),
        "debias2": ("schema-multi.json", {"order_debias": 2}, {}), "t0.5": ("schema-multi.json", {"temperature": 0.5}, {})}


def body(tag):
    name, options, after = TAGS[tag]
    schema = json.loads((HERE / name).read_text())
    fields = copy.deepcopy(schema["fields"])
    for f, parents in after.items():
        fields[f]["after"] = parents
    return {"instructions": schema["instructions"], "fields": fields, "states": ["x"], "options": options}


def main():
    preset = sys.argv[1]
    out = HERE / "task-0" / f"fit-{preset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) == 4 and sys.argv[2] == "--unavailable":
        out.write_text(json.dumps({"unavailable": sys.argv[3]}, indent=1) + "\n")
        print(f"{preset}: recorded as unavailable: {sys.argv[3]}")
        return
    s = _preset_settings(preset)
    cmd = rc.decide_command(s["model"], preset, int(s["decide-seqs"]), dump=False) + ["--info"]
    proc = subprocess.run(cmd, input="".join(json.dumps(body(t)) + "\n" for t in TAGS), capture_output=True, text=True)
    lines = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith("{")]
    if proc.returncode != 0 or len(lines) != len(TAGS):
        sys.exit(f"llama-decide --info exit {proc.returncode}, {len(lines)} lines:\n{proc.stderr[-2000:]}")
    fit = {}
    for t, info in zip(TAGS, lines):
        sch = info.get("schema", {})
        fit[t] = sch.get("states_per_round", 0)
        print(f"{preset} {t}: decide_seqs {info.get('decide_seqs')}, seqs_per_state {sch.get('seqs_per_state')}, "
              f"states_per_round {fit[t]}")
    out.write_text(json.dumps(fit, indent=1) + "\n")
    sys.exit(0 if all(v >= 1 for v in fit.values()) else 1)


if __name__ == "__main__":
    main()
