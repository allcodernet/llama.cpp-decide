# Verification of the public repository, as an outside user would run it (2026-10-02)

A single-commit copy of this repository (commit `ce1c5212a7aa`, 1134 files, made from the development commit `a09f2e5`)
was copied with `tar` over ssh into a fresh directory, `public-verify/`, on the test machine (`user@gpu-host`: RTX 3090,
driver 610.57.04, CUDA 13.3 in `/opt/cuda`, `g++-15` 15.3.0, CMake 4.4.2, uv 0.12.5). Every command below ran inside
that copy, following the README's quick start literally; nothing was read from the development checkout. The engine
was cloned from GitHub (`LOCAL_SEED` unset). The only README step that was not needed was editing the paths of the
`[qwen3.8-27b-vision]` preset: on this machine the model and projector are at the author's paths already written there.

## Build, tree, tests

```
$ ./build-engine.sh                       # LOCAL_SEED=unset; 4 min 28 s wall time
Cloning into 'engine'...
[100%] Built target test-decide-image
build ok: b26bf28d0 on decide, tree c3daa53ab47099be8a69e6d81749f52072768fa7
(exit 0)
$ git -C engine remote -v | head -1
origin	https://github.com/ggml-org/llama.cpp (fetch)
$ git -C engine log --format='%an <%ae> | %cn <%ce>' -1
allcodernet <67543055+allcodernet@users.noreply.github.com> | allcodernet <67543055+allcodernet@users.noreply.github.com>
```

Compiler warnings: none in `tools/decide/` or the decide tests; 47 in upstream files that the patches do not touch
(`tools/mtmd/clip-graph.h` and `qwen3vl.cpp`: bitwise operation between different enumeration types;
`tools/server/server-tools.cpp`: one `std::filesystem` warning).

```
$ git -C engine rev-parse 'HEAD^{tree}'
c3daa53ab47099be8a69e6d81749f52072768fa7
$ ./test-engine.sh
100% tests passed out of 11
$ uv run pytest -q
165 passed, 1 skipped in 1.14s
$ ls engine/build/bin | grep -E '^(llama-server|llama-decide)$'
llama-decide
llama-server
```

## A text request (Qwen2.5-0.5B)

GPU before: 612 MiB in use (the desktop), no `llama-server` running, nothing listening on port 8097.

```
$ mkdir -p models
$ curl -L -o models/Qwen2.5-0.5B-Instruct-Q8_0.gguf https://huggingface.co/lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q8_0.gguf
(exit 0) 531068224 bytes
$ ENGINE_PRESETS=models-engine-3090.ini ./serve-engine.sh      # in the background
$ curl -s http://127.0.0.1:8097/v1/decide -H 'Content-Type: application/json' -d '{ ... the README request ... }'
{"object":"decide","results":[{"answers":{"queue":{"value":"technical","probabilities":{"billing":0.1847517096143914,"technical":0.8152482903856085},"confidence":0.8152482903856085},"angry":{"value":false,"p_true":0.15593135078378675,"confidence":0.8440686492162133}},"usage":{"state_tokens":11,"scored_tokens":2}}],"usage":{"prompt_tokens":111,"cached_tokens":0,"state_tokens":11,"scored_tokens":2},"timings":{"prefill_ms":19.665,"prefix_ms":15.004,"scoring_ms":4.764,"total_ms":28.246,"rounds":1,"decodes":3,"retries":0},"engine":{"scoring":"tree","order_debias":1,"temperature":1.0,"phases":1,"seqs_per_state":3,"states_per_round":6,"quote_split_fields":[],"prefix_cache":{"capacity":8,"entries":0,"hits":0,"misses":1}}}
```

The 0.5B gets this billing ticket wrong; the request only shows the mechanics (the README says so).

## The bundled data and the evaluation scripts

```
$ ENGINE_PRESETS=models-engine-3090.ini uv run decide_client.py --model qwen2.5-0.5b --batch 5 --out-dir /tmp/pv-qs
 80/80     24.4 ms  rounds=1
assistant_prefix tail: '<|im_end|>\n<|im_start|>assistant\n'
wrote /tmp/pv-qs/preds/decide-qwen2.5-0.5b-b5-keywords.jsonl and /tmp/pv-qs/runs/decide-qwen2.5-0.5b-b5-keywords.json
$ uv run score.py --preds-dir /tmp/pv-qs/preds
run                                 queue    urg  urg±1  angry  ECE(q) Brier(a)  NLL(q)  p50 ms
-----------------------------------------------------------------------------------------------
decide-qwen2.5-0.5b-b5-keywords     33.8%  32.5%  67.5%  75.0%   0.232    0.150    2.17       4
```

## An image request (Qwen3.8-27B with its projector, preset `qwen3.8-27b-vision`)

```
$ curl -L -o cats.jpg http://images.cocodataset.org/val2017/000000039769.jpg
(exit 0)
$ uv run python - <<'EOF' ... the README snippet ... EOF
{'answers': {'cat': {'value': True, 'p_true': 0.9997993661457202, 'confidence': 0.9997993661457202}}, 'usage': {'state_tokens': 7, 'scored_tokens': 1, 'images': 1, 'image_cells': 1038, 'image_positions': 39}}
```

GPU with the server: 16440 MiB. The server was stopped by the PID it was started with; GPU after: 612 MiB, no
`llama-server` running.

## Result

Every step of the quick start worked as written; no README defect was found. The repository's Python tests give 165
passed, 1 skipped (the skipped one is the optional System One SDK acceptance test, skipped without `SYSTEMONE_URL`).

The regression and the server smoke under `publish/recheck/` (the smoke JSON does not name its build) ran on the
development build `d5cdf773a` of the same tree, `c3daa53ab47099be8a69e6d81749f52072768fa7`.
