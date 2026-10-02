# Third-party material

This repository's own code, documents and data are under the MIT licence in `LICENSE`. It builds on, quotes or
refers to the following.

## llama.cpp (the engine patches)

`patches/decide/*.patch` is a patch series against upstream llama.cpp, https://github.com/ggml-org/llama.cpp, commit
`60b06ab9a9eeec26f8125c9316ccbf4ee4713d1f`. llama.cpp is MIT-licensed, Copyright (c) 2023-2026 The ggml authors. The
patches modify and add files of that tree (mostly under `tools/decide/`, plus `common/`, `tools/server/` and
`tests/`); the patched tree stays under llama.cpp's licence. `build-engine.sh` clones upstream at that commit and
applies the series with `git am`; no llama.cpp source is stored in this repository apart from the context lines of the
patches. The patched tree vendors stb_image (public domain / MIT, by Sean Barrett) through upstream's `vendor/`
directory, which this work uses to decode images.

## Baseline (Phase 1)

The first phase of this work measured a third-party llama.cpp fork with a `POST /v1/decision` endpoint. The fork is
not part of this repository and no code or text of it is included; only this project's own measurements of it are
kept (under `results/`, and as the "Baseline" rows of `REPORT-ENGINE.md`). The engine here was written separately.

## TypeSafe AI's System One API (Jev)

`POST /v1/systemone` implements the request and response shape of TypeSafe AI's System One API (product name Jev),
from its public documentation, so that existing clients can talk to a local server. Jev, System One and TypeSafe are
names of TypeSafe AI; this project is not affiliated with TypeSafe AI. The official Python SDK,
https://github.com/typesafe-ai/typesafe-sdk-python (MIT; tested at v0.7.0, commit `2ce5c65`), is used only by the
optional acceptance test `tests/test_systemone_sdk.py`; it is not a dependency and not included.

## COCO images and annotations

No image is stored in this repository. The image checks download four COCO 2017 validation images and the
evaluation downloads 200 more into the git-ignored `local/` directory. COCO annotations are CC BY 4.0 (COCO
Consortium, https://cocodataset.org); each image keeps the licence of its Flickr source. The credits and licences are
in `results-engine/image/NOTICE.md` (the four integrity images) and `results-engine/image/coco/NOTICE.md` (the
evaluation subset, `subset.jsonl`, which is derived from the annotations).

## Models

No model weights are included. The recorded results name the GGUF files they used (Qwen2.5, Qwen3, Qwen3.5,
Qwen3.6, Qwen3.8, Gemma 3, Gemma 4 and SmolVLM builds from Hugging Face); each model keeps its own licence.

## Data

`data/` holds a synthetic support-ticket set made for this line of work (`data/README.md`); it is released under this
repository's licence.
