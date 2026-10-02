# Task 0: test-machine preparation (2026-10-02)

- Models: models/SmolVLM-500M-Instruct-Q8_0.gguf: OK
- Models: models/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf: OK
- Models: models/gemma-3-4b-it-Q4_K_M.gguf: OK
- Models: models/mmproj-gemma-3-4b-it-f16.gguf: OK
- Models: models/gemma-4-E4B-it-Q4_0.gguf: OK (already present; same LFS SHA-256 as ggml-org/gemma-4-E4B-it-GGUF)
- Models: models/mmproj-gemma-4-E4B-it-BF16.gguf: OK (ggml-org/gemma-4-E4B-it-GGUF, which offers BF16 and Q8_0 projectors)
- Integrity images: 4 COCO JPEGs OK (integrity-images.sha256); coco-000000039769.png SHA-256
  cf6f3c4befa148732c7453e0de5afab00f682427435fead2d88b07a9615cdac2 (Pillow 12.3.0)
- Old build: engine-49785a95d at 49785a95d63510972b4ff0d42dbbe80052cb2fea, tree 32d63de3ad1b00c086fb4abd2b9da53631d8d043,
  targets llama-server llama-decide-cli, same cmake flags as build-engine.sh
- Presets: qwen3.8-27b-vision, smolvlm-500m, gemma-3-4b-vision, qwen3.6-35b-a3b-vision appended; the existing presets are
  unchanged (git diff shows additions only)
- Preset gemma-4-e4b-vision appended (a decision made during the work): substitute for SmolVLM as live coverage of the family
  "sequential positions, causal attention" (spec 5.4); non-binding for the statistic. Engine source: the projector's
  `clip.vision.projector_type` is `gemma4v`; `mtmd_decode_use_non_causal` (tools/mtmd/mtmd.cpp:2170) returns false for
  GEMMA4V when the text model's `n_embd_inp` is 2560 (E4B; the GGUF has `gemma4.embedding_length` 2560, no deepstack), so
  images decode causally; `llama_model_rope_type` gives NEOX for LLM_ARCH_GEMMA4, which mtmd.cpp:550-566 maps to
  MTMD_POS_TYPE_NORMAL (sequential positions, no M-RoPE). The projector file also holds an audio projector (`gemma4a`).
- Fit of the Task 6 reference tags (task-0/fit-<preset>.json): qwen3.8-27b-vision: all six fit at decide-seqs 16
- Fit of the Task 6 reference tags (task-0/fit-<preset>.json): smolvlm-500m: unavailable, does not load: the engine's
  join check rejects its tokenizer ("tokenizer does not tokenise the trunk/branch join independently (add_space_prefix?);
  model not supported: llama 8B Q8_0, vocab type BPE: prefix/state join "...end_of_utterance>\nUser: " + "Hi" tokenises
  jointly as [21041], separately as [216 26843] (from token 8, left part has 9)"); Task 6 reports it as skipped (spec 8)
- Fit of the Task 6 reference tags (task-0/fit-<preset>.json): gemma-3-4b-vision: all six fit at decide-seqs 16
- Fit of the Task 6 reference tags (task-0/fit-<preset>.json): qwen3.6-35b-a3b-vision: all six fit at decide-seqs 16
- Fit of the Task 6 reference tags (task-0/fit-<preset>.json): gemma-4-e4b-vision: all six fit at decide-seqs 16
