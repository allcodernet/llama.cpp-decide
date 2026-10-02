# /// script
# dependencies = ["httpx"]
# ///
"""Spike (throwaway): ask the stock /v1/chat/completions a yes/no question per image, thinking off, and read
the probability of the first answer token from top_logprobs. p_yes = P(Yes) / (P(Yes) + P(No)) over the case
variants of the two words among the top 20 first tokens.

usage: reference_chat.py IMAGE_DIR OUT_JSON [question]
"""
import base64, json, math, pathlib, sys, time

import httpx

URL = "http://127.0.0.1:8097"
IMAGES = ["cat03", "kitten", "coco_39769", "wildcat", "lynx", "fox", "labrador", "coco_776", "coco_139", "coco_632"]
QUESTION = "Is there a cat in this picture? Answer with yes or no."


def ask(client, path, question):
    b64 = base64.b64encode(path.read_bytes()).decode()
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": question}]}],
        "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = client.post(f"{URL}/v1/chat/completions", json=body)
    r.raise_for_status()
    j = r.json()
    top = j["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
    no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
    return {
        "image": path.stem, "token": j["choices"][0]["message"]["content"], "p_yes_raw": yes, "p_no_raw": no,
        "p_yes": yes / (yes + no) if yes + no > 0 else None, "prompt_tokens": j["usage"]["prompt_tokens"],
        "timings": j.get("timings"), "wall_s": time.time() - t0, "top5": top[:5],
    }


def main():
    img_dir, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    question = sys.argv[3] if len(sys.argv) > 3 else QUESTION
    with httpx.Client(timeout=600) as client:
        rows = [ask(client, img_dir / f"{name}.jpg", question) for name in IMAGES]
    for r in rows:
        print(f"{r['image']:12s} {r['token']!r:8s} p_yes={r['p_yes']:.6f} (raw {r['p_yes_raw']:.4f}/{r['p_no_raw']:.4f}) "
              f"prompt_tokens={r['prompt_tokens']} wall={r['wall_s']:.2f}s")
    out.write_text(json.dumps({"question": question, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
