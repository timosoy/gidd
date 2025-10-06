
import os
import re
import json
import argparse
from typing import List, Dict, Any
import requests
from tqdm import tqdm

def _normalize_host(h: str) -> str:
    h = (h or "").strip()
    if not h:
        return "http://localhost:11434"
    # 0.0.0.0 is a bind address; replace with a connectable host
    if h.startswith("0.0.0.0"):
        h = h.replace("0.0.0.0", "localhost", 1)
    # Prepend scheme if missing
    if not (h.startswith("http://") or h.startswith("https://")):
        h = "http://" + h
    return h.rstrip("/")

OLLAMA_HOST = _normalize_host(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

PROMPT_TEMPLATE = """1. Clarity and coherence: Keeping in mind that the text may be cut off in the beginning
and at the end due to it being an excerpt, how clear and understandable is the text?
2. Grammaticality: Are there any grammatical errors in the text?
3. Factuality: If applicable, is the factually verifiable information stated in the text
(e.g. facts about geography, history, etc.) accurate and reliable?
4. Writing style: How well is the text written in terms of style and fluency? Do the
sentences flow well, is the vocabulary appropriate?
5. Creativity: How original and creative is the text?
For each category, give a short justification before providing the final score. Your
answer should be following the JSON format, with one top-level key for each aspect (‘
clarity‘, ‘grammaticality‘, ‘factuality‘, ‘style‘, and ‘creativity‘).
Each aspect, in turn, should be a JSON object consisting of a ‘reasoning‘ and ‘score‘ key
in that order. The ‘reasoning‘ key should contain a short justification for the score,
and the ‘score‘ key should contain the score itself.
Please keep the following in mind:
- Give your justification first before deciding on a final score.
- Only output the JSON containing the justifications and scores and nothing else.
- Keep in mind that the presented paragraph may be an excerpt from a longer document, so
it may not be fully self-contained. Do not deduct points for issues arising from this.
The text to be graded is as follows:
‘‘‘
{text}
‘‘‘
"""

SYSTEM_HINT = (
    "You are a strict grader."
    "Return ONLY valid JSON. Do not include markdown fences or extra text."
)

def load_samples_from_file(filename: str) -> List[str]:
    """
    Load samples from a text file that may contain either 'Sample N:' or
    'Corrected Sample N:' section headers.
    """
    samples: List[str] = []
    current_lines: List[str] = []

    def flush_current() -> None:
        if current_lines:
            samples.append("".join(current_lines).strip())
            current_lines.clear()

    header_prefixes = ("Sample ", "Corrected Sample ")
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # Start of a new sample block -> flush previous
            if line.startswith(header_prefixes):
                flush_current()
                continue  # skip header line itself
            # Accumulate non-empty content lines
            if line.strip():
                current_lines.append(line)

    flush_current()
    return samples

def extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    s = text.strip()
    if "{" in s and "}" in s:
        s = s[s.index("{"): s.rindex("}") + 1]
        try:
            return json.loads(s)
        except Exception:
            pass
    return {}

def call_ollama_chat(model: str, user_prompt: str, debug: bool = False) -> str:
    """
    Call Ollama /api/chat with a system + user message.
    We set format='json' to ask for strict JSON output.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    base_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_HINT},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.0, "top_p": 1.0},
    }

    # Attempt 1: strict JSON formatting
    try:
        payload = dict(base_payload)
        payload["format"] = "json"
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")
    except requests.HTTPError as e:
        if debug:
            try:
                dbg = r.text if 'r' in locals() else str(e)
            except Exception:
                dbg = str(e)
            print(f"[chat strict-json HTTPError] {e}; body={dbg}")
    except requests.RequestException as e:
        if debug:
            print(f"[chat strict-json RequestException] {e}")

    # Attempt 2: without JSON formatting constraint
    try:
        payload = dict(base_payload)
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")
    except requests.HTTPError as e:
        if debug:
            try:
                dbg = r.text if 'r' in locals() else str(e)
            except Exception:
                dbg = str(e)
            print(f"[chat fallback HTTPError] {e}; body={dbg}")
    except requests.RequestException as e:
        if debug:
            print(f"[chat fallback RequestException] {e}")

    return ""

def summarize(json_list: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = ["clarity", "grammaticality", "factuality", "style", "creativity"]
    sums = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}
    for obj in json_list:
        for k in keys:
            v = obj.get(k, {})
            if isinstance(v, dict) and "score" in v:
                try:
                    s = float(v["score"])
                    sums[k] += s
                    counts[k] += 1
                except Exception:
                    pass
    return {k: (sums[k]/counts[k] if counts[k] > 0 else float("nan")) for k in keys}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama model name, e.g. gemma3:12b-instruct")
    ap.add_argument("--input", required=True, help="Samples/generated_samples.txt or corrected file")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--summary_out", default="", help="Summary JSON path; default alongside out")
    ap.add_argument("--debug", action="store_true", help="Print HTTP errors and fallback attempts")
    args = ap.parse_args()

    texts = load_samples_from_file(args.input)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary_path = args.summary_out or (os.path.splitext(args.out)[0] + "_summary.json")

    all_json = []
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, t in enumerate(tqdm(texts, desc="Grading via Ollama")):
            prompt = PROMPT_TEMPLATE.format(text=t)
            try:
                raw = call_ollama_chat(args.model, prompt, debug=args.debug)
                js = extract_json(raw) if raw else {}
            except requests.HTTPError as e:
                if args.debug:
                    print(f"[main HTTPError] {e}")
                js = {}
            except requests.RequestException as e:
                if args.debug:
                    print(f"[main RequestException] {e}")
                js = {}

            rec = {"index": i, "input_file": args.input, "result": js}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            all_json.append(js)

    means = summarize(all_json)
    with open(summary_path, "w", encoding="utf-8") as fsum:
        json.dump({
            "input_file": args.input,
            "model": args.model,
            "count": len(all_json),
            "mean_scores": means
        }, fsum, ensure_ascii=False, indent=2)

    print(f"\nSaved details to: {args.out}")
    print(f"Saved summary to: {summary_path}")

if __name__ == "__main__":
    main()
