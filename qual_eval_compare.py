import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple
import random
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

# System hint to force strict JSON
SYSTEM_HINT = (
    "You are a strict grader. Return ONLY valid JSON. Do not include markdown fences or extra text."
)

# Pairwise comparison prompt (updated per request)
COMPARISON_PROMPT_TEMPLATE = """Compare two texts on five aspects. Judge each aspect **in isolation** (a decision for one aspect must not affect any other aspect). For **each aspect**, provide:

* a brief **reasoning**; and
* a single **winner** (“A” or “B”). **No ties**.

Aspects to judge:

* Clarity and coherence — Each text may be cut at the beginning or end because it is an excerpt. Do not deduct points for truncation alone.
* Grammaticality — Are there grammatical errors?
* Factuality — If applicable, is the verifiable information accurate and reliable?
* Writing style and fluency — Do sentences flow well? Is the vocabulary appropriate?
* Creativity — How original and inventive is the text?

**Factuality special rule**

* If a text has no verifiable factual content, mark its factuality as “not applicable” in your reasoning. You must still select a winner for factuality **unless both texts are not applicable**.
* If **both** texts are not applicable for factuality, set the factuality **winner** to **"N/A"**.
* When exactly one text is not applicable, prefer the other text **only if** its factual statements are not incorrect or unsupported; otherwise choose the text with fewer incorrect or unsupported claims.

**Tie-break guidance (when both seem equally strong for a non-factuality aspect)**
Use these checks in order until one text is better:

1. Fewer and milder issues for that aspect.
2. Fewer issues per 100 words (length-normalised).
3. Clearer structure/flow or more precise choices for that aspect.

**Output format**
Return **valid JSON only**, no extra text. Use exactly this schema:

{{
"clarity":        {{ "reasoning": "...", "winner": "A" | "B" }},
"grammaticality": {{ "reasoning": "...", "winner": "A" | "B" }},
"factuality":     {{ "reasoning": "state N/A if no verifiable facts", "winner": "A" | "B" | "N/A" }},
"style":          {{ "reasoning": "...", "winner": "A" | "B" }},
"creativity":     {{ "reasoning": "...", "winner": "A" | "B" }}
}}

Texts to judge:

Text A:
'''
{text_A}
'''

Text B:
'''
{text_B}
'''
"""


def load_samples_from_file(filename: str) -> List[str]:
    """Load samples from a text file that may contain either 'Sample N:' or 'Corrected Sample N:' headers."""
    samples: List[str] = []
    current_lines: List[str] = []

    def flush_current() -> None:
        if current_lines:
            samples.append("".join(current_lines).strip())
            current_lines.clear()

    header_prefixes = ("Sample ", "Corrected Sample ")
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(header_prefixes):
                flush_current()
                continue
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
    """Call Ollama /api/chat with a system + user message. Try strict JSON, then fallback."""
    url = f"{OLLAMA_HOST}/api/chat"
    base_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_HINT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "top_p": 1.0},
    }

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


def _aligned_pairs(texts_a: List[str], texts_b: List[str], num_pairs: int) -> List[Tuple[int, int]]:
    na, nb = len(texts_a), len(texts_b)
    n = min(num_pairs, na, nb)
    return [(i, i) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama model name, e.g. gemma3:12b-instruct")
    ap.add_argument("--input_a", required=True, help="File for Text A candidates (e.g., generated_samples.txt)")
    ap.add_argument("--input_b", required=True, help="File for Text B candidates (e.g., corrected_samples.txt)")
    ap.add_argument("--pairs", type=int, default=-1, help="Number of pairs to compare (default=min(len(A),len(B)))")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for A/B position shuffling")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--summary_out", default="", help="Summary JSON path; default alongside out")
    ap.add_argument("--debug", action="store_true", help="Print HTTP errors and fallback attempts")
    args = ap.parse_args()

    texts_a = load_samples_from_file(args.input_a)
    texts_b = load_samples_from_file(args.input_b)
    if not texts_a or not texts_b:
        raise SystemExit("Input A or B is empty. Check files.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary_path = args.summary_out or (os.path.splitext(args.out)[0] + "_summary.json")

    # Determine number of pairs (strictly aligned by index)
    max_pairs = min(len(texts_a), len(texts_b))
    num_pairs = max_pairs if args.pairs is None or args.pairs < 0 else min(args.pairs, max_pairs)
    pairs = _aligned_pairs(texts_a, texts_b, num_pairs=num_pairs)

    rng = random.Random(args.seed)

    all_json: List[Dict[str, Any]] = []
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, (ia, ib) in enumerate(tqdm(pairs, desc="Pairwise grading via Ollama")):
            # Randomize which source appears as Text A vs Text B
            if rng.random() < 0.5:
                text_A, text_B = texts_a[ia], texts_b[ib]
                mapping = {
                    "text_A": {"source": "file_a", "index": ia},
                    "text_B": {"source": "file_b", "index": ib},
                }
            else:
                text_A, text_B = texts_b[ib], texts_a[ia]
                mapping = {
                    "text_A": {"source": "file_b", "index": ib},
                    "text_B": {"source": "file_a", "index": ia},
                }

            prompt = COMPARISON_PROMPT_TEMPLATE.format(text_A=text_A, text_B=text_B)
            try:
                raw = call_ollama_chat(args.model, prompt, debug=args.debug)
                js = extract_json(raw) if raw else {}
            except requests.HTTPError as e:
                if args.debug:
                    print(f"[pairwise HTTPError] {e}")
                js = {}
            except requests.RequestException as e:
                if args.debug:
                    print(f"[pairwise RequestException] {e}")
                js = {}

            rec = {
                "pair_index": i,
                "input_file_a": args.input_a,
                "input_file_b": args.input_b,
                "index_a": ia,
                "index_b": ib,
                "position_mapping": mapping,  # which source went to Text A/B
                "result": js,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            all_json.append(js)

    # Build summary: per-aspect winner counts by true source (file_a/file_b)
    aspects = ["clarity", "grammaticality", "factuality", "style", "creativity"]
    aspect_winner_counts_by_source: Dict[str, Dict[str, int]] = {
        asp: {"file_a": 0, "file_b": 0, "N/A": 0} for asp in aspects
    }

    # Re-iterate pairs with the same RNG to reconstruct mappings in the same order
    rng2 = random.Random(args.seed)
    for (ia, ib), js in zip(pairs, all_json):
        # Rebuild mapping for this pair (same RNG path)
        if rng2.random() < 0.5:
            pos_to_source = {"A": ("file_a", ia), "B": ("file_b", ib)}
        else:
            pos_to_source = {"A": ("file_b", ib), "B": ("file_a", ia)}

        if not isinstance(js, dict):
            continue

        for asp in aspects:
            asp_obj = js.get(asp)
            if not isinstance(asp_obj, dict):
                continue
            winner = asp_obj.get("winner")
            if winner == "A":
                src_label, _ = pos_to_source["A"]
                aspect_winner_counts_by_source[asp][src_label] += 1
            elif winner == "B":
                src_label, _ = pos_to_source["B"]
                aspect_winner_counts_by_source[asp][src_label] += 1
            elif winner == "N/A":
                aspect_winner_counts_by_source[asp]["N/A"] += 1
            else:
                continue

    summary_obj = {
        "input_file_a": args.input_a,
        "input_file_b": args.input_b,
        "model": args.model,
        "count": len(all_json),
        # Per-aspect winner counts mapped to original source files
        "aspect_winner_counts_by_source": aspect_winner_counts_by_source,
    }

    with open(summary_path, "w", encoding="utf-8") as fsum:
        json.dump(summary_obj, fsum, ensure_ascii=False, indent=2)

    print(f"\nSaved details to: {args.out}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()


