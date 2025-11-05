import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


STEPS: List[int] = [32, 64, 128, 256]


def file_exists(path: str) -> bool:
    return os.path.isfile(path)


def try_load_json(paths: Sequence[str]) -> Optional[Dict[str, Any]]:
    for path in paths:
        if not path:
            continue
        if file_exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def safe_get(d: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def extract_metrics(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    # Numeric aggregations we need from corrected_samples_metrics*.json
    return {
        "avg_self_accuracy": to_float(data.get("average_self_accuracy")),
        "avg_self_ppl": to_float(safe_get(data, ["self_ppl_metrics", "average_perplexity"])),
        "gen_ppl": to_float(safe_get(data, ["external_metrics", "ppl"])),
        "gen_accuracy": to_float(safe_get(data, ["external_metrics", "accuracy"])),
        "entropy_per_token": to_float(safe_get(data, ["entropy_metrics", "ent_per_token"])),
        "bleu_score": to_float(safe_get(data, ["bleu", "corrected_vs_original", "bleu"])) or to_float(safe_get(data, ["bleu", "bleu"]))
    }


def extract_qual_scores(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    # Expect structure: { "mean_scores": { "Clarity": x, ... } }
    keys = ["Clarity", "Grammaticality", "Factuality", "Style", "Creativity"]
    out: Dict[str, Optional[float]] = {k: None for k in keys + ["Average"]}
    mean_scores = data.get("mean_scores")
    if isinstance(mean_scores, dict):
        lower_keys = {kk.lower(): kk for kk in mean_scores.keys()}
        vals: List[float] = []
        for k in keys:
            if k in mean_scores:
                v = to_float(mean_scores.get(k))
            else:
                v = None
                if k.lower() in lower_keys:
                    v = to_float(mean_scores.get(lower_keys[k.lower()]))
            out[k] = v
            if v is not None:
                vals.append(v)
        if vals:
            out["Average"] = sum(vals) / len(vals)
    return out


def build_group_file_maps(samples_dir: str) -> List[Dict[str, Any]]:
    # For each group, define how to locate metrics and qual files per step.
    # 128 maps to the non-suffix version as requested.
    def p(*parts: str) -> str:
        return os.path.join(samples_dir, *parts)

    groups: List[Dict[str, Any]] = [
        {
            "label": "Corrected (baseline)",
            "metrics_paths": {
                32: [p("corrected_samples_metrics_32_steps.json")],
                64: [p("corrected_samples_metrics_64_steps.json")],
                128: [p("corrected_samples_metrics.json")],
                256: [p("corrected_samples_metrics_256_steps.json")],
            },
            "qual_paths": {
                32: [p("qual_eval_corrected_32_steps_summary.json")],
                64: [p("qual_eval_corrected_64_steps_summary.json")],
                128: [p("qual_eval_corrected_summary.json")],
                256: [p("qual_eval_corrected_256_steps_summary.json")],
            },
        },
        {
            "label": "Multitoken 10",
            "metrics_paths": {
                32: [p("corrected_samples_metrics_multitoken_10_32_steps.json")],
                64: [p("corrected_samples_metrics_multitoken_10_64_steps.json")],
                128: [p("corrected_samples_metrics_multitoken_10.json")],
                256: [p("corrected_samples_metrics_multitoken_10_256_steps.json")],
            },
            "qual_paths": {
                32: [p("qual_eval_corrected_multitoken_10_32_steps_summary.json")],
                64: [p("qual_eval_corrected_multitoken_10_64_steps_summary.json")],
                128: [p("qual_eval_corrected_summary_multitoken_10.json")],
                256: [p("qual_eval_corrected_multitoken_10_256_steps_summary.json")],
            },
        },
        {
            "label": "NLL temp 0.1",
            "metrics_paths": {
                32: [p("corrected_samples_metrics_nll_temp_0.1_32_steps.json")],
                64: [p("corrected_samples_metrics_nll_temp_0.1_64_steps.json")],
                128: [p("corrected_samples_metrics_nll_temp_0.1.json")],
                256: [p("corrected_samples_metrics_nll_temp_0.1_256_steps.json")],
            },
            "qual_paths": {
                32: [p("qual_eval_corrected_nll_temp_0.1_32_steps_summary.json")],
                64: [p("qual_eval_corrected_nll_temp_0.1_64_steps_summary.json")],
                128: [
                    p("qual_eval_corrected_nll_temp_0.1_summary.json"),
                    p("qual_eval_corrected_summary_nll.json"),
                    p("qual_eval_corrected_summary.json"),
                ],
                256: [p("qual_eval_corrected_nll_temp_0.1_256_steps_summary.json")],
            },
        },
        {
            "label": "NLL temp 0.5",
            "metrics_paths": {
                32: [p("corrected_samples_metrics_nll_temp_0.5_32_steps.json")],
                64: [p("corrected_samples_metrics_nll_temp_0.5_64_steps.json")],
                128: [p("corrected_samples_metrics_nll_temp_0.5.json")],
                256: [p("corrected_samples_metrics_nll_temp_0.5_256_steps.json")],
            },
            "qual_paths": {
                32: [p("qual_eval_corrected_nll_temp_0.5_32_steps_summary.json")],
                64: [p("qual_eval_corrected_nll_temp_0.5_64_steps_summary.json")],
                128: [p("qual_eval_corrected_nll_temp_0.5_summary.json")],
                256: [p("qual_eval_corrected_nll_temp_0.5_256_steps_summary.json")],
            },
        },
    ]
    return groups


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_lines(x_steps: Sequence[int], series: List[Tuple[str, List[Optional[float]]]], title: str, ylabel: str, out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    for label, values in series:
        # Convert to (x, y) pairs skipping None
        xs: List[int] = []
        ys: List[float] = []
        for s, v in zip(x_steps, values):
            if v is None:
                continue
            xs.append(s)
            ys.append(v)
        if xs:
            plt.plot(xs, ys, marker="o", label=label)

    # Use log-scale on the x-axis (base 2 fits 32/64/128/256 nicely)
    try:
        plt.xscale("log", base=2)
    except TypeError:
        # Fallback for older matplotlib versions
        plt.xscale("log")
    plt.xticks(list(x_steps), [str(s) for s in x_steps])
    plt.xlabel("Inference Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    # Assume running from repo root; Samples at ./Samples
    samples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Samples")
    out_dir = os.path.join(samples_dir, "plots")
    ensure_dir(out_dir)

    groups = build_group_file_maps(samples_dir)

    # Aggregate metric values for each group across steps
    metric_keys = [
        ("avg_self_accuracy", "Average self-accuracy", "avg_self_accuracy.png"),
        ("avg_self_ppl", "Average self-PPL", "avg_self_ppl.png"),
        ("gen_ppl", "Generative PPL", "gen_ppl.png"),
        ("gen_accuracy", "Generative Accuracy", "gen_accuracy.png"),
        ("entropy_per_token", "Entropy per Token", "entropy_per_token.png"),
        ("bleu_score", "BLEU Score", "bleu_score.png"),
    ]

    # Qualitative categories
    qual_keys = [
        ("Clarity", "Clarity", "clarity.png"),
        ("Grammaticality", "Grammaticality", "grammaticality.png"),
        ("Factuality", "Factuality", "factuality.png"),
        ("Style", "Style", "style.png"),
        ("Creativity", "Creativity", "creativity.png"),
        ("Average", "Average Qual Score", "qual_average.png"),
    ]

    # Preload data for all groups/steps
    per_group_metrics: List[Dict[int, Dict[str, Optional[float]]]] = []
    per_group_qual: List[Dict[int, Dict[str, Optional[float]]]] = []

    for group in groups:
        g_metrics_by_step: Dict[int, Dict[str, Optional[float]]] = {}
        g_qual_by_step: Dict[int, Dict[str, Optional[float]]] = {}

        for step in STEPS:
            # Metrics
            metrics_json = try_load_json(group["metrics_paths"].get(step, []))
            if metrics_json is not None:
                g_metrics_by_step[step] = extract_metrics(metrics_json)
            else:
                g_metrics_by_step[step] = {k: None for k, _, _ in metric_keys}

            # Qual summary
            qual_json = try_load_json(group["qual_paths"].get(step, []))
            if qual_json is not None:
                g_qual_by_step[step] = extract_qual_scores(qual_json)
            else:
                g_qual_by_step[step] = {k: None for k, _, _ in qual_keys}

        per_group_metrics.append(g_metrics_by_step)
        per_group_qual.append(g_qual_by_step)

    # Plot numeric metrics
    for key, ylabel, filename in metric_keys:
        series: List[Tuple[str, List[Optional[float]]]] = []
        for group, g_metrics in zip(groups, per_group_metrics):
            values = [g_metrics.get(step, {}).get(key) for step in STEPS]
            series.append((group["label"], values))
        plot_lines(STEPS, series, title=ylabel + " vs. Inference Steps", ylabel=ylabel, out_path=os.path.join(out_dir, filename))

    # Plot qualitative metrics
    for key, ylabel, filename in qual_keys:
        series = []
        for group, g_qual in zip(groups, per_group_qual):
            values = [g_qual.get(step, {}).get(key) for step in STEPS]
            series.append((group["label"], values))
        plot_lines(STEPS, series, title=ylabel + " vs. Inference Steps", ylabel=ylabel, out_path=os.path.join(out_dir, filename))

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()


