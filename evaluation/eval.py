import argparse
import concurrent.futures
import csv
import importlib
import os
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm

from evaluation.model_registry import (
    FORMULA_JUDGE_MODEL,
    configure_openai_compatible_client,
    get_api_key,
    get_model_config,
    multimodal_model_names,
)
from utils.eval_data import load_eval_data
from utils.image_inputs import image_count


METHOD_NAMES = ("base", "tool", "correction", "consistency")
METHOD_IMPORTS = {
    "base": ("methods.base", "base"),
    "tool": ("methods.tool_augmentation", "tool_augmentation"),
    "correction": ("methods.self_correction", "self_correction"),
    "consistency": ("methods.self_consistency", "self_consistency"),
}

# Kept for analysis scripts that import this symbol.
MULTIMODAL_MODELS = multimodal_model_names()


def parse_args():
    parser = argparse.ArgumentParser(description="Run MatSciBench evaluation.")
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model registry key from evaluation/model_registry.py.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="base",
        choices=METHOD_NAMES,
        help="Prompting method to evaluate.",
    )
    parser.add_argument(
        "--llm_judge",
        action="store_true",
        help="Also run the LLM judge for FORMULA questions. The rule judge always runs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel model requests.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional number of filtered samples to evaluate.",
    )
    return parser.parse_args()


def load_method(method_name):
    module_name, function_name = METHOD_IMPORTS[method_name]
    return getattr(importlib.import_module(module_name), function_name)


def filter_data(data, method_name, is_multimodal):
    if method_name in {"tool", "correction"}:
        filtered = [entry for entry in data if image_count(entry) == 0]
        print(f"{method_name}: using {len(filtered)}/{len(data)} text-only questions.")
        return filtered
    if not is_multimodal:
        filtered = [entry for entry in data if image_count(entry) == 0]
        print(f"Text-only model: using {len(filtered)}/{len(data)} questions without images.")
        return filtered
    print(f"Multimodal model: using all {len(data)} questions.")
    return data


def run_model_requests(entries, method, model_name, max_tokens, temperature, is_multimodal, num_workers):
    responses = [None] * len(entries)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(
                method,
                entry,
                model_name,
                max_tokens,
                temperature,
                is_multimodal,
            ): index
            for index, entry in enumerate(entries)
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_index),
            total=len(future_to_index),
            desc="Generating answers",
        ):
            index = future_to_index[future]
            try:
                result = future.result()
            except Exception as exc:
                entry = entries[index]
                print(f"Error processing question {entry.get('qid', 'unknown')}: {exc}")
                continue
            if result:
                result["has_image"] = bool(result.get("image_count", 0))
                responses[index] = result
    return [response for response in responses if response is not None]


def output_fieldnames(rows):
    preferred = [
        "qid",
        "question_type",
        "question",
        "final_answer",
        "correct_answer",
        "is_correct",
        "judge_reasoning",
        "rule_is_correct",
        "rule_judge_reasoning",
        "llm_is_correct",
        "llm_judge_reasoning",
        "full_output",
        "correct_solution",
        "unit",
        "number_of_answers",
        "domain",
        "new_token_nums",
        "image",
        "image_count",
        "has_image",
        "error",
    ]
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    return [key for key in preferred if key in all_keys] + sorted(all_keys - set(preferred))


def safe_model_filename(model_key):
    return model_key.split("/")[-1].lower().replace(":", "_").replace(" ", "_")


def main():
    args = parse_args()

    try:
        model_config = get_model_config(args.model)
        configure_openai_compatible_client(model_config)
        if args.llm_judge:
            get_api_key(get_model_config(FORMULA_JUDGE_MODEL))
    except (KeyError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return

    try:
        data = load_eval_data()
    except Exception as exc:
        print(f"Error loading MatSciBench dataset: {exc}")
        return

    print(f"Loaded {len(data)} questions from MatSciBench.")
    filtered_data = filter_data(data, args.method, model_config.multimodal)
    if args.sample_size is not None:
        filtered_data = filtered_data[:args.sample_size]
        print(f"Sample size: {len(filtered_data)} questions.")

    method = load_method(args.method)
    temperature = 0.6 if args.method == "consistency" else args.temperature
    print(
        "Evaluating "
        f"{model_config.model_name} via {model_config.endpoint_url} "
        f"with method={args.method}, workers={args.num_workers}."
    )

    responses = run_model_requests(
        filtered_data,
        method,
        model_config.model_name,
        args.max_tokens,
        temperature,
        model_config.multimodal,
        args.num_workers,
    )

    for response in responses:
        response["model"] = args.model

    from evaluation.auto_judge import judge_responses

    decisions = judge_responses(
        responses,
        max_workers=32,
        use_llm_formula=args.llm_judge,
    )

    if not decisions:
        print(f"Warning: no decisions were generated; responses={len(responses)}.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_output_path = os.path.join(
        args.output_dir,
        f"{safe_model_filename(args.model)}_{args.method}_{timestamp}.csv",
    )

    fieldnames = output_fieldnames(decisions)
    for decision in decisions:
        for key in fieldnames:
            decision.setdefault(key, None)

    with open(csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decisions)

    print(f"Saved responses and decisions to {csv_output_path}")


if __name__ == "__main__":
    main()
