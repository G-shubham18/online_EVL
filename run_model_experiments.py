"""
Model Experiment Runner for EchoVision:
Automates evaluating models one-by-one across:
1. LLM Models:
   - qwen2.5-v1-72b-instruct
   - gemma-4-31b
   - phi-3.5-vision-instruct
2. Captioning Models:
   - Salesforce/blip-image-captioning-base
   - Salesforce/blip-image-captioning-large
   - HuggingFaceTB/SmolVLM-256M-Instruct
   - wraps/moondream-caption
3. Audio Embedding Model:
   - FacebookAI/roberta-base
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional

from main import process_dataset_pipeline, AblationConfig
from config import (
    SUPPORTED_LLM_MODELS,
    SUPPORTED_CAPTION_MODELS,
    SUPPORTED_AUDIO_EMBED_MODELS,
    ACTIVE_LLM_MODEL,
    ACTIVE_CAPTION_MODEL,
    ACTIVE_AUDIO_EMBED_MODEL,
    HF_TOKEN,
    USE_API,
    LLM_API_BASE
)

# Benchmark model suites
BENCHMARK_LLM_MODELS = [
    "qwen2.5-v1-72b-instruct",
    "gemma-4-31b",
    "phi-3.5-vision-instruct",
]

BENCHMARK_CAPTION_MODELS = [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    "wraps/moondream-caption",
]

BENCHMARK_AUDIO_MODELS = [
    "FacebookAI/roberta-base",
]


def format_markdown_report(results_list: List[Dict[str, Any]]) -> str:
    """Creates a formatted markdown comparison table from model experiment runs."""
    headers = [
        "Experiment",
        "Type",
        "Model Tested",
        "Audio Embed Model",
        "Exact Match %",
        "Relaxed Acc %",
        "Evaluated",
        "Time (s)"
    ]

    header_row = "| " + " | ".join(headers) + " |"
    divider_row = "| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |"
    rows = []

    for res in results_list:
        m = res.get("metrics", {})
        em = m.get("exact_match_accuracy", 0.0)
        rel = m.get("relaxed_accuracy", 0.0)
        total_eval = m.get("total_evaluated", 0)
        dur = res.get("duration_sec", 0.0)

        row = [
            f"**{res['exp_name']}**",
            f"`{res['exp_type']}`",
            f"`{res['model_tested']}`",
            f"`{res.get('audio_embed_model', '')}`",
            f"**{em:.2f}%**",
            f"{rel:.2f}%",
            f"{total_eval}",
            f"{dur:.1f}s"
        ]
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join([header_row, divider_row] + rows)


def run_experiments(
    mode: str = "llm",
    videos_dir: str = "smoketest/videos",
    json_dir: str = "smoketest/json",
    base_output_dir: str = "output/model_benchmarks",
    use_api: bool = True,
    hf_token: Optional[str] = None,
    llm_api_base: Optional[str] = None,
    auto_ingest: bool = True,
    force_reindex: bool = False,
    custom_llm: Optional[str] = None,
    custom_caption: Optional[str] = None,
    custom_audio: Optional[str] = None
):
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    os.makedirs(base_output_dir, exist_ok=True)

    experiments = []

    if custom_llm or custom_caption or custom_audio:
        # Run a specific custom combination
        llm = custom_llm or ACTIVE_LLM_MODEL
        cap = custom_caption or ACTIVE_CAPTION_MODEL
        aud = custom_audio or ACTIVE_AUDIO_EMBED_MODEL
        exp_id = f"custom_{llm.split('/')[-1]}_{cap.split('/')[-1]}_{aud.split('/')[-1]}".replace(":", "_").replace("-", "_")
        experiments.append({
            "exp_id": exp_id,
            "exp_name": f"Custom Run: {llm}",
            "exp_type": "Custom",
            "model_tested": llm,
            "llm_model": llm,
            "caption_model": cap,
            "audio_embed_model": aud
        })

    elif mode == "llm":
        # Test each LLM model one by one
        cap = ACTIVE_CAPTION_MODEL
        aud = ACTIVE_AUDIO_EMBED_MODEL
        for llm in BENCHMARK_LLM_MODELS:
            exp_id = f"llm_{llm.split('/')[-1]}".replace(":", "_").replace("-", "_")
            experiments.append({
                "exp_id": exp_id,
                "exp_name": f"LLM: {llm}",
                "exp_type": "LLM",
                "model_tested": llm,
                "llm_model": llm,
                "caption_model": cap,
                "audio_embed_model": aud
            })

    elif mode == "caption":
        # Test each captioning model one by one
        llm = ACTIVE_LLM_MODEL
        aud = ACTIVE_AUDIO_EMBED_MODEL
        for cap in BENCHMARK_CAPTION_MODELS:
            exp_id = f"caption_{cap.split('/')[-1]}".replace(":", "_").replace("-", "_")
            experiments.append({
                "exp_id": exp_id,
                "exp_name": f"Caption: {cap.split('/')[-1]}",
                "exp_type": "Captioning",
                "model_tested": cap,
                "llm_model": llm,
                "caption_model": cap,
                "audio_embed_model": aud
            })

    elif mode == "audio":
        # Test audio embedding models
        llm = ACTIVE_LLM_MODEL
        cap = ACTIVE_CAPTION_MODEL
        for aud in BENCHMARK_AUDIO_MODELS:
            exp_id = f"audio_{aud.split('/')[-1]}".replace(":", "_").replace("-", "_")
            experiments.append({
                "exp_id": exp_id,
                "exp_name": f"Audio Embed: {aud.split('/')[-1]}",
                "exp_type": "AudioEmbedding",
                "model_tested": aud,
                "llm_model": llm,
                "caption_model": cap,
                "audio_embed_model": aud
            })

    elif mode == "all":
        # Test all LLMs, Captions, and Audio one by one
        for llm in BENCHMARK_LLM_MODELS:
            exp_id = f"llm_{llm.split('/')[-1]}".replace(":", "_").replace("-", "_")
            experiments.append({
                "exp_id": exp_id,
                "exp_name": f"LLM: {llm}",
                "exp_type": "LLM",
                "model_tested": llm,
                "llm_model": llm,
                "caption_model": ACTIVE_CAPTION_MODEL,
                "audio_embed_model": ACTIVE_AUDIO_EMBED_MODEL
            })
        for cap in BENCHMARK_CAPTION_MODELS:
            exp_id = f"caption_{cap.split('/')[-1]}".replace(":", "_").replace("-", "_")
            experiments.append({
                "exp_id": exp_id,
                "exp_name": f"Caption: {cap.split('/')[-1]}",
                "exp_type": "Captioning",
                "model_tested": cap,
                "llm_model": ACTIVE_LLM_MODEL,
                "caption_model": cap,
                "audio_embed_model": ACTIVE_AUDIO_EMBED_MODEL
            })
        for aud in BENCHMARK_AUDIO_MODELS:
            exp_id = f"audio_{aud.split('/')[-1]}".replace(":", "_").replace("-", "_")
            experiments.append({
                "exp_id": exp_id,
                "exp_name": f"Audio Embed: {aud.split('/')[-1]}",
                "exp_type": "AudioEmbedding",
                "model_tested": aud,
                "llm_model": ACTIVE_LLM_MODEL,
                "caption_model": ACTIVE_CAPTION_MODEL,
                "audio_embed_model": aud
            })

    print(f"\n=======================================================")
    print(f"      ECHOVISION MULTI-MODEL BENCHMARK RUNNER          ")
    print(f"=======================================================")
    print(f"Total experiments to execute one by one: {len(experiments)}")
    print(f"Base Output Directory                  : {os.path.abspath(base_output_dir)}")
    print(f"API Mode Enabled                       : {use_api}")
    print(f"=======================================================\n")

    results_summary = []

    for idx, exp in enumerate(experiments, start=1):
        exp_id = exp["exp_id"]
        exp_dir = os.path.join(base_output_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n[{idx}/{len(experiments)}] Running Experiment: {exp['exp_name']}")
        print(f"  - Target LLM Model        : {exp['llm_model']}")
        print(f"  - Target Captioning Model : {exp['caption_model']}")
        print(f"  - Target Audio Embed Model: {exp['audio_embed_model']}")
        print(f"  - Isolated Output Folder  : {exp_dir}")

        start_t = time.time()
        try:
            res = process_dataset_pipeline(
                videos_dir=videos_dir,
                json_dir=json_dir,
                output_dir=exp_dir,
                force_reindex=force_reindex,
                auto_ingest=auto_ingest,
                ablation_config=AblationConfig(name=exp_id),
                llm_model=exp["llm_model"],
                caption_model=exp["caption_model"],
                audio_embed_model=exp["audio_embed_model"],
                use_api=use_api,
                hf_token=hf_token,
                llm_api_base=llm_api_base
            )
            elapsed = time.time() - start_t
            metrics = res.get("eval_metrics", {})
            print(f" -> Completed in {elapsed:.1f}s | Exact Match: {metrics.get('exact_match_accuracy', 0.0)}% | Relaxed: {metrics.get('relaxed_accuracy', 0.0)}%")
            
            exp_record = {
                "exp_id": exp_id,
                "exp_name": exp["exp_name"],
                "exp_type": exp["exp_type"],
                "model_tested": exp["model_tested"],
                "llm_model": exp["llm_model"],
                "caption_model": exp["caption_model"],
                "audio_embed_model": exp["audio_embed_model"],
                "metrics": metrics,
                "duration_sec": elapsed,
                "status": "success"
            }
            results_summary.append(exp_record)

        except Exception as e:
            elapsed = time.time() - start_t
            print(f" -> [ERROR in {exp['exp_name']}]: {e}")
            results_summary.append({
                "exp_id": exp_id,
                "exp_name": exp["exp_name"],
                "exp_type": exp["exp_type"],
                "model_tested": exp["model_tested"],
                "metrics": {"exact_match_accuracy": 0.0, "relaxed_accuracy": 0.0, "total_evaluated": 0},
                "duration_sec": elapsed,
                "status": f"error: {str(e)}"
            })

    # Save summary report
    summary_path = os.path.join(base_output_dir, "master_benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    md_table = format_markdown_report(results_summary)
    md_path = os.path.join(base_output_dir, "master_benchmark_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# EchoVision Model Benchmark Report\n\n")
        f.write(md_table)
        f.write("\n")

    print("\n" + "=" * 60)
    print("           MASTER BENCHMARK EVALUATION REPORT             ")
    print("=" * 60)
    print(md_table)
    print("=" * 60)
    print(f"\nAll benchmark results saved to: {os.path.abspath(base_output_dir)}")
    print(f"  - Markdown Summary: {md_path}")
    print(f"  - JSON Summary    : {summary_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EchoVision Model Evaluations One by One")
    parser.add_argument("--mode", type=str, default="llm", choices=["llm", "caption", "audio", "all", "custom"], help="Experiment mode to run one by one")
    parser.add_argument("--videos_dir", type=str, default="smoketest/videos", help="Directory containing videos")
    parser.add_argument("--json_dir", type=str, default="smoketest/json", help="Directory containing QA JSONs")
    parser.add_argument("--output_dir", type=str, default="output/model_benchmarks", help="Base output directory")
    parser.add_argument("--use_api", action="store_true", default=True, help="Enable API inference for models")
    parser.add_argument("--local", action="store_true", help="Force local model inference instead of API")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--llm_api_base", type=str, default=None, help="Custom OpenAI-compatible API base URL")
    parser.add_argument("--force_reindex", action="store_true", help="Force re-indexing Stage 1")
    parser.add_argument("--auto_ingest", action="store_true", default=True, help="Auto-run ingestion if store missing")

    # Specific overrides
    parser.add_argument("--llm_model", type=str, default=None, help="Specific LLM model to run")
    parser.add_argument("--caption_model", type=str, default=None, help="Specific Captioning model to run")
    parser.add_argument("--audio_embed_model", type=str, default=None, help="Specific Audio Embedding model to run")

    args = parser.parse_args()

    api_flag = False if args.local else args.use_api

    run_experiments(
        mode=args.mode,
        videos_dir=args.videos_dir,
        json_dir=args.json_dir,
        base_output_dir=args.output_dir,
        use_api=api_flag,
        hf_token=args.hf_token,
        llm_api_base=args.llm_api_base,
        auto_ingest=args.auto_ingest,
        force_reindex=args.force_reindex,
        custom_llm=args.llm_model,
        custom_caption=args.caption_model,
        custom_audio=args.audio_embed_model
    )
