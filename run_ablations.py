import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any

from main import process_dataset_pipeline, AblationConfig

# Definitions of standard ablations
ABLATION_SUITE = [
    {
        "id": "full",
        "name": "Full EchoVision (Decoupled AV RAG)",
        "config": AblationConfig(store_mode="separated", rerank_mode="full", sufficiency_mode="adaptive", modality_mode="audio_visual", name="full"),
        "description": "Baseline: Decoupled stores + Audio-aware lift + Adaptive sufficiency gate + Full AV"
    },
    {
        "id": "joint_store",
        "name": "Joined Store (Merged Audio+Visual)",
        "config": AblationConfig(store_mode="joint", rerank_mode="full", sufficiency_mode="adaptive", modality_mode="audio_visual", name="joint_store"),
        "description": "Ablation 1: Merged audio & visual unified store. Tests central claim: separation reduces cross-modal hallucination."
    },
    {
        "id": "no_audio_lift",
        "name": "Re-ranking Without Audio Lift",
        "config": AblationConfig(store_mode="separated", rerank_mode="no_audio_lift", sufficiency_mode="adaptive", modality_mode="audio_visual", name="no_audio_lift"),
        "description": "Ablation 2a: Cross-encoder re-ranking without beta(q)*AudioBonus. Tests how much 'silence' failure is fixed."
    },
    {
        "id": "no_rerank",
        "name": "Re-ranking Off (Raw Retriever Scores)",
        "config": AblationConfig(store_mode="separated", rerank_mode="off", sufficiency_mode="adaptive", modality_mode="audio_visual", name="no_rerank"),
        "description": "Ablation 2b: Bypasses cross-encoder re-ranking completely."
    },
    {
        "id": "fixed_cutoff",
        "name": "Fixed Cutoff (No Adaptive Gate)",
        "config": AblationConfig(store_mode="separated", rerank_mode="full", sufficiency_mode="fixed", modality_mode="audio_visual", name="fixed_cutoff"),
        "description": "Ablation 3: Fixed evidence cutoff without adaptive gate or loopbacks. Tests 'don't answer sound without sound'."
    },
    {
        "id": "no_audio",
        "name": "Audio Removed (Visual-Only Baseline)",
        "config": AblationConfig(store_mode="separated", rerank_mode="full", sufficiency_mode="adaptive", modality_mode="visual_only", name="no_audio"),
        "description": "Ablation 4: Audio disabled entirely. Honesty check: measures total gain contributed by audio reasoning."
    }
]


def format_markdown_table(results_summary: List[Dict[str, Any]]) -> str:
    """Generates a clean Markdown comparison table from ablation results."""
    # Find full baseline accuracy if present
    baseline_em = None
    baseline_relaxed = None
    for res in results_summary:
        if res["id"] == "full" and res.get("metrics"):
            baseline_em = res["metrics"].get("exact_match_accuracy")
            baseline_relaxed = res["metrics"].get("relaxed_accuracy")
            break

    headers = [
        "Ablation Experiment",
        "Store Mode",
        "Re-ranking",
        "Sufficiency Gate",
        "Modality",
        "Exact Match %",
        "Relaxed Acc %",
        "Δ (vs Full)"
    ]

    header_row = "| " + " | ".join(headers) + " |"
    divider_row = "| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |"
    data_rows = []

    for res in results_summary:
        m = res.get("metrics", {})
        em = m.get("exact_match_accuracy", 0.0)
        rel = m.get("relaxed_accuracy", 0.0)
        
        delta_str = "0.00%"
        if baseline_em is not None and res["id"] != "full":
            diff = em - baseline_em
            delta_str = f"{diff:+.2f}%"

        cfg = res["config"]
        row = [
            f"**{res['name']}**",
            f"`{cfg.store_mode}`",
            f"`{cfg.rerank_mode}`",
            f"`{cfg.sufficiency_mode}`",
            f"`{cfg.modality_mode}`",
            f"**{em:.2f}%**",
            f"{rel:.2f}%",
            f"`{delta_str}`"
        ]
        data_rows.append("| " + " | ".join(row) + " |")

    return "\n".join([header_row, divider_row] + data_rows)


def run_ablation_suite(
    dataset_dir: str = "dataset",
    videos_dir: str = None,
    json_dir: str = None,
    output_dir: str = "output/ablations",
    selected_ablations: List[str] = None,
    force_reindex: bool = False,
    auto_ingest: bool = False
):
    v_dir = videos_dir if videos_dir else os.path.join(dataset_dir, "videos")
    j_dir = json_dir if json_dir else os.path.join(dataset_dir, "json")

    os.makedirs(output_dir, exist_ok=True)

    suite_to_run = []
    for item in ABLATION_SUITE:
        if selected_ablations:
            if item["id"] in selected_ablations or "all" in selected_ablations:
                suite_to_run.append(item)
        else:
            suite_to_run.append(item)

    print("\n" + "=" * 65)
    print("        ECHOVISION ABLATION EXPERIMENT BENCHMARK SUITE")
    print("=" * 65)
    print(f"Dataset Directory : {os.path.abspath(dataset_dir)}")
    print(f"Videos Directory  : {os.path.abspath(v_dir)}")
    print(f"JSON Directory    : {os.path.abspath(j_dir)}")
    print(f"Output Directory  : {os.path.abspath(output_dir)}")
    print(f"Total Ablations   : {len(suite_to_run)}")
    print("Ablations Scheduled:")
    for idx, abl in enumerate(suite_to_run, start=1):
        print(f"  {idx}. [{abl['id']}] {abl['name']}: {abl['description']}")
    print("=" * 65 + "\n")

    results_summary = []
    start_time = time.time()

    for idx, abl in enumerate(suite_to_run, start=1):
        abl_id = abl["id"]
        abl_name = abl["name"]
        abl_cfg = abl["config"]
        abl_out_dir = os.path.join(output_dir, abl_id)

        print("\n" + "#" * 65)
        print(f" [RUN {idx}/{len(suite_to_run)}] Executing Ablation: {abl_name} ({abl_id})")
        print("#" * 65)

        run_result = process_dataset_pipeline(
            videos_dir=v_dir,
            json_dir=j_dir,
            output_dir=abl_out_dir,
            force_reindex=force_reindex,
            auto_ingest=auto_ingest,
            ablation_config=abl_cfg
        )

        metrics = run_result.get("eval_metrics", {})
        results_summary.append({
            "id": abl_id,
            "name": abl_name,
            "description": abl["description"],
            "config": abl_cfg,
            "metrics": metrics,
            "total_questions": run_result.get("total_questions", 0),
            "successful_questions": run_result.get("successful_questions", 0)
        })

    elapsed = time.time() - start_time

    # Generate Markdown Table & JSON summary
    md_table = format_markdown_table(results_summary)
    
    summary_report_path = os.path.join(output_dir, "ablation_summary.md")
    with open(summary_report_path, "w", encoding="utf-8") as md_f:
        md_f.write("# EchoVision Ablation Study Benchmark Summary\n\n")
        md_f.write(f"- **Dataset**: `{os.path.abspath(dataset_dir)}`\n")
        md_f.write(f"- **Execution Time**: `{elapsed:.2f}s`\n\n")
        md_f.write("### Comparative Results Table\n\n")
        md_f.write(md_table)
        md_f.write("\n\n### Ablation Hypothesis Analysis\n\n")
        md_f.write("1. **Joined vs. Separated Stores (`joint_store`)**:\n")
        md_f.write("   - Merging audio and visual into a single store tests whether modality separation reduces cross-modal hallucination and prevents visual descriptions from overshadowing acoustic events.\n")
        md_f.write("2. **Re-ranking Lift (`no_audio_lift` & `no_rerank`)**:\n")
        md_f.write("   - Removing the audio-aware lift $\\beta(q) \\cdot \\text{AudioBonus}$ measures how effectively cross-modal re-ranking resolves the 'silence' failure on sound questions.\n")
        md_f.write("3. **Sufficiency Stop vs. Fixed Cutoff (`fixed_cutoff`)**:\n")
        md_f.write("   - Replacing adaptive gating with a static evidence cutoff proves the value of 'never answer a sound question without sound evidence.'\n")
        md_f.write("4. **Audio Removed Entirely (`no_audio`)**:\n")
        md_f.write("   - Visual-only baseline establishes the ground truth contribution of acoustic evidence across the benchmark.\n")

    json_report_path = os.path.join(output_dir, "ablation_summary.json")
    with open(json_report_path, "w", encoding="utf-8") as json_f:
        json_f.write(json.dumps({
            "elapsed_seconds": round(elapsed, 2),
            "total_ablations": len(results_summary),
            "results": [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "description": r["description"],
                    "store_mode": r["config"].store_mode,
                    "rerank_mode": r["config"].rerank_mode,
                    "sufficiency_mode": r["config"].sufficiency_mode,
                    "modality_mode": r["config"].modality_mode,
                    "metrics": r["metrics"]
                }
                for r in results_summary
            ]
        }, indent=4))

    print("\n" + "=" * 65)
    print("      ECHOVISION ABLATION SUITE COMPLETED SUCCESSFULLY       ")
    print("=" * 65)
    print(f"Elapsed Time : {elapsed:.2f} seconds")
    print(f"Report (MD)  : {summary_report_path}")
    print(f"Report (JSON): {json_report_path}")
    print("\n" + md_table + "\n")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoVision Automated Ablation Study Runner")
    parser.add_argument("--dataset_dir", type=str, default="smoketest", help="Path to dataset directory containing videos/ and json/")
    parser.add_argument("--videos_dir", type=str, default=None, help="Path to videos directory (defaults to dataset_dir/videos)")
    parser.add_argument("--json_dir", type=str, default=None, help="Path to json directory (defaults to dataset_dir/json)")
    parser.add_argument("--output_dir", type=str, default="output/ablations", help="Path to directory for ablation reports and outputs")
    parser.add_argument("--ablations", nargs="+", default=["all"], help="Specific ablations to run: full, joint_store, no_audio_lift, no_rerank, fixed_cutoff, no_audio, all")
    parser.add_argument("--force-reindex", action="store_true", help="Force re-indexing of Stage 1")
    parser.add_argument("--auto-ingest", action="store_true", help="Automatically ingest missing Stage 1 video indices")

    args = parser.parse_args()

    selected = args.ablations
    if "all" in selected:
        selected = None

    run_ablation_suite(
        dataset_dir=args.dataset_dir,
        videos_dir=args.videos_dir,
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        selected_ablations=selected,
        force_reindex=args.force_reindex,
        auto_ingest=args.auto_ingest
    )
