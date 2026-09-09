import os
import hashlib
import traceback
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from config import (
    VECTOR_STORE_DIR, CONCURRENT_INGESTION,
    ACTIVE_CAPTION_MODEL, ACTIVE_AUDIO_EMBED_MODEL,
    USE_API, HF_TOKEN
)
from stage1_offline.vector_indexer import VectorIndexer, is_video_indexed_on_disk

# Supported video file extensions for recursive scanning
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".3gp", ".ts"
}


def discover_videos(videos_dir: str) -> List[str]:
    """
    Recursively scans the given videos directory and returns a sorted list of absolute paths
    for all supported video files.
    """
    videos_path = Path(videos_dir)
    if not videos_path.exists():
        print(f"[Ingestion] Warning: Videos directory '{videos_dir}' does not exist.")
        return []

    discovered = []
    for root, _, files in os.walk(videos_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in SUPPORTED_VIDEO_EXTENSIONS:
                abs_path = os.path.abspath(os.path.join(root, file))
                discovered.append(abs_path)

    discovered.sort()
    return discovered


def get_video_store_dir(
    video_path: str, 
    base_store_dir: Optional[str] = None,
    audio_embed_model: Optional[str] = None,
    caption_model: Optional[str] = None
) -> str:
    """
    Generates a unique, isolated vector store directory for a specific video file
    and model configuration to ensure tests between different models never collide.
    """
    base_dir = base_store_dir if base_store_dir else VECTOR_STORE_DIR
    abs_path = os.path.abspath(video_path)
    stem = Path(abs_path).stem
    safe_stem = "".join([c if c.isalnum() or c in ('-', '_') else '_' for c in stem])
    
    a_name = audio_embed_model if audio_embed_model else os.getenv("AUDIO_EMBED_MODEL", ACTIVE_AUDIO_EMBED_MODEL)
    c_name = caption_model if caption_model else os.getenv("CAPTION_MODEL", ACTIVE_CAPTION_MODEL)
    
    a_tag = a_name.split("/")[-1].replace("-", "_").lower()
    c_tag = c_name.split("/")[-1].replace("-", "_").lower()

    path_hash = hashlib.md5(f"{abs_path}_{a_tag}_{c_tag}".encode('utf-8')).hexdigest()[:8]
    folder_name = f"{safe_stem}_{a_tag}_{c_tag}_{path_hash}"
    return os.path.join(base_dir, folder_name)


class Stage1Ingestor:
    """
    Handles Stage 1 (Offline Extraction & Indexing) independently for every video in the dataset.
    Stores Stage 1 output/index for each video separately so that data from one video
    never overwrites or mixes with another video.
    """
    def __init__(
        self, 
        base_store_dir: Optional[str] = None, 
        concurrent: Optional[bool] = None,
        caption_model: Optional[str] = None,
        audio_embed_model: Optional[str] = None,
        use_api: Optional[bool] = None,
        hf_token: Optional[str] = None,
        **kwargs
    ):
        self.base_store_dir = base_store_dir if base_store_dir else VECTOR_STORE_DIR
        self.concurrent = concurrent if concurrent is not None else CONCURRENT_INGESTION
        self.caption_model = caption_model if caption_model else os.getenv("CAPTION_MODEL", ACTIVE_CAPTION_MODEL)
        self.audio_embed_model = audio_embed_model if audio_embed_model else os.getenv("AUDIO_EMBED_MODEL", ACTIVE_AUDIO_EMBED_MODEL)
        self.use_api = use_api if use_api is not None else USE_API
        self.hf_token = hf_token if hf_token else (HF_TOKEN if HF_TOKEN else os.getenv("HF_TOKEN", ""))
        self.audio_extractor = None
        self.visual_extractor = None

    def _lazy_init_extractors(self):
        """Lazily instantiates extractor models once to reuse model weights across videos."""
        if self.audio_extractor is None:
            print("[Stage 1 Ingestion] Initializing AudioExtractor model (Whisper & Audio Events)...")
            from stage1_offline.audio_extractor import AudioExtractor
            self.audio_extractor = AudioExtractor()
        if self.visual_extractor is None:
            print(f"[Stage 1 Ingestion] Initializing VisualExtractor model ({self.caption_model})...")
            from stage1_offline.visual_extractor import VisualExtractor
            self.visual_extractor = VisualExtractor(
                caption_model=self.caption_model, 
                use_api=self.use_api,
                hf_token=self.hf_token
            )

    def process_single_video(
        self,
        video_path: str,
        force_reindex: bool = False
    ) -> Tuple[Optional[VectorIndexer], bool, Optional[str]]:
        """
        Runs Stage 1 (Offline Extraction & Indexing) independently for a single video.

        Returns:
            (indexer: Optional[VectorIndexer], was_reused: bool, error_message: Optional[str])
        """
        abs_video_path = os.path.abspath(video_path)
        video_store_dir = get_video_store_dir(
            abs_video_path, 
            self.base_store_dir,
            audio_embed_model=self.audio_embed_model,
            caption_model=self.caption_model
        )
        v_name = os.path.basename(abs_video_path)

        print(f"\n--------------------------------------------------")
        print(f"[Stage 1 Ingestion] Video File: {abs_video_path}")
        print(f"[Stage 1 Ingestion] Isolated Vector Store: {video_store_dir}")

        # Fast check on disk: if index already exists and force_reindex is False, skip extraction
        if not force_reindex and is_video_indexed_on_disk(abs_video_path, video_store_dir):
            print(f"[Stage 1 Ingestion] [SKIP] Stage 1 index already exists for '{v_name}'. Reusing existing vector store.")
            temp_indexer = VectorIndexer(
                store_dir=video_store_dir,
                audio_embed_model=self.audio_embed_model,
                use_api=self.use_api
            )
            return temp_indexer, True, None

        temp_indexer = VectorIndexer(
            store_dir=video_store_dir,
            audio_embed_model=self.audio_embed_model,
            use_api=self.use_api
        )
        if not force_reindex and temp_indexer.is_indexed(abs_video_path):
            print(f"[Stage 1 Ingestion] [SKIP] Stage 1 index already exists for '{v_name}'. Reusing existing vector store.")
            return temp_indexer, True, None

        print(f"[Stage 1 Ingestion] [PROCESSING] Starting Stage 1 processing for '{v_name}'...")
        try:
            self._lazy_init_extractors()

            if self.concurrent:
                print(f" -> Stage 1: Concurrently extracting audio facts and visual keyframe descriptions...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    audio_future = executor.submit(self.audio_extractor.process_video, abs_video_path)
                    visual_future = executor.submit(self.visual_extractor.process_video, abs_video_path)
                    audio_facts = audio_future.result()
                    visual_facts = visual_future.result()
            else:
                # Sequential mode: avoids GPU VRAM memory spikes on standard 16GB GPUs (e.g. T4)
                print(f" -> Stage 1: Extracting audio facts (Whisper & CLAP)...")
                audio_facts = self.audio_extractor.process_video(abs_video_path)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f" -> Stage 1: Extracting visual facts & keyframe descriptions (Qwen2.5-VL & CLIP)...")
                visual_facts = self.visual_extractor.process_video(abs_video_path)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f" -> Stage 1: Building isolated vector stores (ChromaDB, FAISS Visual, & FAISS Joint)...")
            temp_indexer.clear_index()
            temp_indexer.index_audio_facts(audio_facts)
            temp_indexer.index_visual_facts(visual_facts)
            temp_indexer.index_joint_facts(audio_facts, visual_facts)
            temp_indexer.set_indexed_video(abs_video_path)

            print(f"[Stage 1 Ingestion] Stage 1 COMPLETED successfully for '{v_name}'.")
            return temp_indexer, False, None

        except Exception as e:
            err_msg = f"Failed Stage 1 processing for video '{abs_video_path}': {str(e)}\n{traceback.format_exc()}"
            print(f"[Stage 1 Ingestion ERROR] {err_msg}")
            return None, False, str(e)

    def process_dataset(
        self,
        videos_dir: str,
        force_reindex: bool = False
    ) -> Dict[str, Dict]:
        """
        Scans videos_dir recursively and runs Stage 1 for all discovered videos.
        Automatically skips videos that are already ingested.

        Returns a dictionary mapping video_path -> {
            'indexer': VectorIndexer,
            'store_dir': str,
            'status': 'success' | 'failed',
            'reused': bool,
            'error': Optional[str]
        }
        """
        discovered_videos = discover_videos(videos_dir)
        total_videos = len(discovered_videos)
        print(f"\n==================================================")
        print(f"[Stage 1 Dataset Ingestion] Discovered {total_videos} video(s) in '{videos_dir}'")
        print(f"==================================================")

        results = {}
        for idx, video_path in enumerate(discovered_videos, start=1):
            print(f"\n[Overall Progress: Video {idx}/{total_videos}] Current Video: {os.path.basename(video_path)}")
            indexer, reused, err = self.process_single_video(video_path, force_reindex=force_reindex)

            store_dir = get_video_store_dir(
                video_path, 
                self.base_store_dir,
                audio_embed_model=self.audio_embed_model,
                caption_model=self.caption_model
            )
            if indexer is not None:
                results[video_path] = {
                    'indexer': indexer,
                    'store_dir': store_dir,
                    'status': 'success',
                    'reused': reused,
                    'error': None
                }
            else:
                results[video_path] = {
                    'indexer': None,
                    'store_dir': store_dir,
                    'status': 'failed',
                    'reused': False,
                    'error': err
                }

        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 Ingestion: Extract & Index Video Features")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Root dataset directory (containing videos/)")
    parser.add_argument("--videos_dir", type=str, default=None, help="Directory containing videos (defaults to dataset_dir/videos)")
    parser.add_argument("--force-reindex", action="store_true", help="Force re-indexing even if already indexed")
    parser.add_argument("--concurrent", action="store_true", help="Run audio & visual feature extraction concurrently")
    parser.add_argument("--sequential", action="store_true", help="Run audio & visual extraction sequentially (recommended for 16GB GPUs like T4)")
    parser.add_argument("--caption_model", type=str, default=None, help="Captioning model (e.g. Salesforce/blip-image-captioning-base, Salesforce/blip-image-captioning-large, HuggingFaceTB/SmolVLM-256M-Instruct, wraps/moondream-caption)")
    parser.add_argument("--audio_embed_model", type=str, default=None, help="Audio embedding model (e.g. FacebookAI/roberta-base, laion/clap-htsat-unfused)")
    parser.add_argument("--use_api", action="store_true", help="Use Hugging Face Inference API for models instead of local inference")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token for API inference")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    v_dir = args.videos_dir
    if not v_dir:
        candidate_v_dir = os.path.join(args.dataset_dir, "videos")
        if os.path.exists(candidate_v_dir):
            v_dir = candidate_v_dir
        else:
            v_dir = args.dataset_dir

    concurrent_flag = None
    if args.concurrent:
        concurrent_flag = True
    elif args.sequential:
        concurrent_flag = False

    ingestor = Stage1Ingestor(
        concurrent=concurrent_flag,
        caption_model=args.caption_model,
        audio_embed_model=args.audio_embed_model,
        use_api=args.use_api if args.use_api else None
    )
    results = ingestor.process_dataset(v_dir, force_reindex=args.force_reindex)

    total_videos = len(results)
    successful_videos = sum(1 for res in results.values() if res['status'] == 'success')
    reused_videos = sum(1 for res in results.values() if res.get('reused', False))
    newly_indexed_videos = successful_videos - reused_videos
    failed_videos = total_videos - successful_videos

    print("\n" + "=" * 50)
    print("      STAGE 1 INGESTION COMPLETE                 ")
    print("=" * 50)
    print(f"Total Videos Discovered    : {total_videos}")
    print(f"Already Indexed (Skipped)  : {reused_videos}")
    print(f"Newly Indexed (Processed)  : {newly_indexed_videos}")
    print(f"Failed                     : {failed_videos}")
    print("=" * 50)
    print("Stage 1 vector stores are ready! Next step:")
    print("Run: python main.py")
    print("=" * 50 + "\n")

