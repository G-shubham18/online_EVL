import os
import json
import re
from typing import Optional, List, Dict, Any, Tuple
import requests
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scenedetect import detect, ContentDetector
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from sentence_transformers import SentenceTransformer

from config import (
    QWEN_VL_MODEL,
    CLIP_MODEL,
    SIMILARITY_THRESHOLD,
    DEVICE,
    TORCH_DTYPE,
    SCENE_THRESHOLD,
    MAX_SCENE_WINDOW_SEC,
    CLIP_BATCH_SIZE,
    VLM_BATCH_SIZE,
    MIN_KEYFRAMES,
    MAX_KEYFRAMES,
    DYNAMIC_KEYFRAME_INTERVAL_SEC,
    MIN_VISION_PIXELS,
    MAX_VISION_PIXELS,
    VLM_MAX_NEW_TOKENS,
    VLM_IMAGE_RESIZE_MAX,
    IS_GPU,
    BLUR_THRESHOLD,
    DARK_THRESHOLD,
    BRIGHTNESS_MAX_THRESHOLD,
    LOW_CONTRAST_THRESHOLD,
    MIN_FRAME_WIDTH,
    MIN_FRAME_HEIGHT,
    CANDIDATE_SAMPLES_PER_SCENE,
    KEYFRAMES_SAVE_DIR,
    OLLAMA_MODEL,
    OLLAMA_HOST,
    OPENAI_API_KEY,
    GPT_MODEL,
)
from stage1_offline.object_detector import ObjectDetector

# Dynamically import Qwen / vision model classes supported by HF Transformers

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None

# Structured Prompt for Visual Analysis (Dense, factual, front-loads subjects, clothing colors, objects, spatial layout & actions)

STRUCTURED_VLM_PROMPT = """Analyze this video frame and provide a factual, concise visual breakdown:

Environment & Setting: Environment type (e.g. indoor room, dining room, kitchen, living room, church, stage, outdoor, street, park).
Subjects, Clothing & Quadrants: Identify persons, their clothing/colors (e.g. purple shirt, blue coat, red dress, white top, gray jacket), and their 2D quadrant (Top-Left, Top-Right, Bottom-Left, Bottom-Right, or Center). Note body parts (hands, head).
Objects, Furniture & Spatial Positions: Visible items/furniture (foreground, midground, background) and their spatial relations (behind, in front of, under, on, beside person - e.g. shelf/rack behind person, table/cabinet in front of person, chair, mirror, bread, sink, plate, musical instrument, ski).
Actions, Hand Movements & Trajectories: Observable physical actions (drinking, holding hairbrush, tying up hair, blowing hair, playing violin, dancing, folding paper, moving chair) and motion direction (left-to-right, stationary).
Visible Text & Labels: Any readable text or brands.
Be factual, specific, and concise without filler."""

class VisualExtractor:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        self.gpt_model = os.getenv("GPT_MODEL", GPT_MODEL)
        self.model = None
        self.processor = None

        # Load CLIP embedding model once in constructor
        print(f"Loading CLIP Embedding Model: {CLIP_MODEL}")
        self.clip_model = SentenceTransformer(CLIP_MODEL, device=DEVICE)
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.blur_threshold = BLUR_THRESHOLD
        self.dark_threshold = DARK_THRESHOLD
        self.brightness_max_threshold = BRIGHTNESS_MAX_THRESHOLD
        self.low_contrast_threshold = LOW_CONTRAST_THRESHOLD
        self.min_frame_width = MIN_FRAME_WIDTH
        self.min_frame_height = MIN_FRAME_HEIGHT
        self.candidate_samples_per_scene = CANDIDATE_SAMPLES_PER_SCENE
        self.keyframes_save_dir = KEYFRAMES_SAVE_DIR
        self.latest_candidate_metadata = []

        # Object Detection & Tracking (RT-DETR + GroundingDINO + SimpleSort)
        self.object_detector = ObjectDetector()

        # If no OpenAI API key is configured, load local VLM eagerly
        if not self.openai_api_key:
            self._load_local_vlm()

    def _load_local_vlm(self):
        if self.model is not None:
            return
        print(f"Loading Local Vision Model: {QWEN_VL_MODEL}")
        kwargs = {
            "torch_dtype": TORCH_DTYPE,
            "device_map": "auto" if DEVICE == "cuda" else None,
        }

        if "2.5" in QWEN_VL_MODEL and Qwen2_5_VLForConditionalGeneration is not None:
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    QWEN_VL_MODEL, **kwargs
                )
            except Exception as e:
                print(f"Failed to load via Qwen2_5_VLForConditionalGeneration: {e}")

        if self.model is None and AutoModelForImageTextToText is not None:
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    QWEN_VL_MODEL, **kwargs
                )
            except Exception as e:
                print(f"Failed to load via AutoModelForImageTextToText: {e}")

        if self.model is None and Qwen2VLForConditionalGeneration is not None:
            try:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    QWEN_VL_MODEL, **kwargs
                )
            except Exception as e:
                print(f"Failed to load via Qwen2VLForConditionalGeneration: {e}")

        if self.model is None:
            raise RuntimeError(f"Could not load vision model '{QWEN_VL_MODEL}'.")

        if DEVICE != "cuda":
            self.model = self.model.to(DEVICE)

        self.processor = AutoProcessor.from_pretrained(QWEN_VL_MODEL)

    def caption_frame_gpt4o_mini(self, pil_image: Image.Image, tracking_summary: str = "") -> Optional[str]:
        """Calls GPT-4o-mini to caption the keyframe with image + tracked objects context."""
        api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        if not api_key:
            return None
        import base64
        import io
        try:
            img_copy = pil_image.copy()
            img_copy.thumbnail((512, 512))
            buf = io.BytesIO()
            img_copy.save(buf, format="JPEG", quality=85)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

            prompt = STRUCTURED_VLM_PROMPT
            if tracking_summary and tracking_summary != "None detected.":
                prompt += f"\n\nDetected Visible Objects & Tracks:\n{tracking_summary}"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.gpt_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_str}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 250,
                "temperature": 0.0
            }
            res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=18.0)
            if res.status_code == 200:
                desc = res.json()["choices"][0]["message"]["content"].strip()
                return desc
            else:
                print(f"[VisualExtractor Warning] GPT-4o-mini API returned status {res.status_code}")
                return None
        except Exception as e:
            print(f"[VisualExtractor Warning] GPT-4o-mini frame captioning error: {e}")
            return None

    def evaluate_image_quality(self, frame_bgr: np.ndarray) -> dict:
        """
        Evaluates image quality across Sharpness, Brightness, Contrast, and Resolution.
        Returns raw metrics, continuous quality score (0.0 to 100.0),
        quality status (CLEAR, BLURRY, TOO_DARK, LOW_CONTRAST, LOW_QUALITY),
        and rejection reason if applicable.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "sharpness": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
                "resolution": [0, 0],
                "quality_score": 0.0,
                "status": "LOW_QUALITY",
                "rejection_reason": "EMPTY_FRAME"
            }

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        resolution = [w, h]

        status = "CLEAR"
        rejection_reason = None

        if brightness < self.dark_threshold:
            status = "TOO_DARK"
            rejection_reason = "TOO_DARK"
        elif brightness > self.brightness_max_threshold:
            status = "LOW_QUALITY"
            rejection_reason = "OVEREXPOSED"
        elif sharpness < self.blur_threshold:
            status = "BLURRY"
            rejection_reason = "BLURRY"
        elif contrast < self.low_contrast_threshold:
            status = "LOW_CONTRAST"
            rejection_reason = "LOW_CONTRAST"
        elif w < self.min_frame_width or h < self.min_frame_height:
            status = "LOW_QUALITY"
            rejection_reason = "LOW_RESOLUTION"

        # Compute continuous quality score (0.0 to 100.0 scale)
        sharp_norm = min(50.0, (sharpness / max(1.0, self.blur_threshold * 2.0)) * 50.0)
        contrast_norm = min(30.0, (contrast / max(1.0, self.low_contrast_threshold * 2.0)) * 30.0)

        if brightness < self.dark_threshold:
            bright_norm = max(0.0, (brightness / max(1.0, self.dark_threshold)) * 10.0)
        elif brightness > self.brightness_max_threshold:
            bright_norm = max(0.0, ((255.0 - brightness) / max(1.0, 255.0 - self.brightness_max_threshold)) * 10.0)
        else:
            bright_norm = 20.0 - min(10.0, (abs(brightness - 128.0) / 128.0) * 10.0)

        quality_score = round(float(sharp_norm + contrast_norm + bright_norm), 2)
        if status != "CLEAR":
            quality_score = round(quality_score * 0.5, 2)

        return {
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "resolution": resolution,
            "quality_score": quality_score,
            "status": status,
            "rejection_reason": rejection_reason
        }

    def is_dark_frame(self, frame_bgr: np.ndarray, threshold: float = None) -> bool:
        """Checks if a frame is mostly black/dark."""
        eval_res = self.evaluate_image_quality(frame_bgr)
        return eval_res["status"] == "TOO_DARK"

    def is_blurry_frame(self, frame_bgr: np.ndarray, threshold: float = None) -> bool:
        """Checks if a frame is blurry using Laplacian variance."""
        eval_res = self.evaluate_image_quality(frame_bgr)
        return eval_res["status"] == "BLURRY"

    def save_keyframes_to_disk(
        self, video_path: str, keyframes: list, output_dir: str = None, candidate_metadata: list = None
    ) -> list:
        """
        Saves final accepted keyframes to disk as JPG files and exports metadata.json
        into the video directory containing complete tracking data for all evaluated candidate frames.
        """
        if output_dir is None:
            output_dir = self.keyframes_save_dir

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_folder = os.path.join(output_dir, video_name)
        os.makedirs(save_folder, exist_ok=True)

        print(f"Saving {len(keyframes)} accepted keyframes to disk in '{save_folder}'...")
        for kf in keyframes:
            frame_filename = f"frame_{kf['frame_id']:04d}_{kf['mid_time']:.2f}s.jpg"
            save_path = os.path.join(save_folder, frame_filename)
            if "image" in kf and kf["image"] is not None:
                kf["image"].save(save_path, quality=95)
            kf["image_path"] = save_path

        source_metadata = candidate_metadata if candidate_metadata is not None else getattr(self, "latest_candidate_metadata", keyframes)

        all_metadata_records = []
        for record in source_metadata:
            rec_copy = {
                "video_id": record.get("video_id", video_name),
                "frame_id": record.get("frame_id"),
                "scene_id": record.get("scene_id"),
                "candidate_id": record.get("candidate_id"),
                "start_time": record.get("start_time"),
                "end_time": record.get("end_time"),
                "mid_time": record.get("mid_time"),
                "image_path": record.get("image_path"),
                "sharpness_score": record.get("sharpness_score"),
                "brightness_score": record.get("brightness_score"),
                "contrast_score": record.get("contrast_score"),
                "quality_score": record.get("quality_score"),
                "quality_status": record.get("quality_status"),
                "clip_similarity": record.get("clip_similarity"),
                "importance_score": record.get("importance_score"),
                "novelty_score": record.get("novelty_score"),
                "accepted": record.get("accepted", False),
                "rejection_reason": record.get("rejection_reason"),
            }
            all_metadata_records.append(rec_copy)

        metadata_path = os.path.join(save_folder, "metadata.json")
        metadata_wrapper = {
            "video_id": video_name,
            "video_path": os.path.abspath(video_path),
            "total_candidates_evaluated": len(all_metadata_records),
            "accepted_keyframes_count": len(keyframes),
            "keyframes": all_metadata_records
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_wrapper, f, indent=4, ensure_ascii=False)

        print(f"Saved keyframe metadata to '{metadata_path}'.")
        return keyframes

    def compute_frame_diversity_score(self, frame_bgr_1: np.ndarray, frame_bgr_2: np.ndarray) -> float:
        """
        Computes a fast, lightweight visual diversity score in [0.0, 1.0] between two frames
        using 2D HSV color histograms and downsampled spatial luminance difference.
        Avoids heavy neural network / CLIP model calls during candidate pre-selection.
        """
        if frame_bgr_1 is None or frame_bgr_2 is None or frame_bgr_1.size == 0 or frame_bgr_2.size == 0:
            return 0.5
        try:
            # 1. Color distribution distance in HSV space
            hsv1 = cv2.cvtColor(frame_bgr_1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(frame_bgr_2, cv2.COLOR_BGR2HSV)
            hist1 = cv2.calcHist([hsv1], [0, 1], None, [16, 16], [0, 180, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hist_sim = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
            hist_diff = 1.0 - max(0.0, min(1.0, (hist_sim + 1.0) / 2.0))

            # 2. Downsampled 32x32 spatial luminance difference
            gray1 = cv2.resize(cv2.cvtColor(frame_bgr_1, cv2.COLOR_BGR2GRAY), (32, 32))
            gray2 = cv2.resize(cv2.cvtColor(frame_bgr_2, cv2.COLOR_BGR2GRAY), (32, 32))
            pixel_diff = float(np.mean(np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))) / 255.0)

            # Combined visual diversity score
            diversity = 0.5 * hist_diff + 0.5 * pixel_diff
            return round(float(max(0.0, min(1.0, diversity))), 4)
        except Exception:
            return 0.5

    def filter_keyframes(
        self,
        keyframes: list,
        clip_batch_size: int = CLIP_BATCH_SIZE,
        max_keyframes: int = MAX_KEYFRAMES,
    ):
        """
        Stage 8: Importance-Aware Diverse Keyframe Selection
        1. Quality-prioritized CLIP redundancy filtering (evaluates highest quality frames first).
        2. Neutral novelty initialization (0.5 / None) for the primary reference frame.
        3. Importance scoring based mainly on Visual Quality (0.50) + Visual Novelty (0.50).
        4. Temporal-bin diversity allocation capped at MAX_KEYFRAMES with chronological output ordering.
        """
        if not keyframes:
            return []

        if max_keyframes <= 0:
            for kf in keyframes:
                kf["accepted"] = False
                kf["rejection_reason"] = "MAX_KEYFRAMES_EXCEEDED"
            return []

        if len(keyframes) == 1:
            kf = keyframes[0]
            kf["clip_similarity"] = None
            kf["novelty_score"] = None
            norm_q = max(0.0, min(1.0, float(kf.get("quality_score", 50.0)) / 100.0))
            kf["importance_score"] = round(0.50 * norm_q + 0.50 * 0.50, 4)
            kf["accepted"] = True
            kf["rejection_reason"] = None
            return keyframes

        images = [kf["image"] for kf in keyframes]

        print(
            f"Generating batched CLIP embeddings for {len(images)} representative keyframes (batch_size={clip_batch_size})..."
        )
        with torch.inference_mode():
            embeddings = self.clip_model.encode(
                images,
                batch_size=clip_batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

        # ---------------------------------------------------------------------------------
        # Step 1: Quality-Prioritized Redundancy Filtering (Fix 3 & Fix 1)
        # ---------------------------------------------------------------------------------
        # Sort candidate indices by quality_score descending so highest quality frames
        # are evaluated first and cannot be rejected by prior lower-quality similar frames.
        sorted_indices = sorted(
            range(len(keyframes)),
            key=lambda idx: keyframes[idx].get("quality_score", 0.0),
            reverse=True,
        )

        retained_indices = []
        retained_embeddings = []

        for idx in sorted_indices:
            curr_kf = keyframes[idx]
            curr_embedding = embeddings[idx]

            if len(retained_indices) == 0:
                # Primary reference anchor frame (highest quality frame in the pool)
                # Novelty is treated as neutral/unknown (None) because it has no predecessor to compare against
                curr_kf["accepted"] = True
                curr_kf["rejection_reason"] = None
                curr_kf["clip_similarity"] = None
                curr_kf["novelty_score"] = None
                retained_indices.append(idx)
                retained_embeddings.append(curr_embedding)
            else:
                # Compare against all currently retained (higher quality) frame embeddings
                retained_matrix = torch.stack(retained_embeddings)
                sims = torch.matmul(retained_matrix, curr_embedding)
                max_sim = float(torch.max(sims).item())
                curr_kf["clip_similarity"] = round(max_sim, 4)

                novelty = max(0.0, min(1.0, 1.0 - max_sim))
                curr_kf["novelty_score"] = round(novelty, 4)

                if max_sim > self.similarity_threshold:
                    print(
                        f"Skipping frame {curr_kf['frame_id']} at {curr_kf['mid_time']:.2f}s "
                        f"(Max CLIP sim: {max_sim:.4f} > {self.similarity_threshold}, redundant with higher-quality frame)"
                    )
                    curr_kf["accepted"] = False
                    curr_kf["rejection_reason"] = "CLIP_REDUNDANT"
                else:
                    print(
                        f"Keeping frame {curr_kf['frame_id']} at {curr_kf['mid_time']:.2f}s "
                        f"(Max CLIP sim: {max_sim:.4f} <= {self.similarity_threshold})"
                    )
                    curr_kf["accepted"] = True
                    curr_kf["rejection_reason"] = None
                    retained_indices.append(idx)
                    retained_embeddings.append(curr_embedding)

        retained = [keyframes[i] for i in retained_indices]

        print(
            f"Adaptive Scene Sampling: Retained {len(retained)} / {len(keyframes)} keyframes after quality-prioritized redundancy filtering."
        )

        if not retained:
            return []

        # ---------------------------------------------------------------------------------
        # Step 2: Importance Scoring (Visual Quality + Visual Novelty) (Fix 2)
        # ---------------------------------------------------------------------------------
        # Temporal coverage is excluded from importance score because it is enforced by temporal bins.
        # Importance balances Normalized Quality (0.50) + Visual Novelty (0.50).
        for kf in retained:
            # 1. Normalized Quality Score [0.0, 1.0]
            norm_quality = max(0.0, min(1.0, float(kf.get("quality_score", 50.0)) / 100.0))

            # 2. Visual Novelty Score [0.0, 1.0] (neutral 0.50 if unknown/first frame)
            novelty_val = kf.get("novelty_score")
            visual_novelty = 0.50 if novelty_val is None else max(0.0, min(1.0, float(novelty_val)))

            # Importance calculation: 50% Quality + 50% Novelty
            importance = 0.50 * norm_quality + 0.50 * visual_novelty
            kf["importance_score"] = round(importance, 4)

        # ---------------------------------------------------------------------------------
        # Step 3: Temporal Diversity-Aware Selection (Capping at MAX_KEYFRAMES)
        # ---------------------------------------------------------------------------------
        # Ensure retained keyframes are in chronological order for temporal binning
        retained.sort(key=lambda x: x["mid_time"])

        if len(retained) > max_keyframes:
            print(
                f"Importance-Aware Selection: Capping {len(retained)} retained keyframes to MAX_KEYFRAMES ({max_keyframes}) using temporal diversity bins."
            )

            t_min = retained[0]["mid_time"]
            t_max = retained[-1]["mid_time"]
            duration_span = max(0.001, t_max - t_min)

            num_bins = max_keyframes
            bin_duration = duration_span / num_bins

            # Divide timeline into temporal bins
            temporal_bins = [[] for _ in range(num_bins)]
            for kf in retained:
                b_idx = int((kf["mid_time"] - t_min) / bin_duration)
                b_idx = max(0, min(num_bins - 1, b_idx))
                temporal_bins[b_idx].append(kf)

            selected_keyframes = []
            remaining_pool = []

            # Pass 1: Select top importance frame from each non-empty temporal bin to guarantee broad coverage
            for b_idx in range(num_bins):
                bin_frames = temporal_bins[b_idx]
                if bin_frames:
                    bin_frames_sorted = sorted(
                        bin_frames,
                        key=lambda x: x.get("importance_score", 0.0),
                        reverse=True,
                    )
                    selected_keyframes.append(bin_frames_sorted[0])
                    remaining_pool.extend(bin_frames_sorted[1:])

            # Pass 2: Redistribute unused slots from empty bins by selecting highest-importance remaining candidates
            needed_slots = max_keyframes - len(selected_keyframes)
            if needed_slots > 0 and remaining_pool:
                remaining_pool.sort(
                    key=lambda x: x.get("importance_score", 0.0),
                    reverse=True,
                )
                selected_keyframes.extend(remaining_pool[:needed_slots])

            # Safety guarantee: Ensure exact max_keyframes count
            if len(selected_keyframes) > max_keyframes:
                selected_keyframes.sort(
                    key=lambda x: x.get("importance_score", 0.0),
                    reverse=True,
                )
                selected_keyframes = selected_keyframes[:max_keyframes]

            selected_frame_ids = {kf["frame_id"] for kf in selected_keyframes}

            # Update acceptance metadata on all retained candidates
            for kf in retained:
                if kf["frame_id"] in selected_frame_ids:
                    kf["accepted"] = True
                    kf["rejection_reason"] = None
                else:
                    kf["accepted"] = False
                    kf["rejection_reason"] = "MAX_KEYFRAMES_EXCEEDED"

            retained = selected_keyframes

        # Chronological ordering guarantee
        retained.sort(key=lambda x: x["mid_time"])

        # Free GPU memory after CLIP encoding if applicable
        if DEVICE == "cuda" or torch.cuda.is_available():
            torch.cuda.empty_cache()

        return retained

    def extract_keyframes(
        self,
        video_path: str,
        scene_threshold: float = SCENE_THRESHOLD,
        max_scene_window_sec: float = MAX_SCENE_WINDOW_SEC,
        clip_batch_size: int = CLIP_BATCH_SIZE,
        max_keyframes: int = MAX_KEYFRAMES,
        candidate_samples: int = CANDIDATE_SAMPLES_PER_SCENE,
    ):
        """
        Extracts keyframes using full pipeline:
        Scene Detection -> Candidate Frame Sampling -> Image Quality Evaluation -> Adaptive Diverse Frame Selection -> CLIP Redundancy & Importance Selection -> Final Keyframes
        """
        print(
            f"Detecting scenes in {video_path} (SCENE_THRESHOLD={scene_threshold}, MAX_WINDOW={max_scene_window_sec}s)"
        )
        scene_list = detect(video_path, ContentDetector(threshold=scene_threshold))

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps and fps > 0 else 0

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        scenes_to_process = []
        max_window = max_scene_window_sec

        if not scene_list:
            print(
                f"No scene cuts detected. Sampling candidate frames across {max_window:.1f}s video windows."
            )
            curr = 0.0
            sc_idx = 0
            while curr < duration:
                end_t = min(curr + max_window, duration)
                if end_t > curr:
                    scenes_to_process.append((sc_idx, curr, end_t))
                    sc_idx += 1
                curr += max_window
        else:
            sc_idx = 0
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                # Subdivide long scenes into max_window segments to ensure fine-grained temporal coverage
                curr = start_time
                while curr < end_time:
                    sub_end = min(curr + max_window, end_time)
                    if sub_end > curr:
                        scenes_to_process.append((sc_idx, curr, sub_end))
                        sc_idx += 1
                    curr += max_window

        all_candidate_records = []
        selected_scene_candidates = []
        overall_frame_id = 0

        for sc_idx, start_time, end_time in scenes_to_process:
            scene_duration = max(0.01, end_time - start_time)
            scene_candidates = []

            # Sample candidate_samples timestamps per scene
            if candidate_samples <= 1:
                timestamps = [start_time + scene_duration / 2.0]
            else:
                margin = scene_duration / (candidate_samples + 1)
                timestamps = [start_time + margin * (k + 1) for k in range(candidate_samples)]

            for cand_idx, mid_t in enumerate(timestamps):
                cap.set(cv2.CAP_PROP_POS_MSEC, mid_t * 1000)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Run multi-attribute image quality evaluation
                q_eval = self.evaluate_image_quality(frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                cand_record = {
                    "video_id": video_name,
                    "frame_id": overall_frame_id,
                    "scene_id": sc_idx,
                    "candidate_id": cand_idx,
                    "start_time": start_time,
                    "end_time": end_time,
                    "mid_time": mid_t,
                    "image": pil_img,
                    "sharpness_score": q_eval["sharpness"],
                    "brightness_score": q_eval["brightness"],
                    "contrast_score": q_eval["contrast"],
                    "resolution": q_eval["resolution"],
                    "quality_score": q_eval["quality_score"],
                    "quality_status": q_eval["status"],
                    "clip_similarity": None,
                    "importance_score": None,
                    "novelty_score": None,
                    "accepted": False,
                    "rejection_reason": q_eval["rejection_reason"],
                    "image_path": None,
                    "_frame_bgr": frame,
                }

                scene_candidates.append(cand_record)
                all_candidate_records.append(cand_record)
                overall_frame_id += 1

            if not scene_candidates:
                continue

            # Adaptive representative frame selection per scene
            clear_candidates = [c for c in scene_candidates if c["quality_status"] == "CLEAR"]
            pool = clear_candidates if clear_candidates else scene_candidates
            primary_cand = max(pool, key=lambda c: c["quality_score"])
            scene_selected = [primary_cand]

            # For longer or dynamic scenes (>= 2.0s duration), select a 2nd representative candidate (Fix 4)
            # that combines visual quality, temporal separation, and visual diversity against primary candidate
            if scene_duration >= 2.0 and len(pool) >= 2:
                min_time_gap = max(0.8, scene_duration * 0.25)
                temporally_separated = [
                    c for c in pool
                    if c["frame_id"] != primary_cand["frame_id"]
                    and abs(c["mid_time"] - primary_cand["mid_time"]) >= min_time_gap
                ]
                if temporally_separated:
                    scored_candidates = []
                    for c in temporally_separated:
                        div_score = self.compute_frame_diversity_score(
                            c.get("_frame_bgr"), primary_cand.get("_frame_bgr")
                        )
                        norm_q = max(0.0, min(1.0, float(c.get("quality_score", 50.0)) / 100.0))
                        cand_score = 0.50 * norm_q + 0.50 * div_score
                        scored_candidates.append((c, cand_score, div_score))

                    scored_candidates.sort(key=lambda item: item[1], reverse=True)
                    best_secondary, best_score, best_div = scored_candidates[0]

                    if best_secondary["quality_score"] >= 30.0 and best_div >= 0.10:
                        scene_selected.append(best_secondary)

            # Clean up temporary _frame_bgr references
            for c in scene_candidates:
                c.pop("_frame_bgr", None)

            scene_selected.sort(key=lambda c: c["mid_time"])
            selected_ids = {c["frame_id"] for c in scene_selected}

            # Mark non-selected candidates in the scene
            for c in scene_candidates:
                if c["frame_id"] not in selected_ids:
                    c["accepted"] = False
                    if c["rejection_reason"] is None:
                        c["rejection_reason"] = "LOWER_QUALITY_CANDIDATE_IN_SCENE"

            selected_scene_candidates.extend(scene_selected)

        cap.release()

        # Dynamic keyframe allocation based on video duration
        duration_budget = max(MIN_KEYFRAMES, int(round(duration / DYNAMIC_KEYFRAME_INTERVAL_SEC))) if duration > 0 else MIN_KEYFRAMES
        effective_max_keyframes = min(max_keyframes, duration_budget)
        print(f"Video duration: {duration:.2f}s -> Dynamic Keyframe Target: {effective_max_keyframes} (max_cap={max_keyframes})")

        # Apply CLIP similarity redundancy filtering and importance-aware selection on candidate frames
        accepted_keyframes = self.filter_keyframes(
            selected_scene_candidates,
            clip_batch_size=clip_batch_size,
            max_keyframes=effective_max_keyframes
        )

        self.latest_candidate_metadata = all_candidate_records
        return accepted_keyframes

    def generate_descriptions(
        self, keyframes: list, vlm_batch_size: int = VLM_BATCH_SIZE
    ):
        """Passes accepted keyframes to Vision Model (GPT-4o-mini or local VLM) with structured visual-analysis prompt and object tracking context."""
        if not keyframes:
            return []

        results = []
        num_keyframes = len(keyframes)

        # Step 1: Run RT-DETR + GroundingDINO + SimpleSort Tracking across keyframes sequentially
        tracking_summaries = []
        if self.object_detector and self.object_detector.enabled:
            print(f"Running RT-DETR + GroundingDINO + SimpleSort object detection on {num_keyframes} keyframes...")
            for kf in keyframes:
                pil_img = kf["image"]
                frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                timestamp = kf.get("mid_time", 0.0)
                try:
                    tracked_objs, summary = self.object_detector.process_frame(frame_bgr, timestamp=timestamp)
                except Exception as e:
                    print(f"  [ObjectDetector Warning] Error processing frame at {timestamp:.2f}s: {e}")
                    summary = "None detected."
                tracking_summaries.append(summary)
        else:
            tracking_summaries = ["None detected."] * num_keyframes

        # Step 2: Generate descriptions using GPT-4o-mini if API key is present
        api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        if api_key:
            print(f"Generating visual descriptions using GPT-4o-mini for {num_keyframes} keyframes...")
            all_gpt_success = True
            gpt_results = []
            for kf, track_summary in zip(keyframes, tracking_summaries):
                desc = self.caption_frame_gpt4o_mini(kf["image"], tracking_summary=track_summary)
                if desc:
                    if track_summary and track_summary != "None detected." and track_summary not in desc:
                        desc += f"\n- Tracked Objects & Motion: {track_summary}"
                    print(f"  [Frame {kf['frame_id']} @ {kf['mid_time']:.2f}s]: Generated structured description via GPT-4o-mini.")
                    gpt_results.append(
                        {
                            "type": "visual",
                            "frame_id": kf["frame_id"],
                            "start_time": kf["start_time"],
                            "end_time": kf["end_time"],
                            "text": desc,
                        }
                    )
                else:
                    all_gpt_success = False
                    break

            if all_gpt_success and len(gpt_results) == num_keyframes:
                return gpt_results
            else:
                print("GPT-4o-mini captioning incomplete or failed. Falling back to local VLM...")

        # Step 3: Local VLM Inference Fallback
        self._load_local_vlm()

        for i in range(0, num_keyframes, vlm_batch_size):
            batch_kfs = keyframes[i : i + vlm_batch_size]
            batch_summaries = tracking_summaries[i : i + vlm_batch_size]
            batch_end = min(i + vlm_batch_size, num_keyframes)
            print(
                f"Generating local VLM structured descriptions for keyframes {i+1}-{batch_end}/{num_keyframes} (batch_size={vlm_batch_size})..."
            )

            batch_messages = []
            for kf, track_summary in zip(batch_kfs, batch_summaries):
                img = kf["image"]
                if max(img.size) > VLM_IMAGE_RESIZE_MAX:
                    img_scaled = img.copy()
                    img_scaled.thumbnail((VLM_IMAGE_RESIZE_MAX, VLM_IMAGE_RESIZE_MAX))
                else:
                    img_scaled = img

                prompt = STRUCTURED_VLM_PROMPT
                if track_summary and track_summary != "None detected.":
                    prompt += f"\n\nDetected Visible Objects & Tracks:\n{track_summary}"

                msg = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img_scaled,
                                "min_pixels": MIN_VISION_PIXELS,
                                "max_pixels": MAX_VISION_PIXELS,
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ]
                batch_messages.append(msg)

            # Prepare inputs for batch inference
            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in batch_messages
            ]
            image_inputs, video_inputs = process_vision_info(batch_messages)

            with torch.inference_mode():
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                model_device = getattr(self.model, "device", DEVICE)
                inputs = inputs.to(model_device)

                max_toks = max(VLM_MAX_NEW_TOKENS, 256)
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_toks,
                    do_sample=False,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                output_texts = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            for kf, track_summary, out_text in zip(batch_kfs, batch_summaries, output_texts):
                desc = out_text.strip()
                if track_summary and track_summary != "None detected." and track_summary not in desc:
                    desc += f"\n- Tracked Objects & Motion: {track_summary}"
                print(f"  [Frame {kf['frame_id']} @ {kf['mid_time']:.2f}s]: Generated structured description.")
                results.append(
                    {
                        "type": "visual",
                        "frame_id": kf["frame_id"],
                        "start_time": kf["start_time"],
                        "end_time": kf["end_time"],
                        "text": desc,
                    }
                )

            del inputs, generated_ids, generated_ids_trimmed
            if IS_GPU and (DEVICE == "cuda" or torch.cuda.is_available()):
                torch.cuda.empty_cache()

        return results

    def _extract_structured_field(self, text: str, field_name: str) -> str:
        """Extracts content of a specific section from structured visual descriptions."""
        if not text:
            return ""
        pattern = rf"(?:^|\n)[-•*]?\s*{re.escape(field_name)}[\s:]+([^\n]+(?:\n(?![-•*]|\d+\.)[^\n]+)*)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _is_ollama_available(self) -> bool:
        """Checks if local Ollama server is responsive."""
        try:
            res = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=0.8)
            return res.status_code == 200
        except Exception:
            return False

    def generate_temporal_scene_notes(self, visual_facts: list) -> list:
        """
        Performs fast, high-accuracy temporal reasoning across consecutive keyframes.
        Uses local Ollama text reasoning when available (2-3 seconds) or high-density
        structured semantic diffing (instant) as fallback, avoiding heavy multimodal model bottlenecks.
        """
        if len(visual_facts) < 2:
            return visual_facts

        num_pairs = len(visual_facts) - 1
        print(f"Generating optimized temporal reasoning for {num_pairs} keyframe transition pair(s)...")

        transition_facts = []
        use_ollama = self._is_ollama_available()
        if use_ollama:
            print(f"  [Temporal Reasoning] Using fast local Ollama ({OLLAMA_MODEL}) for temporal transitions.")
        else:
            print("  [Temporal Reasoning] Using high-density structured semantic diffing.")

        for i in range(num_pairs):
            curr_fact = visual_facts[i]
            next_fact = visual_facts[i + 1]

            t_start = curr_fact["start_time"]
            t_end = next_fact["end_time"]

            c_action = self._extract_structured_field(curr_fact["text"], "Actions & Movements")
            n_action = self._extract_structured_field(next_fact["text"], "Actions & Movements")
            c_obj = self._extract_structured_field(curr_fact["text"], "Objects & Spatial Layout")
            n_obj = self._extract_structured_field(next_fact["text"], "Objects & Spatial Layout")

            note_text = None

            # 1. Fast Ollama Text LLM Generation
            if use_ollama:
                try:
                    prompt = (
                        f"You are a Video Action, Spatial & Temporal Transition Analyzer.\n"
                        f"Compare these two consecutive video timeframe observations:\n\n"
                        f"[Frame A at {t_start:.1f}s]:\n{curr_fact['text'][:220]}\n\n"
                        f"[Frame B at {t_end:.1f}s]:\n{next_fact['text'][:220]}\n\n"
                        f"In 1 or 2 concise, direct sentences, describe the transition:\n"
                        f"- What action started, finished, or evolved from Frame A to Frame B.\n"
                        f"- Any visible quadrant shift (top-left, top-right, bottom-left, bottom-right) or trajectory (moved left to right, background to foreground).\n"
                        f"- Any object movement or positional change.\n"
                        f"Output ONLY the concise factual statement."
                    )
                    payload = {
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 60,
                        }
                    }
                    res = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=4.0)
                    if res.status_code == 200:
                        raw_desc = res.json().get("response", "").strip()
                        if raw_desc:
                            note_text = f"Temporal Transition [{t_start:.2f}s to {t_end:.2f}s]:\n{raw_desc}"
                except Exception:
                    pass

            # 2. High-Density Structured Semantic Diffing (Fallback or Standalone)
            if not note_text:
                parts = []
                if c_action and n_action:
                    if c_action.strip().lower() == n_action.strip().lower():
                        parts.append(f"Ongoing action: {c_action}.")
                    else:
                        parts.append(f"Action sequence: After [{c_action}], the action transitions to [{n_action}].")
                elif n_action:
                    parts.append(f"Action begins: {n_action}.")
                elif c_action:
                    parts.append(f"Previous action [{c_action}] concludes.")

                if c_obj and n_obj and c_obj.strip().lower() != n_obj.strip().lower():
                    parts.append(f"Spatial & object state changes from [{c_obj}] to [{n_obj}].")

                if not parts:
                    c_clean = curr_fact["text"].replace("\n", " ").strip()
                    n_clean = next_fact["text"].replace("\n", " ").strip()
                    parts.append(f"State evolves from [{c_clean[:90]}] into [{n_clean[:90]}].")

                diff_summary = " ".join(parts)
                note_text = f"Temporal Transition [{t_start:.2f}s to {t_end:.2f}s]:\n{diff_summary}"

            print(f"  [Temporal Transition {i}->{i+1}]: {note_text.splitlines()[-1][:90]}...")
            transition_facts.append({
                "type": "visual",
                "frame_id": f"trans_{i}_{i+1}",
                "start_time": t_start,
                "end_time": t_end,
                "text": note_text,
            })

        all_visual_facts = visual_facts + transition_facts
        all_visual_facts.sort(key=lambda x: x["start_time"])
        return all_visual_facts

    def process_video(self, video_path: str, save_keyframes: bool = True):
        """Runs the full visual extraction pipeline."""
        keyframes = self.extract_keyframes(video_path)
        if save_keyframes:
            self.save_keyframes_to_disk(video_path, keyframes)
        visual_facts = self.generate_descriptions(keyframes)
        visual_facts_complete = self.generate_temporal_scene_notes(visual_facts)
        return visual_facts_complete

if __name__ == "__main__":
    pass