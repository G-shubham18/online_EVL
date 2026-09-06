import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional

from config import (
    RTDETR_MODEL,
    GROUNDING_DINO_MODEL,
    ENABLE_OBJECT_DETECTION,
    SIMPLYSORT_MAX_AGE,
    SIMPLYSORT_MIN_HITS,
    SIMPLYSORT_IOU_THRESHOLD,
    DEVICE,
    TORCH_DTYPE,
    IS_GPU
)
from stage1_offline.simple_sort import SimpleSort, compute_quadrant

class ObjectDetector:
    """
    Unified Object Detection & Tracking system combining:
    1. RT-DETR: High-speed real-time general object detection
    2. GroundingDINO: Text-guided open-vocabulary zero-shot object detection
    3. SimpleSort: Multi-object Kalman/IoU tracker across video keyframes
    """
    def __init__(self):
        self.enabled = ENABLE_OBJECT_DETECTION
        self.tracker = SimpleSort(
            max_age=SIMPLYSORT_MAX_AGE,
            min_hits=SIMPLYSORT_MIN_HITS,
            iou_threshold=SIMPLYSORT_IOU_THRESHOLD
        )
        self.rtdetr_model = None
        self.rtdetr_processor = None
        self.dino_model = None
        self.dino_processor = None
        self._models_loaded = False

    def reset_tracker(self):
        """Resets the tracking state for a new video."""
        self.tracker.reset()

    def _lazy_init(self):
        """Lazily loads RT-DETR and GroundingDINO models on first use."""
        if self._models_loaded or not self.enabled:
            return

        # 1. Load RT-DETR
        try:
            print(f"[ObjectDetector] Loading Real-Time Detector: {RTDETR_MODEL}...")
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            self.rtdetr_processor = AutoImageProcessor.from_pretrained(RTDETR_MODEL)
            kwargs = {"torch_dtype": TORCH_DTYPE} if IS_GPU else {}
            self.rtdetr_model = AutoModelForObjectDetection.from_pretrained(RTDETR_MODEL, **kwargs).to(DEVICE)
            self.rtdetr_model.eval()
            print(f"[ObjectDetector] Successfully loaded RT-DETR on {DEVICE}.")
        except Exception as e:
            print(f"[ObjectDetector Warning] Could not load RT-DETR ({e}). Skipping RT-DETR branch.")
            self.rtdetr_model = None

        # 2. Load GroundingDINO
        try:
            print(f"[ObjectDetector] Loading Text-Guided Detector: {GROUNDING_DINO_MODEL}...")
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.dino_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
            kwargs = {"torch_dtype": TORCH_DTYPE} if IS_GPU else {}
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL, **kwargs).to(DEVICE)
            self.dino_model.eval()
            print(f"[ObjectDetector] Successfully loaded GroundingDINO on {DEVICE}.")
        except Exception as e:
            print(f"[ObjectDetector Warning] Could not load GroundingDINO ({e}). Skipping GroundingDINO branch.")
            self.dino_model = None

        self._models_loaded = True

    def detect_rtdetr(self, pil_img: Image.Image, confidence_threshold: float = 0.35) -> List[Dict[str, Any]]:
        """Runs RT-DETR inference on image and returns detected bounding boxes."""
        if self.rtdetr_model is None or self.rtdetr_processor is None:
            return []
        try:
            inputs = self.rtdetr_processor(images=pil_img, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                outputs = self.rtdetr_model(**inputs)
            
            target_sizes = torch.tensor([pil_img.size[::-1]]).to(DEVICE)
            results = self.rtdetr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=confidence_threshold
            )[0]

            w, h = pil_img.size
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]  # [x1, y1, x2, y2]
                class_name = self.rtdetr_model.config.id2label.get(label.item(), f"obj_{label.item()}")
                norm_box = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
                detections.append({
                    "bbox": norm_box,
                    "score": float(score.item()),
                    "class_name": class_name,
                    "source": "RT-DETR"
                })
            return detections
        except Exception as e:
            print(f"[ObjectDetector Warning] RT-DETR inference error: {e}")
            return []

    def detect_grounding_dino(
        self, 
        pil_img: Image.Image, 
        text_queries: str = "person . chair . table . musical instrument . mirror . shelf . clothing . cup . guitar . violin .",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """Runs GroundingDINO open-vocabulary detection based on text prompt."""
        if self.dino_model is None or self.dino_processor is None:
            return []
        try:
            inputs = self.dino_processor(images=pil_img, text=text_queries, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                outputs = self.dino_model(**inputs)

            target_sizes = torch.tensor([pil_img.size[::-1]]).to(DEVICE)
            results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes
            )[0]

            w, h = pil_img.size
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                norm_box = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
                detections.append({
                    "bbox": norm_box,
                    "score": float(score.item()),
                    "class_name": label.strip(),
                    "source": "GroundingDINO"
                })
            return detections
        except Exception as e:
            print(f"[ObjectDetector Warning] GroundingDINO inference error: {e}")
            return []

    def nms_filter(self, detections: List[Dict[str, Any]], iou_thresh: float = 0.5) -> List[Dict[str, Any]]:
        """Applies Non-Maximum Suppression to remove overlapping duplicate detections."""
        if not detections:
            return []
        
        sorted_dets = sorted(detections, key=lambda x: x["score"], reverse=True)
        selected = []

        for det in sorted_dets:
            keep = True
            for sel in selected:
                # compute IoU
                b1 = det["bbox"]
                b2 = sel["bbox"]
                xx1 = max(b1[0], b2[0])
                yy1 = max(b1[1], b2[1])
                xx2 = min(b1[2], b2[2])
                yy2 = min(b1[3], b2[3])
                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union = a1 + a2 - inter
                iou = (inter / union) if union > 0 else 0.0

                if iou > iou_thresh and det["class_name"].lower() == sel["class_name"].lower():
                    keep = False
                    break
            if keep:
                selected.append(det)

        return selected

    def process_frame(
        self, 
        frame_bgr: np.ndarray, 
        timestamp: float,
        text_queries: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Runs RT-DETR + GroundingDINO detection and SimpleSort tracking on a frame.
        Returns:
            (tracked_objects: List[Dict], tracking_summary_string: str)
        """
        if not self.enabled:
            return [], ""

        self._lazy_init()

        # Convert to PIL Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        all_detections = []

        # 1. Real-time RT-DETR detection
        rtdetr_dets = self.detect_rtdetr(pil_img)
        all_detections.extend(rtdetr_dets)

        # 2. Text-guided GroundingDINO detection
        dino_prompt = text_queries if text_queries else "person . chair . table . musical instrument . mirror . shelf . clothing . cup . guitar . violin ."
        dino_dets = self.detect_grounding_dino(pil_img, text_queries=dino_prompt)
        all_detections.extend(dino_dets)

        # 3. Fuse & deduplicate detections via NMS
        fused_detections = self.nms_filter(all_detections, iou_thresh=0.5)

        # 4. Track objects across frames with SimpleSort
        tracked_objects = self.tracker.update(fused_detections, timestamp=timestamp)

        # 5. Format tracking summary text
        summary = self.tracker.format_tracking_summary(tracked_objects)

        return tracked_objects, summary
