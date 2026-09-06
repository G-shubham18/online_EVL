import numpy as np
from typing import List, Dict, Any, Tuple, Optional

def compute_iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    wh = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    denom = area_test + area_gt - wh
    if denom <= 0:
        return 0.0
    return float(wh / denom)

def compute_quadrant(bbox: List[float], img_w: float = 1.0, img_h: float = 1.0) -> str:
    """Determines 2D quadrant (Top-Left, Top-Right, Bottom-Left, Bottom-Right, or Center)."""
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    
    # Normalize if coordinates are pixels
    norm_x = cx / img_w if img_w > 1.0 else cx
    norm_y = cy / img_h if img_h > 1.0 else cy

    if 0.35 <= norm_x <= 0.65 and 0.35 <= norm_y <= 0.65:
        return "Center"
    
    x_part = "Left" if norm_x < 0.5 else "Right"
    y_part = "Top" if norm_y < 0.5 else "Bottom"
    return f"{y_part}-{x_part}"

class Track:
    """Represents a single persistent object track across video frames."""
    _count = 0

    def __init__(self, bbox: List[float], score: float, class_name: str, timestamp: float):
        Track._count += 1
        self.track_id = Track._count
        self.class_name = class_name
        self.score = score
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.timestamps = [timestamp]
        self.history = [bbox]
        self.hits = 1
        self.time_since_update = 0

    def update(self, bbox: List[float], score: float, timestamp: float):
        self.bbox = bbox
        self.score = score
        self.timestamps.append(timestamp)
        self.history.append(bbox)
        self.hits += 1
        self.time_since_update = 0

    def get_trajectory(self) -> Dict[str, Any]:
        """Calculates trajectory, motion vector, and direction across frames."""
        if len(self.history) < 2:
            return {
                "movement": "stationary",
                "dx": 0.0,
                "dy": 0.0,
                "start_quadrant": compute_quadrant(self.history[0]),
                "current_quadrant": compute_quadrant(self.history[-1])
            }
        
        first_box = self.history[0]
        last_box = self.history[-1]
        
        c_first_x = (first_box[0] + first_box[2]) / 2.0
        c_first_y = (first_box[1] + first_box[3]) / 2.0
        c_last_x = (last_box[0] + last_box[2]) / 2.0
        c_last_y = (last_box[1] + last_box[3]) / 2.0
        
        dx = c_last_x - c_first_x
        dy = c_last_y - c_first_y
        
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.05:
            direction = "stationary"
        elif abs(dx) > abs(dy) * 1.5:
            direction = "moved from left to right" if dx > 0 else "moved from right to left"
        elif abs(dy) > abs(dx) * 1.5:
            direction = "moved from top to bottom" if dy > 0 else "moved from bottom to top"
        else:
            x_dir = "right" if dx > 0 else "left"
            y_dir = "bottom" if dy > 0 else "top"
            direction = f"moved diagonally towards {y_dir}-{x_dir}"

        return {
            "movement": direction,
            "dx": round(float(dx), 3),
            "dy": round(float(dy), 3),
            "start_quadrant": compute_quadrant(first_box),
            "current_quadrant": compute_quadrant(last_box)
        }

class SimpleSort:
    """
    Simple Online and Realtime Tracking (SimpleSort / SORT).
    Associates object detections across video keyframes using IoU matching,
    assigning persistent track IDs and calculating spatial movement trajectories.
    """
    def __init__(self, max_age: int = 5, min_hits: int = 1, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        Track._count = 0  # Reset track ID counter for each video

    def reset(self):
        """Resets tracker state for a new video."""
        self.tracks = []
        Track._count = 0

    def update(self, detections: List[Dict[str, Any]], timestamp: float) -> List[Dict[str, Any]]:
        """
        Updates tracks with detections from current frame.
        detections: list of dicts: {"bbox": [x1, y1, x2, y2], "score": float, "class_name": str}
        Returns list of active tracked object states with persistent track IDs.
        """
        for t in self.tracks:
            t.time_since_update += 1

        matched_indices = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        if len(self.tracks) > 0 and len(detections) > 0:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, trk in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    # Prefer matching objects of the same class
                    base_iou = compute_iou(np.array(trk.bbox), np.array(det["bbox"]))
                    if trk.class_name.lower() == det["class_name"].lower():
                        iou_matrix[t_idx, d_idx] = base_iou + 0.1  # Bonus for class match
                    else:
                        iou_matrix[t_idx, d_idx] = base_iou

            # Greedy matching by maximum IoU
            while True:
                if iou_matrix.size == 0 or np.max(iou_matrix) < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if iou_matrix[t_idx, d_idx] < self.iou_threshold:
                    break
                matched_indices.append((t_idx, d_idx))
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0
                if t_idx in unmatched_tracks:
                    unmatched_tracks.remove(t_idx)
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)

        # Update matched tracks
        for t_idx, d_idx in matched_indices:
            det = detections[d_idx]
            self.tracks[t_idx].update(det["bbox"], det["score"], timestamp)

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_track = Track(det["bbox"], det["score"], det["class_name"], timestamp)
            self.tracks.append(new_track)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Format output
        active_tracked = []
        for t in self.tracks:
            if t.hits >= self.min_hits and t.time_since_update == 0:
                traj = t.get_trajectory()
                active_tracked.append({
                    "track_id": t.track_id,
                    "class_name": t.class_name,
                    "score": round(float(t.score), 3),
                    "bbox": [round(float(c), 3) for c in t.bbox],
                    "quadrant": compute_quadrant(t.bbox),
                    "trajectory": traj["movement"],
                    "dx": traj["dx"],
                    "dy": traj["dy"],
                    "hits": t.hits
                })

        return active_tracked

    def format_tracking_summary(self, tracked_objects: List[Dict[str, Any]]) -> str:
        """Formats tracked objects into a concise factual string for scene descriptions."""
        if not tracked_objects:
            return "None detected."
        lines = []
        for obj in tracked_objects:
            t_id = obj["track_id"]
            c_name = obj["class_name"]
            quad = obj["quadrant"]
            traj = obj["trajectory"]
            lines.append(f"{c_name} [Track #{t_id}]: located in {quad}, {traj}")
        return "; ".join(lines)
