import os
import torch

# Base Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_stores")

# Create data directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Hardware Configuration & Dual Mode (GPU if available, else optimized CPU fallback)
IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available() if hasattr(torch, "xpu") else False

IS_DML_AVAILABLE = False
try:
    import torch_directml
    if torch_directml.is_available():
        IS_DML_AVAILABLE = True
except ImportError:
    IS_DML_AVAILABLE = False

if IS_CUDA_AVAILABLE:
    DEVICE = "cuda"
elif IS_DML_AVAILABLE:
    import torch_directml
    DEVICE = torch_directml.device()
elif IS_XPU_AVAILABLE:
    DEVICE = "xpu"
else:
    DEVICE = "cpu"

IS_GPU = DEVICE != "cpu"

# Precision Configuration
if DEVICE == "cuda":
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
elif IS_GPU:
    TORCH_DTYPE = torch.float16
else:
    TORCH_DTYPE = torch.float32

print(f"[Hardware Setup] Pipeline Running in Dual Mode: {'GPU (' + str(DEVICE) + ')' if IS_GPU else 'CPU (Optimized Fallback Mode)'}")
print(f"[Hardware Setup] Selected Data Type: {TORCH_DTYPE}")

# Model Identifiers (Configurable via Environment Variables for Cloud/Colab/Kaggle)
# Multi-Model Benchmark Configurations
SUPPORTED_LLM_MODELS = {
    "qwen2.5-v1-72b-instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    "qwen2.5-vl-72b-instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    "gemma-4-31b": "google/gemma-4-31b-it",
    "gemma-2-27b": "google/gemma-2-27b-it",
    "phi-3.5-vision-instruct": "microsoft/Phi-3.5-vision-instruct",
}

SUPPORTED_CAPTION_MODELS = [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    "wraps/moondream-caption",
]

SUPPORTED_AUDIO_EMBED_MODELS = [
    "FacebookAI/roberta-base",
    "laion/clap-htsat-unfused",
]

# API Keys & Serverless Endpoints
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACEHUB_API_TOKEN", "")))
USE_API = os.getenv("USE_API", "1").lower() in ("1", "true", "yes")
LLM_API_BASE = os.getenv("LLM_API_BASE", os.getenv("OPENAI_BASE_URL", ""))
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", "")))

# Active Model Selectors (Configurable via CLI or Env Vars)
ACTIVE_LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-v1-72b-instruct")
ACTIVE_CAPTION_MODEL = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
ACTIVE_AUDIO_EMBED_MODEL = os.getenv("AUDIO_EMBED_MODEL", "FacebookAI/roberta-base")

# Stage 1: Extraction & Indexing
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")  # Timestamped speech transcription
AST_MODEL = os.getenv("AST_MODEL", "MIT/ast-finetuned-audioset-10-10-0.4593")  # Audio Spectrogram Transformer
CLAP_MODEL = os.getenv("CLAP_MODEL", "laion/clap-htsat-unfused")  # Optional CLAP fallback
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")  # Visual keyframe selection
RTDETR_MODEL = os.getenv("RTDETR_MODEL", "PekingU/rtdetr_r50vd")  # Real-time object detection
GROUNDING_DINO_MODEL = os.getenv("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny")  # Text-based object detection
ENABLE_OBJECT_DETECTION = os.getenv("ENABLE_OBJECT_DETECTION", "1").lower() in ("1", "true", "yes")
SIMPLYSORT_MAX_AGE = int(os.getenv("SIMPLYSORT_MAX_AGE", "5"))
SIMPLYSORT_MIN_HITS = int(os.getenv("SIMPLYSORT_MIN_HITS", "1"))
SIMPLYSORT_IOU_THRESHOLD = float(os.getenv("SIMPLYSORT_IOU_THRESHOLD", "0.3"))

# Vision & Language Models
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")  # Frame captioning, query classification & answer generation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")  # Local VLM fallback
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.90"))

# Stage 2: Dense Text Embeddings & BM25 Keyword Re-Ranking
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")  # SOTA 1024-dim dense embedder
MODALITY_ESTIMATOR_MODEL = os.getenv("MODALITY_ESTIMATOR_MODEL", "BAAI/bge-large-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
ENABLE_BM25 = os.getenv("ENABLE_BM25", "1").lower() in ("1", "true", "yes")
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))

# Stage 3: Generation (GPT-4o-mini primary with Ollama/HF in-process fallback)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LLM_BACKEND = os.getenv("LLM_BACKEND", "api" if (HF_TOKEN or LLM_API_KEY or os.getenv("OPENAI_API_KEY")) else "auto")  # "api", "gpt", "auto", "ollama", "hf"

# Ingestion concurrency (Sequential default on single GPU like T4 prevents VRAM spikes)
CONCURRENT_INGESTION = os.getenv("CONCURRENT_INGESTION", "0").lower() in ("1", "true", "yes")

# Retrieval & Threshold Configurations
CHROMA_AUDIO_COLLECTION = "audio_collection"
FAISS_VISUAL_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "visual_index.faiss")
VISUAL_METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "visual_metadata.json")
FAISS_JOINT_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "joint_index.faiss")
JOINT_METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "joint_metadata.json")

# Hyperparameters
KA_DEFAULT = 5 # Default number of audio chunks to retrieve
KV_DEFAULT = 10 # Default number of visual chunks to retrieve
NMS_OVERLAP_THRESHOLD = 0.8 # 80% time overlap drops redundant lower-scoring evidence
AUDIO_BONUS = 1.0 # Added to reranker score if candidate is audio and beta(q) is high
TEMPORAL_AGREEMENT_BONUS = 0.5
MIN_AUDIO_EVIDENCE_THRESHOLD = 1
MODALITY_THRESHOLD = 0.6 # If beta(q) > 0.6, it's considered an audio-heavy question

# Ablation Configuration Presets & Modes
STORE_MODES = ["separated", "joint"]
RERANK_MODES = ["full", "no_audio_lift", "off"]
SUFFICIENCY_MODES = ["adaptive", "fixed"]
MODALITY_MODES = ["audio_visual", "visual_only"]
ABLATION_PRESETS = ["none", "joint_store", "no_rerank", "no_audio_lift", "fixed_cutoff", "no_audio", "visual_only"]

# Audio Extraction Configurations
AUDIO_CHUNK_LENGTH = 3.0 # Duration of each audio chunk in seconds
AUDIO_OVERLAP = 1.0 # Overlap between consecutive audio chunks in seconds
AUDIO_THRESHOLD = 0.3 # Confidence threshold for CLAP detection
AUDIO_BATCH_SIZE = 16 if IS_GPU else 4 # Adaptive batch size for CLAP inference
MIN_EVENT_DURATION = 0.5 # Minimum duration for sound events in seconds
AUDIO_RMS_THRESHOLD = 0.001 # RMS energy threshold to skip silent/low-energy chunks
STEREO_BALANCE_THRESHOLD = 0.15 # Energy ratio threshold for Left vs Right channel sound source localization

# Visual Extraction Configurations
SCENE_THRESHOLD = 27.0 # PySceneDetect ContentDetector threshold
MAX_SCENE_WINDOW_SEC = 4.0 # Maximum time window per keyframe segment to capture intra-scene actions
CLIP_BATCH_SIZE = 32 if IS_GPU else 8 # Batch size for CLIP image embedding generation
VLM_BATCH_SIZE = 1 # Batch size for Vision Model inference
MIN_KEYFRAMES = 8 # Minimum keyframe floor for short videos
MAX_KEYFRAMES = 60 if IS_GPU else 30 # Maximum keyframe cap for long videos (optimized for latency)
DYNAMIC_KEYFRAME_INTERVAL_SEC = 2.5 # 1 keyframe target per 2.5s of video

# Dual Mode Vision Optimization Settings
# Capping vision resolution and max output tokens on CPU prevents 50-minute delays
MIN_VISION_PIXELS = 256 * 14 * 14 if not IS_GPU else 256 * 28 * 28
MAX_VISION_PIXELS = 384 * 14 * 14 if not IS_GPU else 512 * 28 * 28
VLM_MAX_NEW_TOKENS = 128 if not IS_GPU else 256
VLM_IMAGE_RESIZE_MAX = 512 if not IS_GPU else 768
OLLAMA_MAX_TOKENS = 30 # Generation max new tokens for concise QA answers

# Visual Extraction Image Quality Check Configurations
BLUR_THRESHOLD = 100.0 # Laplacian variance threshold below which image is deemed blurry
DARK_THRESHOLD = 15.0 # Mean pixel brightness threshold below which image is deemed dark/black
BRIGHTNESS_MAX_THRESHOLD = 240.0 # Upper brightness threshold above which frame is deemed overexposed
LOW_CONTRAST_THRESHOLD = 20.0 # Standard deviation of pixel intensities threshold below which frame is low contrast
MIN_FRAME_WIDTH = 128 # Minimum frame width requirement in pixels
MIN_FRAME_HEIGHT = 128 # Minimum frame height requirement in pixels
CANDIDATE_SAMPLES_PER_SCENE = 5 # Number of candidate frames sampled per detected scene

KEYFRAMES_SAVE_DIR = os.path.join(DATA_DIR, "keyframes")
os.makedirs(KEYFRAMES_SAVE_DIR, exist_ok=True)







