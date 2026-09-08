import os
import json
import numpy as np
import chromadb
import faiss
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
import torch

from config import (
    VECTOR_STORE_DIR, CHROMA_AUDIO_COLLECTION, 
    FAISS_VISUAL_INDEX_PATH, VISUAL_METADATA_PATH,
    CLAP_MODEL, CLIP_MODEL, TEXT_EMBEDDING_MODEL, DEVICE,
    ACTIVE_AUDIO_EMBED_MODEL, SUPPORTED_AUDIO_EMBED_MODELS,
    HF_TOKEN, USE_API
)

def is_video_indexed_on_disk(video_path: str, store_dir: str) -> bool:
    """Fast check on disk without loading heavy models or ChromaDB."""
    if not os.path.exists(store_dir):
        return False
    info_path = os.path.join(store_dir, "indexed_video.json")
    if not os.path.exists(info_path):
        return False
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            indexed_data = json.load(f)
            indexed_vid = indexed_data.get("video_path")
            if not indexed_vid:
                return False
            norm_target = os.path.normpath(os.path.abspath(video_path)).lower()
            norm_indexed = os.path.normpath(os.path.abspath(indexed_vid)).lower()
            if norm_target == norm_indexed or os.path.basename(norm_target) == os.path.basename(norm_indexed):
                faiss_path = os.path.join(store_dir, "visual_index.faiss")
                chroma_dir = os.path.join(store_dir, "chroma")
                if os.path.exists(faiss_path) or os.path.exists(chroma_dir):
                    return True
    except Exception:
        return False
    return False

class VectorIndexer:
    _shared_clap_model = None
    _shared_clap_processor = None
    _shared_audio_embedders = {}
    _shared_visual_embedder = None
    _models_initialized = False

    @classmethod
    def _init_shared_models(cls, audio_embed_model: str = None):
        if audio_embed_model is None:
            audio_embed_model = os.getenv("AUDIO_EMBED_MODEL", ACTIVE_AUDIO_EMBED_MODEL)

        # Audio Embedding Model (e.g. FacebookAI/roberta-base or CLAP)
        if audio_embed_model not in cls._shared_audio_embedders:
            if "clap" in audio_embed_model.lower():
                print(f"Loading CLAP Model for Audio Store: {audio_embed_model}")
                try:
                    from transformers import ClapModel, ClapProcessor
                    cls._shared_clap_model = ClapModel.from_pretrained(audio_embed_model).to(DEVICE)
                    cls._shared_clap_processor = ClapProcessor.from_pretrained(audio_embed_model)
                    cls._shared_audio_embedders[audio_embed_model] = "clap"
                    print(f"[VectorIndexer] Successfully loaded CLAP text encoder for audio store.")
                except Exception as e:
                    print(f"[VectorIndexer Warning] Could not load CLAP via transformers: {e}. Falling back to sentence-transformers.")
                    cls._shared_audio_embedders[audio_embed_model] = SentenceTransformer(TEXT_EMBEDDING_MODEL, device=DEVICE)
            else:
                print(f"Loading Audio Embedding Model ({audio_embed_model})...")
                try:
                    cls._shared_audio_embedders[audio_embed_model] = SentenceTransformer(audio_embed_model, device=DEVICE)
                    print(f"[VectorIndexer] Successfully loaded Audio Embedding Model: {audio_embed_model}")
                except Exception as e:
                    print(f"[VectorIndexer Warning] SentenceTransformer could not load {audio_embed_model} ({e}). Falling back to {TEXT_EMBEDDING_MODEL}.")
                    cls._shared_audio_embedders[audio_embed_model] = SentenceTransformer(TEXT_EMBEDDING_MODEL, device=DEVICE)

        # High-Speed SOTA Visual Text Embedding Model
        if cls._shared_visual_embedder is None:
            print(f"Loading Dense Semantic Text Embedder for Visual Store: {TEXT_EMBEDDING_MODEL}")
            cls._shared_visual_embedder = SentenceTransformer(TEXT_EMBEDDING_MODEL, device=DEVICE)

    def __init__(self, store_dir: str = None, audio_embed_model: str = None, use_api: bool = None):
        self.store_dir = os.path.abspath(store_dir) if store_dir else VECTOR_STORE_DIR
        os.makedirs(self.store_dir, exist_ok=True)
        
        self.audio_embed_model = audio_embed_model if audio_embed_model else os.getenv("AUDIO_EMBED_MODEL", ACTIVE_AUDIO_EMBED_MODEL)
        self.use_api = use_api if use_api is not None else USE_API
        self.hf_token = HF_TOKEN
        self._hf_client = None

        self.faiss_path = os.path.join(self.store_dir, "visual_index.faiss")
        self.metadata_path = os.path.join(self.store_dir, "visual_metadata.json")
        self.joint_faiss_path = os.path.join(self.store_dir, "joint_index.faiss")
        self.joint_metadata_path = os.path.join(self.store_dir, "joint_metadata.json")
        self.video_info_path = os.path.join(self.store_dir, "indexed_video.json")
        self.chroma_dir = os.path.join(self.store_dir, "chroma")

        self._init_shared_models(self.audio_embed_model)
        self.clap_model = self._shared_clap_model
        self.clap_processor = self._shared_clap_processor
        self.audio_embedder = self._shared_audio_embedders.get(self.audio_embed_model)
        self.visual_embedder = self._shared_visual_embedder

        # Initialize ChromaDB for Audio
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        self.audio_collection = self.chroma_client.get_or_create_collection(name=CHROMA_AUDIO_COLLECTION)

        # Initialize FAISS for Visual
        get_dim_fn = getattr(self.visual_embedder, "get_embedding_dimension", getattr(self.visual_embedder, "get_sentence_embedding_dimension", None))
        self.visual_dim = get_dim_fn() if get_dim_fn else 384
        if os.path.exists(self.faiss_path) and os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.visual_metadata = json.load(f)
                self.visual_index = faiss.read_index(self.faiss_path)
                
                # If index dimension changed, rebuild FAISS index in milliseconds from cached visual_metadata
                if self.visual_index.d != self.visual_dim:
                    print(f"[VectorIndexer] Migrating FAISS visual index dimension ({self.visual_index.d} -> {self.visual_dim}) at '{os.path.basename(self.store_dir)}'...")
                    self.visual_index = faiss.IndexFlatIP(self.visual_dim)
                    if self.visual_metadata:
                        texts = [m["text"] for m in self.visual_metadata]
                        embeddings = self.visual_embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                        self.visual_index.add(embeddings.astype("float32"))
                        faiss.write_index(self.visual_index, self.faiss_path)
            except Exception as e:
                print(f"[VectorIndexer Warning] Re-initializing FAISS index due to: {e}")
                self.visual_index = faiss.IndexFlatIP(self.visual_dim)
                self.visual_metadata = []
        else:
            self.visual_index = faiss.IndexFlatIP(self.visual_dim) # Inner product for cosine sim
            self.visual_metadata = []

        # Initialize FAISS for Unified Joint Store (Audio + Visual merged)
        if os.path.exists(self.joint_faiss_path) and os.path.exists(self.joint_metadata_path):
            try:
                with open(self.joint_metadata_path, "r", encoding="utf-8") as f:
                    self.joint_metadata = json.load(f)
                self.joint_index = faiss.read_index(self.joint_faiss_path)
            except Exception as e:
                print(f"[VectorIndexer Warning] Re-initializing FAISS joint index due to: {e}")
                self.joint_index = faiss.IndexFlatIP(self.visual_dim)
                self.joint_metadata = []
        else:
            self.joint_index = faiss.IndexFlatIP(self.visual_dim)
            self.joint_metadata = []

    def clear_index(self):
        """Clears all stored audio, visual, and joint vector indices and metadata."""
        print(f"[VectorIndexer] Clearing vector store at '{self.store_dir}'...")
        try:
            self.chroma_client.delete_collection(name=CHROMA_AUDIO_COLLECTION)
        except Exception:
            pass
        self.audio_collection = self.chroma_client.get_or_create_collection(name=CHROMA_AUDIO_COLLECTION)

        self.visual_index = faiss.IndexFlatIP(self.visual_dim)
        self.visual_metadata = []
        self.joint_index = faiss.IndexFlatIP(self.visual_dim)
        self.joint_metadata = []

        if os.path.exists(self.faiss_path):
            os.remove(self.faiss_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        if os.path.exists(self.joint_faiss_path):
            os.remove(self.joint_faiss_path)
        if os.path.exists(self.joint_metadata_path):
            os.remove(self.joint_metadata_path)
        if os.path.exists(self.video_info_path):
            os.remove(self.video_info_path)

    def get_indexed_video(self):
        """Returns the absolute path of the video currently indexed, or None if empty."""
        if os.path.exists(self.video_info_path):
            try:
                with open(self.video_info_path, "r") as f:
                    return json.load(f).get("video_path")
            except Exception:
                return None
        return None

    def set_indexed_video(self, video_path: str):
        """Saves the absolute path of the newly indexed video along with metadata."""
        with open(self.video_info_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_path": os.path.abspath(video_path),
                "audio_embed_model": self.audio_embed_model
            }, f, indent=2)

    def is_indexed(self, video_path: str) -> bool:
        """Returns True if the specified video is already indexed in this store."""
        indexed_vid = self.get_indexed_video()
        if indexed_vid:
            norm_target = os.path.normpath(os.path.abspath(video_path)).lower()
            norm_indexed = os.path.normpath(os.path.abspath(indexed_vid)).lower()
            if norm_target == norm_indexed or os.path.basename(norm_target) == os.path.basename(norm_indexed):
                if os.path.exists(self.video_info_path) and (os.path.exists(self.faiss_path) or os.path.exists(self.chroma_dir)):
                    return True
        return False

    def embed_audio_text(self, text: str):
        """Embeds audio text facts using configured model (e.g. FacebookAI/roberta-base, CLAP, or SentenceTransformer)."""
        embed = None
        # 1. Try Hugging Face API feature extraction if enabled
        if self.use_api and self.hf_token and "clap" not in self.audio_embed_model.lower():
            try:
                from huggingface_hub import InferenceClient
                if self._hf_client is None:
                    self._hf_client = InferenceClient(token=self.hf_token)
                res = self._hf_client.feature_extraction(text=text, model=self.audio_embed_model)
                if isinstance(res, np.ndarray) and res.size > 0:
                    embed = res.flatten()
            except Exception:
                pass

        # 2. Local Model Inference
        if embed is None:
            if "clap" in self.audio_embed_model.lower() and getattr(self, "clap_model", None) is not None and getattr(self, "clap_processor", None) is not None:
                inputs = self.clap_processor(text=text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                with torch.no_grad():
                    outputs = self.clap_model.get_text_features(**inputs)
                    if isinstance(outputs, torch.Tensor):
                        tensor = outputs
                    elif hasattr(outputs, "text_embeds"):
                        tensor = outputs.text_embeds
                    elif hasattr(outputs, "pooler_output"):
                        tensor = outputs.pooler_output
                    else:
                        tensor = outputs[0]
                embed = tensor.cpu().numpy()[0]
            elif self.audio_embedder is not None and hasattr(self.audio_embedder, "encode"):
                embed = self.audio_embedder.encode(text, convert_to_numpy=True)
            else:
                embed = self.visual_embedder.encode(text, convert_to_numpy=True)

        norm = np.linalg.norm(embed)
        if norm > 0:
            embed = embed / norm
        return embed.astype("float32")

    def embed_visual_text(self, text: str):
        """Embeds text descriptions using dense semantic text embedder with L2 normalization."""
        embed = self.visual_embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embed.astype("float32")

    def index_audio_facts(self, audio_facts: list):
        """Indexes audio transcripts and sound events into ChromaDB."""
        if not audio_facts:
            return
            
        print(f"Indexing {len(audio_facts)} audio facts into ChromaDB...")
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for i, fact in enumerate(audio_facts):
            fact_id = f"audio_{i}"
            text = fact["text"]
            
            ids.append(fact_id)
            documents.append(text)
            embeddings.append(self.embed_audio_text(text).tolist())
            
            metadata = {
                "type": fact["type"],
                "start_time": fact["start_time"],
                "end_time": fact["end_time"]
            }
            if "score" in fact:
                metadata["score"] = fact["score"]
            metadatas.append(metadata)
            
        self.audio_collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def index_visual_facts(self, visual_facts: list):
        """Indexes visual descriptions into FAISS."""
        if not visual_facts:
            return
            
        print(f"Indexing {len(visual_facts)} visual facts into FAISS...")
        
        embeddings = []
        
        for fact in visual_facts:
            text = fact["text"]
            embed = self.embed_visual_text(text)
            embeddings.append(embed)
            
            self.visual_metadata.append({
                "type": "visual",
                "frame_id": fact["frame_id"],
                "start_time": fact["start_time"],
                "end_time": fact["end_time"],
                "text": text
            })
            
        # Add to FAISS
        embeddings_np = np.vstack(embeddings).astype('float32')
        self.visual_index.add(embeddings_np)
        
        # Save to disk
        faiss.write_index(self.visual_index, self.faiss_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self.visual_metadata, f)

    def index_joint_facts(self, audio_facts: list, visual_facts: list):
        """Indexes combined audio and visual facts into a unified FAISS index for ablation."""
        combined_facts = []
        if audio_facts:
            for fact in audio_facts:
                item = {
                    "type": fact.get("type", "sound"),
                    "start_time": fact.get("start_time", 0.0),
                    "end_time": fact.get("end_time", 0.0),
                    "text": fact.get("text", "")
                }
                if "score" in fact:
                    item["score"] = fact["score"]
                combined_facts.append(item)
        if visual_facts:
            for fact in visual_facts:
                item = {
                    "type": "visual",
                    "frame_id": fact.get("frame_id", 0),
                    "start_time": fact.get("start_time", 0.0),
                    "end_time": fact.get("end_time", 0.0),
                    "text": fact.get("text", "")
                }
                combined_facts.append(item)

        if not combined_facts:
            return

        print(f"Indexing {len(combined_facts)} combined facts into Unified Joint Store...")
        texts = [f["text"] for f in combined_facts]
        embeddings = self.visual_embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        self.joint_index = faiss.IndexFlatIP(self.visual_dim)
        self.joint_index.add(embeddings.astype("float32"))
        self.joint_metadata = combined_facts

        faiss.write_index(self.joint_index, self.joint_faiss_path)
        with open(self.joint_metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.joint_metadata, f)

    def ensure_joint_index(self):
        """Lazily ensures unified joint index is loaded or built on the fly from existing stores."""
        if getattr(self, "joint_index", None) is not None and self.joint_index.ntotal > 0:
            return

        # Check if joint index exists on disk
        if os.path.exists(self.joint_faiss_path) and os.path.exists(self.joint_metadata_path):
            try:
                with open(self.joint_metadata_path, "r", encoding="utf-8") as f:
                    self.joint_metadata = json.load(f)
                self.joint_index = faiss.read_index(self.joint_faiss_path)
                if self.joint_index.ntotal > 0:
                    return
            except Exception as e:
                print(f"[VectorIndexer Warning] Could not load existing joint index: {e}")

        # Construct joint index on the fly from visual_metadata and audio_collection
        all_facts = []
        if getattr(self, "visual_metadata", None):
            for item in self.visual_metadata:
                all_facts.append(dict(item))

        try:
            chroma_data = self.audio_collection.get()
            if chroma_data and chroma_data.get("documents"):
                for doc, meta in zip(chroma_data["documents"], chroma_data.get("metadatas", [])):
                    fact = {
                        "type": meta.get("type", "sound") if meta else "sound",
                        "start_time": meta.get("start_time", 0.0) if meta else 0.0,
                        "end_time": meta.get("end_time", 0.0) if meta else 0.0,
                        "text": doc
                    }
                    if meta and "score" in meta:
                        fact["score"] = meta["score"]
                    all_facts.append(fact)
        except Exception as e:
            print(f"[VectorIndexer Warning] Could not query Chroma audio collection: {e}")

        if all_facts:
            print(f"[VectorIndexer] Dynamically constructing Unified Joint Index for {len(all_facts)} audio+visual facts...")
            texts = [f["text"] for f in all_facts]
            embeddings = self.visual_embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            self.joint_index = faiss.IndexFlatIP(self.visual_dim)
            self.joint_index.add(embeddings.astype("float32"))
            self.joint_metadata = all_facts
            try:
                faiss.write_index(self.joint_index, self.joint_faiss_path)
                with open(self.joint_metadata_path, "w", encoding="utf-8") as f:
                    json.dump(self.joint_metadata, f)
            except Exception as e:
                print(f"[VectorIndexer Warning] Could not save dynamically created joint index: {e}")

if __name__ == "__main__":
    pass
