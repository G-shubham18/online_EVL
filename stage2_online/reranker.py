import torch
import numpy as np
import math
import re
from collections import Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import (
    RERANKER_MODEL, 
    DEVICE, 
    AUDIO_BONUS, 
    TEMPORAL_AGREEMENT_BONUS, 
    TORCH_DTYPE, 
    IS_GPU, 
    ENABLE_BM25, 
    BM25_WEIGHT
)

def tokenize_text(text: str) -> list:
    """Tokenizes text into lowercase alphanumeric terms for BM25."""
    return re.findall(r'\b\w+\b', text.lower())

class BM25Scorer:
    """
    Fast BM25Okapi implementation for keyword-based re-ranking.
    Uses rank_bm25 if installed, with a self-contained NumPy fallback.
    """
    def __init__(self, corpus: list, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / max(1, self.corpus_size)
        
        # Calculate document frequencies
        self.df = Counter()
        for doc in corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.df[term] += 1
                
        # Precompute IDF scores
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            
        self.corpus = corpus

    def score(self, query_tokens: list) -> list:
        """Scores all documents in corpus against query tokens."""
        scores = []
        for i, doc in enumerate(self.corpus):
            doc_len = self.doc_lengths[i]
            tf = Counter(doc)
            score = 0.0
            for term in query_tokens:
                if term in tf:
                    term_freq = tf[term]
                    idf_term = self.idf.get(term, 0.0)
                    denom = term_freq + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avgdl)))
                    score += idf_term * (term_freq * (self.k1 + 1.0)) / max(1e-6, denom)
            scores.append(score)
        return scores

class ReRanker:
    def __init__(self):
        print(f"Loading Re-Ranker (BGE + BM25): {RERANKER_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        kwargs = {"torch_dtype": TORCH_DTYPE} if IS_GPU else {}
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL, **kwargs).to(DEVICE)
        self.model.eval()
        self.enable_bm25 = ENABLE_BM25
        self.bm25_weight = BM25_WEIGHT

    def compute_temporal_agreement(self, candidate, all_candidates):
        """
        Adds extra weight to evidence sharing timestamp windows with other top candidates.
        """
        c_start = candidate["metadata"]["start_time"]
        c_end = candidate["metadata"]["end_time"]
        
        agreement_score = 0.0
        for other in all_candidates:
            if other["id"] == candidate["id"]:
                continue
            
            o_start = other["metadata"]["start_time"]
            o_end = other["metadata"]["end_time"]
            
            # Check for overlap
            overlap_start = max(c_start, o_start)
            overlap_end = min(c_end, o_end)
            if overlap_start < overlap_end:
                agreement_score += TEMPORAL_AGREEMENT_BONUS
                
        return agreement_score

    def score_candidates(
        self, 
        question: str, 
        candidates: list, 
        beta_q: float,
        rerank_mode: str = "full"
    ):
        """
        Scores candidates using hybrid Cross-Encoder + BM25 keyword re-ranking:
        Formula:
          FinalScore = CrossEncoderRelevance + (BM25Norm * BM25Weight) + (beta(q) * AudioBonus) + TemporalAgreement
        """
        if not candidates:
            return []

        # 1. BM25 Keyword-Based Scoring
        bm25_scores = [0.0] * len(candidates)
        if self.enable_bm25 and len(candidates) > 0:
            tokenized_corpus = [tokenize_text(cand["text"]) for cand in candidates]
            query_tokens = tokenize_text(question)
            try:
                bm25 = BM25Scorer(tokenized_corpus)
                raw_bm25 = bm25.score(query_tokens)
                max_b = max(raw_bm25) if raw_bm25 else 1.0
                min_b = min(raw_bm25) if raw_bm25 else 0.0
                # Min-max normalize BM25 scores to [0, 1]
                if max_b > min_b:
                    bm25_scores = [(s - min_b) / (max_b - min_b) for s in raw_bm25]
                else:
                    bm25_scores = [0.5] * len(candidates)
            except Exception as e:
                print(f"[ReRanker Warning] BM25 scoring error: {e}")

        # Ablation 2 (Full Bypass): Re-ranking completely off
        if rerank_mode in ["off", "none", "no_rerank"]:
            print("[ReRanker Ablation: OFF] Cross-encoder bypassed. Combining retriever dense score and BM25.")
            for i, cand in enumerate(candidates):
                raw_score = float(cand.get("score", 0.0))
                bm25_val = bm25_scores[i] * self.bm25_weight if self.enable_bm25 else 0.0
                cand["relevance_score"] = raw_score
                cand["bm25_score"] = bm25_scores[i]
                cand["final_score"] = raw_score + bm25_val
            return sorted(candidates, key=lambda x: x["final_score"], reverse=True)

        pairs = [[question, cand["text"]] for cand in candidates]
        scores = []
        
        # Mini-batched inference with max_length=192 (optimized for fast CPU/GPU throughput)
        batch_size = 6
        with torch.inference_mode():
            for i in range(0, len(pairs), batch_size):
                b_pairs = pairs[i : i + batch_size]
                inputs = self.tokenizer(
                    b_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=192
                ).to(DEVICE)
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()
                if logits.ndim == 0:
                    scores.append(float(logits))
                else:
                    scores.extend(logits.tolist())
            
        # Apply formula: Score = Relevance + (BM25 * Weight) + (beta(q) * AudioBonus) + TemporalAgreement
        for i, cand in enumerate(candidates):
            relevance = scores[i] if i < len(scores) else 0.0
            bm25_lift = bm25_scores[i] * self.bm25_weight if self.enable_bm25 else 0.0
            
            # Ablation 2 (No Audio Lift): Remove audio-aware lift term
            if rerank_mode == "no_audio_lift":
                audio_bonus_term = 0.0
            else:
                is_audio = 1.0 if cand["metadata"]["type"] in ["speech", "sound"] else 0.0
                audio_bonus_term = beta_q * (AUDIO_BONUS if is_audio else 0.0)
            
            temporal_agreement = self.compute_temporal_agreement(cand, candidates)
            
            final_score = relevance + bm25_lift + audio_bonus_term + temporal_agreement
            
            cand["relevance_score"] = float(relevance)
            cand["bm25_score"] = float(bm25_scores[i])
            cand["final_score"] = float(final_score)
            
        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)

# Alias for backward compatibility
CrossEncoderReranker = ReRanker

