import os
import re
import json
import requests
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from config import MODALITY_ESTIMATOR_MODEL, DEVICE, OPENAI_API_KEY, GPT_MODEL

class QuestionClassifier:
    """
    Classifies questions and estimates the audio dependency weight beta(q) in [0, 1].
    Uses GPT-4o-mini query classification when OPENAI_API_KEY is available,
    with BGE-large semantic concept embeddings & keyword heuristic fallback.
    """
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        self.gpt_model = os.getenv("GPT_MODEL", GPT_MODEL)
        
        print(f"Loading Modality Estimator Embedder: {MODALITY_ESTIMATOR_MODEL}")
        self.model = SentenceTransformer(MODALITY_ESTIMATOR_MODEL, device=DEVICE)
        
        # Audio concept descriptions (speech, acoustic events, sounds, spoken words, music, ambient audio)
        self.audio_anchors = [
            "questions about spoken speech, dialog, words said, or spoken conversation",
            "questions about sound effects, background noise, audio events, music, or acoustic signals",
            "what noise or sound is heard in the video",
            "did someone say or speak something",
            "what did the speaker or person say",
            "who spoke or yelled in the audio track",
            "listening to music, sirens, laughter, applause, or environmental sounds",
            "audio transcript, voice, conversation, or speech utterance"
        ]
        
        # Visual concept descriptions (objects, subjects, colors, actions, spatial layout, physical appearance)
        self.visual_anchors = [
            "questions about visual appearance, colors, clothes, objects, or physical subjects",
            "questions about spatial layout, location, left, right, background, or visible scene",
            "what color is the shirt, vehicle, object, or item",
            "where is the person, object, or item positioned visually",
            "what action is visually being performed in the video frame",
            "who is visible or seen on screen in the video",
            "describe the visual scene, appearance, or background details",
            "visible text, physical objects on table, or optical keyframes"
        ]
        
        self.audio_embeds = self.model.encode(self.audio_anchors, convert_to_tensor=True)
        self.visual_embeds = self.model.encode(self.visual_anchors, convert_to_tensor=True)

        self.audio_keywords = {
            "say", "said", "spoke", "speaking", "talk", "talking", "sound", "noise",
            "listen", "heard", "hear", "music", "singing", "yell", "shout", "whisper",
            "voice", "applause", "laughter", "siren", "alarm", "speech", "dialogue"
        }

        self.visual_keywords = {
            "color", "wearing", "clothes", "shirt", "pants", "dress", "visible",
            "look", "see", "seen", "where", "behind", "next", "left", "right",
            "background", "foreground", "object", "car", "table", "holding", "standing"
        }

    def classify_with_gpt(self, question: str) -> dict:
        """Uses GPT-4o-mini for query classification and modality weight estimation."""
        api_key = os.getenv("OPENAI_API_KEY", self.api_key)
        if not api_key:
            return None

        prompt = f"""You are an expert multi-modal Video QA classifier.
Analyze the user's question about a video:
Question: "{question}"

Classify into JSON with these exact keys:
- "category": one of ["spatial", "temporal", "spatiotemporal", "audio", "counting", "yes_no", "general"]
- "modality": one of ["audio", "visual", "audio_visual"]
- "beta": a float between 0.0 and 1.0 representing audio dependency:
  * 0.85 to 1.0: questions strictly about speech, dialogue, sounds, musical instruments, acoustics
  * 0.40 to 0.75: questions requiring both audio and visual (e.g. which instrument is playing on the left, does the person speak while moving)
  * 0.0 to 0.20: questions strictly about visual appearance, colors, objects, locations, furniture, clothing, physical actions
- "reasoning": 1 short sentence

Output ONLY valid JSON."""

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.gpt_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 120
            }
            res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=8.0)
            if res.status_code == 200:
                raw_text = res.json()["choices"][0]["message"]["content"].strip()
                # Parse JSON block
                clean_json = re.sub(r"^```json\s*", "", raw_text, flags=re.IGNORECASE)
                clean_json = re.sub(r"\s*```$", "", clean_json)
                parsed = json.loads(clean_json)
                return parsed
        except Exception as e:
            print(f"[QuestionClassifier Warning] GPT-4o-mini classification error: {e}")
        return None

    def estimate_beta(self, question: str) -> float:
        """
        Estimates the audio dependency weight beta(q) in [0, 1].
        1.0 means highly audio-dependent, 0.0 means highly visual-dependent.
        """
        # 1. Try GPT-4o-mini Query Classification
        gpt_res = self.classify_with_gpt(question)
        if gpt_res and "beta" in gpt_res:
            beta = max(0.0, min(1.0, float(gpt_res["beta"])))
            cat = gpt_res.get("category", "unknown")
            print(f"Question Classifier (GPT-4o-mini): category={cat}, beta(q)={beta:.2f} for '{question}'")
            return beta

        # 2. Dense Semantic Anchors + Keyword Heuristics Fallback (BGE Embeddings)
        q_lower = question.lower()
        words = set(re.findall(r'\b\w+\b', q_lower))

        audio_kw_count = len(words.intersection(self.audio_keywords))
        visual_kw_count = len(words.intersection(self.visual_keywords))

        q_embed = self.model.encode([question], convert_to_tensor=True)
        
        # Mean + Max similarity to audio and visual semantic concepts
        audio_sims = cos_sim(q_embed, self.audio_embeds)[0]
        visual_sims = cos_sim(q_embed, self.visual_embeds)[0]

        audio_score = float((audio_sims.mean() * 0.5 + audio_sims.max() * 0.5).item())
        visual_score = float((visual_sims.mean() * 0.5 + visual_sims.max() * 0.5).item())

        # Adjust scores using keyword priors
        audio_score += audio_kw_count * 0.15
        visual_score += visual_kw_count * 0.15

        audio_score = max(0.001, audio_score)
        visual_score = max(0.001, visual_score)
        
        beta = audio_score / (audio_score + visual_score)
        beta = max(0.0, min(1.0, float(beta)))
        
        print(f"Question Classifier (BGE Fallback): beta(q) = {beta:.2f} for question: '{question}'")
        return beta

if __name__ == "__main__":
    qc = QuestionClassifier()
    print(qc.estimate_beta("what did the person say?"))
    print(qc.estimate_beta("what color is the car?"))
