import os
import requests
import json
import re
from typing import Optional, List, Dict, Any
import torch
from config import (
    OLLAMA_MODEL, 
    OLLAMA_HOST, 
    OLLAMA_MAX_TOKENS, 
    HF_LLM_MODEL, 
    LLM_BACKEND, 
    TORCH_DTYPE, 
    DEVICE, 
    IS_GPU,
    OPENAI_API_KEY,
    GPT_MODEL
)

class Generator:
    def __init__(self, backend: str = None, model: str = None):
        self.host = OLLAMA_HOST
        self.model = model if model else OLLAMA_MODEL
        self.max_tokens = OLLAMA_MAX_TOKENS
        self.backend = (backend if backend else LLM_BACKEND).lower().strip()
        self.hf_model_id = os.getenv("HF_LLM_MODEL", HF_LLM_MODEL)
        self.gpt_model = os.getenv("GPT_MODEL", GPT_MODEL)
        self.api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        self._hf_pipeline = None
        self._hf_tokenizer = None

    def _generate_gpt(self, prompt: str, question: str = "") -> Optional[str]:
        """Generates answer using GPT-4o-mini via OpenAI API."""
        api_key = os.getenv("OPENAI_API_KEY", self.api_key)
        if not api_key:
            return None
        try:
            print(f"[Stage 3 Generator] Generating answer via {self.gpt_model}...")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.gpt_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": self.max_tokens
            }
            res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15.0)
            if res.status_code == 200:
                raw_ans = res.json()["choices"][0]["message"]["content"].strip()
                cleaned = self.clean_answer(raw_ans, question=question)
                return cleaned
            else:
                print(f"[Generator Warning] GPT API returned {res.status_code}: {res.text[:100]}")
        except Exception as e:
            print(f"[Generator Warning] GPT generation failed: {e}")
        return None

    def _is_ollama_available(self) -> bool:
        """Quick check if local Ollama server is running and responsive."""
        try:
            res = requests.get(f"{self.host}/api/tags", timeout=1.0)
            return res.status_code == 200
        except Exception:
            return False

    def _lazy_init_hf(self):
        """Lazily instantiates the Hugging Face CausalLM pipeline in-process."""
        if self._hf_pipeline is None:
            print(f"[Stage 3 Generator] Initializing in-process Hugging Face LLM ({self.hf_model_id})...")
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_id, 
                trust_remote_code=True
            )
            kwargs = {
                "torch_dtype": TORCH_DTYPE if IS_GPU else torch.float32,
                "device_map": "auto" if DEVICE == "cuda" else None,
                "trust_remote_code": True,
            }
            hf_model = AutoModelForCausalLM.from_pretrained(self.hf_model_id, **kwargs)
            if DEVICE != "cuda" and DEVICE != "cpu":
                hf_model = hf_model.to(DEVICE)
            self._hf_pipeline = pipeline(
                "text-generation",
                model=hf_model,
                tokenizer=self._hf_tokenizer,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
            print(f"[Stage 3 Generator] Hugging Face LLM ({self.hf_model_id}) ready on {DEVICE}.")

    def _generate_hf(self, prompt: str, question: str = "") -> str:
        """Generates answer using the in-process Hugging Face model without external servers."""
        self._lazy_init_hf()
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = self._hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            formatted = prompt

        outputs = self._hf_pipeline(formatted, max_new_tokens=self.max_tokens, return_full_text=False)
        raw_ans = outputs[0]["generated_text"].strip()
        cleaned = self.clean_answer(raw_ans, question=question)
        return cleaned

    def format_context(self, candidates: list) -> str:
        """
        Formats the context from audio and visual candidates.
        """
        context_lines = []
        for cand in sorted(candidates, key=lambda x: x["metadata"]["start_time"]):
            md = cand["metadata"]
            start = md["start_time"]
            end = md["end_time"]
            fact_type = md["type"]
            text = cand["text"].strip()
            
            if fact_type == "visual":
                context_lines.append(f"[Visual - {start:.2f}s to {end:.2f}s] {text}")
            elif fact_type == "speech":
                context_lines.append(f"[Speech - {start:.2f}s to {end:.2f}s] {text}")
            elif fact_type == "sound":
                context_lines.append(f"[Sound - {start:.2f}s to {end:.2f}s] {text}")
                
        return "\n".join(context_lines)

    def clean_answer(self, raw_answer: str, question: str = "") -> str:
        """
        Strips conversational prefixes, filler explanations, markdown artifacts,
        and applies benchmark-aligned normalization.
        """
        if not raw_answer:
            return ""
        
        ans = raw_answer.strip().replace("**", "").replace("*", "").replace("`", "").replace('"', '').replace("'", "")
        
        # Split into lines and take the first informative line
        lines = [line.strip() for line in ans.split("\n") if line.strip()]
        if lines:
            ans = lines[0]
            
        # Strip common verbose lead-ins iteratively
        verbose_patterns = [
            r"^short answer\s*:\s*",
            r"^answer\s*:\s*",
            r"^based on (?:the )?(?:provided )?(?:visual |video |audio |speech )*(?:context|evidence|scenes?|timestamps?|descriptions?)[,\s:]*",
            r"^in the (?:provided )?(?:visual |video |audio )*(?:scenes?|video)[,\s:]*",
            r"^from the (?:provided )?(?:visual |video |audio )*(?:evidence|video)[,\s:]*",
            r"^the answer is\s*:\s*",
            r"^the answer is\s*",
            r"^it takes place in\s*(?:a|an)?\s*",
            r"^the performance is (?:located )?(?:in|at)\s*(?:a|an)?\s*",
            r"^the person (?:is|appears to be|was)?\s*(?:standing|sitting|positioned|seen)?\s*(?:in front of|behind|near|beside|next to|on|under)\s*",
            r"^there is (?:a|an)\s*",
            r"^it is (?:a|an)\s*",
            r"^[-•*]\s+",
            r"^\d+[\.\)]\s+",
            r"^[-•*0-9]+[.)]\s*",
        ]
        
        changed = True
        while changed:
            changed = False
            for pat in verbose_patterns:
                new_ans = re.sub(pat, "", ans, flags=re.IGNORECASE).strip()
                if new_ans != ans:
                    ans = new_ans
                    changed = True
            
        # Strip trailing periods for short answers
        if len(ans.split()) <= 8:
            ans = ans.rstrip(" .;,")

        q_lower = question.lower() if question else ""
        ans_lower = ans.lower().strip()

        # 1. Yes/No binary question routing
        yes_no_starters = ["is there", "is the", "are there", "are the", "did ", "does ", "was there", "were there", "can ", "could ", "would ", "given the"]
        if any(q_lower.startswith(starter) for starter in yes_no_starters):
            if "yes" in ans_lower:
                return "yes"
            elif "no" in ans_lower:
                return "no"

        # 2. Spatial channel localization (MusicAVQA: Which object / left or right)
        if "sound" in q_lower and ("which" in q_lower or "left" in q_lower or "right" in q_lower):
            if "left" in ans_lower and "right" not in ans_lower:
                return "left"
            elif "right" in ans_lower and "left" not in ans_lower:
                return "right"

        # 3. Environment normalizer ("outdoor setting" -> "outdoor")
        if "where is the performance" in q_lower or "where does the performance" in q_lower:
            if any(w in ans_lower for w in ["outdoor", "street", "park", "outside"]):
                return "outdoor"
            elif any(w in ans_lower for w in ["indoor", "inside", "hall", "room", "auditorium"]):
                return "indoor"

        # 4. Number word normalization for counting questions
        word_to_num = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }
        if q_lower.startswith("how many") or "count" in q_lower:
            # Extract first number or digit
            num_match = re.search(r'\b(\d+)\b', ans)
            if num_match:
                return num_match.group(1)
            for w, n in word_to_num.items():
                if re.search(rf'\b{w}\b', ans_lower):
                    return n

        return ans

    def generate_answer(self, question: str, context: str) -> str:
        """
        Calls Ollama to generate a direct, concise answer based on multi-modal evidence.
        """
        prompt = f"""You are an expert Video Question Answering model.
Answer the question accurately, directly, and concisely using the provided video evidence (visual descriptions, temporal transitions, speech, and sound events).

Task Instructions & Reasoning Rules:
1. Short Direct Answer: Output ONLY a short, direct answer (typically 1 to 5 words, e.g. "mirror", "shelf", "table", "tie up hair", "hair styling", "dance", "moved from right to left", "yes", "no", "1", "2", "outdoor", "left", "right").
2. Spatial & Spatiotemporal Questions:
   - For positions (behind, in front of, under, on): identify the primary object or furniture directly adjacent to or behind the subject (e.g. table, shelf, cabinet, stage, mirror, sink, bread).
   - For left vs right sound source questions (e.g. "Which <Object> makes the sound?"): check for "[Audio Source: Left Side]" or "[Audio Source: Right Side]" in the sound tags and output "left" or "right".
   - For performance location: if outdoor/street/park, answer "outdoor"; if inside a building/room, answer the specific room (e.g. "dining room", "church", "indoor").
3. Yes/No Verification Questions:
   - For questions starting with "Is there...", "Is the...", "Did...", "Does...", "Was...", output strictly "yes" or "no".
4. Temporal Sequence Questions (what happened before / after / when...):
   - Track chronological transitions across timestamps to find the immediate previous or next action in the sequence.
5. Counting Questions (how many people / chairs / instruments):
   - Output the exact number (e.g. "1", "2", "3", "4").
6. Formatting & Reliability:
   - Do NOT write explanations, reasoning steps, conversational filler, or timestamps. Output ONLY the concise final answer.
   - NEVER output "Unknown", "I don't know", or "Unclear". Always make a best-effort prediction using the most relevant visual/audio scene evidence.

Evidence:
{context if context.strip() else "[No specific evidence retrieved; predict best probable answer from scene context]"}

Question: {question}
Short Answer:"""

        # 1. Priority: GPT-4o-mini if configured or OPENAI_API_KEY is available
        if self.backend == "gpt" or (self.backend == "auto" and os.getenv("OPENAI_API_KEY", self.api_key)):
            gpt_ans = self._generate_gpt(prompt, question=question)
            if gpt_ans and not gpt_ans.lower().startswith("error"):
                return gpt_ans
            print("[Stage 3 Generator] GPT generation unavailable/failed. Falling back to local backends...")

        # 2. Determine whether to use Ollama or HuggingFace
        use_ollama = False
        if self.backend == "ollama":
            use_ollama = True
        elif self.backend == "auto":
            use_ollama = self._is_ollama_available()
        # If backend == "hf", use_ollama remains False

        if not use_ollama:
            print(f"[Stage 3 Generator] Generating via in-process Hugging Face LLM ({self.hf_model_id})...")
            try:
                ans = self._generate_hf(prompt, question=question)
                return ans
            except Exception as e:
                print(f"[Stage 3 Generator ERROR] Hugging Face generation failed: {e}")
                return f"Error: Hugging Face generation failed ({e})"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_predict": self.max_tokens,
            },
            "keep_alive": "5m"
        }
        
        try:
            print(f"Sending request to Ollama ({self.model})...")
            response = requests.post(f"{self.host}/api/generate", json=payload, timeout=25.0)
            response.raise_for_status()
            result = response.json()
            raw_ans = result.get("response", "").strip()
            cleaned = self.clean_answer(raw_ans, question=question)
            
            # Robust fallback if model still returned Unknown
            if cleaned.lower() in ["unknown", "unknown.", "n/a", "none", "i don't know", "unclear"]:
                fallback_prompt = f"""Given the video evidence below, what is the single most likely object, entity, or action for the question?
Evidence:
{context}
Question: {question}
Output only the object name or action (1 to 3 words, e.g. mirror, shelf, table, styling hair):"""
                fb_payload = dict(payload)
                fb_payload["prompt"] = fallback_prompt
                fb_res = requests.post(f"{self.host}/api/generate", json=fb_payload, timeout=20.0)
                if fb_res.ok:
                    fb_raw = fb_res.json().get("response", "").strip()
                    fb_cleaned = self.clean_answer(fb_raw, question=question)
                    if fb_cleaned and fb_cleaned.lower() not in ["unknown", "unknown."]:
                        return fb_cleaned

            return cleaned
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"[Generator Warning] Model '{self.model}' not found in Ollama. Falling back to 'qwen2:1.5b'...")
                fallback_payload = dict(payload)
                fallback_payload["model"] = "qwen2:1.5b"
                try:
                    res = requests.post(f"{self.host}/api/generate", json=fallback_payload, timeout=20.0)
                    res.raise_for_status()
                    raw_ans = res.json().get("response", "").strip()
                    return self.clean_answer(raw_ans, question=question)
                except Exception as fb_err:
                    if self.backend == "auto":
                        print(f"[Generator Warning] Ollama fallback failed. Trying Hugging Face ({self.hf_model_id})...")
                        return self._generate_hf(prompt, question=question)
                    return f"Error: Model '{self.model}' not pulled in Ollama. Run 'ollama pull {self.model}'."
            if self.backend == "auto":
                print(f"[Generator Warning] Ollama HTTP error ({e}). Trying Hugging Face ({self.hf_model_id})...")
                return self._generate_hf(prompt, question=question)
            print(f"Error generating answer with Ollama: {e}")
            return f"Error: Could not generate answer ({e}). Ensure Ollama is running."
        except Exception as e:
            if self.backend == "auto":
                print(f"[Generator Warning] Ollama connection failed ({e}). Automatically falling back to Hugging Face ({self.hf_model_id})...")
                try:
                    return self._generate_hf(prompt, question=question)
                except Exception as hf_err:
                    return f"Error: Both Ollama and Hugging Face generation failed ({hf_err})"
            print(f"Error generating answer with Ollama: {e}")
            return f"Error: Could not generate answer. Ensure Ollama is running and model '{self.model}' is pulled."

# Alias for backward compatibility
AnswerGenerator = Generator
