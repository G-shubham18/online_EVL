import os
import subprocess
import tempfile
import wave
import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import pipeline
import shutil

from config import (
    WHISPER_MODEL,
    DEVICE,
    IS_CUDA_AVAILABLE,
    IS_GPU,
    AST_MODEL,
    CLAP_MODEL,
    AUDIO_CHUNK_LENGTH,
    AUDIO_OVERLAP,
    AUDIO_THRESHOLD,
    AUDIO_BATCH_SIZE,
    MIN_EVENT_DURATION,
    AUDIO_RMS_THRESHOLD,
    STEREO_BALANCE_THRESHOLD,
)


class AudioExtractor:
    def __init__(self):
        print(f"Loading Whisper speech transcription model: {WHISPER_MODEL}")
        whisper_dev = "cuda" if IS_CUDA_AVAILABLE else "cpu"
        whisper_compute = "float16" if IS_CUDA_AVAILABLE else "int8"
        try:
            self.whisper_model = WhisperModel(
                WHISPER_MODEL,
                device=whisper_dev,
                compute_type=whisper_compute,
            )
        except Exception as e:
            print(f"[AudioExtractor Warning] Whisper initialization failed with {whisper_compute} ({e}). Retrying with int8/cpu fallback...")
            try:
                self.whisper_model = WhisperModel(
                    WHISPER_MODEL,
                    device=whisper_dev,
                    compute_type="int8_float16" if whisper_dev == "cuda" else "int8",
                )
            except Exception:
                self.whisper_model = WhisperModel(
                    WHISPER_MODEL,
                    device="cpu",
                    compute_type="int8",
                )

        # Initialize AST (Audio Spectrogram Transformer) for Audio Event Classification
        print(f"Loading Audio Event Model (AST): {AST_MODEL}")
        self.ast_extractor = None
        self.ast_model = None
        self.use_ast = False

        try:
            from transformers import ASTForAudioClassification, AutoFeatureExtractor
            self.ast_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL)
            self.ast_model = ASTForAudioClassification.from_pretrained(AST_MODEL).to(DEVICE)
            self.ast_model.eval()
            self.use_ast = True
            print(f"[AudioExtractor] Successfully loaded AST model on {DEVICE}.")
        except Exception as e:
            print(f"[AudioExtractor Warning] Could not load AST model ({e}). Falling back to CLAP.")
            self.use_ast = False

        # Fallback to zero-shot CLAP if AST is not loaded
        self.audio_classifier = None
        if not self.use_ast:
            try:
                clap_dev = 0 if IS_CUDA_AVAILABLE else -1
                self.audio_classifier = pipeline(
                    task="zero-shot-audio-classification",
                    model=CLAP_MODEL,
                    device=clap_dev,
                )
            except Exception as clap_err:
                print(f"[AudioExtractor Warning] CLAP fallback also failed: {clap_err}")

        # Define comprehensive sound event taxonomy to cover everyday, mechanical, musical, and contact acoustics
        self.sound_event_labels = [
            "violin or fiddle playing",
            "acoustic guitar strumming",
            "electric guitar",
            "piano or keyboard",
            "cello playing",
            "flute or recorder",
            "clarinet or woodwind",
            "trumpet or brass horn",
            "drums or percussion",
            "accordion",
            "musical instrument performance",
            "singing or vocal performance",
            "faint plastic click",
            "sharp plastic thud",
            "muffled wooden thud",
            "metallic scraping sound",
            "metallic squeak",
            "metallic clinking",
            "ceramic clinking",
            "clank or metal impact",
            "glass clinking or breaking",
            "footsteps or walking",
            "door opening",
            "door closing or slamming",
            "cabinet door opening or closing",
            "laptop lid opening or closing",
            "paper folding or crinkling",
            "running water",
            "water dripping or faucet squeak",
            "liquid pouring",
            "water splashing",
            "tattoo machine buzzing",
            "electric motor buzzing or humming",
            "hair dryer blowing air",
            "steady white noise from a fan",
            "appliances humming",
            "computer keyboard typing",
            "mouse clicking",
            "car engine",
            "siren or alarm",
            "phone ringing",
            "applause and clapping",
            "laughter and cheering",
            "intermittent voices or talking",
            "dialogue or conversation",
            "breathing or sighing",
            "whispering",
            "wind or background breeze",
            "rain or thunder",
            "silence or quiet room",
        ]

    @staticmethod
    def _is_working_ffmpeg(bin_path: str) -> bool:
        if not bin_path:
            return False
        try:
            res = subprocess.run(
                [bin_path, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            return res.returncode == 0
        except Exception:
            return False

    def _get_ffmpeg_cmd(self):
        if hasattr(self, "_cached_ffmpeg") and self._cached_ffmpeg:
            return self._cached_ffmpeg

        # 1. Try imageio_ffmpeg (robust precompiled static binary)
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if self._is_working_ffmpeg(exe):
                self._cached_ffmpeg = exe
                return exe
        except Exception:
            pass

        # 2. Check local workspace directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_ffmpeg = os.path.join(base_dir, "ffmpeg.exe")
        if self._is_working_ffmpeg(local_ffmpeg):
            self._cached_ffmpeg = local_ffmpeg
            return local_ffmpeg

        # 3. Check system PATH
        ffmpeg_bin = shutil.which("ffmpeg")
        if self._is_working_ffmpeg(ffmpeg_bin):
            self._cached_ffmpeg = ffmpeg_bin
            return ffmpeg_bin

        # 4. Auto-install imageio-ffmpeg into active Python environment
        try:
            import sys

            print("Working FFmpeg executable not found. Auto-installing imageio-ffmpeg...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "imageio-ffmpeg"], check=True
            )
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if self._is_working_ffmpeg(exe):
                self._cached_ffmpeg = exe
                return exe
        except Exception as e:
            print(f"Warning: Failed to auto-install imageio-ffmpeg: {e}")

        # 5. Direct download of standalone static ffmpeg.exe for Windows (Windows only)
        if sys.platform == "win32":
            try:
                import urllib.request
                import zipfile

                print("Downloading standalone FFmpeg binary for Windows...")
                url = "https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v4.4.1/ffmpeg-4.4.1-win-64.zip"
                zip_path = local_ffmpeg + ".zip"
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(base_dir)
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                if self._is_working_ffmpeg(local_ffmpeg):
                    self._cached_ffmpeg = local_ffmpeg
                    return local_ffmpeg
            except Exception as e:
                print(f"Warning: Failed to download standalone FFmpeg: {e}")

        self._cached_ffmpeg = "ffmpeg"
        return "ffmpeg"

    def extract_audio_from_video(self, video_path: str, output_wav_path: str):
        """Extracts the audio track from a video file as a 16kHz stereo .wav file."""
        print(f"Extracting stereo audio from {video_path} to {output_wav_path}")
        ffmpeg_bin = self._get_ffmpeg_cmd()
        command = [
            ffmpeg_bin,
            "-y",
            "-i",
            video_path,
            "-vn",  # Disable video
            "-acodec",
            "pcm_s16le",  # 16-bit PCM
            "-ar",
            "16000",  # 16kHz sampling rate
            "-ac",
            "2",  # 2 channels (Stereo) to preserve Left vs Right spatial acoustics
            output_wav_path,
        ]

        try:
            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() if e.stderr else str(e)
            raise RuntimeError(
                f"FFmpeg audio extraction failed for '{video_path}' using '{ffmpeg_bin}' (exit {e.returncode}): {err_msg}"
            ) from e
        return output_wav_path

    def transcribe_speech(self, audio_path: str):
        """Transcribes speech using faster-whisper with torch.inference_mode() and returns timestamped segments."""
        print(f"Transcribing speech from {audio_path}")
        with torch.inference_mode():
            segments, info = self.whisper_model.transcribe(audio_path, beam_size=1)

            results = []
            for segment in segments:
                results.append(
                    {
                        "type": "speech",
                        "start_time": segment.start,
                        "end_time": segment.end,
                        "text": segment.text.strip(),
                    }
                )
        return results

    def _merge_adjacent_sound_events(
        self, events: list, min_event_duration: float = MIN_EVENT_DURATION
    ):
        """Merges adjacent identical sound events and filters out events shorter than min_event_duration."""
        if not events:
            return []

        # Sort by start time
        events.sort(key=lambda x: x["start_time"])

        merged = []
        current = None

        for ev in events:
            if current is None:
                current = dict(ev)
            else:
                # Merge if it's the same sound event label/text and adjacent/overlapping
                if (
                    ev["text"] == current["text"]
                    and ev["start_time"] <= current["end_time"]
                ):
                    current["end_time"] = max(current["end_time"], ev["end_time"])
                    current["score"] = max(current["score"], ev["score"])
                else:
                    if (
                        current["end_time"] - current["start_time"]
                    ) >= min_event_duration:
                        merged.append(current)
                    current = dict(ev)

        if current is not None:
            if (current["end_time"] - current["start_time"]) >= min_event_duration:
                merged.append(current)

        return merged

    def detect_sound_events(
        self,
        audio_path: str,
        chunk_length_s: float = AUDIO_CHUNK_LENGTH,
        overlap_s: float = AUDIO_OVERLAP,
        threshold: float = AUDIO_THRESHOLD,
        batch_size: int = AUDIO_BATCH_SIZE,
        min_event_duration: float = MIN_EVENT_DURATION,
        rms_threshold: float = AUDIO_RMS_THRESHOLD,
        stereo_threshold: float = STEREO_BALANCE_THRESHOLD,
    ):
        """Detects sound events and spatial Left/Right balance by sliding a window over the stereo audio."""
        print(f"Detecting sound events and spatial audio in {audio_path}")

        # Load audio using standard wave module & numpy
        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        if sampwidth == 2:
            y = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            y = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            y = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        if n_channels == 2:
            y_stereo = y.reshape(-1, 2)
            y_mono = y_stereo.mean(axis=1)
        elif n_channels > 2:
            y_multi = y.reshape(-1, n_channels)
            y_stereo = y_multi[:, :2]
            y_mono = y_multi.mean(axis=1)
        else:
            y_stereo = np.column_stack([y, y])
            y_mono = y

        duration = float(len(y_mono)) / float(sr) if sr > 0 else 0.0
        step_s = chunk_length_s - overlap_s

        chunks_to_process = []
        current_time = 0.0
        total_chunks = 0

        # Extract audio chunks and calculate spatial Left/Right energy balance
        while current_time < duration:
            end_time = min(current_time + chunk_length_s, duration)
            if end_time - current_time < 0.5:  # Skip very short final chunks
                break

            total_chunks += 1
            start_sample = int(current_time * sr)
            end_sample = int(end_time * sr)
            chunk_mono = y_mono[start_sample:end_sample]
            chunk_st = y_stereo[start_sample:end_sample]

            # RMS energy threshold check to skip low-energy (silent) chunks
            rms = np.sqrt(np.mean(chunk_mono**2)) if len(chunk_mono) > 0 else 0.0

            if rms >= rms_threshold:
                # Spatial audio balance: Left vs Right channel energy ratio
                l_rms = float(np.sqrt(np.mean(chunk_st[:, 0]**2))) if len(chunk_st) > 0 else 0.0
                r_rms = float(np.sqrt(np.mean(chunk_st[:, 1]**2))) if len(chunk_st) > 0 else 0.0
                tot_rms = l_rms + r_rms

                spatial_label = "center"
                if tot_rms > 1e-4:
                    bal = (l_rms - r_rms) / tot_rms
                    if bal > stereo_threshold:
                        spatial_label = "left"
                    elif bal < -stereo_threshold:
                        spatial_label = "right"

                chunks_to_process.append(
                    {
                        "start_time": current_time,
                        "end_time": end_time,
                        "audio": chunk_mono,
                        "spatial_label": spatial_label,
                    }
                )

            current_time += step_s

        num_valid_chunks = len(chunks_to_process)
        if total_chunks > num_valid_chunks:
            print(
                f"Skipped {total_chunks - num_valid_chunks}/{total_chunks} low-energy (silent) chunks using RMS threshold ({rms_threshold})."
            )

        raw_events = []

        # 1. AST (Audio Spectrogram Transformer) Inference
        if self.use_ast and self.ast_model is not None and self.ast_extractor is not None:
            print(f"[AudioExtractor] Running AST (Audio Spectrogram Transformer) inference over {num_valid_chunks} audio chunk(s)...")
            for item in chunks_to_process:
                chunk_audio = item["audio"]
                sp_label = item.get("spatial_label", "center")
                sp_str = f" on {sp_label} side" if sp_label in ["left", "right"] else ""

                try:
                    inputs = self.ast_extractor(chunk_audio, sampling_rate=16000, return_tensors="pt").to(DEVICE)
                    with torch.inference_mode():
                        logits = self.ast_model(**inputs).logits
                        probs = torch.sigmoid(logits)[0]
                        top_vals, top_idx = torch.topk(probs, k=2)

                    for score_val, idx_val in zip(top_vals, top_idx):
                        score = float(score_val.item())
                        label = self.ast_model.config.id2label.get(idx_val.item(), f"sound_{idx_val.item()}")
                        if score > 0.15 and "silence" not in label.lower():
                            raw_events.append(
                                {
                                    "type": "sound",
                                    "start_time": item["start_time"],
                                    "end_time": item["end_time"],
                                    "text": f"Sound of {label}{sp_str}",
                                    "score": score,
                                    "spatial_source": sp_label,
                                }
                            )
                except Exception as ast_err:
                    print(f"[AudioExtractor Warning] AST chunk inference failed: {ast_err}")

            if DEVICE == "cuda" or torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 2. CLAP Batch Inference Fallback
        elif self.audio_classifier is not None:
            for i in range(0, num_valid_chunks, batch_size):
                batch_items = chunks_to_process[i : i + batch_size]
                batch_audios = [item["audio"] for item in batch_items]

                chunk_idx_display = min(i + batch_size, num_valid_chunks)
                print(f"Processing audio chunk {chunk_idx_display}/{num_valid_chunks}")

                with torch.inference_mode():
                    batch_classifications = self.audio_classifier(
                        batch_audios, candidate_labels=self.sound_event_labels
                    )

                    if (
                        len(batch_audios) == 1
                        and isinstance(batch_classifications, list)
                        and len(batch_classifications) > 0
                        and isinstance(batch_classifications[0], dict)
                    ):
                        batch_classifications = [batch_classifications]

                    for item, classifications in zip(
                        batch_items, batch_classifications
                    ):
                        sp_label = item.get("spatial_label", "center")
                        sp_str = f" on {sp_label} side" if sp_label in ["left", "right"] else ""

                        for top_class in classifications[:2]:
                            if (
                                top_class["score"] > 0.12
                                and "silence" not in top_class["label"].lower()
                            ):
                                raw_events.append(
                                    {
                                        "type": "sound",
                                        "start_time": item["start_time"],
                                        "end_time": item["end_time"],
                                        "text": f"Sound of {top_class['label']}{sp_str}",
                                        "score": top_class["score"],
                                        "spatial_source": sp_label,
                                    }
                                )

                if DEVICE == "cuda" or torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Merge adjacent identical sound events
        merged_events = self._merge_adjacent_sound_events(
            raw_events, min_event_duration=min_event_duration
        )
        return merged_events

    def process_video(self, video_path: str):
        """Runs the full audio extraction pipeline."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav_path = temp_wav.name

        try:
            self.extract_audio_from_video(video_path, wav_path)

            speech_segments = self.transcribe_speech(wav_path)
            sound_events = self.detect_sound_events(wav_path)

            # Combine and sort by start time
            all_audio_facts = speech_segments + sound_events
            all_audio_facts.sort(key=lambda x: x["start_time"])

            return all_audio_facts
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)


if __name__ == "__main__":
    # Simple test
    pass

