#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
import os
import re
import tempfile
import zipfile
from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# For torchaudio alignment
try:
    from torchaudio.pipelines import MMS_FA as ALIGN_BUNDLE
    ALIGN_MODEL = None  # Lazy load
except ImportError:
    ALIGN_BUNDLE = None


def smart_split(text, language="eng"):
    """
    Split text into smaller chunks based on language rules.
    Balances line lengths and ensures punctuation is attached to the previous line.
    English: 10±4 words
    CJK: 14±4 characters
    """
    if not text:
        return []

    # Clean text: remove multiple newlines and whitespace
    text = re.sub(r"\s+", " ", text).strip()
    lang_lower = language.lower() if language else "eng"

    # Punctuation that should not start a line
    FORBIDDEN_START = "，。！？、）】》”’；：,.!?;:)]}>"

    def split_into_balanced_chunks(items, max_val, soft_limit, join_str=""):
        if not items:
            return []
        total = len(items)
        if total <= soft_limit:
            return [join_str.join(items)]
        
        # Calculate number of chunks needed
        num_chunks = (total + max_val - 1) // max_val
        target = total // num_chunks
        
        # Try to find punctuation within tolerance range (target ± 4)
        best_split = -1
        PUNCT_SPLIT = "，。！？、；：,.!?;:"
        
        search_start = max(1, target - 4)
        search_end = min(total - 1, target + 4)
        
        # Search from right to left to maximize the first chunk within limits
        for j in range(search_end, search_start - 1, -1):
            char_or_word = items[j-1]
            if char_or_word[-1] in PUNCT_SPLIT:
                best_split = j
                break
        
        # Fallback to balanced split if no punctuation found
        if best_split == -1:
            best_split = target
            
        first = join_str.join(items[:best_split])
        remaining = items[best_split:]
        return [first] + split_into_balanced_chunks(remaining, max_val, soft_limit, join_str)

    def finalize_chunks(chunks):
        if not chunks:
            return []
        processed = [chunks[0]]
        for i in range(1, len(chunks)):
            current = chunks[i].strip()
            if not current:
                continue
            
            # Pull up leading punctuation to the previous chunk
            while current and (current[0] in FORBIDDEN_START or current[0] == "…"):
                punct = current[0]
                processed[-1] += punct
                current = current[1:].strip()
            
            if current:
                processed.append(current)
        return processed

    if lang_lower in ["eng", "en", "auto"]:
        # 10±4 words
        max_val, soft_limit = 10, 14
        sentences = re.split(r"(?<=[.!?]) +", text)
        all_chunks = []
        for s in sentences:
            words = s.split()
            all_chunks.extend(split_into_balanced_chunks(words, max_val, soft_limit, " "))
        return finalize_chunks(all_chunks)

    # CJK (Chinese, Japanese, Korean)
    elif lang_lower in ["cmn", "zh", "jpn", "ja", "kor", "ko"]:
        # 14±4 characters
        max_val, soft_limit = 14, 18
        sentences = re.split(r"(?<=[。！？])", text)
        all_chunks = []
        for s in sentences:
            s = s.strip()
            if not s: continue
            all_chunks.extend(split_into_balanced_chunks(list(s), max_val, soft_limit, ""))
        return finalize_chunks(all_chunks)

    # Default fallback for other languages
    else:
        # Generic word-based split if space exists, otherwise char-based
        if " " in text:
            return finalize_chunks(split_into_balanced_chunks(text.split(), 10, 14, " "))
        else:
            return finalize_chunks(split_into_balanced_chunks(list(text), 14, 18, ""))


def text_to_srt_with_timestamps(text: str, audio_tuple, language="eng") -> str:
    """
    Generate SRT subtitle content using torchaudio forced alignment.
    This replaces the unreliable aeneas library for better compatibility.
    """
    if not text or audio_tuple is None:
        return "No text or audio to align."
    
    if not ALIGN_BUNDLE:
        return f"Alignment error: torchaudio alignment bundle not available.\n\nFalling back to plain text:\n{text}"

    global ALIGN_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 1. Load model (Lazy load Meta MMS Aligner)
        if ALIGN_MODEL is None:
            ALIGN_MODEL = ALIGN_BUNDLE.get_model().to(device)
        
        sr, audio_arr = audio_tuple
        # Ensure audio is float32 for alignment
        if audio_arr.dtype == np.int16:
            audio_arr = audio_arr.astype(np.float32) / 32767.0
        
        audio_tensor = torch.from_numpy(audio_arr).float()
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        
        # MMS_FA expects 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_tensor = resampler(audio_tensor)
        
        audio_tensor = audio_tensor.to(device)
        
        # 2. Get segments using our smart_split
        segments = smart_split(text, language)
        if not segments:
            return text
            
        # 3. Tokenize text segment-by-segment to track token counts
        tokenizer = ALIGN_BUNDLE.get_tokenizer()
        aligner = ALIGN_BUNDLE.get_aligner()
        
        all_tokens = []
        segment_token_counts = []
        
        for seg in segments:
            # Clean and split into words to avoid space character issues
            words = re.sub(r'[^\w\s\u4e00-\u9fff]', '', seg).lower().split()
            seg_tokens = []
            for w in words:
                w_t = tokenizer(w)
                if w_t:
                    seg_tokens.extend(w_t)
            
            all_tokens.extend(seg_tokens)
            segment_token_counts.append(len(seg_tokens))
        
        if not all_tokens:
             return text
             
        # 4. Run Alignment
        with torch.inference_mode():
            emission, _ = ALIGN_MODEL(audio_tensor)
            token_spans = aligner(emission[0], all_tokens)
        
        if not token_spans:
            return text
            
        # Frame-to-Second mapping
        num_frames = emission.shape[1]
        total_duration = len(audio_arr) / sr
        frame_to_sec = total_duration / num_frames
        
        # 5. Map spans back to segments
        srt_lines = []
        token_offset = 0
        
        def get_val(obj, attr):
            # Recursively drill down into nested lists to find the span attribute
            while isinstance(obj, list) and len(obj) > 0:
                obj = obj[0] if attr == 'start' else obj[-1]
            return getattr(obj, attr, 0)

        for i, (seg, count) in enumerate(zip(segments, segment_token_counts)):
            if count == 0:
                continue
                
            start_token_idx = token_offset
            end_token_idx = token_offset + count
            
            if start_token_idx < len(token_spans):
                # Use the recursive helper to avoid 'list' attribute errors
                start_frame = get_val(token_spans[start_token_idx], 'start')
                actual_end_idx = min(end_token_idx - 1, len(token_spans) - 1)
                end_frame = get_val(token_spans[actual_end_idx], 'end')
                
                start_time = start_frame * frame_to_sec
                end_time = end_frame * frame_to_sec
                
                # Ensure timing doesn't overlap backwards
                if srt_lines and start_time < srt_lines[-1][2]:
                    start_time = srt_lines[-1][2]
                
                srt_lines.append((i + 1, start_time, end_time, seg))
            
            token_offset += count

        # Format as SRT
        def format_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds % 1) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        res = ""
        for idx, start, end, t in srt_lines:
            res += f"{idx}\n{format_time(start)} --> {format_time(end)}\n{t}\n\n"
            
        return res.strip() if res else text

    except Exception as e:
        return f"Alignment error: {str(e)}\n\nFalling back to plain text:\n{text}"


def get_slug(text, max_tokens=8):
    """Generate a filename-safe slug from the first few words/characters."""
    # Remove punctuation and special characters, keep letters, numbers and CJK
    clean_text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    # Tokenize: CJK characters or English words
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", clean_text)
    selected = tokens[:max_tokens]
    if not selected:
        return "output"

    # Join tokens with space if they are English words, or keep together for CJK
    res = selected[0]
    for i in range(1, len(selected)):
        prev, curr = selected[i - 1], selected[i]
        # Add space if either is English/Number, or between English and CJK
        if re.match(r"[a-zA-Z0-9]", prev) or re.match(r"[a-zA-Z0-9]", curr):
            res += " " + curr
        else:
            res += curr
    return res.strip()


def get_best_device():
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
) -> gr.Blocks:

    sampling_rate = model.sampling_rate

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        generate_srt=True,
        ref_text=None,
    ):
        if not text or not text.strip():
            return None, "", None, "Please enter the text to synthesize."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 0.5,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, "", None, "Please upload a reference audio."
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, "", None, f"Error: {type(e).__name__}: {e}"

        waveform = (audio[0] * 32767).astype(np.int16)
        audio_tuple = (sampling_rate, waveform)

        # Generate slug for filenames
        slug = get_slug(text)
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"{slug}.wav")
        sf.write(audio_path, waveform, sampling_rate)

        srt_content = ""
        download_path = audio_path

        if generate_srt:
            srt_content = text_to_srt_with_timestamps(text, audio_tuple, language=lang)
            # Create a zip file containing both audio and srt
            zip_path = os.path.join(temp_dir, f"{slug}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(audio_path, arcname=f"{slug}.wav")
                srt_path = os.path.join(temp_dir, f"{slug}.srt")
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                zipf.write(srt_path, arcname=f"{slug}.srt")
            download_path = zip_path

        return audio_path, srt_content, download_path, "Done."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Language (optional) / 语种 (可选)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Keep as Auto to auto-detect the language.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Generation Settings (optional)", open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=0.9,
                step=0.05,
                label="Speed",
                info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set.",
            )
            du = gr.Number(
                value=None,
                label="Duration (seconds)",
                info=(
                    "Leave empty to use speed."
                    " Set a fixed duration to override speed."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="Inference Steps",
                info="Default: 32. Lower = faster, higher = better quality.",
            )
            dn = gr.Checkbox(
                label="Denoise",
                value=True,
                info="Default: enabled. Uncheck to disable denoising.",
            )
            gs = gr.Slider(
                0.0,
                5.0,
                value=0.5,
                step=0.1,
                label="Guidance Scale (CFG)",
                info="Default: 0.5.",
            )
            pp = gr.Checkbox(
                label="Preprocess Prompt",
                value=True,
                info="apply silence removal and trimming to the reference "
                "audio, add punctuation in the end of reference text (if not already)",
            )
            po = gr.Checkbox(
                label="Postprocess Output",
                value=True,
                info="Remove long silences from generated audio.",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        gr.Markdown(
            """
# OmniVoice Demo

State-of-the-art text-to-speech model for **600+ languages**, supporting:

- **Voice Clone** — Clone any voice from a reference audio
- **Voice Design** — Create custom voices with speaker attributes

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi AI Lab Next-gen Kaldi team.
"""
        )

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="Reference Audio / 参考音频",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Recommended: 3–10 seconds audio. "
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label=("Reference Text (optional)" " / 参考音频文本（可选）"),
                            lines=2,
                            placeholder="Transcript of the reference audio. Leave empty"
                            " to auto-transcribe via ASR models.",
                        )
                        vc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        vc_gen_srt = gr.Checkbox(
                            label="Generate Subtitles (SRT) / 生成字幕",
                            value=True,
                            info="If enabled, audio and SRT will be packed in a ZIP file.",
                        )
                        with gr.Accordion("Instruct (optional)", open=False):
                            vc_instruct = gr.Textbox(label="Instruct", lines=2)
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="filepath",
                        )
                        vc_srt = gr.Textbox(
                            label="Subtitle (SRT) - Aligned from your input text",
                            lines=12,
                            show_copy_button=True,
                            interactive=False,
                        )
                        vc_download = gr.DownloadButton(
                            "📥 Download Results (Audio + SRT)", visible=False
                        )
                        vc_status = gr.Textbox(label="Status / 状态", lines=2)

                def _clone_fn(
                    text,
                    lang,
                    ref_aud,
                    ref_text,
                    instruct,
                    ns,
                    gs,
                    dn,
                    sp,
                    du,
                    pp,
                    po,
                    gen_srt,
                ):
                    return _gen(
                        text,
                        lang,
                        ref_aud,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        generate_srt=gen_srt,
                        ref_text=ref_text or None,
                    )

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                        vc_gen_srt,
                    ],
                    outputs=[vc_audio, vc_srt, vc_download, vc_status],
                ).then(
                    lambda dl: gr.update(visible=True, value=dl),
                    inputs=[vc_download],
                    outputs=[vc_download],
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vd_lang = _lang_dropdown()
                        vd_gen_srt = gr.Checkbox(
                            label="Generate Subtitles (SRT) / 生成字幕",
                            value=True,
                            info="If enabled, audio and SRT will be packed in a ZIP file.",
                        )

                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        vd_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="filepath",
                        )
                        vd_srt = gr.Textbox(
                            label="Subtitle (SRT) - Aligned from your input text",
                            lines=12,
                            show_copy_button=True,
                            interactive=False,
                        )
                        vd_download = gr.DownloadButton(
                            "📥 Download Results (Audio + SRT)", visible=False
                        )
                        vd_status = gr.Textbox(label="Status / 状态", lines=2)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # Dialects have no English equivalent
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, gen_srt, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                        generate_srt=gen_srt,
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                        vd_gen_srt,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_srt, vd_download, vd_status],
                ).then(
                    lambda dl: gr.update(visible=True, value=dl),
                    inputs=[vd_download],
                    outputs=[vd_download],
                )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
