#!/usr/bin/env python3
import argparse
import logging
import os
import re
import tempfile
import zipfile
import torch
import numpy as np
import soundfile as sf
import torchaudio
import gradio as gr
from typing import Any, Dict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import ctc_forced_aligner
from ctc_forced_aligner import align, unflatten

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# --- Subtitle Generation (Model-based / Transformers) ---
def smart_split(text, language="zh"):
    if not text: return []
    text = re.sub(r"\s+", " ", text).strip()
    lang_lower = language.lower() if language else "zh"
    FORBIDDEN_START = "，。！？、）】》”’；：,.!?;:)]}>"
    
    def split_into_balanced_chunks(items, max_val, soft_limit, join_str=""):
        if not items: return []
        total = len(items)
        if total <= soft_limit: return [join_str.join(items)]
        num_chunks = (total + max_val - 1) // max_val
        target = total // num_chunks
        best_split = -1
        PUNCT_SPLIT = "，。！？、；：,.!?;:"
        search_start = max(1, target - 4)
        search_end = min(total - 1, target + 4)
        for j in range(search_end, search_start - 1, -1):
            if items[j-1][-1] in PUNCT_SPLIT:
                best_split = j
                break
        if best_split == -1: best_split = target
        return [join_str.join(items[:best_split])] + split_into_balanced_chunks(items[best_split:], max_val, soft_limit, join_str)

    def finalize_chunks(chunks):
        if not chunks: return []
        processed = [chunks[0]]
        for i in range(1, len(chunks)):
            current = chunks[i].strip()
            if not current: continue
            while current and (current[0] in FORBIDDEN_START or current[0] == "…"):
                processed[-1] += current[0]
                current = current[1:].strip()
            if current: processed.append(current)
        return processed

    if lang_lower in ["eng", "en", "auto"]:
        max_val, soft_limit = 10, 14
        sentences = re.split(r"(?<=[.!?]) +", text)
        all_chunks = []
        for s in sentences:
            all_chunks.extend(split_into_balanced_chunks(s.split(), max_val, soft_limit, " "))
        return finalize_chunks(all_chunks)
    else:
        max_val, soft_limit = 14, 18
        sentences = re.split(r"(?<=[。！？])", text)
        all_chunks = []
        for s in sentences:
            s = s.strip()
            if s: all_chunks.extend(split_into_balanced_chunks(list(s), max_val, soft_limit, ""))
        return finalize_chunks(all_chunks)

def format_timestamp(seconds):
    td = float(seconds)
    hours = int(td // 3600)
    minutes = int((td % 3600) // 60)
    secs = int(td % 60)
    millis = int((td % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# Global model cache for robustness
ALIGN_MODEL = None
ALIGN_PROCESSOR = None

def load_align_model():
    global ALIGN_MODEL, ALIGN_PROCESSOR
    if ALIGN_MODEL is None:
        model_path = "./models/zh_alignment"
        if not os.path.exists(model_path):
            model_path = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
        print(f"Loading alignment model from {model_path}...")
        ALIGN_PROCESSOR = Wav2Vec2Processor.from_pretrained(model_path)
        ALIGN_MODEL = Wav2Vec2ForCTC.from_pretrained(model_path)
    return ALIGN_MODEL, ALIGN_PROCESSOR

def text_to_srt_robust(text: str, audio_tuple, language="zh") -> str:
    model, processor = load_align_model()
    sr, waveform_int16 = audio_tuple
    
    # 1. Prep Audio
    waveform = waveform_int16.astype(np.float32) / 32767.0
    audio_pt = torch.from_numpy(waveform)
    if sr != 16000:
        audio_pt = torchaudio.transforms.Resample(sr, 16000)(audio_pt)
    
    # 2. Prep Text (Smart Split first)
    segments = smart_split(text, language)
    full_text = "".join(segments)
    
    # 3. Align
    vocab = processor.tokenizer.get_vocab()
    clean_text = "".join([c for c in full_text if c in vocab])
    if not clean_text: return "No alignable text."
    
    tokens = [vocab[c] for c in clean_text]
    with torch.inference_mode():
        logits = model(audio_pt.unsqueeze(0)).logits
        emissions = torch.log_softmax(logits, dim=-1).cpu()

    aligned_tokens, alignment_scores = align(emissions, tokens, "cpu")
    import torchaudio.functional as F
    token_spans = F.merge_tokens(torch.tensor(aligned_tokens).flatten(), torch.tensor(alignment_scores).flatten())
    word_spans = unflatten(token_spans, [1]*len(clean_text))
    
    stride = 320 # Wav2Vec2 standard
    word_results = []
    for i, span in enumerate(word_spans):
        if not span: continue
        word_results.append({
            'start': (span[0].start * stride) / 16000,
            'end': (span[-1].end * stride) / 16000
        })

    # 4. Group into SRT lines
    srt_output = ""
    current_idx = 0
    for i, line in enumerate(segments, 1):
        line_chars = [c for c in line if c in vocab]
        if not line_chars: continue
        
        line_start = None
        line_end = None
        count = 0
        while current_idx < len(word_results) and count < len(line_chars):
            if line_start is None: line_start = word_results[current_idx]['start']
            line_end = word_results[current_idx]['end']
            current_idx += 1
            count += 1
            
        if line_start is not None:
            srt_output += f"{i}\n{format_timestamp(line_start)} --> {format_timestamp(line_end)}\n{line}\n\n"
            
    return srt_output

def get_slug(text, max_tokens=8):
    clean_text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", clean_text)
    selected = tokens[:max_tokens]
    if not selected: return "output"
    res = selected[0]
    for i in range(1, len(selected)):
        prev, curr = selected[i-1], selected[i]
        if re.match(r"[a-zA-Z0-9]", prev) or re.match(r"[a-zA-Z0-9]", curr):
            res += " " + curr
        else:
            res += curr
    return res.strip()

# --- Main Logic ---
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

def build_demo(model: OmniVoice):
    sampling_rate = model.sampling_rate

    def _gen_core(text, language, ref_audio, instruct, num_step, guidance_scale, denoise, speed, duration, preprocess_prompt, postprocess_output, mode, generate_srt=True, ref_text=None):
        if not text or not text.strip(): return None, "", None, "Text required."
        
        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale or 0.5),
            denoise=bool(denoise),
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output)
        )
        
        lang = language if language != "Auto" else None
        kw = dict(text=text.strip(), language=lang, generation_config=gen_config)
        if speed and float(speed) != 1.0: kw["speed"] = float(speed)
        if duration and float(duration) > 0: kw["duration"] = float(duration)
        if instruct: kw["instruct"] = instruct.strip()
        
        if mode == "clone":
            if not ref_audio: return None, "", None, "Ref audio required."
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, "", None, str(e)

        waveform = (audio[0] * 32767).astype(np.int16)
        audio_tuple = (sampling_rate, waveform)
        slug = get_slug(text)
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"{slug}.wav")
        sf.write(audio_path, waveform, sampling_rate)

        srt_content = ""
        download_path = audio_path
        if generate_srt:
            srt_content = text_to_srt_robust(text, audio_tuple, language=lang)
            zip_path = os.path.join(temp_dir, f"{slug}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(audio_path, arcname=f"{slug}.wav")
                srt_path = os.path.join(temp_dir, f"{slug}.srt")
                with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)
                zipf.write(srt_path, arcname=f"{slug}.srt")
            download_path = zip_path

        return audio_path, srt_content, download_path, "Done."

    with gr.Blocks(title="OmniVoice (Robust-Model)") as demo:
        gr.Markdown("# OmniVoice - Robust Model-based Alignment")
        with gr.Tabs():
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column():
                        vc_text = gr.Textbox(label="Text", lines=4)
                        vc_ref_audio = gr.Audio(label="Ref Audio", type="filepath")
                        vc_ref_text = gr.Textbox(label="Ref Text (optional)")
                        vc_lang = gr.Dropdown(label="Language", choices=_ALL_LANGUAGES, value="Auto")
                        vc_gen_srt = gr.Checkbox(label="Generate SRT", value=True)
                        with gr.Accordion("Settings", open=False):
                            vc_sp = gr.Slider(0.5, 1.5, value=0.9, label="Speed")
                            vc_ns = gr.Slider(4, 64, value=32, label="Steps")
                            vc_gs = gr.Slider(0, 5, value=0.5, label="Guidance")
                        vc_btn = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        vc_audio = gr.Audio(label="Output")
                        vc_srt = gr.Textbox(label="SRT", lines=10)
                        vc_dl = gr.DownloadButton("Download ZIP", visible=False)
                        vc_status = gr.Textbox(label="Status")
                
                vc_btn.click(lambda *args: _gen_core(*args, mode="clone"), 
                    inputs=[vc_text, vc_lang, vc_ref_audio, gr.State(""), vc_ns, vc_gs, gr.State(True), vc_sp, gr.State(None), gr.State(True), gr.State(True), gr.State("clone"), vc_gen_srt, vc_ref_text],
                    outputs=[vc_audio, vc_srt, vc_dl, vc_status]
                ).then(lambda dl: gr.update(visible=True, value=dl), inputs=[vc_dl], outputs=[vc_dl])

    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OmniVoice(checkpoint=args.model, device=device)
    demo = build_demo(model)
    demo.launch(server_port=args.port, share=True)

if __name__ == "__main__":
    main()
