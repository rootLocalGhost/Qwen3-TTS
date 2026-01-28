import os
import gc
import sys
import time
import logging
import warnings
import torch
import numpy as np
import gradio as gr
from typing import Optional
from qwen_tts import Qwen3TTSModel

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Qwen3-TTS-XPU")

# Enable Hugging Face progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

# Show model loading details
logging.getLogger("transformers").setLevel(logging.INFO)

# Device configuration
DEVICE = "xpu"
DTYPE = torch.bfloat16
ATTN_IMPL = "sdpa"

# Global variables to manage model state
current_model: Optional[Qwen3TTSModel] = None
current_model_type: str = ""

# Available models for different TTS tasks
MODELS = {
    "CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "VoiceClone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
}

# Supported speakers and languages
SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee"
]
LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian"
]

# Memory management utilities

def log_memory():
    """Log current XPU memory usage"""
    if hasattr(torch, 'xpu'):
        mem_alloc = torch.xpu.memory_allocated() / 1024**3
        mem_res = torch.xpu.memory_reserved() / 1024**3
        logger.info(f"XPU Memory: Allocated={mem_alloc:.2f}GB, Reserved={mem_res:.2f}GB")

def clean_memory():
    """Clear XPU cache and garbage collect"""
    global current_model
    logger.info("Cleaning memory...")
    if hasattr(torch, 'xpu'):
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    gc.collect()
    log_memory()

def load_model(task_type: str) -> str:
    """Load the specified model if not already loaded"""
    global current_model, current_model_type

    if current_model is not None and current_model_type == task_type:
        return f"Ready: {task_type}"

    logger.info(f"🔁 Request to switch model to: {task_type}")

    if current_model is not None:
        logger.info("Unloading previous model...")
        del current_model
        clean_memory()

    try:
        t_start = time.time()
        logger.info(f"Downloading/Loading {MODELS[task_type]} to {DEVICE}...")

        current_model = Qwen3TTSModel.from_pretrained(
            MODELS[task_type],
            device_map=DEVICE,
            dtype=DTYPE,
            attn_implementation=ATTN_IMPL,
        )
        current_model_type = task_type

        t_end = time.time()
        logger.info(f"✅ Model Loaded successfully in {t_end - t_start:.2f} seconds.")
        log_memory()
        return f"✅ Loaded {task_type}"
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return f"❌ Load Error: {str(e)}"

def normalize_gradio_audio(audio_data):
    """Convert audio data to float32 and handle stereo/mono conversion"""
    if audio_data is None: return None
    sr, wav = audio_data
    logger.info(f"Processing Input Audio: SR={sr}, Shape={wav.shape}, Dtype={wav.dtype}")

    # Convert to float32
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    elif wav.dtype == np.int32:
        wav = wav.astype(np.float32) / 2147483648.0

    # Stereo to Mono
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)

    return (wav, sr)

# Generation handler functions

def run_custom_voice(text, speaker, language, instruct, progress=gr.Progress()):
    """Generate custom voice audio"""
    msg = load_model("CustomVoice")
    if "Error" in msg: return None, msg

    logger.info(f"--- START CustomVoice Generation ---")
    logger.info(f"Speaker: {speaker} | Lang: {language} | Text Len: {len(text)}")

    t0 = time.time()
    if hasattr(torch, 'xpu'): torch.xpu.synchronize()

    try:
        wavs, sr = current_model.generate_custom_voice(
            text=text,
            language=language if language != "Auto" else None,
            speaker=speaker,
            instruct=instruct if instruct.strip() != "" else None,
            non_streaming_mode=True
        )

        if hasattr(torch, 'xpu'): torch.xpu.synchronize()
        t1 = time.time()
        logger.info(f"--- END CustomVoice Generation. Time: {t1-t0:.3f}s ---")
        return (sr, wavs[0]), "✅ Done"
    except Exception as e:
        logger.error("Generation Failed", exc_info=True)
        return None, f"❌ Error: {str(e)}"

def run_voice_design(text, instruct, language, progress=gr.Progress()):
    """Generate voice designed from text description"""
    msg = load_model("VoiceDesign")
    if "Error" in msg: return None, msg

    logger.info(f"--- START VoiceDesign Generation ---")
    logger.info(f"Prompt: {instruct[:50]}... | Lang: {language}")

    t0 = time.time()
    if hasattr(torch, 'xpu'): torch.xpu.synchronize()

    try:
        wavs, sr = current_model.generate_voice_design(
            text=text,
            language=language if language != "Auto" else None,
            instruct=instruct,
            non_streaming_mode=True
        )

        if hasattr(torch, 'xpu'): torch.xpu.synchronize()
        t1 = time.time()
        logger.info(f"--- END VoiceDesign Generation. Time: {t1-t0:.3f}s ---")
        return (sr, wavs[0]), "✅ Done"
    except Exception as e:
        logger.error("Generation Failed", exc_info=True)
        return None, f"❌ Error: {str(e)}"

def run_voice_clone(text, ref_audio, ref_text, language, xvec_only, progress=gr.Progress()):
    """Generate voice clone from reference audio"""
    logger.info("--- Validating Clone Inputs ---")
    if ref_audio is None:
        logger.warning("Reference Audio Missing")
        return None, "❌ Error: Reference Audio required."

    if not xvec_only and (ref_text is None or ref_text.strip() == ""):
        logger.warning("ICL Mode selected but Reference Text is missing")
        return None, "❌ Error: Input 'Reference Text' or enable 'X-Vector Only'."

    msg = load_model("VoiceClone")
    if "Error" in msg: return None, msg

    logger.info(f"--- START VoiceClone Generation ---")
    logger.info(f"Mode: {'X-Vector' if xvec_only else 'ICL (High Quality)'}")

    # Pre-process Audio
    t_process = time.time()
    processed_audio = normalize_gradio_audio(ref_audio)
    logger.info(f"Audio Processing Time: {time.time() - t_process:.4f}s")

    t0 = time.time()
    if hasattr(torch, 'xpu'): torch.xpu.synchronize()

    try:
        logger.info("Calling generate_voice_clone (non_streaming_mode=True)...")

        wavs, sr = current_model.generate_voice_clone(
            text=text,
            language=language if language != "Auto" else None,
            ref_audio=processed_audio,
            ref_text=ref_text,
            x_vector_only_mode=xvec_only,
            non_streaming_mode=True
        )

        if hasattr(torch, 'xpu'): torch.xpu.synchronize()
        t1 = time.time()

        output_len_sec = len(wavs[0]) / sr
        logger.info(f"--- END VoiceClone Generation. Time: {t1-t0:.3f}s ---")
        logger.info(f"Output Audio Length: {output_len_sec:.2f}s")

        return (sr, wavs[0]), "✅ Done"
    except Exception as e:
        logger.error("Clone Generation Failed", exc_info=True)
        return None, f"❌ Error: {str(e)}"

# UI layout configuration
CSS = """
footer {visibility: hidden}
.container {max-width: 900px; margin: auto; padding-top: 30px}
.btn-primary {background-color: #2563eb !important; color: white !important; font-size: 1.1em;}
.grp-box {border-radius: 10px; border: 1px solid #e5e7eb; padding: 15px; background: white;}
"""

with gr.Blocks(title="Qwen3-TTS XPU Verbose", css=CSS, theme=gr.themes.Soft()) as app:

    with gr.Column(elem_classes="container"):
        gr.Markdown("## 🎙️ Qwen3-TTS XPU")
        gr.Markdown("**Note:** Check your terminal for detailed execution logs.")
        status_box = gr.Textbox(label="Status", value="Ready.", interactive=False)

        with gr.Tabs():
            # Custom Voice tab
            with gr.Tab("🗣️ Custom Voice"):
                with gr.Group(elem_classes="grp-box"):
                    with gr.Row():
                        cv_speaker = gr.Dropdown(SPEAKERS, value="Vivian", label="Select Speaker")
                        cv_lang = gr.Dropdown(LANGUAGES, value="Auto", label="Language")
                    cv_text = gr.Textbox(label="Text to Speak", lines=4, value="Hello! This is running locally on my Intel ARC GPU.")
                    cv_instruct = gr.Textbox(label="Emotional Instruction", placeholder="e.g. Speak in a sad, whispering tone.")

                cv_btn = gr.Button("Generate", variant="primary", elem_classes="btn-primary")
                cv_audio = gr.Audio(label="Output", interactive=False, autoplay=True)
                cv_btn.click(run_custom_voice, [cv_text, cv_speaker, cv_lang, cv_instruct], [cv_audio, status_box])

            # Voice Design tab
            with gr.Tab("🎨 Voice Design"):
                with gr.Group(elem_classes="grp-box"):
                    vd_instruct = gr.Textbox(label="Voice Description", lines=2, value="A deep, raspy male voice.")
                    vd_text = gr.Textbox(label="Text to Speak", lines=4, value="I am a voice created entirely from your text description.")
                    vd_lang = gr.Dropdown(LANGUAGES, value="Auto", label="Language")

                vd_btn = gr.Button("Generate", variant="primary", elem_classes="btn-primary")
                vd_audio = gr.Audio(label="Output", interactive=False, autoplay=True)
                vd_btn.click(run_voice_design, [vd_text, vd_instruct, vd_lang], [vd_audio, status_box])

            # Voice Cloning tab
            with gr.Tab("🧬 Voice Cloning"):
                with gr.Group(elem_classes="grp-box"):
                    gr.Markdown("### 1. Reference Input")
                    vc_ref_audio = gr.Audio(label="Upload Audio (3-10s)", type="numpy")
                    vc_ref_text = gr.Textbox(label="Transcript", placeholder="Type exactly what is said in the audio above.")
                    vc_xvec = gr.Checkbox(label="Use X-Vector Only (Skip text, lower quality)", value=False)

                with gr.Group(elem_classes="grp-box"):
                    gr.Markdown("### 2. Target Output")
                    vc_text = gr.Textbox(label="Text to Speak", lines=3, value="This is my cloned voice speaking.")
                    vc_lang = gr.Dropdown(LANGUAGES, value="Auto", label="Language")

                vc_btn = gr.Button("Clone (Raw Output)", variant="primary", elem_classes="btn-primary")
                vc_audio = gr.Audio(label="Output Audio", interactive=False, autoplay=True)
                vc_btn.click(run_voice_clone, [vc_text, vc_ref_audio, vc_ref_text, vc_lang, vc_xvec], [vc_audio, status_box])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)