Here is your complete **Intel ARC (XPU) User Guide** for Qwen3-TTS.

This guide explains every file written for XPU inference, what it does, and how to use it to get the most out of your GPU.

---

## 🛠️ Prerequisites

Before running any file, ensure you have:
1.  **PyTorch XPU**: Installed in your environment.
2.  **SoX**: Install via pip `pip install sox` or install on your OS (`conda install -c conda-forge sox` or `sudo apt install sox`).

---

## 📂 File Breakdown

I created **6 specific files** adapted for Intel XPU. Here is what each one does:

### 1. The "Preset Voice" Generator
*   **File:** `examples/test_model_12hz_custom_voice_xpu.py`
*   **Model Used:** `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
*   **What it does:**
    *   Uses high-quality **preset speakers** (like "Vivian", "Ryan", "Uncle_Fu").
    *   Allows you to control emotions using text instructions (e.g., "Speak in a sad tone").
    *   **Key XPU Change:** Replaced `flash_attn` with `sdpa` (Scaled Dot Product Attention) and `cuda` with `xpu`.
*   **Run command:**
    ```bash
    python examples/test_model_12hz_custom_voice_xpu.py
    ```

### 2. The "Voice Designer"
*   **File:** `examples/test_model_12hz_voice_design_xpu.py`
*   **Model Used:** `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
*   **What it does:**
    *   Creates a **brand new voice** from a text description.
    *   You don't pick a speaker name; you write a prompt like *"A deep, raspy old man voice."*
*   **Run command:**
    ```bash
    python examples/test_model_12hz_voice_design_xpu.py
    ```

### 3. The "Voice Cloner"
*   **File:** `examples/test_model_12hz_base_xpu.py`
*   **Model Used:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
*   **What it does:**
    *   Takes a short audio file (3-10s) of **your voice** (or anyone's).
    *   Uses that audio as a reference to speak new text in that same voice.
    *   Supports "Zero-Shot" cloning (no training required).
*   **Run command:**
    ```bash
    python examples/test_model_12hz_base_xpu.py
    ```

### 4. The Web Interface (GUI)
*   **File:** `qwen_tts/cli/demo_xpu.py`
*   **What it does:**
    *   Launches a local web server (Gradio).
    *   Gives you a nice UI to type text, click buttons, and play audio.
    *   **Key XPU Change:** Removed the mandatory Flash Attention check and defaults to `xpu`.
*   **Run command:**
    *   *For Presets:* `python -m qwen_tts.cli.demo_xpu Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
    *   *For Design:* `python -m qwen_tts.cli.demo_xpu Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
    *   *For Cloning:* `python -m qwen_tts.cli.demo_xpu Qwen/Qwen3-TTS-12Hz-1.7B-Base`

### 5. Fine-Tuning: Data Prep
*   **File:** `finetuning/prepare_data_xpu.py`
*   **What it does:**
    *   Takes your `jsonl` file (list of audio paths and text).
    *   Uses the Tokenizer model to convert your audio files into "codes" that the AI understands.
    *   **Why use it:** This is step 1 of training the model on your own data.
*   **Run command:**
    ```bash
    python finetuning/prepare_data_xpu.py --input_jsonl train.jsonl --output_jsonl train_codes.jsonl
    ```

### 6. Fine-Tuning: Training Script
*   **File:** `finetuning/sft_12hz_xpu.py`
*   **What it does:**
    *   Actually trains the AI model on the data prepared in step 5.
    *   **Key XPU Change:** Modified the `Accelerator` and Model loading to ensure `bfloat16` and `sdpa` are used on Intel hardware to prevent crashes.
*   **Run command:**
    ```bash
    python finetuning/sft_12hz_xpu.py --train_jsonl train_codes.jsonl --batch_size 1 --num_epochs 3
    ```

---

## 🚀 Quick Start Workflows

### Scenario A: "I just want to generate text-to-speech quickly."
1.  Open `examples/test_model_12hz_custom_voice_xpu.py` in a text editor.
2.  Change the `text=` line to whatever you want to say.
3.  Change the `speaker=` to "Vivian", "Ryan", or "Uncle_Fu".
4.  Run: `python examples/test_model_12hz_custom_voice_xpu.py`
5.  Listen to the `.wav` file generated in the folder.

### Scenario B: "I want to experiment with a visual interface."
1.  Run the web demo:
    ```bash
    python -m qwen_tts.cli.demo_xpu Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    ```
2.  Open your browser to `http://127.0.0.1:8000`.
3.  Type text and click Generate.

### Scenario C: "I want to clone a specific character from an audio file."
1.  Get a clean `.wav` file of the character talking (3 to 10 seconds long).
2.  Open `examples/test_model_12hz_base_xpu.py`.
3.  Change `ref_audio_path_1` to point to your local `.wav` file.
4.  Change `ref_text_single` to match exactly what they said in that audio file.
5.  Change `syn_text_single` to what you *want* them to say.
6.  Run: `python examples/test_model_12hz_base_xpu.py`

---

## ❓ FAQ / Troubleshooting

**Q: I see "Warning: flash-attn is not installed."**
**A:** **Ignore it.** This warning comes from the original library. We are bypassing Flash Attention and using SDPA (Scaled Dot Product Attention), which is the native acceleration method for Intel ARC.

**Q: The first run is slow.**
**A:** This is normal for Intel ARC (XPU). The first time you run a model, it compiles "kernels" for your GPU. The second run will be significantly faster.

**Q: I get an Out of Memory (OOM) error.**
**A:** The A770 has 16GB VRAM, which is plenty for inference.
*   If generating: Ensure you aren't generating excessively long paragraphs in one go.
*   If training: Reduce `--batch_size` in `sft_12hz_xpu.py` to `1`.

**Q: Can I use `float16`?**
**A:** It is highly recommended to use `bfloat16` (Brain Float 16) on Intel ARC for AI workloads, as it is more stable than `float16`. My scripts default to this.