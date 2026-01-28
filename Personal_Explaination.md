### 1. How many models are there?
There are **6 specific models** released in this package, categorized by size (1.7B or 0.6B) and function:

#### The Tokenizer (The Engine)
1.  **Qwen3-TTS-Tokenizer-12Hz**: The core component that compresses audio into tokens. Used by all models below.

#### 1.7B Parameters (Higher Quality, recommended for your 16GB VRAM)
2.  **Qwen3-TTS-12Hz-1.7B-CustomVoice**:
    *   **Best for:** Using preset high-quality speakers (Vivian, Ryan, etc.) and controlling their emotion/style via text instructions.
3.  **Qwen3-TTS-12Hz-1.7B-VoiceDesign**:
    *   **Best for:** Creating a brand new voice from scratch by describing it (e.g., "A deep, raspy old man's voice").
4.  **Qwen3-TTS-12Hz-1.7B-Base**:
    *   **Best for:** Voice Cloning (Zero-shot) using a reference audio clip, and for Fine-Tuning.

#### 0.6B Parameters (Faster, Lower VRAM)
5.  **Qwen3-TTS-12Hz-0.6B-CustomVoice**: Same features as the 1.7B version, just smaller/faster.
6.  **Qwen3-TTS-12Hz-0.6B-Base**: Same features as the 1.7B version, just smaller/faster.

---

### 2. What can we do?

*   **Custom Voice Generation:** You pick a preset speaker (e.g., "Vivian"), type text, and give an instruction like "Say this in a crying tone" or "Say this very quickly and happily."
*   **Voice Design:** You don't pick a speaker. Instead, you type a prompt like: *"A young female voice, slightly high-pitched, speaking in a professional news-anchor tone."* The AI generates that voice.
*   **Voice Cloning:** You give the model a 3-10 second `.wav` file of *your* voice (or anyone else's). The model then speaks your text using that voice.
*   **Fine-Tuning:** You can train the model on a specific person's voice data to make it permanently sound like them (requires the Base model).

---

### 3. How to Run (On Intel ARC A770)

Since you are using an Intel ARC GPU, **do not** use the standard `qwen-tts-demo` command directly, as it defaults to CUDA. Use the scripts I provided in the previous answer.

#### Prerequisite
Ensure you have the modified files saved:
1.  Save the second code block from my previous answer as `qwen_tts/cli/demo_xpu.py`.
2.  Save the first code block as `examples/test_model_12hz_custom_voice_xpu.py`.

#### Option A: Run the Web UI (Gradio)
This is the easiest way to try all features.

1.  **Run Custom Voice (Presets & Instructions):**
    ```bash
    python -m qwen_tts.cli.demo_xpu Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    ```

2.  **Run Voice Design (Prompt-based creation):**
    ```bash
    python -m qwen_tts.cli.demo_xpu Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
    ```

3.  **Run Voice Cloning (Reference Audio):**
    ```bash
    python -m qwen_tts.cli.demo_xpu Qwen/Qwen3-TTS-12Hz-1.7B-Base
    ```

*After running a command, open the local URL (usually `http://0.0.0.0:8000`) in your browser.*

#### Option B: Run via Python Script (Command Line)
If you want to run generation programmatically without a UI:

```bash
python examples/test_model_12hz_custom_voice_xpu.py
```
*(This script downloads the CustomVoice model and generates audio files in the current directory).*

#### Option C: Fine-Tuning on ARC
To train the model on your own voice data:

1.  Prepare your `train_raw.jsonl` (format is in the README).
2.  **Step 1 (Extract Codes):**
    ```bash
    python finetuning/prepare_data_xpu.py \
      --input_jsonl train_raw.jsonl \
      --output_jsonl train_with_codes.jsonl
    ```
3.  **Step 2 (Train):**
    ```bash
    python finetuning/sft_12hz_xpu.py \
      --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
      --train_jsonl train_with_codes.jsonl \
      --batch_size 1 \
      --num_epochs 3
    ```
    *(Note: I set batch_size to 1 to be safe with VRAM, but on an A770 16GB, you might be able to push it to 2 or 4).*