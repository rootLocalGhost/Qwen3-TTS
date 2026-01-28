import os
import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

def ensure_dir(d: str):
    """Create directory if it doesn't exist"""
    os.makedirs(d, exist_ok=True)

def run_case(tts: Qwen3TTSModel, out_dir: str, case_name: str, call_fn):
    """Run a test case and save the output"""
    if hasattr(torch, 'xpu'): torch.xpu.synchronize()
    t0 = time.time()

    wavs, sr = call_fn()

    if hasattr(torch, 'xpu'): torch.xpu.synchronize()
    t1 = time.time()

    print(f"[{case_name}] time: {t1 - t0:.3f}s, n_wavs={len(wavs)}")
    for i, w in enumerate(wavs):
        filename = os.path.join(out_dir, f"{case_name}_{i}.wav")
        sf.write(filename, w, sr)
        print(f"  -> Saved: {filename}")

def main():
    device = "xpu"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    OUT_DIR = "qwen3_tts_test_voice_clone_output_xpu"
    ensure_dir(OUT_DIR)

    print(f"Loading Base (Cloning) model on {device}...")
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # Reference audio files for cloning
    ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
    ref_audio_path_2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"

    ref_text_single = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    syn_text_single = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."

    # Common generation settings
    common_gen_kwargs = dict(
        max_new_tokens=2048,
        do_sample=True,
        top_k=50,
        temperature=0.9,
    )

    print("\n--- Case 1: Zero-shot Voice Cloning (using URL as reference) ---")
    run_case(
        tts, OUT_DIR, "clone_single_direct",
        lambda: tts.generate_voice_clone(
            text=syn_text_single,
            language="Auto",
            ref_audio=ref_audio_path_1,
            ref_text=ref_text_single,
            x_vector_only_mode=False, # Use ICL mode for better quality
            **common_gen_kwargs,
        ),
    )

    print("\n--- Case 2: Prompt Reuse (Faster for multiple lines) ---")
    def _case_reuse_prompt():
        print("  Extracting features from reference audio...")
        prompt_items = tts.create_voice_clone_prompt(
            ref_audio=ref_audio_path_1,
            ref_text=ref_text_single,
            x_vector_only_mode=False,
        )
        print("  Generating audio using cached prompt...")
        return tts.generate_voice_clone(
            text=syn_text_single,
            language="Auto",
            voice_clone_prompt=prompt_items,
            **common_gen_kwargs,
        )

    run_case(
        tts, OUT_DIR, "clone_single_prompt_reuse",
        _case_reuse_prompt,
    )

    print("\n--- Case 3: Batch Cloning (2 different voices at once) ---")
    ref_text_batch = [
        "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
        "There was even a situation where transactions almost came to a halt.",
    ]
    syn_text_batch = [
        "This is the first cloned voice speaking English.",
        "This is the second cloned voice speaking English.",
    ]

    run_case(
        tts, OUT_DIR, "clone_batch_mixed",
        lambda: tts.generate_voice_clone(
            text=syn_text_batch,
            language=["English", "English"],
            ref_audio=[ref_audio_path_1, ref_audio_path_2],
            ref_text=ref_text_batch,
            x_vector_only_mode=[False, False],
            **common_gen_kwargs,
        ),
    )

if __name__ == "__main__":
    main()