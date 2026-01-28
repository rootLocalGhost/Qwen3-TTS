import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

def main():
    device = "xpu"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

    print(f"Loading Voice Design model on {device}...")
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # Required for XPU compatibility
    )

    # Single inference example
    print("\n--- Generating Single Voice Design ---")
    text = "Big brother, you're back! I've been waiting for you for so long, give me a hug!"
    instruct = "Express a cutesy, childish girl's voice with a high pitch and obvious fluctuations, creating a clingy, affected, and deliberately cute auditory effect."

    if hasattr(torch, 'xpu'): torch.xpu.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_voice_design(
        text=text,
        language="English",
        instruct=instruct,
    )

    if hasattr(torch, 'xpu'): torch.xpu.synchronize()
    t1 = time.time()
    print(f"Time: {t1 - t0:.3f}s")

    filename = "qwen3_tts_voice_design_single_xpu.wav"
    sf.write(filename, wavs[0], sr)
    print(f"Saved: {filename}")

    # Batch inference example
    print("\n--- Generating Batch Voice Design ---")
    texts = [
        "Big brother, you're back! I've been waiting for you for so long, give me a hug!",
        "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
    ]
    languages = ["English", "English"]
    instructs = [
        "Express a cutesy, childish girl's voice with a high pitch and obvious fluctuations, creating a clingy, affected, and deliberately cute auditory effect.",
        "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
    ]

    if hasattr(torch, 'xpu'): torch.xpu.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_voice_design(
        text=texts,
        language=languages,
        instruct=instructs,
        max_new_tokens=2048,
    )

    if hasattr(torch, 'xpu'): torch.xpu.synchronize()
    t1 = time.time()
    print(f"Time: {t1 - t0:.3f}s")

    for i, w in enumerate(wavs):
        filename = f"qwen3_tts_voice_design_batch_xpu_{i}.wav"
        sf.write(filename, w, sr)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    main()