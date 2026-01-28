import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

def main():
    device = "xpu"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    print(f"Loading model: {MODEL_PATH}")
    print(f"Device: {device}")

    try:
        tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",  # Using SDPA for XPU compatibility
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Tip: Ensure you have internet access to download weights from HuggingFace.")
        return

    # Single Inference
    print("\n--- Running Single Inference ---")
    if hasattr(torch, 'xpu'):
        torch.xpu.synchronize()

    t0 = time.time()
    wavs, sr = tts.generate_custom_voice(
        text="Actually, I have realized that I am someone who is particularly good at observing others' emotions.",
        language="English",
        speaker="Vivian",
        instruct="Say it with particular anger",
    )

    if hasattr(torch, 'xpu'):
        torch.xpu.synchronize()
    t1 = time.time()

    print(f"[CustomVoice Single] time: {t1 - t0:.3f}s")
    sf.write("qwen3_tts_test_custom_single_xpu.wav", wavs[0], sr)
    print("Saved: qwen3_tts_test_custom_single_xpu.wav")

    # Batch Inference
    print("\n--- Running Batch Inference ---")
    texts = ["Actually, I have realized that I am someone who is particularly good at observing others' emotions.", "She said she would be here by noon."]
    languages = ["English", "English"]
    speakers = ["Vivian", "Ryan"]
    instructs = ["", "Very happy."]

    if hasattr(torch, 'xpu'):
        torch.xpu.synchronize()

    t0 = time.time()
    wavs, sr = tts.generate_custom_voice(
        text=texts,
        language=languages,
        speaker=speakers,
        instruct=instructs,
        max_new_tokens=2048,
    )

    if hasattr(torch, 'xpu'):
        torch.xpu.synchronize()
    t1 = time.time()

    print(f"[CustomVoice Batch] time: {t1 - t0:.3f}s")
    for i, w in enumerate(wavs):
        filename = f"qwen3_tts_test_custom_batch_xpu_{i}.wav"
        sf.write(filename, w, sr)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    main()