# FILE: scripts/local_ai/verify_stack.py
r"""
Quick verification of the Local AI stack.
Run after install_local_ai_stack.ps1 to confirm everything works.

Usage:  python D:\Orb\scripts\local_ai\verify_stack.py
"""
import sys
import time
import json

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

def check_whisper():
    """Verify faster-whisper loads on CUDA with large-v3."""
    print("\n--- 1. faster-whisper (STT) ---")
    try:
        from faster_whisper import WhisperModel
        print("  Import: OK")
    except ImportError as e:
        print(f"  {FAIL} Import failed: {e}")
        return False

    try:
        t0 = time.time()
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        dt = time.time() - t0
        print(f"  {PASS} large-v3 loaded on CUDA (float16) in {dt:.1f}s")
        return True
    except Exception as e:
        print(f"  {WARN} CUDA failed: {e}")
        try:
            model = WhisperModel("large-v3", device="cpu", compute_type="int8")
            print(f"  {WARN} Loaded on CPU only — check CUDA/cuDNN")
            return True
        except Exception as e2:
            print(f"  {FAIL} Cannot load model at all: {e2}")
            return False


def check_ollama():
    """Verify Ollama is running and has the model."""
    print("\n--- 2. Ollama (Local LLM) ---")
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            print(f"  Ollama running, {len(models)} model(s) available")
            has_qwen = any("qwen2.5-coder" in m for m in models)
            if has_qwen:
                print(f"  {PASS} qwen2.5-coder found")
            else:
                print(f"  {WARN} qwen2.5-coder not found. Available: {models}")
                print(f"        Run: ollama pull qwen2.5-coder:14b")
            return True
    except Exception as e:
        print(f"  {FAIL} Ollama not reachable at 127.0.0.1:11434")
        print(f"        Is Ollama running?  Start it with: ollama serve")
        print(f"        Error: {e}")
        return False


def check_piper():
    """Verify piper-tts is importable and voice exists."""
    print("\n--- 3. Piper TTS ---")
    try:
        import piper
        print("  Import: OK")
    except ImportError as e:
        print(f"  {FAIL} Import failed: {e}")
        return False

    import os
    voice_path = r"D:\LocalAI\models\piper-voices\en_GB-alan-medium.onnx"
    if os.path.exists(voice_path):
        size_mb = os.path.getsize(voice_path) / (1024 * 1024)
        print(f"  {PASS} Voice model found ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  {WARN} Voice model not found at {voice_path}")
        print(f"        Re-run the installer or download manually")
        return False


def check_cuda():
    """Check CUDA availability via ctranslate2 (used by faster-whisper)."""
    print("\n--- 0. CUDA / GPU ---")
    try:
        import ctranslate2
        types = ctranslate2.get_supported_compute_types("cuda")
        print(f"  {PASS} CUDA available, compute types: {types}")
        return True
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  {PASS} CUDA via PyTorch: {name} ({vram:.1f} GB)")
            return True
        else:
            print(f"  {FAIL} torch.cuda.is_available() = False")
            return False
    except ImportError:
        print(f"  {WARN} Neither ctranslate2 nor torch available for CUDA check")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("  ASTRA Local AI Stack Verification")
    print("=" * 50)

    results = {
        "CUDA":    check_cuda(),
        "Whisper": check_whisper(),
        "Ollama":  check_ollama(),
        "Piper":   check_piper(),
    }

    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {name:12s} {status}")

    all_ok = all(results.values())
    print()
    if all_ok:
        print("  All checks passed. Stack is ready for ASTRA.")
    else:
        print("  Some checks failed. See details above.")
    
    sys.exit(0 if all_ok else 1)
