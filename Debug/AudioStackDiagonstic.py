import sys
import traceback
import importlib
import pkg_resources
import platform

print("\n" + "="*70)
print("🔍 GRIMLOCK HARD DEBUG — AUDIO STACK")
print("="*70)

# --------------------------------------------------
# 🧠 ENVIRONMENT INFO
# --------------------------------------------------
print("\n🧠 PYTHON ENVIRONMENT")
print("-"*50)
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Platform:", platform.platform())

print("\n📦 INSTALLED KEY PACKAGES:")
for pkg in ["tensorflow", "tensorflow_hub", "basic_pitch", "keras"]:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f"  ✅ {pkg}=={version}")
    except:
        print(f"  ❌ {pkg} NOT FOUND")

# --------------------------------------------------
# 🔥 FORCE IMPORT WITH REAL ERRORS
# --------------------------------------------------
def test_import(name):
    print(f"\n🔍 Testing import: {name}")
    try:
        module = importlib.import_module(name)
        print(f"  ✅ SUCCESS: {name}")
        return module
    except Exception as e:
        print(f"  ❌ FAILED: {name}")
        traceback.print_exc()
        return None

tf = test_import("tensorflow")
hub = test_import("tensorflow_hub")
bp = test_import("basic_pitch")

# --------------------------------------------------
# ⚙️ TENSORFLOW DEEP CHECK
# --------------------------------------------------
if tf:
    print("\n⚙️ TensorFlow Deep Check")
    print("-"*50)
    try:
        print("TF version:", tf.__version__)
        print("GPU available:", tf.config.list_physical_devices('GPU'))
    except Exception:
        traceback.print_exc()

# --------------------------------------------------
# 🔥 TF-HUB MODEL LOAD TEST (SPICE)
# --------------------------------------------------
if hub:
    print("\n🔥 Testing TF-Hub SPICE Model Load")
    print("-"*50)
    try:
        model_url = "https://tfhub.dev/google/spice/2"
        model = hub.load(model_url)
        print("  ✅ SPICE model LOADED SUCCESSFULLY")
    except Exception as e:
        print("  ❌ SPICE MODEL LOAD FAILED")
        traceback.print_exc()

# --------------------------------------------------
# 🎵 BASIC PITCH MODEL LOAD TEST
# --------------------------------------------------
if bp:
    print("\n🎵 Testing Basic Pitch Model Load")
    print("-"*50)
    try:
        from basic_pitch.inference import predict

        print("  ✅ basic_pitch.predict import OK")

        # dummy test (no audio needed, just checking function call structure)
        print("  ⚠️ Skipping full inference (requires audio file)")
    except Exception:
        print("  ❌ BASIC PITCH FAILED DURING IMPORT/SETUP")
        traceback.print_exc()

# --------------------------------------------------
# 🚨 FINAL DIAGNOSIS
# --------------------------------------------------
print("\n" + "="*70)
print("📊 FINAL DIAGNOSIS")
print("="*70)

if not tf:
    print("❌ TensorFlow is fundamentally broken in this environment.")
if tf and not hub:
    print("⚠️ TensorFlow works, but TF-Hub is broken (version mismatch likely).")
if hub:
    print("✅ TF-Hub import works — failures likely occur during model load.")

if not bp:
    print("❌ Basic Pitch is broken (dependency or TF issue).")
else:
    print("✅ Basic Pitch import works — check runtime inference separately.")

print("\n🧠 If you see stack traces above — THAT is your real problem.")
print("🧠 Copy them. That’s what we fix next.")
print("="*70)

import time
import traceback


def diagnose_demucs_hang(audio_path):
    print(f"Diagnosing Demucs on: {audio_path}")

    # Step 1: Check file
    import librosa
    duration = librosa.get_duration(path=audio_path)
    print(f"Audio duration: {duration:.2f} seconds")

    # Step 2: Test Demucs loading
    print("Loading Demucs...")
    start = time.time()
    try:
        from demucs import separate
        print(f"Demucs import took: {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Import failed: {e}")
        return

    # Step 3: Test with small segment
    print("Testing with 10-second segment...")
    y, sr = librosa.load(audio_path, duration=10, sr=44100)

    import tempfile
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, y, sr)

        try:
            separator = demucs.api.Separator(progress=True)
            start = time.time()
            stems = separator.separate(tmp.name)
            print(f"✅ 10-second test passed in {time.time() - start:.2f}s")
        except Exception as e:
            print(f"❌ 10-second test failed: {e}")
            traceback.print_exc()


# Run diagnosis
diagnose_demucs_hang("your_audio_file.wav")