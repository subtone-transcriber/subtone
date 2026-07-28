import sys
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

# Automatically import correct Demucs backend
IS_MLX = False
try:
    from demucs_mlx.api import Separator, save_audio
    IS_MLX = True
except ImportError:
    import torch
    from demucs.api import Separator, save_audio

def process_file(file_path_str, separator, duration=None):
    audio_path = Path(file_path_str).resolve()

    if not audio_path.exists():
        print(f"❌ Error: File not found: {audio_path}")
        return

    out_dir = Path(f"./stems_{audio_path.stem}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==================================================")
    print(f"🎵 Processing: {audio_path.name}")
    if duration:
        print(f"⏱️  Duration limited to: {duration} seconds")
    print(f"==================================================")

    # Let demucs process the whole file natively via Apple Silicon GPU / Metal
    if duration is None:
        print(f"[1/2] Separating stems natively...")
        if IS_MLX:
            _, stems = separator.separate_audio_file(audio_path)
        else:
            origin, stems = separator.separate_audio_file(audio_path)
    else:
        # If duration flag is set, load and trim in memory using librosa
        import librosa
        print(f"[1/2] Loading {duration}s slice into memory...")
        y, sr = librosa.load(audio_path, sr=None, mono=False, duration=duration)

        # Ensure 2D array shape (channels, samples)
        if y.ndim == 1:
            y = np.stack([y, y])

        print(f"[2/2] Separating stems...")
        if IS_MLX:
            stems = separator.separate_tensor(y)
        else:
            tensor_y = torch.from_numpy(y).float()
            stems = separator.separate_tensor(tensor_y)

    print("🎼 Exporting stem files...")
    for name, stem_audio in stems.items():
        stem_path = out_dir / f"{name}.wav"
        save_audio(np.asarray(stem_audio), stem_path, samplerate=separator.samplerate)
        print(f"  ✓ Saved stem: {stem_path}")

def main():
    parser = argparse.ArgumentParser(description="Process audio files with Demucs stem separation.")
    parser.add_argument("files", nargs="+", help="Audio files to process")
    parser.add_argument("-d", "--duration", type=float, default=None, help="Optional duration limit in seconds")

    args = parser.parse_args()

    print("Initializing Demucs separator model...")
    if IS_MLX:
        print("⚡ Operating in Apple Silicon MLX GPU mode.")
        separator = Separator(model="htdemucs_6s")
    else:
        print("⚡ Operating in PyTorch CPU mode.")
        separator = Separator(model="htdemucs_6s", device="cpu")

    for file_path in args.files:
        try:
            process_file(file_path, separator, duration=args.duration)
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")

if __name__ == "__main__":
    main()
