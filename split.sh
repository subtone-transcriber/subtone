#!/usr/bin/env bash
# split.sh
# Usage: ./split.sh [-d DURATION_SEC] path/to/file1.mp3 [path/to/file2.wav ...]

set -euo pipefail

# Parse optional --duration / -d flag while keeping remaining positional arguments
DURATION=""
FILES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

# Check if at least one file argument was passed
if [ ${#FILES[@]} -eq 0 ]; then
    echo "❌ Error: Missing input audio file target(s)."
    echo "Usage: $0 [-d DURATION_IN_SECONDS] <path_to_audio_file1.mp3> [path_to_audio_file2.mp3 ...]"
    exit 1
fi

echo "=== 🎼 Deploying Stem Separation Engine ==="

ENV_DIR=".venv_profound"

# ------------------------------------------------------------------------------
# 1. Environment Resolution (Skips ALL checks if .venv_profound exists)
# ------------------------------------------------------------------------------
if [ -d "$ENV_DIR" ]; then
    echo "⚡ Environment '${ENV_DIR}' detected. Skipping all system checks."
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
else
    echo "📦 Initializing environment for the first time..."

    OS="$(uname -s)"
    ARCH="$(uname -m)"

    if [[ "${OS}" == "Darwin" ]]; then
        if [ -d "/opt/homebrew/bin" ]; then
            export PATH="/opt/homebrew/bin:$PATH"
        fi
    elif [[ "${OS}" == "Linux" ]]; then
        if command -v apt-get &> /dev/null; then
            export DEBIAN_FRONTEND=noninteractive
            MISSING=()
            command -v ffmpeg &> /dev/null || MISSING+=("ffmpeg")
            dpkg -s libsndfile1 &> /dev/null || dpkg -s libsndfile &> /dev/null || MISSING+=("libsndfile1")
            if [ ${#MISSING[@]} -gt 0 ]; then
                echo " Installing missing Linux dependencies: ${MISSING[*]}"
                sudo DEBIAN_FRONTEND=noninteractive apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${MISSING[@]}"
            fi
        fi
    fi

    # Locate Python 3 binary dynamically
    PY_CMD=""
    for py in python3.11 python3.12 python3.10 python3; do
        if command -v "$py" &> /dev/null; then
            PY_CMD="$(command -v "$py")"
            break
        fi
    done

    if [[ -z "${PY_CMD}" ]]; then
        echo "❌ Error: Python 3.10 or later is required but was not found."
        exit 1
    fi

    echo "📦 Creating virtual environment in ${ENV_DIR} using ${PY_CMD}..."
    "$PY_CMD" -m venv "$ENV_DIR"

    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"

    echo "⚙️  Installing Python packages..."
    pip install --upgrade pip "setuptools<82"

    if [[ "${OS}" == "Darwin" && "${ARCH}" == "arm64" ]]; then
        # Apple Silicon optimized
        pip install "demucs-mlx[convert]" librosa soundfile
    else
        # Linux / x86_64 fallback (Demucs explicitly on CPU)
        pip install demucs librosa soundfile torch --extra-index-url https://download.pytorch.org/whl/cpu
    fi
fi

# ------------------------------------------------------------------------------
# 2. Dynamic Python Script Generation
# ------------------------------------------------------------------------------
cat << 'EOF' > run_split.py
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
EOF

# ------------------------------------------------------------------------------
# 3. Pipeline Execution
# ------------------------------------------------------------------------------
PY_ARGS=()
if [ -n "$DURATION" ]; then
    PY_ARGS+=("-d" "$DURATION")
fi

python run_split.py ${PY_ARGS[@]+"${PY_ARGS[@]}"} "${FILES[@]}"