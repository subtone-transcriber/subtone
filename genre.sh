#!/bin/bash
# trying genre detection but it's pretty busted. 
# genre.sh
# Usage: 
#   ./genre.sh path/to/file.mp3
#   ./genre.sh path/to/folder/
#   ./genre.sh *.mp3

set -euo pipefail

# Determine the absolute path of the script's directory for persistent caching
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ENV_DIR="$SCRIPT_DIR/.venv_essentia"
MODEL_CACHE_DIR="$ENV_DIR/cached_models"

FILES=()
Q
# Parse positional arguments
while [[ $# -gt 0 ]]; do
    FILES+=("$1")
    shift
done

# Check if at least one file argument was passed
if [ ${#FILES[@]} -eq 0 ]; then
    echo "❌ Error: Missing input audio file or directory target(s)."
    echo "Usage: $0 <file_or_directory_1> [file_or_directory_2 ...]"
    exit 1
fi

echo "=== 🎼 Deploying Essentia Discogs Classification Engine ==="

if [ -d "/opt/homebrew/bin" ]; then export PATH="/opt/homebrew/bin:$PATH"; fi
if command -v python3.11 &> /dev/null; then PY_CMD="python3.11"; elif command -v python3 &> /dev/null; then PY_CMD="python3"; else echo "❌ Error: Python 3 required."; exit 1; fi

# Setup Virtual Environment
if [ ! -d "$ENV_DIR" ]; then 
    echo "📦 Creating virtual environment at $ENV_DIR..."
    $PY_CMD -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"
    echo "📦 Installing dependencies..."
    pip install --upgrade pip "setuptools<70.0.0" -q
    pip install essentia-tensorflow requests urllib3 -q
else
    source "$ENV_DIR/bin/activate"
fi

export ESSENTIA_MODEL_CACHE="$MODEL_CACHE_DIR"

cat << 'EOF' > "$SCRIPT_DIR/run_genre.py"
import sys
import os
import argparse
import urllib.request
import json
import re
from pathlib import Path
import essentia.standard as es

# Define model paths using the persistent cache directory
cache_dir = os.environ.get("ESSENTIA_MODEL_CACHE", "./essentia_models")
MODEL_DIR = Path(cache_dir)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = MODEL_DIR / "discogs-effnet-bs64-1.pb"
HEAD_MODEL = MODEL_DIR / "genre_discogs400-discogs-effnet-1.pb"
LABELS_JSON = MODEL_DIR / "genre_discogs400-discogs-effnet-1.json"

def download_models():
    """Downloads the necessary pre-trained Discogs models from Essentia's official hub."""
    base_url_extract = "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"
    base_url_head = "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb"
    base_url_json = "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json"

    if not EMBED_MODEL.exists():
        print("⬇️ Downloading Discogs Effnet Embedding Model (~12MB)...")
        urllib.request.urlretrieve(base_url_extract, EMBED_MODEL)
    else:
        print(f"✅ Embedding model found in cache: {EMBED_MODEL.parent.name}/{EMBED_MODEL.name}")

    if not HEAD_MODEL.exists():
        print("⬇️ Downloading Discogs Classification Head (~2MB)...")
        urllib.request.urlretrieve(base_url_head, HEAD_MODEL)
    else:
        print(f"✅ Classification head found in cache: {HEAD_MODEL.parent.name}/{HEAD_MODEL.name}")

    if not LABELS_JSON.exists():
        print("⬇️ Downloading Discogs Genre Labels...")
        urllib.request.urlretrieve(base_url_json, LABELS_JSON)

def gather_audio_files(input_paths):
    """Scans provided paths and returns a list of valid audio files."""
    valid_exts = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    audio_files = []

    for p_str in input_paths:
        p = Path(p_str).resolve()
        if p.is_file() and p.suffix.lower() in valid_exts:
            audio_files.append(p)
        elif p.is_dir():
            print(f"📂 Scanning directory: {p.name}")
            for f in p.rglob('*'):
                if f.is_file() and f.suffix.lower() in valid_exts:
                    audio_files.append(f)
        else:
            print(f"⚠️ Warning: Path not found or unsupported: {p}")

    return sorted(list(set(audio_files))) # Return sorted unique files

def process_file(audio_path, classes):
    print(f"\n==================================================")
    print(f"🎵 Processing: {audio_path.name}")
    print(f"==================================================")
    print(f"  [1/2] Loading audio and computing embeddings...")

    try:
        # Essentia's EffNet models expect 16kHz audio
        audio = es.MonoLoader(filename=str(audio_path), sampleRate=16000)()

        # Extract embeddings using the base model
        embedder = es.TensorflowPredictEffnetDiscogs(graphFilename=str(EMBED_MODEL), output="PartitionedCall:1")
        embeddings = embedder(audio)

        print(f"  [2/2] Classifying genre against Discogs dataset...")

        # Pass embeddings through the classification head with explicit node specs
        predictor = es.TensorflowPredict2D(
            graphFilename=str(HEAD_MODEL),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0"
        )
        predictions = predictor(embeddings)

        # Take the mean of the predictions across all frames to get the overall track genre
        mean_predictions = predictions.mean(axis=0)

        # Find the highest scoring genre
        top_idx = mean_predictions.argmax()
        top_genre = classes[top_idx]

        # Sanitize the genre string for file systems (e.g., "Electronic---House" -> "Electronic_House")
        safe_genre = re.sub(r'[^A-Za-z0-9]+', '_', top_genre).strip('_')

        # Check if already renamed to avoid double-prepending if run twice
        if audio_path.name.startswith(f"{safe_genre}_"):
            print(f"  ✓ File is already prefixed with {safe_genre}. Skipping rename.")
            return

        new_name = f"{safe_genre}_{audio_path.name}"
        new_path = audio_path.parent / new_name

        # Rename the file
        audio_path.rename(new_path)

        print(f"  🏆 Top Genre: {top_genre} (Score: {mean_predictions[top_idx]:.2f})")
        print(f"  💾 Renamed to: {new_name}")

    except Exception as e:
        print(f"❌ Failed to process {audio_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Genre and rename MP3s using Essentia Discogs model.")
    parser.add_argument("files", nargs="+", help="Audio files or directories to process")
    args = parser.parse_args()

    print("\n--- Verifying Model Cache ---")
    download_models()

    # Load genre labels once
    with open(LABELS_JSON, 'r') as f:
        labels_metadata = json.load(f)
    classes = labels_metadata['classes']

    # Gather all files (including scanning directories)
    target_files = gather_audio_files(args.files)

    if not target_files:
        print("❌ No valid audio files found to process.")
        sys.exit(1)

    print(f"\n🚀 Found {len(target_files)} audio file(s) to process.")

    # Process all gathered files
    for file_path in target_files:
        process_file(file_path, classes)

if __name__ == "__main__":
    main()
EOF

# Execute the pipeline
python "$SCRIPT_DIR/run_genre.py" "${FILES[@]}"