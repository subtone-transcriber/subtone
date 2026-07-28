#!/usr/bin/env bash

# Default flags (0 = exclude, 1 = include)
INCLUDE_OUTPUT=0
INCLUDE_MIDI=0

# Parse command line options
while [[ $# -gt 0 ]]; do
  case $1 in
    --include-output|-o)
      INCLUDE_OUTPUT=1
      shift
      ;;
    --include-midi|-m)
      INCLUDE_MIDI=1
      shift
      ;;
    --help|-h)
      echo "Usage: ./zip_subtone.sh [options]"
      echo "Options:"
      echo "  -o, --include-output  Include subtone/output_bass directory"
      echo "  -m, --include-midi    Include subtone/midi directory"
      echo "  -h, --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Base exclusions scoped relative to subtone directory
EXCLUDES=(
  "subtone/*stems_*"
  "subtone/*__pycache__*"
  "subtone/*.pyc"
  "subtone/*__MACOSX*"
  "subtone/.*"
  "subtone/*egg-info*"
  "subtone/*testoutput*"
)

# Conditionally exclude output_bass and midi if flags are not set
if [ $INCLUDE_OUTPUT -eq 0 ]; then
  EXCLUDES+=("subtone/output_bass/*")
fi

if [ $INCLUDE_MIDI -eq 0 ]; then
  EXCLUDES+=("subtone/midi/*")
fi

# Build zip command array
ZIP_CMD=("zip" "-r" "subtone.zip" "subtone")
for pattern in "${EXCLUDES[@]}"; do
  ZIP_CMD+=("-x" "$pattern")
done

# Navigate to parent directory and execute zip
cd .. && "${ZIP_CMD[@]}"
