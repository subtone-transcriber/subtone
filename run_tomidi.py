import os
import sys

# Force eager loading for all libraries using lazy_loader (librosa, scipy, etc.)
os.environ['PYTHONLAZYLOADING'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['COREML_SILENT_LOGGING'] = '1'

import logging
import contextlib
import argparse
from pathlib import Path

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('basic_pitch').setLevel(logging.ERROR)

import numpy as np

# Strict direct module imports
import librosa
import librosa.beat
import librosa.effects
import librosa.feature
import librosa.onset

import pretty_midi
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

STEM_CONFIGS = {
    "guitar": {
        "onset_threshold": 0.35,
        "frame_threshold": 0.20,
        "minimum_note_length": 30.0,
        "minimum_frequency": 80.0,
        "maximum_frequency": 1200.0,
    },
    "vocal": {
        "onset_threshold": 0.30,
        "frame_threshold": 0.18,
        "minimum_note_length": 60.0,
        "minimum_frequency": 80.0,
        "maximum_frequency": 1000.0,
    },
    "vocals": {
        "onset_threshold": 0.30,
        "frame_threshold": 0.18,
        "minimum_note_length": 60.0,
        "minimum_frequency": 80.0,
        "maximum_frequency": 1000.0,
    },
    "piano": {
        "onset_threshold": 0.32,
        "frame_threshold": 0.18,
        "minimum_note_length": 35.0,
        "minimum_frequency": 27.5,
        "maximum_frequency": 4186.0,
    },
    "other": {
        "onset_threshold": 0.35,
        "frame_threshold": 0.20,
        "minimum_note_length": 35.0,
    }
}

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def create_pretty_midi_with_tempo(y: np.ndarray, sr: int) -> tuple[pretty_midi.PrettyMIDI, float]:
    pm = pretty_midi.PrettyMIDI()
    tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)

    if isinstance(tempo_raw, np.ndarray):
        tempo = float(tempo_raw[0]) if len(tempo_raw) > 0 else 120.0
    else:
        tempo = float(tempo_raw)

    if tempo <= 0 or np.isnan(tempo):
        tempo = 120.0

    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))
    return pm, tempo

def transcribe_with_basic_pitch(wav_path: Path, output_dir: Path, stem_name: str) -> None:
    if "bass" in stem_name or "drum" in stem_name:
        return

    print(f"  └─ [Basic Pitch Preprocessor] Extracting note events from {wav_path.name}...")
    coreml_model_path = ICASSP_2022_MODEL_PATH.parent / "nmp.mlmodel"
    model_to_use = coreml_model_path if coreml_model_path.exists() else ICASSP_2022_MODEL_PATH

    matching_key = next((key for key in STEM_CONFIGS if key in stem_name), "other")
    config = STEM_CONFIGS[matching_key]

    with suppress_stdout_stderr():
        predict_and_save(
            audio_path_list=[str(wav_path)],
            output_directory=str(output_dir),
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=model_to_use,
            **config
        )

    generated_midi = output_dir / f"{wav_path.stem}_basic_pitch.mid"
    target_midi = output_dir / f"{stem_name}.midi"

    if generated_midi.exists():
        if target_midi.exists():
            target_midi.unlink()
        generated_midi.rename(target_midi)
        print(f"     ✓ Extracted event map: {target_midi.name}")
    else:
        candidates = list(output_dir.glob(f"*{stem_name}*.mid*"))
        if not candidates:
            raise FileNotFoundError(f"Strict Error: Basic Pitch produced no output MIDI file for {wav_path.name}")
        alt = candidates[0]
        if target_midi.exists():
            target_midi.unlink()
        alt.rename(target_midi)
        print(f"     ✓ Extracted event map: {target_midi.name}")

def transcribe_drums_adaptive(wav_path: Path, output_midi_path: Path) -> None:
    print(f"  └─ [Drum Event Extractor] Multi-band transient & crosstalk NMS on {wav_path.name}...")
    sr = 22050
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    if len(y) == 0 or np.max(np.abs(y)) < 1e-4:
        print("     ⚠️ Track silent. Skipping drum event extraction.")
        return

    pm, tempo = create_pretty_midi_with_tempo(y, sr)
    drum_inst = pretty_midi.Instrument(program=0, is_drum=True)
    hop_length = 512

    bands = {
        "low":  {"fmin": 20,   "fmax": 200,   "pitch": 36, "delta": 0.12, "apply_preemph": False},
        "mid":  {"fmin": 200,  "fmax": 3000,  "pitch": 38, "delta": 0.08, "apply_preemph": True},
        "high": {"fmin": 3000, "fmax": 11000, "pitch": 42, "delta": 0.05, "apply_preemph": True}
    }

    band_envelopes = {}
    band_rms = {}

    for b_name, b_info in bands.items():
        y_proc = librosa.effects.preemphasis(y) if b_info["apply_preemph"] else y
        S = np.abs(librosa.stft(y_proc, n_fft=2048, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        idx_min = np.searchsorted(freqs, b_info["fmin"])
        idx_max = np.searchsorted(freqs, b_info["fmax"])

        S_band = S[idx_min:idx_max, :]
        env = librosa.onset.onset_strength(S=S_band, sr=sr)
        max_env = np.max(env)
        band_envelopes[b_name] = env / max_env if max_env > 0 else env

        b_rms = np.mean(S_band, axis=0)
        max_b_rms = np.max(b_rms)
        band_rms[b_name] = b_rms / max_b_rms if max_b_rms > 0 else b_rms

    candidate_events = []
    for b_name, b_info in bands.items():
        frames = librosa.onset.onset_detect(
            onset_envelope=band_envelopes[b_name],
            sr=sr,
            hop_length=hop_length,
            backtrack=True,
            delta=b_info["delta"],
            wait=2
        )
        for f in frames:
            candidate_events.append((f, b_name))

    candidate_events.sort(key=lambda x: x[0])

    frame_window = 2
    grouped_frames = {}
    for f, b_name in candidate_events:
        matched_key = None
        for key_frame in grouped_frames:
            if abs(key_frame - f) <= frame_window:
                matched_key = key_frame
                break
        if matched_key is None:
            grouped_frames[f] = [b_name]
        else:
            grouped_frames[matched_key].append(b_name)

    total_notes = 0
    for f_key, active_bands in grouped_frames.items():
        onset_time = librosa.frames_to_time(f_key, sr=sr, hop_length=hop_length)

        low_e = band_envelopes["low"][f_key] if f_key < len(band_envelopes["low"]) else 0
        mid_e = band_envelopes["mid"][f_key] if f_key < len(band_envelopes["mid"]) else 0
        high_e = band_envelopes["high"][f_key] if f_key < len(band_envelopes["high"]) else 0

        dominant_band = max([("low", low_e), ("mid", mid_e), ("high", high_e)], key=lambda x: x[1])[0]

        for b_name in set(active_bands):
            e_val = band_envelopes[b_name][f_key] if f_key < len(band_envelopes[b_name]) else 0
            if b_name != dominant_band and (e_val < 0.45 * max(low_e, mid_e, high_e)):
                continue

            pitch = bands[b_name]["pitch"]
            rms_val = band_rms[b_name][f_key] if f_key < len(band_rms[b_name]) else 0.1
            velocity = int(np.clip(40 + (rms_val * 87), 40, 127))

            drum_inst.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=onset_time,
                    end=onset_time + 0.08
                )
            )
            total_notes += 1

    drum_inst.notes.sort(key=lambda x: x.start)
    pm.instruments.append(drum_inst)
    pm.write(str(output_midi_path))
    print(f"     ✓ Saved drum events: {output_midi_path.name} ({total_notes} transients extracted @ {tempo:.1f} BPM)")

def transcribe_bass_pyyin(wav_path: Path, output_midi_path: Path) -> None:
    print(f"  └─ [Bass Event & Pitch Bend Extractor] Continuous pYIN on {wav_path.name}...")
    sr = 22050
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    if len(y) == 0 or np.max(np.abs(y)) < 1e-4:
        print("     ⚠️ Track silent. Skipping bass event extraction.")
        return

    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    frame_length = 4096
    hop_length = 512
    fmin = 28.0   # ~A0
    fmax = 380.0  # ~F#4

    pm, tempo = create_pretty_midi_with_tempo(y, sr)
    bass_inst = pretty_midi.Instrument(program=33)

    # Calling librosa.pyin strictly
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    )

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    max_rms = np.max(rms) if np.max(rms) > 0 else 1.0

    frame_duration = hop_length / sr
    current_base_midi = None
    start_time = 0.0
    start_frame = 0
    min_note_duration = 0.04

    def compute_peak_attack_velocity(s_frame: int, length: int = 4) -> int:
        end_f = min(len(rms), s_frame + length)
        window_rms = rms[s_frame:end_f]
        peak = np.max(window_rms) if len(window_rms) > 0 else 0.1
        return int(np.clip(35 + ((peak / max_rms) * 92), 35, 127))

    for frame_idx, (f0_val, is_voiced) in enumerate(zip(f0, voiced_flag)):
        time = frame_idx * frame_duration

        if is_voiced and not np.isnan(f0_val) and f0_val > 0:
            exact_midi = 69.0 + 12.0 * np.log2(f0_val / 440.0)
            base_midi = int(np.round(exact_midi))

            pitch_dev = exact_midi - base_midi
            bend_val = int(np.clip(pitch_dev * 4096.0, -8192, 8191))
            bass_inst.pitch_bends.append(pretty_midi.PitchBend(pitch=bend_val, time=time))

            if current_base_midi is None:
                current_base_midi = base_midi
                start_time = time
                start_frame = frame_idx
            elif abs(base_midi - current_base_midi) > 1:
                if time - start_time >= min_note_duration:
                    vel = compute_peak_attack_velocity(start_frame)
                    bass_inst.notes.append(pretty_midi.Note(
                        velocity=vel, pitch=current_base_midi, start=start_time, end=time
                    ))
                current_base_midi = base_midi
                start_time = time
                start_frame = frame_idx
        else:
            if current_base_midi is not None:
                if time - start_time >= min_note_duration:
                    vel = compute_peak_attack_velocity(start_frame)
                    bass_inst.notes.append(pretty_midi.Note(
                        velocity=vel, pitch=current_base_midi, start=start_time, end=time
                    ))
                current_base_midi = None

    if current_base_midi is not None:
        total_time = len(f0) * frame_duration
        if total_time - start_time >= min_note_duration:
            vel = compute_peak_attack_velocity(start_frame)
            bass_inst.notes.append(pretty_midi.Note(
                velocity=vel, pitch=current_base_midi, start=start_time, end=total_time
            ))

    pm.instruments.append(bass_inst)
    pm.write(str(output_midi_path))
    print(f"     ✓ Extracted bass events: {output_midi_path.name} ({len(bass_inst.notes)} note events, {len(bass_inst.pitch_bends)} pitch bend points @ {tempo:.1f} BPM)")

def process_stem_folder(stem_dir: Path) -> None:
    stem_dir = Path(stem_dir).resolve()
    if not stem_dir.is_dir():
        raise NotADirectoryError(f"Strict Check: Provided target stem path '{stem_dir}' is not a directory.")

    folder_name = stem_dir.name
    file_stem = folder_name.replace("stems_", "", 1) if folder_name.startswith("stems_") else folder_name

    out_midi_dir = Path("./midi").resolve() / file_stem
    out_midi_dir.mkdir(parents=True, exist_ok=True)

    print("\n==================================================")
    print(f"🎹 Preprocessing Events in Stem Folder: {stem_dir.name}")
    print(f"📁 Exporting Event Maps to: {out_midi_dir}")
    print("==================================================")

    wav_files = list(stem_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"Strict Check: No .wav files found in directory {stem_dir}")

    for wav_file in wav_files:
        stem_type = wav_file.stem.lower()

        if "bass" in stem_type:
            bass_midi_out = out_midi_dir / "bass.midi"
            if bass_midi_out.exists():
                print(f"     ⏭️ Skipping {wav_file.name}: {bass_midi_out.name} already exists.")
                continue
            transcribe_bass_pyyin(wav_file, bass_midi_out)
        elif "drum" in stem_type:
            drum_midi_out = out_midi_dir / "drums.midi"
            if drum_midi_out.exists():
                print(f"     ⏭️ Skipping {wav_file.name}: {drum_midi_out.name} already exists.")
                continue
            transcribe_drums_adaptive(wav_file, drum_midi_out)
        else:
            target_midi = out_midi_dir / f"{stem_type}.midi"
            if target_midi.exists():
                print(f"     ⏭️ Skipping {wav_file.name}: {target_midi.name} already exists.")
                continue
            transcribe_with_basic_pitch(wav_file, out_midi_dir, stem_type)

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio events and pitch contours into MIDI preprocessing maps.")
    parser.add_argument("targets", nargs="*", help="Stem directories to process")
    args = parser.parse_args()

    targets = args.targets
    if not targets:
        targets = [str(p) for p in Path(".").glob("stems_*") if p.is_dir()]

    if not targets:
        raise RuntimeError("Strict Error: No target stem folders found to process.")

    for target in targets:
        target_path = Path(target).resolve()
        process_stem_folder(target_path)

if __name__ == "__main__":
    main()
