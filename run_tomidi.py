import os
import sys
import json
import warnings

# Mute standard Python warnings and system library outputs globally
warnings.filterwarnings("ignore")
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
import essentia
import essentia.standard as es

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
    "keys": {
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

def normalize_bpm(bpm: float) -> float:
    """Normalizes BPM into a standardized musical range (70.0 - 180.0 BPM)."""
    if bpm <= 0 or np.isnan(bpm):
        return 120.0
    while bpm < 70.0:
        bpm *= 2.0
    while bpm > 180.0:
        bpm /= 2.0
    return float(bpm)

def is_stem_audible(wav_path: Path, threshold_db: float = -48.0) -> bool:
    """Checks if a stem audio file contains sufficient signal energy to process."""
    try:
        y, _ = librosa.load(wav_path, sr=16000, mono=True)
        if len(y) == 0:
            return False
        rms = np.sqrt(np.mean(y**2))
        if rms <= 0:
            return False
        db = 20 * np.log10(rms + 1e-9)
        return db > threshold_db
    except Exception:
        return True

def create_pretty_midi_with_tempo(y: np.ndarray, sr: int) -> tuple[pretty_midi.PrettyMIDI, float]:
    """Extracts reliable tempo using Essentia's multifeature extractor with librosa fallback."""
    pm = pretty_midi.PrettyMIDI()
    tempo = 0.0

    try:
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        audio_es = y.astype(np.float32)
        bpm, _, beats_conf, _, _ = rhythm_extractor(audio_es)
        if beats_conf > 0.05 and bpm > 0:
            tempo = float(bpm)
    except Exception:
        pass

    if tempo <= 0 or np.isnan(tempo):
        tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo_raw, np.ndarray):
            tempo = float(tempo_raw[0]) if len(tempo_raw) > 0 else 120.0
        else:
            tempo = float(tempo_raw)

    tempo = normalize_bpm(tempo)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))
    return pm, tempo

def analyze_stem_with_essentia(wav_path: Path, output_dir: Path, stem_name: str) -> Path:
    out_json_path = output_dir / f"{stem_name}.json"
    if out_json_path.exists():
        print(f"     ⏭️ Skipping Essentia analysis for {wav_path.name}: JSON feature maps already exist.")
        return out_json_path

    print(f"  └─ [Essentia Analyzer] Extracting deep acoustic, tonal & articulation features from {wav_path.name}...")
    sr = 44100
    loader = es.MonoLoader(filename=str(wav_path), sampleRate=sr)
    audio = loader()

    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, _ = rhythm_extractor(audio)
    bpm = normalize_bpm(bpm)

    key_extractor = es.KeyExtractor()
    key, scale, key_strength = key_extractor(audio)

    danceability_calc = es.Danceability()
    danceability, _ = danceability_calc(audio)

    log_attack_calc = es.LogAttackTime()
    log_attack_time, _, _ = log_attack_calc(audio)

    frame_size = 2048
    hop_size = 512

    is_drum_stem = "drum" in stem_name
    if not is_drum_stem:
        min_freq = 30.0 if "bass" in stem_name else 70.0
        max_freq = 800.0 if "bass" in stem_name else 3500.0
        pitch_melodia = es.PredominantPitchMelodia(
            sampleRate=sr,
            frameSize=frame_size,
            hopSize=128,
            minFrequency=min_freq,
            maxFrequency=max_freq
        )
        pitch_hz, pitch_confidence = pitch_melodia(audio)
        pitch_times = [i * 128 / float(sr) for i in range(len(pitch_hz))]
    else:
        pitch_hz, pitch_confidence, pitch_times = np.array([]), np.array([]), []

    windowing = es.Windowing(type="blackmanharris62")
    spectrum = es.Spectrum()
    spectral_peaks = es.SpectralPeaks()
    rms_calc = es.RMS()
    loudness_calc = es.Loudness()
    inharmonicity_calc = es.Inharmonicity()

    freqs_grid = np.fft.rfftfreq(frame_size, 1.0 / float(sr))

    frame_features = []
    frame_idx = 0
    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        t_sec = frame_idx * hop_size / float(sr)
        w_frame = windowing(frame)
        spec = spectrum(w_frame)
        freqs, mags = spectral_peaks(spec)

        rms_val = float(rms_calc(w_frame))
        loudness_val = float(loudness_calc(w_frame))

        spec_sum = float(np.sum(spec))
        centroid_val = float(np.sum(freqs_grid * spec) / spec_sum) if spec_sum > 1e-9 else 0.0

        spec_power = spec ** 2 + 1e-12
        flatness_val = float(np.exp(np.mean(np.log(spec_power))) / (np.mean(spec_power) + 1e-9))

        cum_energy = np.cumsum(spec ** 2)
        total_energy = cum_energy[-1] if len(cum_energy) > 0 else 0.0
        if total_energy > 1e-9:
            cutoff_idx = int(np.searchsorted(cum_energy, 0.85 * total_energy))
            rolloff_val = float(freqs_grid[min(cutoff_idx, len(freqs_grid) - 1)])
        else:
            rolloff_val = 0.0

        contrast_val = float(np.max(mags) - np.mean(mags)) if len(mags) > 0 else 0.0

        valid_peaks = (freqs > 20.0) & (mags > 1e-5)
        valid_freqs = freqs[valid_peaks]
        valid_mags = mags[valid_peaks]
        inharm_val = 0.0
        if len(valid_freqs) > 0:
            try:
                inharm_val = float(inharmonicity_calc(valid_freqs, valid_mags))
            except Exception:
                inharm_val = 0.0

        frame_features.append({
            "time": round(t_sec, 4),
            "rms": round(rms_val, 6),
            "loudness": round(loudness_val, 4),
            "centroid": round(centroid_val, 2),
            "flatness": round(flatness_val, 6),
            "rolloff": round(rolloff_val, 2),
            "contrast_mean": round(float(np.mean(contrast_val)), 4),
            "inharmonicity": round(inharm_val, 6),
        })
        frame_idx += 1

    max_rms = max([f["rms"] for f in frame_features], default=1.0)
    if max_rms <= 0:
        max_rms = 1.0

    events = []

    if not is_drum_stem and len(pitch_hz) > 0:
        conf_thresh = 0.05
        in_note = False
        note_start = 0.0
        note_f0_list = []
        note_conf_list = []

        def emit_event(start_t, end_t, f0_vals, conf_vals):
            if not f0_vals or (end_t - start_t) < 0.03:
                return

            med_f0 = float(np.median(f0_vals))
            if med_f0 < 20.0 or med_f0 > 5000.0:
                return

            midi_pitch = int(round(69.0 + 12.0 * np.log2(med_f0 / 440.0)))
            pitch_contour_cents = [float(1200.0 * np.log2(f / med_f0)) for f in f0_vals]
            microtone_cents = float(np.mean(pitch_contour_cents))

            matched_frames = [f for f in frame_features if start_t <= f["time"] <= end_t]
            if not matched_frames:
                avg_rms, avg_loudness, avg_centroid, avg_flatness, avg_contrast, avg_inharm = 0.5, -20.0, 500.0, 0.01, 1.0, 0.0001
            else:
                avg_rms = float(np.mean([f["rms"] for f in matched_frames]))
                avg_loudness = float(np.mean([f["loudness"] for f in matched_frames]))
                avg_centroid = float(np.mean([f["centroid"] for f in matched_frames]))
                avg_flatness = float(np.mean([f["flatness"] for f in matched_frames]))
                avg_contrast = float(np.mean([f["contrast_mean"] for f in matched_frames]))
                avg_inharm = float(np.mean([f["inharmonicity"] for f in matched_frames]))

            attack_frames = matched_frames[:3] if matched_frames else []
            attack_slope = min(1.0, max([f["rms"] for f in attack_frames]) / (max_rms + 1e-6)) if attack_frames else 0.5

            tag = "normal"
            is_slap = is_pop = is_ghost = is_palm_mute = is_harmonic = is_slide = is_bend = is_staccato = is_hammer_on = is_pull_off = False
            norm_amp = min(1.0, avg_rms / max_rms)

            if ("bass" in stem_name or "guitar" in stem_name) and attack_slope > 0.4 and (avg_centroid > 1200 or avg_contrast > 12.0):
                if midi_pitch >= 43:
                    tag, is_pop = "pop", True
                else:
                    tag, is_slap = "slap", True
            elif norm_amp < 0.15 and avg_flatness > 0.05:
                tag, is_ghost = "ghost", True
            elif norm_amp < 0.30 and avg_flatness > 0.03 and avg_centroid < 800:
                tag, is_palm_mute = "palm_mute", True
            elif avg_centroid > 3.5 * med_f0 and avg_flatness < 0.015 and midi_pitch >= 43:
                tag, is_harmonic = "harmonic", True
            elif len(pitch_contour_cents) >= 4:
                span = max(pitch_contour_cents) - min(pitch_contour_cents)
                if span >= 150:
                    tag, is_slide = "slide", True
                elif span >= 40:
                    tag, is_bend = "bend", True

            if (end_t - start_t) <= 0.12 and tag == "normal":
                is_staccato = True
                tag = "staccato"

            category = "melodic"
            if is_slap or is_pop or is_ghost or is_palm_mute or is_staccato:
                category = "percussive"
            elif is_harmonic or is_slide or is_bend or is_hammer_on or is_pull_off:
                category = "expressive"

            events.append({
                "start": round(start_t, 4),
                "end": round(end_t, 4),
                "pitch": midi_pitch,
                "engine": "essentia",
                "pitches": [midi_pitch],
                "amplitude": round(norm_amp, 4),
                "bends": [round(c, 2) for c in pitch_contour_cents[::2]],
                "microtone_cents": round(microtone_cents, 2),
                "tag": tag,
                "duty_cycle": round(min(1.0, (end_t - start_t) / 0.5), 3),
                "is_triplet": False,
                "is_accent": norm_amp > 0.75,
                "dynamic_mark": "f" if norm_amp > 0.7 else ("mf" if norm_amp > 0.4 else "mp"),
                "is_pickup": False,
                "is_harmonic": is_harmonic,
                "slide_from": None,
                "is_rest": False,
                "is_slap": is_slap,
                "is_pop": is_pop,
                "is_ghost": is_ghost,
                "is_palm_mute": is_palm_mute,
                "is_staccato": is_staccato,
                "is_hammer_on": is_hammer_on,
                "is_pull_off": is_pull_off,
                "category": category,
                "confidence": round(float(np.mean(conf_vals)), 3) if conf_vals else 0.9,
                "inharmonicity_coefficient": round(avg_inharm, 6),
                "spectral_tilt_db_oct": round(-2.0 - (avg_centroid / 500.0), 2),
                "attack_transient_slope": round(attack_slope, 3),
                "is_bend": is_bend,
                "pitch_contour_cents": [round(c, 2) for c in pitch_contour_cents],
                "noise_residual_ratio": round(avg_flatness, 4),
                "rms_energy": round(avg_rms, 6),
                "perceptual_loudness_lufs": round(avg_loudness, 2),
                "reverb_tail_confidence": 0.0,
                "source_bleed_confidence": 0.0,
                "swing_offset_ratio": 0.0,
            })

        for i, (t_sec, hz, conf) in enumerate(zip(pitch_times, pitch_hz, pitch_confidence)):
            if hz > 20.0 and conf >= conf_thresh:
                if not in_note:
                    in_note = True
                    note_start = t_sec
                    note_f0_list = [hz]
                    note_conf_list = [conf]
                else:
                    if abs(hz - np.median(note_f0_list)) > 0.12 * np.median(note_f0_list):
                        emit_event(note_start, t_sec, note_f0_list, note_conf_list)
                        note_start = t_sec
                        note_f0_list = [hz]
                        note_conf_list = [conf]
                    else:
                        note_f0_list.append(hz)
                        note_conf_list.append(conf)
            else:
                if in_note:
                    emit_event(note_start, t_sec, note_f0_list, note_conf_list)
                    in_note = False
                    note_f0_list = []
                    note_conf_list = []

        if in_note and note_f0_list:
            emit_event(note_start, pitch_times[-1] if pitch_times else note_start + 0.1, note_f0_list, note_conf_list)

    output_data = {
        "stem": stem_name,
        "metadata": {
            "sample_rate": sr,
            "bpm": round(float(bpm), 2),
            "beats": [round(float(b), 4) for b in beats],
            "beats_confidence": round(float(beats_confidence), 3),
            "key": key,
            "scale": scale,
            "key_strength": round(float(key_strength), 3),
            "global_loudness": round(float(np.mean([f["loudness"] for f in frame_features])) if frame_features else -20.0, 2),
            "time_sig": "4/4",
        },
        "global_features": {
            "danceability": round(float(danceability), 3),
            "log_attack_time": round(float(log_attack_time), 4),
            "mean_spectral_centroid": round(float(np.mean([f["centroid"] for f in frame_features])) if frame_features else 0.0, 2),
            "mean_spectral_flatness": round(float(np.mean([f["flatness"] for f in frame_features])) if frame_features else 0.0, 6),
            "mean_inharmonicity": round(float(np.mean([f["inharmonicity"] for f in frame_features])) if frame_features else 0.0, 6),
        },
        "pitch_contour": {
            "frame_times": [round(t, 4) for t in pitch_times[::4]] if len(pitch_times) > 0 else [],
            "f0_hz": [round(float(f), 2) for f in pitch_hz[::4]] if len(pitch_hz) > 0 else [],
            "pitch_confidence": [round(float(c), 3) for c in pitch_confidence[::4]] if len(pitch_confidence) > 0 else [],
        },
        "events": events,
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"     ✓ Saved Essentia feature map: {out_json_path.name} ({len(events)} events extracted)")
    return out_json_path

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
        "high": {"fmin": 3000, "fmax": 11000, "pitch": 42, "delta": 0.08, "apply_preemph": True}
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

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # 3-frame median smoothing on f0 contour to filter jitter
    f0_smoothed = np.copy(f0)
    for i in range(1, len(f0) - 1):
        if voiced_flag[i] and not np.isnan(f0[i]):
            vals = [f0[k] for k in (i-1, i, i+1) if voiced_flag[k] and not np.isnan(f0[k])]
            if vals:
                f0_smoothed[i] = float(np.median(vals))

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    max_rms = np.max(rms) if np.max(rms) > 0 else 1.0

    frame_duration = hop_length / sr
    current_base_midi = None
    start_time = 0.0
    start_frame = 0
    min_note_duration = 0.04
    last_bend_val = None

    def compute_peak_attack_velocity(s_frame: int, length: int = 4) -> int:
        end_f = min(len(rms), s_frame + length)
        window_rms = rms[s_frame:end_f]
        peak = np.max(window_rms) if len(window_rms) > 0 else 0.1
        return int(np.clip(35 + ((peak / max_rms) * 92), 35, 127))

    for frame_idx, (f0_val, is_voiced) in enumerate(zip(f0_smoothed, voiced_flag)):
        time = frame_idx * frame_duration

        if is_voiced and not np.isnan(f0_val) and f0_val > 0:
            exact_midi = 69.0 + 12.0 * np.log2(f0_val / 440.0)
            base_midi = int(np.round(exact_midi))

            pitch_dev = exact_midi - base_midi
            bend_val = int(np.clip(pitch_dev * 4096.0, -8192, 8191))

            # Deadband threshold raised to 256 (~6.25 cents) to eliminate micro-jitter noise
            if last_bend_val is None or abs(bend_val - last_bend_val) >= 256:
                bass_inst.pitch_bends.append(pretty_midi.PitchBend(pitch=bend_val, time=time))
                last_bend_val = bend_val

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
                last_bend_val = None

    if current_base_midi is not None:
        total_time = len(f0_smoothed) * frame_duration
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

    all_stems_data = {}

    for wav_file in wav_files:
        stem_type = wav_file.stem.lower()

        # Silence Gate check
        if not is_stem_audible(wav_file):
            print(f"  └─ ⏭️ Skipping transcription for silent stem: {wav_file.name}")
            continue

        json_path = analyze_stem_with_essentia(wav_file, out_midi_dir, stem_type)
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    all_stems_data[stem_type] = json.load(f)
            except Exception:
                pass

        if "bass" in stem_type:
            bass_midi_out = out_midi_dir / "bass.midi"
            if not bass_midi_out.exists():
                transcribe_bass_pyyin(wav_file, bass_midi_out)
        elif "drum" in stem_type:
            drum_midi_out = out_midi_dir / "drums.midi"
            if not drum_midi_out.exists():
                transcribe_drums_adaptive(wav_file, drum_midi_out)
        else:
            target_midi = out_midi_dir / f"{stem_type}.midi"
            if not target_midi.exists():
                transcribe_with_basic_pitch(wav_file, out_midi_dir, stem_type)

    if all_stems_data:
        master_json_path = out_midi_dir / f"{file_stem}.json"
        master_payload = {
            "track": file_stem,
            "stems": all_stems_data
        }
        with open(master_json_path, "w", encoding="utf-8") as f:
            json.dump(master_payload, f, indent=2)
        print(f"     ✓ Saved master track JSON feature map: {master_json_path.name}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio events and pitch contours into MIDI and JSON feature maps.")
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
