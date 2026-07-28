"""Pitch conversion, key detection, scale logic, and domain-level pitch rules."""

import logging
import re

try:
    import librosa
except ModuleNotFoundError:
    librosa = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from music21 import key, pitch
except ModuleNotFoundError:
    class _PitchFallback:
        def __init__(self, name=None, midi=None):
            PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            if midi is not None:
                self.midi = int(midi)
                note_idx = self.midi % 12
                octave = (self.midi // 12) - 1
                self.name = PITCH_NAMES[note_idx]
                self.nameWithOctave = f"{self.name}{octave}"
                self.step = self.name[0]
                self.octave = octave
                self.alter = 1 if "#" in self.name else (-1 if "b" in self.name else 0)
            elif name:
                self.nameWithOctave = name
                match = re.match(r"([A-Ga-g][#b]?)(-?\d+)?", name)
                if match:
                    self.name = match.group(1).upper()
                    self.step = self.name[0]
                    self.octave = int(match.group(2)) if match.group(2) is not None else 4
                    base_midis = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
                    m = base_midis.get(self.step, 0)
                    if "#" in self.name:
                        m += 1
                        self.alter = 1
                    elif "b" in self.name:
                        m -= 1
                        self.alter = -1
                    else:
                        self.alter = 0
                    self.midi = (self.octave + 1) * 12 + m
                else:
                    self.name = "C"
                    self.step = "C"
                    self.octave = 4
                    self.midi = 60
                    self.alter = 0
            self.accidental = None if "#" not in self.name and "b" not in self.name else type("Accidental", (), {"name": "sharp" if "#" in self.name else "flat"})()

        def getEnharmonic(self):
            return self

        def __str__(self):
            return self.nameWithOctave

    class _TonicFallback:
        def __init__(self, name):
            self.name = name

    class _KeyFallback:
        Key = None

        def __init__(self, name="C", mode="major"):
            self.name = name
            self.mode = mode
            self.tonic = _TonicFallback(name.upper() if name else "C")
            self.tonicPitchNameWithCase = name.lower() if mode == "minor" else name.upper()

        def __str__(self):
            return f"{self.tonicPitchNameWithCase} {self.mode}"

    _KeyFallback.Key = _KeyFallback
    _PitchFallback.Pitch = _PitchFallback
    key = _KeyFallback
    pitch = _PitchFallback

from subtone.settings_loader import MAX_BASS_MIDI, MIN_BASS_MIDI, MODAL_SCALE_OFFSETS
from subtone.schemas import Song

logger = logging.getLogger(__name__)


# --- Pitch & Frequency Conversions ---


def midi_to_frequency(midi_value):
    """Convert a MIDI pitch to frequency in hertz."""
    return float(librosa.midi_to_hz(midi_value))


def midi_to_hz(midi_value):
    """Convert a MIDI pitch to frequency in hertz."""
    return midi_to_frequency(midi_value)


def frequency_to_midi(frequency):
    """Convert a frequency in hertz to a MIDI pitch."""
    if frequency is None:
        return 0.0
    if isinstance(frequency, (int, float)):
        if frequency <= 0.0:
            return 0.0
        return float(librosa.hz_to_midi(frequency))
    elif isinstance(frequency, np.ndarray):
        if frequency.size == 0:
            return np.array([])
        freq_copy = np.copy(frequency)
        mask = freq_copy <= 0.0
        freq_copy[mask] = 1e-5
        res = librosa.hz_to_midi(freq_copy)
        res[mask] = 0.0
        return res
    try:
        f_val = float(frequency)
        if f_val <= 0.0:
            return 0.0
        return float(librosa.hz_to_midi(f_val))
    except (ValueError, TypeError):
        return 0.0


def hz_to_midi(frequency):
    """Convert a frequency in hertz to a MIDI pitch."""
    return frequency_to_midi(frequency)


def get_closest_value(target: float, array):
    """Returns element in array closest to target value."""
    if array is None or len(array) == 0:
        return None
    if isinstance(array, np.ndarray):
        if array.size == 0:
            return None
        idx = np.searchsorted(array, target)
        if idx == 0:
            return float(array[0])
        if idx == len(array):
            return float(array[-1])
        left, right = float(array[idx - 1]), float(array[idx])
        return left if abs(target - left) < abs(target - right) else right
    return min(array, key=lambda x: abs(x - target))


def key_to_midi_pitch(key_str: str, default: int = 36) -> int:
    """Converts a musical key string into a MIDI pitch integer for octave 2."""
    if not key_str:
        return default
    root_note = str(key_str).strip().split()[0].replace("m", "").replace("maj", "")
    if not root_note:
        root_note = str(key_str).strip().split()[0]
    return int(librosa.note_to_midi(f"{root_note}2"))


# --- Key Parsing, Detection & Pitch Classes ---


def normalize_key_str(raw_key: str):
    """Normalize a user key name to music21's spelling convention."""
    if not raw_key:
        return None
    value = raw_key.strip().replace("_", " ")
    value = re.sub(r"(?i)sharp", "#", value)
    value = re.sub(r"(?i)flat", "-", value)

    is_minor = bool(re.search(r"(?i)\b(min|minor)\b", value)) or (
        len(value) > 1 and value.endswith("m") and not value.endswith("-m")
    )
    if is_minor:
        value = re.sub(r"(?i)[\s\-_]*(min|minor|m)$", "", value).strip()
    if re.search(r"(?i)\b(maj|major)\b", value):
        value = re.sub(r"(?i)[\s\-_]*(maj|major)$", "", value).strip()

    value = re.sub(r"([A-Ga-g])b(?![a-zA-Z])", r"\1-", value)
    if not value:
        return "c" if is_minor else "C"
    return f"{value[0].lower() if is_minor else value[0].upper()}{value[1:]}"


def parse_key_object(raw_key):
    """Parse a key representation into a music21 Key object."""
    if isinstance(key.Key, type) and isinstance(raw_key, key.Key):
        return raw_key
    if not raw_key:
        return key.Key("C", "major")
    if isinstance(raw_key, str):
        normalized = normalize_key_str(raw_key)
        is_minor = bool(normalized and normalized[0].islower())
        return key.Key(normalized, "minor" if is_minor else "major")
    if hasattr(raw_key, "tonic") or hasattr(raw_key, "mode"):
        return raw_key
    raise TypeError(f"Invalid key specification type: {type(raw_key).__name__}")


def detect_key_signature(audio_y, sr, parsed_key=None, bass_filter_fn=None):
    """Detect a music21 key from audio, honoring an explicitly parsed key."""
    if parsed_key:
        try:
            return parse_key_object(parsed_key), True
        except (ValueError, TypeError) as exc:
            logger.warning("Could not parse supplied key %r: %s", parsed_key, exc)

    if librosa is None or np is None or audio_y is None or len(audio_y) == 0:
        return parse_key_object("C"), False

    try:
        filtered_y = bass_filter_fn(audio_y, sr, lowcut=30.0, highcut=600.0) if bass_filter_fn else audio_y
        try:
            chroma = librosa.feature.chroma_cqt(y=filtered_y, sr=sr, fmin=librosa.note_to_hz("C1"), n_octaves=4)
        except (ValueError, RuntimeError) as exc:
            logger.warning("CQT key detection failed; retrying with default range: %s", exc)
            chroma = librosa.feature.chroma_cqt(y=filtered_y, sr=sr)
        chroma_sum = np.sum(np.log1p(chroma * 10), axis=1)
        if np.sum(chroma_sum) == 0:
            return key.Key("C"), False
        profiles = [
            (
                np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
                "major",
            ),
            (
                np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.98, 2.69, 3.34, 3.17]),
                "minor",
            ),
            (
                np.array([6.20, 2.10, 3.40, 2.20, 4.30, 4.00, 2.40, 5.10, 2.30, 3.50, 4.80, 2.50]),
                "mixolydian",
            ),
            (
                np.array([6.20, 2.50, 3.50, 5.20, 2.50, 3.80, 2.40, 4.80, 2.80, 4.00, 3.20, 2.80]),
                "dorian",
            ),
            (
                np.array([6.50, 1.80, 2.20, 5.50, 2.00, 5.00, 4.80, 5.20, 1.80, 2.20, 4.50, 2.00]),
                "minor",
            ),
        ]
        names = ["C", "C#", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B"]
        best_score, best_root, best_mode = -float("inf"), "C", "major"
        for index in range(12):
            rotated = np.roll(chroma_sum, -index)
            for profile, mode in profiles:
                score = np.nan_to_num(np.corrcoef(rotated, profile)[0, 1])
                if score > best_score:
                    best_score, best_root, best_mode = score, names[index], mode
        if best_mode in {"major", "minor"}:
            return key.Key(best_root if best_mode == "major" else best_root.lower()), False
        return key.Key(best_root, best_mode), False
    except (ValueError, RuntimeError) as exc:
        logger.warning("Key detection failed; defaulting to C major: %s", exc)
        return key.Key("C"), False


def key_pitch_classes(key_obj) -> set[int]:
    """Return the pitch classes represented by a music21 key object."""
    if key_obj is None:
        return set()
    try:
        pcs = {p.pitchClass for p in key_obj.getPitches()}
        if pcs:
            return pcs
    except (AttributeError, TypeError, ValueError):
        pass

    # Fallback scale derivation
    tonic_str = getattr(key_obj, "name", "C")
    if not isinstance(tonic_str, str):
        tonic_str = str(tonic_str) if tonic_str else "C"

    # Root mapping
    pc_map = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
    root_char = tonic_str[0].lower() if tonic_str else "c"
    root_pc = pc_map.get(root_char, 0)
    if "#" in tonic_str or "sharp" in tonic_str.lower():
        root_pc = (root_pc + 1) % 12
    elif "-" in tonic_str or "b" in tonic_str[1:].lower() or "flat" in tonic_str.lower():
        root_pc = (root_pc - 1) % 12

    mode = getattr(key_obj, "mode", "major")
    if not isinstance(mode, str):
        mode = "minor" if getattr(key_obj, "name", "C")[0].islower() else "major"

    if mode == "minor":
        offsets = [0, 2, 3, 5, 7, 8, 10]
    else:
        offsets = [0, 2, 4, 5, 7, 9, 11]

    return {(root_pc + o) % 12 for o in offsets}


KEY_DIATONIC_SPELLINGS = {
    # Major
    "C major":  {0: "C", 2: "D", 4: "E", 5: "F", 7: "G", 9: "A", 11: "B"},
    "G major":  {7: "G", 9: "A", 11: "B", 0: "C", 2: "D", 4: "E", 6: "F#"},
    "D major":  {2: "D", 4: "E", 6: "F#", 7: "G", 9: "A", 11: "B", 1: "C#"},
    "A major":  {9: "A", 11: "B", 1: "C#", 2: "D", 4: "E", 6: "F#", 8: "G#"},
    "E major":  {4: "E", 6: "F#", 8: "G#", 9: "A", 11: "B", 1: "C#", 3: "D#"},
    "B major":  {11: "B", 1: "C#", 3: "D#", 4: "E", 6: "F#", 8: "G#", 10: "A#"},
    "F# major": {6: "F#", 8: "G#", 10: "A#", 11: "B", 1: "C#", 3: "D#", 5: "E#"},
    "F major":  {5: "F", 7: "G", 9: "A", 10: "Bb", 0: "C", 2: "D", 4: "E"},
    "Bb major": {10: "Bb", 0: "C", 2: "D", 3: "Eb", 5: "F", 7: "G", 9: "A"},
    "Eb major": {3: "Eb", 5: "F", 7: "G", 8: "Ab", 10: "Bb", 0: "C", 2: "D"},
    "Ab major": {8: "Ab", 10: "Bb", 0: "C", 1: "Db", 3: "Eb", 5: "F", 7: "G"},
    "Db major": {1: "Db", 3: "Eb", 5: "F", 6: "Gb", 8: "Ab", 10: "Bb", 0: "C"},
    # Minor
    "A minor":  {9: "A", 11: "B", 0: "C", 2: "D", 4: "E", 5: "F", 7: "G"},
    "E minor":  {4: "E", 6: "F#", 7: "G", 9: "A", 11: "B", 0: "C", 2: "D"},
    "B minor":  {11: "B", 1: "C#", 2: "D", 4: "E", 6: "F#", 7: "G", 9: "A"},
    "F# minor": {6: "F#", 8: "G#", 9: "A", 11: "B", 1: "C#", 2: "D", 4: "E"},
    "C# minor": {1: "C#", 3: "D#", 4: "E", 6: "F#", 8: "G#", 9: "A", 11: "B"},
    "G# minor": {8: "G#", 10: "A#", 11: "B", 1: "C#", 3: "D#", 4: "E", 6: "F#"},
    "D minor":  {2: "D", 4: "E", 5: "F", 7: "G", 9: "A", 10: "Bb", 0: "C"},
    "G minor":  {7: "G", 9: "A", 10: "Bb", 0: "C", 2: "D", 3: "Eb", 5: "F"},
    "C minor":  {0: "C", 2: "D", 3: "Eb", 5: "F", 7: "G", 8: "Ab", 10: "Bb"},
    "F minor":  {5: "F", 7: "G", 8: "Ab", 10: "Bb", 0: "C", 1: "Db", 3: "Eb"},
    "Bb minor": {10: "Bb", 0: "C", 1: "Db", 3: "Eb", 5: "F", 6: "Gb", 8: "Ab"},
    "Eb minor": {3: "Eb", 5: "F", 6: "Gb", 8: "Ab", 10: "Bb", 11: "Cb", 1: "Db"},
}


def get_key_name_str(raw_key) -> str:
    if not raw_key:
        return "C major"
    s = str(raw_key).strip()
    if hasattr(raw_key, "tonic") and hasattr(raw_key, "mode"):
        t_name = getattr(raw_key.tonic, "name", str(raw_key.tonic))
        return f"{t_name} {raw_key.mode}"
    is_minor = bool(re.search(r"(?i)\b(min|minor)\b", s)) or (
        len(s) > 1 and s.endswith("m") and not s.endswith("-m")
    )
    clean = re.sub(r"(?i)(minor|major|min|maj|m|\s)", "", s)
    if clean.endswith("-"):
        clean = clean[:-1] + "b"
    clean_cap = clean.capitalize() if not is_minor else clean.lower()
    t_cap = clean_cap.capitalize()
    return f"{t_cap} minor" if is_minor else f"{t_cap} major"


def spell_pitch(midi_val: int, key_obj=None, prev_midi=None, spelling_preference=None):
    """Create a key-aware, direction-aware music21 pitch."""
    if key_obj is not None:
        key_name = get_key_name_str(key_obj)
        diatonic = KEY_DIATONIC_SPELLINGS.get(key_name)
        pc = int(midi_val) % 12
        octave = (int(midi_val) // 12) - 1

        if diatonic and pc in diatonic:
            note_str = f"{diatonic[pc]}{octave}"
            return pitch.Pitch(note_str)
        else:
            # Chromatic spelling based on key orientation
            is_flat_key = any(k in key_name.lower() for k in ["b", "flat", "f ", "d min", "g min", "c min", "f min"])
            if is_flat_key or spelling_preference == "flat":
                flat_names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
                return pitch.Pitch(f"{flat_names[pc]}{octave}")
            else:
                sharp_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                return pitch.Pitch(f"{sharp_names[pc]}{octave}")

    result = pitch.Pitch(midi=midi_val)
    if spelling_preference == "flat" and getattr(result, "accidental", None) and getattr(result.accidental, "name", "") == "sharp":
        result.getEnharmonic(inPlace=True)
    elif spelling_preference == "sharp" and getattr(result, "accidental", None) and getattr(result.accidental, "name", "") == "flat":
        result.getEnharmonic(inPlace=True)
    elif getattr(result, "accidental", None) and prev_midi is not None and prev_midi != midi_val:
        ascending = midi_val > prev_midi
        if (ascending and getattr(result.accidental, "name", "") == "flat") or (
            not ascending and getattr(result.accidental, "name", "") == "sharp"
        ):
            result.getEnharmonic(inPlace=True)
    return result


# --- Domain-Level Pitch Rules & Scale Snapping ---


def fold_pitch_to_bass_range(midi_pitch: int, min_pitch: int = MIN_BASS_MIDI, max_pitch: int = MAX_BASS_MIDI) -> int:
    """Fold a MIDI pitch into the playable bass register."""
    if min_pitch >= max_pitch:
        return min_pitch
    while midi_pitch < min_pitch:
        midi_pitch += 12
    while midi_pitch > max_pitch:
        midi_pitch -= 12
    return midi_pitch


def snap_pitch_to_scale(
    midi_val: int,
    key_obj,
    level: int = 5,
    next_midi: int = None,
    rigor: str = "strict_diatonic",
    preserve_blue_notes: bool = True,
    modal_fallback: str = None,
) -> int:
    """Apply the project's scale-snapping policy to a MIDI pitch."""
    midi_val = fold_pitch_to_bass_range(midi_val)
    if key_obj is None or rigor == "chromatic_direct":
        return midi_val

    current_pc = midi_val % 12
    if preserve_blue_notes:
        tonic = getattr(key_obj, "tonic", None)
        root_pc = getattr(tonic, "pitchClass", None)
        if root_pc is not None and current_pc in {(root_pc + offset) % 12 for offset in (3, 6, 10)}:
            return midi_val

    if rigor == "modal_permissive" and modal_fallback in MODAL_SCALE_OFFSETS:
        root_pc = getattr(getattr(key_obj, "tonic", None), "pitchClass", 0)
        scale_pcs = {(root_pc + offset) % 12 for offset in MODAL_SCALE_OFFSETS[modal_fallback]}
    else:
        scale_pcs = key_pitch_classes(key_obj)
    if current_pc in scale_pcs or not scale_pcs:
        return midi_val
    if next_midi is not None and abs(next_midi - midi_val) == 1:
        return midi_val

    candidates = []
    for scale_pc in scale_pcs:
        shift = (scale_pc - current_pc + 6) % 12 - 6
        candidates.append((shift, abs(shift)))
    minimum_distance = min(distance for _, distance in candidates)
    best = [shift for shift, distance in candidates if distance == minimum_distance]
    if next_midi is not None and len(best) > 1:
        chosen_shift = max(best) if next_midi > midi_val else min(best)
    else:
        chosen_shift = best[0]
    return midi_val + chosen_shift if level <= 1 or minimum_distance <= 1 else midi_val


def midi_to_pitch_string(midi_val: int, key_obj=None) -> str:
    if midi_val is None:
        return "N/A"
    if key_obj is not None:
        p = spell_pitch(midi_val, key_obj)
        if hasattr(p, "name") and p.name:
            octave = (int(midi_val) // 12) - 1
            return f"{p.name}{octave}"
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[int(midi_val) % 12]}{(int(midi_val) // 12) - 1}"


def get_directional_enharmonic_pitch(
    midi_val: int, key_obj=None, prev_midi: int = None, spelling_preference: str = None
):
    return spell_pitch(midi_val, key_obj, prev_midi, spelling_preference)


def get_key_aware_pitch(midi_val: int, key_obj=None, prev_midi: int = None, spelling_preference: str = None):
    return get_directional_enharmonic_pitch(midi_val, key_obj, prev_midi, spelling_preference)


def snap_song_to_scale(song: Song) -> Song:
    """Snap the processed note stream using the song's own key and target level."""
    target_level = getattr(song, "target_level", 5)
    for index, note in enumerate(song.notes):
        next_pitch = song.notes[index + 1].pitch if index + 1 < len(song.notes) else None
        note.update_pitch(
            snap_pitch_to_scale(
                note.pitch,
                song.key_obj,
                level=target_level,
                next_midi=next_pitch,
            )
        )
    if hasattr(song, "partition_into_measures") and callable(song.partition_into_measures):
        song.partition_into_measures()
    return song


def parse_metadata_from_path(path_str: str, custom_genre: str | None = None, config_path: str | None = None):
    """Parses artist, song title, key signature, and genre configuration from a path string."""
    import os
    from subtone.schemas import Genre

    norm_path = os.path.normpath(str(path_str))
    raw_name = os.path.basename(norm_path)
    for ext in [".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".mid", ".midi", ".xml", ".musicxml"]:
        if raw_name.lower().endswith(ext):
            raw_name = raw_name[:-len(ext)]
            break

    generic_names = {"bass", "drums", "other", "vocals", "guitar", "piano", "stem", "stems", "track"}
    if raw_name.lower() in generic_names:
        parent_dir = os.path.basename(os.path.dirname(norm_path))
        if parent_dir and parent_dir.lower() not in ("midi", "stems", "audio", "test_batch_output", "test_output"):
            raw_name = parent_dir

    if raw_name.startswith("stems_"):
        raw_name = raw_name[6:]

    artist = "Unknown Artist"
    title = raw_name
    key_str = "C major"
    genre_name = custom_genre or "default"

    parts = [p.strip() for p in raw_name.split("_") if p.strip()]
    if len(parts) >= 4:
        genre_name = custom_genre or parts[0]
        key_str = parts[1]
        artist = parts[2]
        title = "_".join(parts[3:])
    elif len(parts) == 3:
        artist = parts[0]
        title = parts[1]
        key_str = parts[2]
    elif len(parts) == 2:
        artist = parts[0]
        title = parts[1]
    elif " - " in raw_name:
        dash_parts = raw_name.split(" - ")
        artist = dash_parts[0].strip()
        title = " - ".join(dash_parts[1:]).strip()

    clean_name = f"{artist} - {title}".strip()
    genre_config = Genre(name=genre_name)

    return artist, title, clean_name, key_str, genre_name, genre_config
