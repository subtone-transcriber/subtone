import argparse
import copy
import fractions
import os
import re

DEFAULT_BPM = 120.0
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_FFT_SIZE = 2048
DEFAULT_TIME_SIGNATURE = "4/4"
DEFAULT_TUNING_TYPE = "4_string_standard"

MAX_FRETBOARD_FRETS = 24
MIN_BASS_MIDI = 28  # E1
MAX_BASS_MIDI = 67  # G4


def parse_metadata_from_path(*args, **kwargs):
    from subtone.pitch_theory import parse_metadata_from_path as _p
    return _p(*args, **kwargs)

MODAL_SCALE_OFFSETS = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
}

FRETBOARD_TUNING_PROFILES = {
    "4_string_standard": [28, 33, 38, 43],  # E1, A1, D2, G2
    "standard": [28, 33, 38, 43],
    "4_string_drop_d": [26, 33, 38, 43],    # D1, A1, D2, G2
    "drop_d": [26, 33, 38, 43],
    "5_string_standard": [23, 28, 33, 38, 43],  # B0, E1, A1, D2, G2
    "5_string": [23, 28, 33, 38, 43],
    "6_string_standard": [23, 28, 33, 38, 43, 48],  # B0, E1, A1, D2, G2, C3
    "6_string": [23, 28, 33, 38, 43, 48],
}

STANDARD_BASS_TUNING_MIDIS = FRETBOARD_TUNING_PROFILES

BASS_STRING_TUNINGS = {
    "4_string_standard": [
        ("E", 1, 28),
        ("A", 1, 33),
        ("D", 2, 38),
        ("G", 2, 43),
    ],
    "4_string_drop_d": [
        ("D", 1, 26),
        ("A", 1, 33),
        ("D", 2, 38),
        ("G", 2, 43),
    ],
    "5_string_standard": [
        ("B", 0, 23),
        ("E", 1, 28),
        ("A", 1, 33),
        ("D", 2, 38),
        ("G", 2, 43),
    ],
    "6_string_standard": [
        ("B", 0, 23),
        ("E", 1, 28),
        ("A", 1, 33),
        ("D", 2, 38),
        ("G", 2, 43),
        ("C", 3, 48),
    ],
}

MEASURE_SCHEMA_ORDER = ["print", "attributes", "direction", "note", "backup", "forward", "barline"]
ATTRIBUTES_SCHEMA_ORDER = ["divisions", "key", "time", "staves", "clef", "transpose", "staff-details"]
NOTE_SCHEMA_ORDER = [
    "pitch",
    "rest",
    "duration",
    "tie",
    "instrument",
    "voice",
    "type",
    "dot",
    "accidental",
    "time-modification",
    "stem",
    "notehead",
    "staff",
    "beam",
    "notations",
    "lyric",
]
NOTATIONS_SCHEMA_ORDER = [
    "tied",
    "slur",
    "tuplet",
    "glissando",
    "slide",
    "ornaments",
    "technical",
    "articulations",
    "dynamics",
    "fermata",
    "other-notation",
]
TECHNICAL_SCHEMA_ORDER = [
    "string",
    "fret",
    "pull-off",
    "hammer-on",
    "harmonic",
    "tap",
    "heel",
    "toe",
    "fingering",
    "pluck",
    "double-tongue",
    "triple-tongue",
    "stopped",
    "snap-thumb",
    "down-bow",
    "up-bow",
    "other-technical",
]

try:
    from music21 import meter
except ModuleNotFoundError:
    meter = None

def _parse_simple_toml(content_str: str) -> dict:
    data = {}
    current_section = data

    for line in content_str.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            sec_name = line[1:-1].strip()
            parts = [p.strip() for p in sec_name.split(".")]
            curr = data
            for p in parts:
                if p not in curr or not isinstance(curr[p], dict):
                    curr[p] = {}
                curr = curr[p]
            current_section = curr
            continue

        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip().strip('"').strip("'")
            val = val.strip()

            if "#" in val and not (val.startswith('"') or val.startswith("'")):
                val = val.split("#", 1)[0].strip()

            current_section[key] = _parse_toml_val(val)

    return data


def _parse_toml_val(val: str):
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.startswith("[") and val.endswith("]"):
        raw_items = val[1:-1].split(",")
        return [_parse_toml_val(i.strip()) for i in raw_items if i.strip()]
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


class _TomlFallback:
    @staticmethod
    def loads(s):
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        return _parse_simple_toml(s)

    @staticmethod
    def load(f):
        content = f.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return _parse_simple_toml(content)


import sys
if sys.version_info >= (3, 11):
    try:
        import tomllib
    except ImportError:
        tomllib = _TomlFallback()
else:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            tomllib = _TomlFallback()

if tomllib is None:
    tomllib = _TomlFallback()

from subtone.schemas import Song, Genre, AudioEvent
# pitch_theory imports moved to function level to prevent circular import


def load_genre_configs(config_path=None):
    """Loads all genre TOML configuration files."""
    if not config_path:
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, "config")

    configs = {}
    if os.path.isdir(config_path) and tomllib is not None:
        for fname in os.listdir(config_path):
            if fname.endswith(".toml"):
                fpath = os.path.join(config_path, fname)
                try:
                    with open(fpath, "rb") as f:
                        data = tomllib.load(f)
                        file_key = os.path.splitext(fname)[0]
                        configs[file_key] = data
                        if isinstance(data, dict):
                            for item_key, item_val in data.items():
                                if isinstance(item_val, dict):
                                    gname = item_val.get("name", item_key)
                                    genre_obj = Genre.from_dict(gname, item_val) if hasattr(Genre, "from_dict") else item_val
                                    configs[item_key] = genre_obj
                                    configs[gname] = genre_obj
                except Exception:
                    pass
    return configs


def resolve_genre(genre_name, all_configs=None):
    """Resolves a genre name against loaded genre configurations."""
    if all_configs is None:
        all_configs = load_genre_configs()

    if genre_name in all_configs:
        return genre_name, all_configs[genre_name]

    clean_key = str(genre_name).lower().replace(" ", "_") if genre_name else "default"
    if clean_key in all_configs:
        return clean_key, all_configs[clean_key]

    for k, v in all_configs.items():
        if k.lower() == clean_key:
            return k, v

    default_cfg = all_configs.get("default", Genre(name="default"))
    return genre_name or "default", default_cfg


def resolve_artist(artist_name, config_path=None):
    """Resolves an artist name to genre association."""
    return "default"


def load_midi_folder_to_event_streams(midi_dir):
    """Loads MIDI files from a directory into event stream dictionary format."""
    import glob

    streams = {}
    if not os.path.isdir(midi_dir):
        return streams

    for mpath in glob.glob(os.path.join(midi_dir, "*.mid*")):
        stem_name = os.path.splitext(os.path.basename(mpath))[0].lower()
        events = []
        try:
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(mpath)
            for inst in pm.instruments:
                for note in inst.notes:
                    events.append(AudioEvent(
                        start=note.start,
                        end=note.end,
                        pitch=note.pitch,
                        amplitude=note.velocity / 127.0
                    ))
        except Exception:
            pass
        streams[stem_name] = {
            "stream_type": "primary" if "bass" in stem_name else "auxiliary",
            "events": events,
            "metadata": {"bpm": 120.0, "time_sig": "4/4"}
        }
    return streams



def _midi_to_name(midi_val: int, key_obj=None) -> str:
    """Helper to convert MIDI integer pitch to musical pitch name (e.g., E1, G#2) with key awareness."""
    if midi_val is None:
        return "N/A"
    try:
        from subtone.pitch_theory import midi_to_pitch_string
        return midi_to_pitch_string(midi_val, key_obj=key_obj)
    except Exception:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return f"{names[int(midi_val) % 12]}{(int(midi_val) // 12) - 1}"


def _get_cfg_dict(cfg, key):
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        val = cfg.get(key, {})
    else:
        val = getattr(cfg, key, {})
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if hasattr(val, "to_dict"):
        return val.to_dict()
    if hasattr(val, "__dict__"):
        return {k: v for k, v in val.__dict__.items() if not k.startswith("_")}
    return val


class AudioTranscriptionPipeline:
    """Run the current service-based transcription pipeline supporting MP3 and cached AudioEvents folders."""

    def __init__(self, output_dir=None, genre_config_path=None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.output_dir = output_dir or os.path.join(project_root, "output_bass")
        self.genre_config_path = genre_config_path

    def run(
        self,
        target_input: str,
        generate_all_levels=False,
        level=5,
        use_gpu=False,
        genre_override=None,
    ):
        del use_gpu  # GPU selection is handled by the installed audio dependencies.

        print("\n======================================================================")
        print("                AUDIO TRANSCRIPTION PIPELINE RUN START                 ")
        print("======================================================================")

        # --- Stage 1: Metadata Parsing ---
        print(f"\n[Stage 1: Metadata Parsing] Loading target input: {target_input}")
        from subtone.pitch_theory import parse_metadata_from_path
        artist_name, song_title, _, parsed_key, genre_name, genre_config = parse_metadata_from_path(
            target_input,
            custom_genre=genre_override,
            config_path=self.genre_config_path,
        )
        print(f"  ├─ Artist:        '{artist_name}'")
        print(f"  ├─ Song Title:    '{song_title}'")
        print(f"  ├─ Key Signature: '{parsed_key}'")
        print(f"  └─ Genre Context: '{genre_name}' (Override: '{genre_override or 'None'}')")

        if genre_config:
            extends = (
                getattr(genre_config, "extends", "default")
                if not isinstance(genre_config, dict)
                else genre_config.get("extends", "default")
            )
            technique = (
                getattr(genre_config, "technique", "default")
                if not isinstance(genre_config, dict)
                else genre_config.get("technique", "default")
            )
            tuning = (
                getattr(genre_config, "tuning", "standard")
                if not isinstance(genre_config, dict)
                else genre_config.get("tuning", "standard")
            )

            micro_timing = _get_cfg_dict(genre_config, "micro_timing")
            articulation = _get_cfg_dict(genre_config, "articulation_intent")
            fret_nav = _get_cfg_dict(genre_config, "fretboard_navigation")
            level_profile = _get_cfg_dict(genre_config, "level_profile")
            notation_engraving = _get_cfg_dict(genre_config, "notation_engraving")
            harmonic_hysteresis = _get_cfg_dict(genre_config, "harmonic_hysteresis")

            print(f"     ├─ Genre Profile: category='{extends}', technique='{technique}', tuning='{tuning}'")
            if micro_timing:
                print(f"     ├─ Micro Timing Config: {micro_timing}")
            if articulation:
                print(f"     ├─ Articulation Intent Rules: {articulation}")
            if fret_nav:
                print(f"     ├─ Fretboard Navigation Rules: {fret_nav}")
            if level_profile:
                print(f"     ├─ Level Profile Settings: {level_profile}")
            if notation_engraving:
                print(f"     ├─ Notation & Engraving Settings: {notation_engraving}")
            if harmonic_hysteresis:
                print(f"     └─ Harmonic Hysteresis Rules: {harmonic_hysteresis}")

        from subtone.engraver import build_and_export_song, stream_quantized_events, filter_song_for_level
        from subtone.dsp import process_audio_target_to_events

        # --- Stage 2: Audio DSP & Transcription ---
        print("\n[Stage 2: Audio DSP & Event Extraction] Processing audio target into event streams...")
        event_streams, song_stem_name, cached_events_path = process_audio_target_to_events(
            target_path=target_input,
            genre_config=genre_config,
            custom_genre=genre_override,
        )
        print(f"  ├─ Active Stem Folder / Name: {song_stem_name}")
        print(f"  ├─ Cached Events Path:        {cached_events_path or 'N/A'}")
        print("  └─ Extracted Event Streams:")
        for stream_name, stream_data in event_streams.items():
            ev_count = len(stream_data.get("events", []))
            st_type = stream_data.get("stream_type", "auxiliary")
            print(f"     ├─ [{stream_name}] type: {st_type}, events: {ev_count}")

        clean_filename = re.sub(r'[\\/*?:"<>|]', "", f"{artist_name} - {song_title}").strip()
        os.makedirs(self.output_dir, exist_ok=True)

        selected_level = level if isinstance(level, int) and 0 <= level <= 5 else 5
        target_levels = range(6) if generate_all_levels else [selected_level]

        pYin_key = next(
            (k for k in event_streams if "pYin" in k or "torch_crepe" in k or "touchcrepe" in k or "crepe" in k),
            None,
        )
        if pYin_key:
            primary_stream_key = pYin_key
        elif "bass" in event_streams:
            primary_stream_key = "bass"
        else:
            primary_stream_key = list(event_streams.keys())[0]

        # --- Stage 3: Song Model Assembly ---
        print(
            f"\n[Stage 3: Song Model Assembly] Initializing Song & Note abstractions for target stem '{primary_stream_key}'..."
        )
        source_song = Song.from_event_streams(
            event_streams=event_streams,
            active_stream_name=primary_stream_key,
            artist_name=artist_name,
            song_title=song_title,
            genres=[genre_name] if genre_name else [],
            genre_config=genre_config,
            parsed_key_str=parsed_key,
            stem_folder=cached_events_path,
        )

        total_bass_events = len(source_song.bass_audio_events)
        total_bass_notes = len(source_song.bass_notes)

        if meter is not None:
            ts = meter.TimeSignature(source_song.time_sig)
            measure_capacity = fractions.Fraction(ts.numerator * (4.0 / ts.denominator)).limit_denominator(64)
        else:
            parts = source_song.time_sig.split("/")
            num = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 4
            den = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 4
            measure_capacity = fractions.Fraction(num * 4, den).limit_denominator(64)

        source_chunks = list(
            stream_quantized_events(
                source_song.bass_notes, source_song.bpm, measure_capacity, source_song.is_compound, level=5
            )
        )
        total_measures = max(
            [getattr(c, "measure_index", getattr(c, "measure_num", 1)) for c in source_chunks],
            default=len(source_song.measures) or 1,
        )

        pitches = [n.pitch for n in source_song.bass_notes if getattr(n, "pitch", None) is not None and n.pitch > 0]
        min_pitch_str = _midi_to_name(min(pitches), key_obj=source_song.key_obj) if pitches else "N/A"
        max_pitch_str = _midi_to_name(max(pitches), key_obj=source_song.key_obj) if pitches else "N/A"
        min_midi = min(pitches) if pitches else 0
        max_midi = max(pitches) if pitches else 0

        avg_notes_per_measure = (total_bass_notes / total_measures) if total_measures > 0 else 0.0

        print(
            f"  ├─ BPM: {source_song.bpm:.2f} | Time Signature: {source_song.time_sig} | Tuning: {source_song.tuning_type}"
        )
        print(f"  ├─ Total Raw Audio Events: {total_bass_events}")
        print(f"  ├─ Total High-Level Note Objs: {total_bass_notes}")
        print(
            f"  ├─ Partitioned Measures: {total_measures} MeasureChunks (Avg Density: {avg_notes_per_measure:.1f} notes/bar)"
        )
        print(f"  └─ Pitch Range: {min_pitch_str} (MIDI {min_midi}) ───> {max_pitch_str} (MIDI {max_midi})")

        # --- Target Level Iteration ---
        for target_level in target_levels:
            print("\n======================================================================")
            print(f"                     PROCESSING TARGET LEVEL {target_level}                     ")
            print("======================================================================")

            level_song = copy.deepcopy(source_song)
            level_song.target_level = target_level

            # Stage 4: Level Filtering
            print(f"\n[Stage 4: Level Filtering] Applying Level {target_level} notation abstraction...")
            pre_filter_notes = list(level_song.bass_notes)
            pre_filter_count = len(pre_filter_notes)
            filter_song_for_level(level_song)
            post_filter_notes = level_song.bass_notes
            post_filter_count = len(post_filter_notes)
            retained_pct = (post_filter_count / pre_filter_count * 100.0) if pre_filter_count > 0 else 0.0
            pre_density = pre_filter_count / total_measures if total_measures > 0 else 0.0
            post_density = post_filter_count / total_measures if total_measures > 0 else 0.0

            print(
                f"  ├─ Note Reduction: {pre_filter_count} ──> {post_filter_count} notes ({retained_pct:.1f}% retained)"
            )
            print(f"  ├─ Note Density Change: {pre_density:.1f} ──> {post_density:.1f} notes/measure")

            # Stage 5: Pitch Theory & Key Snapping
            print(f"\n[Stage 5: Pitch Theory] Snapping notes to scale '{level_song.key_obj}'...")
            from subtone.pitch_theory import snap_song_to_scale
            pitches_before = [n.pitch for n in level_song.bass_notes]
            snap_song_to_scale(level_song)
            pitches_after = [n.pitch for n in level_song.bass_notes]

            snapped_diff_count = sum(1 for b, a in zip(pitches_before, pitches_after) if b != a)
            sample_pitches_str = [
                _midi_to_name(n.pitch, key_obj=level_song.key_obj) for n in level_song.bass_notes[:6]
            ]
            print(f"  ├─ Key Object: {level_song.key_obj}")
            print(f"  ├─ Notes Adjusted to Scale: {snapped_diff_count} / {len(pitches_after)}")
            print(f"  └─ Sample Pitch Sequence (1st 6 Notes): {sample_pitches_str}")

            # Stage 6: Ergonomic Fretboard Solver (HMM)
            print(
                "\n[Stage 6: Ergonomic Fretboard HMM] Calculating optimal string/fret path across Note sequence..."
            )
            from subtone.fretboard import ErgonomicFretboardHMMSolver
            solver = ErgonomicFretboardHMMSolver(song=level_song)
            solver.solve_song(level_song)

            positions = level_song.fretboard_path or []
            sample_positions = positions[:5] if positions else []

            string_counts = {}
            frets = []
            for pos in positions:
                if pos:
                    s_num, f_num, _ = pos
                    string_counts[s_num] = string_counts.get(s_num, 0) + 1
                    if f_num > 0:
                        frets.append(f_num)

            min_fret = min(frets) if frets else 0
            max_fret = max(frets) if frets else 0
            avg_fret = (sum(frets) / len(frets)) if frets else 0.0

            notes = level_song.bass_notes
            slaps = sum(1 for n in notes if getattr(n, "is_slap", False) or getattr(n, "tag", "") == "slap")
            pops = sum(1 for n in notes if getattr(n, "is_pop", False) or getattr(n, "tag", "") == "pop")
            ghosts = sum(1 for n in notes if getattr(n, "is_ghost", False) or getattr(n, "tag", "") == "ghost")
            palm_mutes = sum(
                1 for n in notes if getattr(n, "is_palm_mute", False) or getattr(n, "tag", "") == "palm_mute"
            )
            harmonics = sum(1 for n in notes if getattr(n, "is_harmonic", False) or getattr(n, "tag", "") == "harmonic")
            downpicks = sum(1 for n in notes if getattr(n, "is_downpick", False))

            rakes_cnt = sum(level_song.rakes) if getattr(level_song, "rakes", None) else 0
            legatos_cnt = sum(level_song.legatos) if getattr(level_song, "legatos", None) else 0
            slides_cnt = sum(level_song.slides) if getattr(level_song, "slides", None) else 0

            print(f"  ├─ Solved Positions (1st 5 Notes): {sample_positions}")
            print(f"  ├─ Fretboard Range: Frets {min_fret} to {max_fret} (Average Fret: {avg_fret:.1f})")
            print(f"  ├─ String Usage: {dict(sorted(string_counts.items()))}")
            print("  └─ Articulations Detected on Note Stream:")
            print(f"     ├─ Slaps: {slaps} | Pops: {pops} | Ghosts: {ghosts} | Palm Mutes: {palm_mutes}")
            print(
                f"     └─ Harmonics: {harmonics} | Downpicks: {downpicks} | Legatos: {legatos_cnt} | Slides: {slides_cnt} | Rakes: {rakes_cnt}"
            )

            # Stage 7: Score Building & MusicXML Export
            if generate_all_levels:
                output_name = f"{clean_filename}_Level{target_level}.musicxml"
            else:
                output_name = f"{clean_filename}.musicxml"

            output_path = os.path.join(self.output_dir, output_name)
            level_song.output_xml_path = output_path

            print(
                "\n[Stage 7: Score Building & Engraver Rules] Decomposing Note stream into MeasureChunks & RhythmicAtoms..."
            )
            level_chunks = list(
                stream_quantized_events(
                    level_song.bass_notes,
                    level_song.bpm,
                    measure_capacity,
                    level_song.is_compound,
                    level=target_level,
                )
            )
            level_measures_cnt = max(
                [getattr(c, "measure_index", getattr(c, "measure_num", 1)) for c in level_chunks],
                default=len(level_song.measures) or 1,
            )

            level_atoms = [atom for c in level_chunks for atom in c.atoms]
            total_atoms_cnt = len(level_atoms)
            pitched_atoms_cnt = sum(1 for a in level_atoms if not a.is_rest and a.pitch > 0)
            rest_atoms_cnt = sum(1 for a in level_atoms if a.is_rest or a.pitch == 0)
            tied_atoms_cnt = sum(1 for a in level_atoms if a.tie_type is not None)

            print(f"  ├─ Total MeasureChunks Built: {level_measures_cnt}")
            print(f"  ├─ Total RhythmicAtoms Generated: {total_atoms_cnt}")
            print(f"  │  ├─ Pitched Atoms: {pitched_atoms_cnt}")
            print(f"  │  ├─ Rest Atoms:    {rest_atoms_cnt}")
            print(f"  │  └─ Tied Atoms:    {tied_atoms_cnt}")

            build_and_export_song(level_song)
            print(f"  └─ Output saved: {output_path}")

            # --- Pasteable Song Analysis Summary ---
            print("\n" + "=" * 70)
            print("                     SONG TRANSCRIPTION ANALYSIS SUMMARY                ")
            print("=" * 70)
            print(f" Track:               {artist_name} - {song_title}")
            print(f" Key Signature:       {level_song.key_obj} (Parsed: {parsed_key})")
            print(f" Tempo & Meter:       {level_song.bpm:.2f} BPM | {level_song.time_sig}")
            print(f" Tuning Profile:      {level_song.tuning_type}")
            print(f" Abstraction Level:   Level {target_level}")
            print(f" Total Measures:      {level_measures_cnt}")
            print("-" * 70)
            print(" NOTE & RHYTHMIC ATOM METRICS")
            print(f" • Raw Audio Events:  {total_bass_events}")
            print(f" • Processed Notes:   {post_filter_count} ({retained_pct:.1f}% of level 5)")
            print(f" • Pitch Range:       {min_pitch_str} (MIDI {min_midi}) to {max_pitch_str} (MIDI {max_midi})")
            print(f" • Scale Adjustments: {snapped_diff_count} notes snapped to {level_song.key_obj}")
            print(
                f" • Rhythmic Atoms:    {total_atoms_cnt} total ({pitched_atoms_cnt} pitched, {rest_atoms_cnt} rests, {tied_atoms_cnt} ties)"
            )
            print("-" * 70)
            print(" FRETBOARD & TECHNIQUE BREAKDOWN")
            print(f" • Active Fret Range: Fret {min_fret} to Fret {max_fret} (Avg Fret: {avg_fret:.1f})")
            print(" • String Breakdown:  " + ", ".join([f"Str {s}: {cnt}" for s, cnt in sorted(string_counts.items())]))
            print(f" • Techniques:        Slap={slaps}, Pop={pops}, Ghost={ghosts}, PalmMute={palm_mutes}")
            print(
                f" • Expressive Marks:  Legato={legatos_cnt}, Slide={slides_cnt}, Rake={rakes_cnt}, Harmonic={harmonics}"
            )
            print("-" * 70)
            print(f" Output File:         {output_path}")
            print("=" * 70 + "\n")


def main():
    """Transcribe audio files or compressed AudioEvents folders into MusicXML files."""
    parser = argparse.ArgumentParser(description="Subtone Bass Transcription Engine")
    parser.add_argument("inputs", nargs="+", help="Path to audio file(s) or preprocessed AudioEvents folder(s)")
    parser.add_argument("-a", "--all-levels", action="store_true", help="Generate outputs for all levels")
    parser.add_argument("-o", "--output-dir", help="Custom output directory")
    parser.add_argument("--level", type=int, default=5, help="Complexity level (0-5)")
    parser.add_argument("-g", "--gpu", action="store_true", help="Use GPU-backed audio dependencies")
    parser.add_argument("-c", "--config", help="Path to custom genre/configuration directory or TOML file")
    parser.add_argument("--genre", help="Genre override name")
    args = parser.parse_args()

    pipeline = AudioTranscriptionPipeline(
        output_dir=args.output_dir,
        genre_config_path=args.config,
    )
    for target in args.inputs:
        pipeline.run(
            target_input=target,
            generate_all_levels=args.all_levels,
            level=args.level,
            use_gpu=args.gpu,
            genre_override=args.genre,
        )


if __name__ == "__main__":
    main()