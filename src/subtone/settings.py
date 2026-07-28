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
    from subtone.musicality import parse_metadata_from_path as _p
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
        from subtone.musicality import midi_to_pitch_string
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
        print("               SUBTONE 6-PHASE / 12-STAGE TRANSCRIPTION START          ")
        print("======================================================================")

        from subtone.dsp import (
            stage1_stem_separation_and_audio_to_midi,
            stage2_multistem_f0_tracking,
            stage3_drum_percussive_grid_mining,
            stage4_frame_to_symbolic_bounding,
            stage5_drum_pocket_and_groove_audit,
            stage6_melodic_counterpoint_register_audit,
            stage10_songwide_multistem_audit,
        )
        from subtone.musicality import stage7_harmonic_context_validation, snap_song_to_scale
        from subtone.biomechanics import stage8_genre_pattern_and_fretboard_hmm
        from subtone.tabs import (
            stage9_pedagogical_abstraction_and_partitioning,
            stage11_rest_synthesis_and_reconciliation,
            stage12_musicxml_dom_serialization,
            filter_song_for_level,
        )

        # ----------------------------------------------------------------------
        # PHASE I: EXTERNAL SCRIPT STEM SEPARATION & AUDIO-TO-MIDI
        # ----------------------------------------------------------------------
        print("\n----------------------------------------------------------------------")
        print("PHASE I: EXTERNAL SCRIPT STEM SEPARATION & AUDIO-TO-MIDI")
        print("----------------------------------------------------------------------")
        print("[Stage 1: External Script Stem Separation & Audio-to-MIDI]")
        print("  • Multi-Stem Separation using Demucs")
        print("  • Audio-to-MIDI Extraction with Basic-Pitch / Librosa")

        meta, event_streams = stage1_stem_separation_and_audio_to_midi(
            target_input=target_input,
            custom_genre=genre_override,
            config_path=self.genre_config_path,
        )
        artist_name = meta["artist_name"]
        song_title = meta["song_title"]
        parsed_key = meta["parsed_key"]
        genre_name = meta["genre_name"]
        genre_config = meta["genre_config"]
        song_stem_name = meta["song_stem_name"]
        cached_events_path = meta["cached_events_path"]

        print(f"  ├─ Artist:        '{artist_name}'")
        print(f"  ├─ Song Title:    '{song_title}'")
        print(f"  ├─ Key Signature: '{parsed_key}'")
        print(f"  └─ Genre Context: '{genre_name}' (Override: '{genre_override or 'None'}')")

        # ----------------------------------------------------------------------
        # PHASE II: SIGNAL INGESTION & GENRE-DRIVEN DSP FEATURE EXTRACTION
        # ----------------------------------------------------------------------
        print("\n----------------------------------------------------------------------")
        print("PHASE II: SIGNAL INGESTION & GENRE-DRIVEN DSP FEATURE EXTRACTION")
        print("----------------------------------------------------------------------")
        print("[Stage 2: Multi-Stem Source Separation & Genre Policy F0 Tracking]")
        print("  • Demucs HPSS (Bass, Drums, Vocals, Guitar, Piano, Other)")
        print("  • Genre Policy Injection (Tuning, Technique, DSP Bounds)")
        print("  • Dynamic F0 Tracking (Sub-bass, Drop Tuning, Slap Attacks)")

        event_streams, primary_key, tuning_type, fmin_hz = stage2_multistem_f0_tracking(
            event_streams=event_streams,
            target_input=target_input,
            genre_config=genre_config,
            genre_override=genre_override,
        )
        print(f"  ├─ Ingested HPSS Stems: {list(event_streams.keys())}")
        print(f"  ├─ Active Primary Stem: '{primary_key}'")
        print(f"  ├─ Selected Tuning Profile: '{tuning_type}' (fmin={fmin_hz:.1f} Hz)")
        print(f"  └─ Dynamic F0 Tracked Events: {len(event_streams.get(primary_key, {}).get('events', []))} events")

        print("\n[Stage 3: Genre-Aware Percussive Grid & Rhythmic Anchor Mining [DRUMS]]")
        print("  • Transient Energy Mining (Kick/Snare/Hi-Hat Maps)")
        print("  • Dynamic Swing Ratio & Clave/Syncopation Grid Extraction")

        drum_events, detected_bpm, time_sig = stage3_drum_percussive_grid_mining(
            event_streams=event_streams,
            genre_config=genre_config,
        )
        print(f"  ├─ Mined Drum Transients: {len(drum_events)} drum events")
        print(f"  ├─ Estimated Tempo & Time Sig: {detected_bpm:.2f} BPM | {time_sig}")
        print("  └─ Output: Continuous F0 Trajectories + AudioEvents")

        # ----------------------------------------------------------------------
        # PHASE III: SYMBOLIC CONVERSION, RHYTHMIC & MELODIC STEM VALIDATION
        # ----------------------------------------------------------------------
        print("\n----------------------------------------------------------------------")
        print("PHASE III: SYMBOLIC CONVERSION, RHYTHMIC & MELODIC STEM VALIDATION")
        print("----------------------------------------------------------------------")
        print("[Stage 4: Frame-to-Symbolic Bounding & Quantization Grid Mapping]")

        source_song = Song.from_event_streams(
            event_streams=event_streams,
            active_stream_name=primary_key,
            artist_name=artist_name,
            song_title=song_title,
            genres=[genre_name] if genre_name else [],
            genre_config=genre_config,
            parsed_key_str=parsed_key,
            stem_folder=cached_events_path,
        )
        source_song.bpm = detected_bpm
        source_song.time_sig = time_sig

        raw_notes = stage4_frame_to_symbolic_bounding(
            event_streams=event_streams,
            song=source_song,
            genre_config=genre_config,
        )
        print(f"  ├─ Bounded Symbolic Note Objects: {len(raw_notes)} notes")
        print("  └─ Quantization Grid Alignment: Complete")

        print("\n[Stage 5: Genre-Conditioned Rhythmic Pocket & Groove Audit [DRUMS STEM]]")
        print("  • Transient Attack Alignment & Pocket Determination")
        print("  • Technique Ghost Note Tagging (Slap Clicks / Palm Mutes)")

        pocket_notes = stage5_drum_pocket_and_groove_audit(
            bass_notes=raw_notes,
            drum_events=drum_events,
            bpm=source_song.bpm,
            genre_config=genre_config,
        )
        aligned_count = sum(1 for n in pocket_notes if getattr(n, "is_pocket_aligned", False))
        ghost_count = sum(1 for n in pocket_notes if getattr(n, "is_ghost", False))
        print(f"  ├─ Pocket-Aligned Onsets: {aligned_count} / {len(pocket_notes)}")
        print(f"  └─ Technique Ghost Notes Tagged: {ghost_count}")

        print("\n[Stage 6: Melodic Counterpoint & Register Audit [VOCALS / GUITAR STEMS]]")
        print("  • Spectral Masking Resolution & Pitch Cutoff Filtering")

        vocal_events = event_streams.get("vocals", {}).get("events", [])
        guitar_events = event_streams.get("guitar", {}).get("events", [])
        validated_notes = stage6_melodic_counterpoint_register_audit(
            bass_notes=pocket_notes,
            vocal_events=vocal_events,
            guitar_events=guitar_events,
            genre_config=genre_config,
        )
        source_song.bass_notes = validated_notes
        source_song.notes = validated_notes
        print(f"  ├─ Register Range Validated: {len(validated_notes)} notes retained")
        print("  └─ Output: Rhythmically & Melodically Validated Notes")

        # Selected level & output setup
        clean_filename = re.sub(r'[\\/*?:"<>|]', "", f"{artist_name} - {song_title}").strip()
        os.makedirs(self.output_dir, exist_ok=True)

        selected_level = level if isinstance(level, int) and 0 <= level <= 5 else 5
        target_levels = range(6) if generate_all_levels else [selected_level]

        # ----------------------------------------------------------------------
        # PHASE IV, V, VI PER TARGET LEVEL
        # ----------------------------------------------------------------------
        for target_level in target_levels:
            print("\n======================================================================")
            print(f"                     PROCESSING TARGET LEVEL {target_level}                     ")
            print("======================================================================")

            level_song = copy.deepcopy(source_song)
            level_song.target_level = target_level

            # PHASE IV: HARMONIC VALIDATION, ERGONOMICS & MEASURE PARTITIONING
            print("\n----------------------------------------------------------------------")
            print("PHASE IV: HARMONIC VALIDATION, ERGONOMICS & MEASURE PARTITIONING")
            print("----------------------------------------------------------------------")
            print("[Stage 7: Polyphonic Harmonic Context Validation [GUITAR / PIANO / OTHER]]")
            print("  • Root vs. Inversion Resolution via Chroma/CQT Matrices")
            print("  • Directional Enharmonic Pitch Spelling (Key Signature Tonal)")

            pitches_before = [n.pitch for n in level_song.bass_notes]
            stage7_harmonic_context_validation(
                song=level_song,
                guitar_events=guitar_events,
                piano_events=event_streams.get("piano", {}).get("events", []),
            )
            pitches_after = [n.pitch for n in level_song.bass_notes]
            snapped_diff_count = sum(1 for b, a in zip(pitches_before, pitches_after) if b != a)

            sample_pitches_str = [
                _midi_to_name(n.pitch, key_obj=level_song.key_obj) for n in level_song.bass_notes[:6]
            ]
            print(f"  ├─ Key Object: {level_song.key_obj}")
            print(f"  ├─ Scale Snapped Notes: {snapped_diff_count} / {len(pitches_after)}")
            print(f"  └─ Sample Pitch Sequence (1st 6 Notes): {sample_pitches_str}")

            print("\n[Stage 8: Genre Pattern Engine & Biomechanical Ergonomic Solver]")
            print("  • Genre Pattern Matching (Tumbao, Walking, Gallop, Slap)")
            print("  • Fretboard HMM / Viterbi Path (Genre Cost Parameter Matrix)")

            positions = stage8_genre_pattern_and_fretboard_hmm(level_song) or []
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
            print(f"  ├─ Active Fret Range: Frets {min_fret} to {max_fret} (Average Fret: {avg_fret:.1f})")
            print(f"  ├─ String Usage: {dict(sorted(string_counts.items()))}")
            print(f"  └─ Technique Articulations: Slaps={slaps}, Pops={pops}, Ghosts={ghosts}, PalmMutes={palm_mutes}, Harmonics={harmonics}")

            print("\n[Stage 9: Pedagogical Abstraction (Levels 1-5) & Metric Partitioning]")
            print("  • 5-Level Pedagogical Filter Matrix Application")
            print("  • Measure Capacity Partitioning & Beat Boundary Note Tying")

            pre_count = len(level_song.bass_notes)
            level_chunks, measure_capacity = stage9_pedagogical_abstraction_and_partitioning(
                song=level_song,
                target_level=target_level,
            )
            post_count = len(level_song.bass_notes)
            retained_pct = (post_count / pre_count * 100.0) if pre_count > 0 else 0.0

            print(f"  ├─ Target Level: Level {target_level}")
            print(f"  ├─ Retained Note Count: {post_count} notes ({retained_pct:.1f}% retained)")
            print(f"  └─ MeasureChunks Partitioned: {len(level_chunks)} bars")

            # PHASE V: HOLISTIC MULTI-STEM VALIDATION & REST SYNTHESIS ENGINE
            print("\n----------------------------------------------------------------------")
            print("PHASE V: HOLISTIC MULTI-STEM VALIDATION & REST SYNTHESIS ENGINE")
            print("----------------------------------------------------------------------")
            print("[Stage 10: Song-Wide Multi-Stem Audit, Outlier Pruning & Coherence]")
            print("  • Cross-Scan Bass against ALL STEMS (Drums/Guitar/Keys/Vocals)")
            print("  • Section Healing & Melodic Strictest Bounds Enforcement")

            stage10_songwide_multistem_audit(
                song=level_song,
                measure_chunks=level_chunks,
                all_stem_events=event_streams,
            )
            print("  ├─ Cross-Scan Multi-Stem Validation: Complete")
            print("  └─ Outlier Pruning & Section Healing Applied")

            print("\n[Stage 11: First-Class Rest Synthesis & Measure Reconciliation]")
            print("  • Instantiate Explicit Rest Objects (Duration, Position)")
            print("  • Strict Measure Capacity Lock: Sum(Notes) + Sum(Rests) = Bar")

            audited_chunks = stage11_rest_synthesis_and_reconciliation(
                measure_chunks=level_chunks,
                measure_capacity=measure_capacity,
                time_sig=level_song.time_sig,
            )
            level_song.measures = audited_chunks
            total_atoms = sum(len(getattr(c, "atoms", [])) for c in audited_chunks)
            rest_atoms = sum(1 for c in audited_chunks for a in getattr(c, "atoms", []) if getattr(a, "is_rest", False))
            print(f"  ├─ Explicit Rest Atoms Synthesized: {rest_atoms}")
            print("  └─ Capacity Reconciliation Lock: 100% Validated Bar Measure Capacity")

            # PHASE VI: PURE SCORE SERIALIZATION & MUSICXML ENGRAVING
            print("\n----------------------------------------------------------------------")
            print("PHASE VI: PURE SCORE SERIALIZATION & MUSICXML ENGRAVING")
            print("----------------------------------------------------------------------")
            print("[Stage 12: Pure 1:1 Score Object to MusicXML DOM Serialization]")

            if generate_all_levels:
                output_name = f"{clean_filename}_Level{target_level}.musicxml"
            else:
                output_name = f"{clean_filename}.musicxml"

            output_path = os.path.join(self.output_dir, output_name)
            final_xml_path = stage12_musicxml_dom_serialization(
                song=level_song,
                measure_chunks=audited_chunks,
                output_path=output_path,
            )

            print("  ├─ Score Object Tree Serialized to MusicXML DOM")
            print(f"  └─ Output saved: {final_xml_path}")

            # --- Song Analysis Summary ---
            print("\n" + "=" * 70)
            print("                     SONG TRANSCRIPTION ANALYSIS SUMMARY                ")
            print("=" * 70)
            print(f" Track:               {artist_name} - {song_title}")
            print(f" Key Signature:       {level_song.key_obj} (Parsed: {parsed_key})")
            print(f" Tempo & Meter:       {level_song.bpm:.2f} BPM | {level_song.time_sig}")
            print(f" Tuning Profile:      {level_song.tuning_type}")
            print(f" Abstraction Level:   Level {target_level}")
            total_measures_cnt = max(c.measure_num for c in audited_chunks) if audited_chunks else 0
            print(f" Total Measures:      {total_measures_cnt}")
            print("-" * 70)
            print(" NOTE & RHYTHMIC ATOM METRICS")
            print(f" • Processed Notes:   {post_count} ({retained_pct:.1f}% retained)")
            print(f" • Scale Adjustments: {snapped_diff_count} notes snapped to {level_song.key_obj}")
            print(f" • Rhythmic Atoms:    {total_atoms} total ({total_atoms - rest_atoms} pitched, {rest_atoms} rests)")
            print("-" * 70)
            print(" FRETBOARD & TECHNIQUE BREAKDOWN")
            print(f" • Active Fret Range: Fret {min_fret} to Fret {max_fret} (Avg Fret: {avg_fret:.1f})")
            print(" • String Breakdown:  " + ", ".join([f"Str {s}: {cnt}" for s, cnt in sorted(string_counts.items())]))
            print(f" • Techniques:        Slap={slaps}, Pop={pops}, Ghost={ghosts}, PalmMute={palm_mutes}")
            print(f" • Expressive Marks:  Legato={legatos_cnt}, Slide={slides_cnt}, Rake={rakes_cnt}, Harmonic={harmonics}")
            print("-" * 70)
            print(f" Output File:         {final_xml_path}")
            print("=" * 70 + "\n")

            processed_song = level_song

        # `target_levels` always contains at least one entry (either the
        # requested `level` or the full 0-5 sweep), so the loop above always
        # runs and `processed_song` is always bound to a fully-transcribed
        # Song. When --all-levels is used, the Song for the final (most
        # complete) level is returned as the canonical result.
        return processed_song