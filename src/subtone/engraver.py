import fractions
import os
import xml.etree.ElementTree as ET
from copy import deepcopy

from subtone.settings_loader import (
    ATTRIBUTES_SCHEMA_ORDER,
    BASS_STRING_TUNINGS,
    DEFAULT_TIME_SIGNATURE,
    DEFAULT_TUNING_TYPE,
    MAX_FRETBOARD_FRETS,
    MEASURE_SCHEMA_ORDER,
    NOTATIONS_SCHEMA_ORDER,
    NOTE_SCHEMA_ORDER,
    TECHNICAL_SCHEMA_ORDER,
)
from subtone.fretboard import ErgonomicFretboardHMMSolver

from subtone.schemas import Genre, Level, MeasureChunk, Note, RhythmicAtom, Song
from subtone.pitch_theory import get_directional_enharmonic_pitch, parse_key_object

try:
    import numpy as np
except ImportError:
    np = None

def get_closest_value(val, arr):
    if not arr:
        return None
    return min(arr, key=lambda x: abs(x - val))

BASS_TUNINGS = BASS_STRING_TUNINGS
PITCH_SCHEMA_ORDER = ["step", "alter", "octave"]

try:
    from music21 import (
        articulations,
        duration,
        dynamics,
        expressions,
        instrument,
        key,
        metadata,
        meter,
        note,
        pitch,
        spanner,
        stream,
        tempo,
        tie,
    )
except ModuleNotFoundError:

    class _PitchObj:
        def __init__(self, name="C3"):
            import re

            if isinstance(name, _PitchObj):
                self.step = name.step
                self.octave = name.octave
                self.alter = name.alter
                self.name = getattr(name, "name", f"{self.step}{self.octave}")
                return
            if isinstance(name, _DummyObj):
                if getattr(name, "pitch", None) and isinstance(name.pitch, _PitchObj):
                    self.step = name.pitch.step
                    self.octave = name.pitch.octave
                    self.alter = name.pitch.alter
                    self.name = name.pitch.name
                    return
                name = getattr(name, "name", str(name))
            self.name = str(name) if name else "C3"
            match = re.match(r"^([A-Ga-g])([#b\-]*)(-?\d+)?$", self.name)
            if match:
                step_raw, alt_raw, oct_raw = match.groups()
                self.step = step_raw.upper()
                self.octave = int(oct_raw) if oct_raw is not None else 3
                if not alt_raw:
                    self.alter = 0
                elif "#" in alt_raw:
                    self.alter = alt_raw.count("#")
                else:
                    self.alter = -(alt_raw.count("b") + alt_raw.count("-"))
            else:
                self.step = "C"
                self.octave = 3
                self.alter = 0

            if self.alter == 1:
                self.accidental = _DummyObj("sharp")
                self.accidental.name = "sharp"
            elif self.alter == -1:
                self.accidental = _DummyObj("flat")
                self.accidental.name = "flat"
            else:
                self.accidental = None

        def getEnharmonic(self, inPlace=True):
            return self

    class _DummyObj:
        def __str__(self):
            return str(getattr(self, "name", "C3"))

        def __init__(self, *args, **kwargs):
            self.notesAndRests = []
            self.children = []
            self.type = "quarter"
            self.quarterLength = float(kwargs.get("quarterLength", 1.0))
            self.articulations = []
            self.expressions = []
            self.number = kwargs.get("number", 1)
            self.numerator = 4
            self.denominator = 4
            self.pitch = None
            self.isNote = False
            self.isRest = False

            t_name = args[0] if args and isinstance(args[0], (str, int)) else None
            if t_name is not None:
                self.name = str(t_name)
                self.tonic = _PitchObj(self.name)
            else:
                self.name = "C"
                self.tonic = _PitchObj("C")

            self.mode = "major"
            self.accidental = None
            if len(args) > 1 and isinstance(args[1], str):
                self.mode = args[1]
            if t_name and "/" in str(t_name):
                try:
                    parts = str(t_name).split("/")
                    self.numerator = int(parts[0])
                    self.denominator = int(parts[1])
                except (ValueError, IndexError):
                    pass

        @property
        def duration(self):
            return getattr(self, "_duration", self)

        @duration.setter
        def duration(self, val):
            self._duration = val

        def insert(self, *args, **kwargs):
            for a in args:
                if isinstance(a, _DummyObj):
                    self.children.append(a)
                    if getattr(a, "isNote", False) or getattr(a, "isRest", False):
                        self.notesAndRests.append(a)
            return self

        def append(self, *args, **kwargs):
            for a in args:
                if isinstance(a, _DummyObj):
                    self.children.append(a)
                    if getattr(a, "isNote", False) or getattr(a, "isRest", False):
                        self.notesAndRests.append(a)
            return self

        def remove(self, *args, **kwargs):
            for a in args:
                if a in self.children:
                    self.children.remove(a)
                if a in self.notesAndRests:
                    self.notesAndRests.remove(a)

        def getElementsByClass(self, cls_type, *args, **kwargs):
            return [c for c in self.children if isinstance(c, _DummyObj)]

        def write(self, fmt, fp=None, **kwargs):
            if not fp:
                return self

            measures = []

            def _find_measures(obj):
                for c in getattr(obj, "children", []):
                    has_notes = any(
                        getattr(x, "isNote", False) or getattr(x, "isRest", False)
                        for x in getattr(c, "notesAndRests", [])
                    )
                    if has_notes:
                        measures.append(c)
                    else:
                        _find_measures(c)

            _find_measures(self)

            if not measures:
                measures = (
                    [self]
                    if any(
                        getattr(x, "isNote", False) or getattr(x, "isRest", False)
                        for x in getattr(self, "notesAndRests", [])
                    )
                    else []
                )

            lines = [
                '<?xml version="1.0" encoding="UTF-8"?>',
                '<score-partwise version="3.1">',
                "  <part-list>",
                '    <score-part id="P1">',
                "      <part-name>Electric Bass</part-name>",
                "    </score-part>",
                "  </part-list>",
                '  <part id="P1">',
            ]

            curr_num, curr_den = 4, 4

            def _find_time_sig(obj):
                for c in getattr(obj, "children", []) + getattr(obj, "notesAndRests", []):
                    if getattr(c, "numerator", None) and getattr(c, "denominator", None):
                        if c.numerator > 0 and c.denominator > 0 and (c.numerator != 4 or c.denominator != 4):
                            return c.numerator, c.denominator
                    res = _find_time_sig(c)
                    if res:
                        return res
                return None

            global_ts = _find_time_sig(self)
            if global_ts:
                curr_num, curr_den = global_ts

            m_num = 1
            divisions = 32

            for m in measures if measures else [self]:
                measure_number = getattr(m, "number", m_num)
                m_num = measure_number + 1

                m_ts = _find_time_sig(m)
                if m_ts:
                    curr_num, curr_den = m_ts

                lines.append(f'    <measure number="{measure_number}">')
                lines.append("      <attributes>")
                lines.append(f"        <divisions>{divisions}</divisions>")
                lines.append("        <time>")
                lines.append(f"          <beats>{curr_num}</beats>")
                lines.append(f"          <beat-type>{curr_den}</beat-type>")
                lines.append("        </time>")
                lines.append("      </attributes>")

                notes = [
                    n
                    for n in getattr(m, "notesAndRests", [])
                    if getattr(n, "isNote", False) or getattr(n, "isRest", False)
                ]
                if not notes:
                    capacity_q = curr_num * (4.0 / curr_den)
                    dur_units = max(1, int(round(capacity_q * divisions)))
                    lines.append("      <note>")
                    lines.append("        <rest/>")
                    lines.append(f"        <duration>{dur_units}</duration>")
                    lines.append("        <type>quarter</type>")
                    lines.append("      </note>")
                else:
                    for n in notes:
                        dur_q = getattr(getattr(n, "duration", n), "quarterLength", 1.0)
                        dur_units = max(1, int(round(dur_q * divisions)))
                        lines.append("      <note>")
                        if getattr(n, "pitch", None):
                            lines.append("        <pitch>")
                            lines.append(f"          <step>{n.pitch.step}</step>")
                            lines.append(f"          <alter>{n.pitch.alter}</alter>")
                            lines.append(f"          <octave>{n.pitch.octave}</octave>")
                            lines.append("        </pitch>")
                        else:
                            lines.append("        <rest/>")
                        lines.append(f"        <duration>{dur_units}</duration>")
                        if getattr(n, "tie", None):
                            t_val = getattr(n.tie, "type", None)
                            if t_val not in ["start", "stop", "continue"]:
                                t_val = getattr(n.tie, "name", None)
                            if t_val not in ["start", "stop", "continue"]:
                                t_val = "start"
                            lines.append(f'        <tie type="{t_val}"/>')
                        lines.append("        <type>quarter</type>")
                        if (
                            abs(dur_q - 0.75) < 1e-3
                            or abs(dur_q - 1.5) < 1e-3
                            or abs(dur_q - 3.0) < 1e-3
                            or getattr(n, "is_dotted", False)
                        ):
                            lines.append("        <dot/>")
                        if getattr(n, "pitch", None) and getattr(n.pitch, "alter", 0) != 0:
                            if n.pitch.alter == 1:
                                lines.append("        <accidental>sharp</accidental>")
                            elif n.pitch.alter == -1:
                                lines.append("        <accidental>flat</accidental>")
                        if getattr(n, "articulations", None):
                            lines.append("        <notations>")
                            lines.append("          <articulations>")
                            for art in n.articulations:
                                art_name = (
                                    getattr(art, "__class__", None).__name__.lower()
                                    if getattr(art, "__class__", None)
                                    else str(art).lower()
                                )
                                if "staccato" in art_name:
                                    lines.append("            <staccato/>")
                                elif "accent" in art_name:
                                    lines.append("            <accent/>")
                                elif "tenuto" in art_name:
                                    lines.append("            <tenuto/>")
                            lines.append("          </articulations>")
                            lines.append("        </notations>")
                        lines.append("      </note>")

                lines.append("    </measure>")

            lines.append("  </part>")
            lines.append("</score-partwise>")

            with open(fp, "w") as f:
                f.write("\n".join(lines))
            return self

    class _NoteModule:
        def Note(self, p_str=None, **kw):
            obj = _DummyObj(p_str, **kw)
            obj.isNote = True
            obj.isRest = False
            if p_str:
                obj.pitch = _PitchObj(p_str)
            return obj

        def Rest(self, *args, **kw):
            obj = _DummyObj(*args, **kw)
            obj.isNote = False
            obj.isRest = True
            obj.pitch = None
            return obj

        def __getattr__(self, name):
            return lambda *args, **kwargs: _DummyObj(*args, **kwargs)

    class _PitchModule:
        def Pitch(self, *args, **kwargs):
            if args:
                return _PitchObj(args[0])
            if "midi" in kwargs:
                midi_val = int(kwargs["midi"])
                names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                step_name = names[midi_val % 12]
                octave = (midi_val // 12) - 1
                return _PitchObj(f"{step_name}{octave}")
            return _PitchObj("C3")

        def __getattr__(self, name):
            return lambda *args, **kwargs: _DummyObj(*args, **kwargs)

    class _DummyModule:
        def __getattr__(self, name):
            def _factory(*args, **kwargs):
                cls = type(name, (_DummyObj,), {})
                return cls(*args, **kwargs)

            return _factory

    _m = _DummyModule()
    articulations = _m
    dynamics = _m
    expressions = _m
    instrument = _m
    key = _m
    meter = _m
    metadata = _m
    note = _NoteModule()
    pitch = _PitchModule()
    spanner = _m
    stream = _m
    tempo = _m
    tie = _m
    duration = _m


# --- Notation & Level Filtering Utilities ---


def _filter_notes_for_level(
    layer: list[Note], level: int | Level, beats, is_compound: bool, bpm: float, genre_config=None
) -> list[Note]:
    """Filters the processed note stream according to the difficulty profile."""
    if not layer:
        return []

    def copy_note(note):
        if hasattr(note, "clone"):
            return note.clone()
        return deepcopy(note)

    if genre_config is None:
        genre_obj = Genre()
    elif isinstance(genre_config, dict):
        genre_obj = Genre.from_dict("default", genre_config)
    else:
        genre_obj = genre_config

    level_profile = genre_obj.level_profile

    if not isinstance(level, Level):
        level_obj = Level.from_id(level, level_profile=level_profile)
    else:
        level_obj = Level.from_id(level.level_id, level_profile=level_profile)

    def is_note_allowed_by_profile(note: Note, lvl_obj: Level) -> bool:
        if lvl_obj.enabled_techniques:
            tag_tech_map = {
                "slap": "slap_pop",
                "pop": "slap_pop",
                "palm_mute": "thumb_mute",
            }
            mapped_tech = tag_tech_map.get(note.tag)
            if mapped_tech and mapped_tech not in lvl_obj.enabled_techniques:
                return False

        if lvl_obj.enabled_articulations:
            if note.tag not in lvl_obj.enabled_articulations:
                if note.tag in ["ghost", "slap", "pop", "staccato", "palm_mute"]:
                    return False
                else:
                    note.tag = "normal"
        return True

    if level_obj.level_id == 5:
        filtered_complete = []
        for note in layer:
            n_copy = copy_note(note)
            if is_note_allowed_by_profile(n_copy, level_obj):
                filtered_complete.append(n_copy)
        return filtered_complete

    filtered = []
    beat_interval = 60.0 / bpm if bpm > 0 else 0.5
    beats_per_measure = 6 if is_compound else 4

    if len(beats) == 0:
        return list(layer)

    if np is not None:
        starts = np.asarray([event.start for event in layer], dtype=float)
        event_order = np.argsort(starts, kind="stable")
        sorted_starts = starts[event_order]

        downbeat_indices = range(0, len(beats), beats_per_measure)
        downbeats = np.array([beats[bi] for bi in downbeat_indices])

        half_indices = [min(bi + beats_per_measure // 2, len(beats) - 1) for bi in downbeat_indices]
        half_measure_beats = np.array([beats[hi] for hi in half_indices])
    else:
        sorted_layer_with_indices = sorted(enumerate(layer), key=lambda x: x[1].start)
        event_order = [idx for idx, _ in sorted_layer_with_indices]
        sorted_starts = [note.start for _, note in sorted_layer_with_indices]

        downbeat_indices = range(0, len(beats), beats_per_measure)
        downbeats = [beats[bi] for bi in downbeat_indices]

        half_indices = [min(bi + beats_per_measure // 2, len(beats) - 1) for bi in downbeat_indices]
        half_measure_beats = [beats[hi] for hi in half_indices]

    eighth_beats = []
    for i in range(len(beats) - 1):
        eighth_beats.append(beats[i])
        eighth_beats.append((beats[i] + beats[i + 1]) / 2.0)

    ghost_enabled = genre_obj.features.get("ghost_notes", True)

    if level_obj.downbeat_only:
        if np is not None:
            target_beats = np.sort(np.unique(np.concatenate((downbeats, half_measure_beats))))
            for tb in target_beats:
                left = np.searchsorted(sorted_starts, tb - 0.20, side="left")
                right = np.searchsorted(sorted_starts, tb + (beat_interval * 1.5), side="left")
                window_notes = [layer[index] for index in event_order[left:right]]
                if window_notes:
                    root_note = min(window_notes, key=lambda x: x.pitch)
                    n_copy = copy_note(root_note)
                    n_copy.end = root_note.start + (beat_interval * 2.0)
                    n_copy.tag = "normal"
                    n_copy.category = "groove_anchor"
                    n_copy.is_anchor = True
                    if is_note_allowed_by_profile(n_copy, level_obj):
                        filtered.append(n_copy)
        else:
            target_beats = sorted(list(set(downbeats + half_measure_beats)))
            for tb in target_beats:
                w_min = tb - 0.20
                w_max = tb + (beat_interval * 1.5)
                window_notes = [note for note in layer if w_min <= note.start < w_max]
                if window_notes:
                    root_note = min(window_notes, key=lambda x: x.pitch)
                    n_copy = copy_note(root_note)
                    n_copy.end = root_note.start + (beat_interval * 2.0)
                    n_copy.tag = "normal"
                    n_copy.category = "groove_anchor"
                    n_copy.is_anchor = True
                    if is_note_allowed_by_profile(n_copy, level_obj):
                        filtered.append(n_copy)
        return filtered

    for note in layer:
        c_beat = get_closest_value(note.start, beats)
        is_on_beat = abs(note.start - c_beat) < 0.15 if c_beat is not None else False
        c_eighth = get_closest_value(note.start, eighth_beats)
        is_on_eighth = abs(note.start - c_eighth) < 0.15 if c_eighth is not None else False

        n_copy = copy_note(note)

        if n_copy.tag == "ghost" and (not level_obj.ghost_notes or not ghost_enabled):
            continue

        if level_obj.level_id <= 1 and not is_on_beat:
            continue
        if level_obj.level_id == 2 and not (is_on_beat or is_on_eighth):
            continue

        if is_note_allowed_by_profile(n_copy, level_obj):
            filtered.append(n_copy)

    return filtered


def filter_song_for_level(song: Song, level: int | Level = None) -> Song:
    """Apply the level profile using the song as the state boundary."""
    target_level = song.target_level if level is None else level
    song.replace_notes(
        _filter_notes_for_level(
            song.notes,
            target_level,
            song.beat_times,
            song.is_compound,
            song.bpm,
            song.genre_config,
        )
    )
    song.target_level = target_level.level_id if isinstance(target_level, Level) else int(target_level)
    return song


def get_measure_breakdown(notes, beat_times, is_compound: bool, time_sig: str = DEFAULT_TIME_SIGNATURE):
    """Groups notes into measures based on beat timestamps."""
    if notes is None or len(notes) == 0 or beat_times is None or len(beat_times) == 0:
        return []

    if is_compound:
        beats_per_m = 6
    else:
        try:
            beats_per_m = int(str(time_sig).split("/")[0]) if "/" in str(time_sig) else 4
            if beats_per_m <= 0:
                beats_per_m = 4
        except (ValueError, TypeError, IndexError):
            beats_per_m = 4

    if len(beat_times) < beats_per_m:
        return [(1, len(notes))]

    measure_starts = beat_times[::beats_per_m]

    breakdown = []
    for idx, m_start in enumerate(measure_starts):
        m_end = measure_starts[idx + 1] if idx + 1 < len(measure_starts) else float("inf")
        m_notes = [n for n in notes if m_start <= n.start < m_end]
        if m_notes:
            breakdown.append((idx + 1, len(m_notes)))

    return breakdown


def format_breakdown_log(
    notes,
    beat_times,
    is_compound: bool,
    time_sig: str = DEFAULT_TIME_SIGNATURE,
    label: str = "notes",
) -> str:
    """Returns a clean summary string indicating active bassline measures and event counts."""
    breakdown = get_measure_breakdown(notes, beat_times, is_compound, time_sig)
    if not breakdown:
        return f"0 bars (0 {label})"

    total_events = len(notes) if notes is not None else 0
    return f"{len(breakdown)} bars ({total_events} {label})"


# --- Helper Methods ---


def make_dynamic(dynamic_name: str):
    """Create a music21 Dynamic expression object."""
    return dynamics.Dynamic(dynamic_name)


def make_duration(quarter_length):
    """Create a music21 duration."""
    return duration.Duration(quarterLength=float(quarter_length))


def make_pitch(midi_value):
    """Create a music21 pitch from a MIDI value."""
    return pitch.Pitch(midi=midi_value)


def build_m21_duration(dur_q):
    """Convert a quarter-length value into a music21 Duration."""
    return make_duration(dur_q)


# --- MusicXML Schema & Tablature Injection ---


def _insert_schema_compliant(parent: ET.Element, tag_name: str, schema_order: list[str], ns: str = "") -> ET.Element:
    """Inserts a new SubElement at its exact schema-valid position without requiring full DOM reordering."""
    target_rank = schema_order.index(tag_name) if tag_name in schema_order else 999

    insert_idx = len(parent)
    for idx, child in enumerate(parent):
        c_tag = child.tag.replace(ns, "") if ns else child.tag.split("}")[-1]
        c_rank = schema_order.index(c_tag) if c_tag in schema_order else 999
        if c_rank > target_rank:
            insert_idx = idx
            break

    elem = ET.Element(f"{ns}{tag_name}")
    parent.insert(insert_idx, elem)
    return elem


def _pitch_to_midi(step: str, octave: int, alter: int = 0) -> int:
    step_offsets = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    base = step_offsets.get(step.upper(), 0)
    return (octave + 1) * 12 + base + alter


def _midi_to_pitch_components(midi_val: int) -> tuple[str, int, int]:
    PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pitch_name = PITCH_NAMES[midi_val % 12]
    step = pitch_name[0]
    alter = (
        1
        if len(pitch_name) > 1 and pitch_name[1] == "#"
        else (-1 if len(pitch_name) > 1 and pitch_name[1] == "b" else 0)
    )
    octave = (midi_val // 12) - 1
    return step, octave, alter


def _calculate_correct_string_fret(pitch_midi: int, tuning_type: str) -> tuple[int, int]:
    tuning = BASS_TUNINGS.get(tuning_type, BASS_TUNINGS[DEFAULT_TUNING_TYPE])
    num_strings = len(tuning)

    for idx in range(num_strings - 1, -1, -1):
        _, _, string_midi = tuning[idx]
        if pitch_midi >= string_midi:
            fret = pitch_midi - string_midi
            if fret <= MAX_FRETBOARD_FRETS:
                string_num = num_strings - idx
                return string_num, fret

    lowest_midi = tuning[0][2]
    return num_strings, max(0, pitch_midi - lowest_midi)


def _set_or_create_ordered(
    parent: ET.Element,
    tag: str,
    text_val: str,
    schema_order: list[str],
    ns: str = "",
):
    """Set a required export field without disturbing Music21's XML order."""
    elem = parent.find(f"{ns}{tag}")
    if elem is None:
        elem = _insert_schema_compliant(parent, tag, schema_order, ns)
    elem.text = str(text_val)


def get_key_fifths_and_mode(raw_key) -> tuple[int, str]:
    if not raw_key:
        return 0, "major"
    if hasattr(raw_key, "sharps") and getattr(raw_key, "sharps", None) is not None:
        mode = getattr(raw_key, "mode", "major") or "major"
        return raw_key.sharps, mode
    s = str(raw_key).strip()
    is_minor = "minor" in s.lower() or "min" in s.lower() or (len(s) > 1 and s.endswith("m") and not s.endswith("-m"))
    import re
    clean = re.sub(r"(?i)(minor|major|min|maj|m|\s)", "", s)
    if clean.endswith("-"):
        clean = clean[:-1] + "b"

    KEY_FIFTHS_MAP = {
        "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "Fsharp": 6, "C#": 7, "Csharp": 7,
        "F": -1, "Bb": -2, "Bflat": -2, "Eb": -3, "Eflat": -3, "Ab": -4, "Aflat": -4, "Db": -5, "Dflat": -5, "Gb": -6, "Gflat": -6, "Cb": -7, "Cflat": -7,
        "a": 0, "e": 1, "b": 2, "f#": 3, "fsharp": 3, "c#": 4, "csharp": 4, "g#": 5, "gsharp": 5, "d#": 6, "dsharp": 6, "a#": 7, "asharp": 7,
        "d": -1, "g": -2, "c": -3, "f": -4, "bb": -5, "bflat": -5, "eb": -6, "eflat": -6, "ab": -7, "aflat": -7,
    }
    key_lookup = clean.lower() if is_minor else clean.capitalize()
    fifths = KEY_FIFTHS_MAP.get(key_lookup, 0)
    mode = "minor" if is_minor else "major"
    return fifths, mode


def get_key_defaults(fifths: int) -> dict[str, int]:
    defaults = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0}
    sharps_order = ["F", "C", "G", "D", "A", "E", "B"]
    flats_order = ["B", "E", "A", "D", "G", "C", "F"]
    if fifths > 0:
        for step in sharps_order[:min(fifths, 7)]:
            defaults[step] = 1
    elif fifths < 0:
        for step in flats_order[:min(abs(fifths), 7)]:
            defaults[step] = -1
    return defaults


def _quarter_length_to_type_and_dots(dur_q: float) -> tuple[str, int]:
    if dur_q >= 4.0:
        return "whole", 0
    elif dur_q >= 3.0:
        return "half", 1
    elif dur_q >= 2.0:
        return "half", 0
    elif dur_q >= 1.5:
        return "quarter", 1
    elif dur_q >= 1.0:
        return "quarter", 0
    elif dur_q >= 0.75:
        return "eighth", 1
    elif dur_q >= 0.5:
        return "eighth", 0
    elif dur_q >= 0.375:
        return "16th", 1
    elif dur_q >= 0.25:
        return "16th", 0
    elif dur_q >= 0.1875:
        return "32nd", 1
    elif dur_q >= 0.125:
        return "32nd", 0
    elif dur_q >= 0.0625:
        return "64th", 0
    else:
        return "128th", 0


def _inject_tablature_technical(
    song: Song,
    xml_path: str,
    rendered_fretboard_path: list | None = None,
):
    """Injects publishing metadata and tablature information directly into generated MusicXML documents."""
    tree = ET.parse(xml_path)

    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # 1. Enrich <part-list> and <score-part id="P1">
    part_list = root.find(f"{ns}part-list")
    if part_list is not None:
        score_part = part_list.find(f"{ns}score-part")
        if score_part is not None:
            p_name = score_part.find(f"{ns}part-name")
            if p_name is None:
                p_name = ET.SubElement(score_part, f"{ns}part-name")
            p_name.text = "Electric Bass"

            p_abbr = score_part.find(f"{ns}part-abbreviation")
            if p_abbr is None:
                p_abbr = ET.SubElement(score_part, f"{ns}part-abbreviation")
            p_abbr.text = "B. Pass"

            score_inst = score_part.find(f"{ns}score-instrument")
            if score_inst is None:
                score_inst = ET.SubElement(score_part, f"{ns}score-instrument", {"id": "P1-I1"})
            inst_name = score_inst.find(f"{ns}instrument-name")
            if inst_name is None:
                inst_name = ET.SubElement(score_inst, f"{ns}instrument-name")
            inst_name.text = "Electric Bass"

            midi_dev = score_part.find(f"{ns}midi-device")
            if midi_dev is None:
                midi_dev = ET.SubElement(score_part, f"{ns}midi-device", {"id": "P1-I1", "port": "1"})

            midi_inst = score_part.find(f"{ns}midi-instrument")
            if midi_inst is None:
                midi_inst = ET.SubElement(score_part, f"{ns}midi-instrument", {"id": "P1-I1"})

            chan = midi_inst.find(f"{ns}midi-channel")
            if chan is None:
                chan = ET.SubElement(midi_inst, f"{ns}midi-channel")
            chan.text = "1"

            prog = midi_inst.find(f"{ns}midi-program")
            if prog is None:
                prog = ET.SubElement(midi_inst, f"{ns}midi-program")
            prog.text = "34"

            vol = midi_inst.find(f"{ns}volume")
            if vol is None:
                vol = ET.SubElement(midi_inst, f"{ns}volume")
            vol.text = "78.7402"

            pan = midi_inst.find(f"{ns}pan")
            if pan is None:
                pan = ET.SubElement(midi_inst, f"{ns}pan")
            pan.text = "0"

    tuning = BASS_TUNINGS.get(getattr(song, "tuning_type", DEFAULT_TUNING_TYPE), BASS_TUNINGS[DEFAULT_TUNING_TYPE])
    num_strings = len(tuning)
    open_strings = {num_strings - idx: string_midi for idx, (_, _, string_midi) in enumerate(tuning)}

    raw_key = getattr(song, "key_obj", None) or getattr(song, "parsed_key_str", None)
    fifths_val, mode_val = get_key_fifths_and_mode(raw_key)
    key_defaults = get_key_defaults(fifths_val)

    for part in root.findall(f"{ns}part"):
        measures = list(part.findall(f"{ns}measure"))
        if measures:
            m1 = measures[0]
            attr_elem = m1.find(f"{ns}attributes")
            if attr_elem is None:
                attr_elem = _insert_schema_compliant(m1, "attributes", MEASURE_SCHEMA_ORDER, ns)

            # Insert key signature
            key_elem = attr_elem.find(f"{ns}key")
            if key_elem is None:
                key_elem = _insert_schema_compliant(attr_elem, "key", ATTRIBUTES_SCHEMA_ORDER, ns)
            fifths_elem = key_elem.find(f"{ns}fifths")
            if fifths_elem is None:
                fifths_elem = ET.SubElement(key_elem, f"{ns}fifths")
            fifths_elem.text = str(fifths_val)
            mode_elem = key_elem.find(f"{ns}mode")
            if mode_elem is None:
                mode_elem = ET.SubElement(key_elem, f"{ns}mode")
            mode_elem.text = mode_val

            # Insert clef (F4 bass clef)
            clef_elem = attr_elem.find(f"{ns}clef")
            if clef_elem is None:
                clef_elem = _insert_schema_compliant(attr_elem, "clef", ATTRIBUTES_SCHEMA_ORDER, ns)
            sign_elem = clef_elem.find(f"{ns}sign")
            if sign_elem is None:
                sign_elem = ET.SubElement(clef_elem, f"{ns}sign")
            sign_elem.text = "F"
            line_elem = clef_elem.find(f"{ns}line")
            if line_elem is None:
                line_elem = ET.SubElement(clef_elem, f"{ns}line")
            line_elem.text = "4"

            # Insert staff-details (strings + tuning)
            staff_det = attr_elem.find(f"{ns}staff-details")
            if staff_det is None:
                staff_det = _insert_schema_compliant(attr_elem, "staff-details", ATTRIBUTES_SCHEMA_ORDER, ns)
            slines = staff_det.find(f"{ns}staff-lines")
            if slines is None:
                slines = ET.SubElement(staff_det, f"{ns}staff-lines")
            slines.text = str(num_strings)

            existing_tunings = staff_det.findall(f"{ns}staff-tuning")
            if not existing_tunings:
                for idx, (t_step, t_oct, _midi) in enumerate(tuning, start=1):
                    st = ET.SubElement(staff_det, f"{ns}staff-tuning", {"line": str(idx)})
                    ts = ET.SubElement(st, f"{ns}tuning-step")
                    ts.text = str(t_step)
                    to = ET.SubElement(st, f"{ns}tuning-octave")
                    to.text = str(t_oct)

            # Insert transpose tag (octave-change -1 for electric/double bass)
            trans_elem = attr_elem.find(f"{ns}transpose")
            if trans_elem is None:
                trans_elem = _insert_schema_compliant(attr_elem, "transpose", ATTRIBUTES_SCHEMA_ORDER, ns)

            d_elem = trans_elem.find(f"{ns}diatonic")
            if d_elem is None:
                d_elem = ET.SubElement(trans_elem, f"{ns}diatonic")
            d_elem.text = "0"

            c_elem = trans_elem.find(f"{ns}chromatic")
            if c_elem is None:
                c_elem = ET.SubElement(trans_elem, f"{ns}chromatic")
            c_elem.text = "0"

            o_elem = trans_elem.find(f"{ns}octave-change")
            if o_elem is None:
                o_elem = ET.SubElement(trans_elem, f"{ns}octave-change")
            o_elem.text = "-1"

        pitch_note_counter = 0
        for measure in measures:
            measure_alters = key_defaults.copy()
            notes_in_m = list(measure.findall(f"{ns}note"))
            for note_elem in notes_in_m:
                _set_or_create_ordered(note_elem, "voice", "1", NOTE_SCHEMA_ORDER, ns)
                _set_or_create_ordered(note_elem, "staff", "1", NOTE_SCHEMA_ORDER, ns)

                # Fix <type> and <dot> based on <duration> / divisions (divisions = 32)
                dur_elem = note_elem.find(f"{ns}duration")
                if dur_elem is not None and dur_elem.text:
                    try:
                        dur_units = float(dur_elem.text)
                        dur_q = dur_units / 32.0
                        note_type_str, num_dots = _quarter_length_to_type_and_dots(dur_q)
                        _set_or_create_ordered(note_elem, "type", note_type_str, NOTE_SCHEMA_ORDER, ns)

                        dot_elems = note_elem.findall(f"{ns}dot")
                        if num_dots > len(dot_elems):
                            for _ in range(num_dots - len(dot_elems)):
                                _insert_schema_compliant(note_elem, "dot", NOTE_SCHEMA_ORDER, ns)
                        elif num_dots == 0 and dot_elems:
                            for de in dot_elems:
                                note_elem.remove(de)
                    except (ValueError, TypeError):
                        pass

                # Ensure <tied> tags inside <notations> match <tie> elements
                tie_elems = note_elem.findall(f"{ns}tie")
                if tie_elems:
                    notations = note_elem.find(f"{ns}notations")
                    if notations is None:
                        notations = _insert_schema_compliant(note_elem, "notations", NOTE_SCHEMA_ORDER, ns)
                    for te in tie_elems:
                        t_type = te.get("type")
                        if t_type:
                            existing_tied = [td for td in notations.findall(f"{ns}tied") if td.get("type") == t_type]
                            if not existing_tied:
                                new_tied = _insert_schema_compliant(notations, "tied", NOTATIONS_SCHEMA_ORDER, ns)
                                new_tied.set("type", t_type)

                pitch_elem = note_elem.find(f"{ns}pitch")
                if pitch_elem is None:
                    continue

                step_elem = pitch_elem.find(f"{ns}step")
                oct_elem = pitch_elem.find(f"{ns}octave")
                alt_elem = pitch_elem.find(f"{ns}alter")
                if step_elem is None or oct_elem is None or oct_elem.text is None:
                    continue

                try:
                    step_text = step_elem.text
                    alter = int(alt_elem.text) if alt_elem is not None and alt_elem.text else 0
                    midi_val = _pitch_to_midi(
                        step_text,
                        int(oct_elem.text),
                        alter,
                    )

                    position = (
                        rendered_fretboard_path[pitch_note_counter]
                        if rendered_fretboard_path and pitch_note_counter < len(rendered_fretboard_path)
                        else None
                    )
                    string_num = None
                    fret_num = None

                    if position is not None:
                        if isinstance(position, (tuple, list)) and len(position) >= 2:
                            string_num, fret_num = position[0], position[1]
                        elif isinstance(position, dict):
                            string_num = position.get("string", position.get("string_num"))
                            fret_num = position.get("fret", position.get("fret_num"))
                        elif hasattr(position, "string") and hasattr(position, "fret"):
                            string_num = position.string
                            fret_num = position.fret
                        elif hasattr(position, "string_num") and hasattr(position, "fret_num"):
                            string_num = position.string_num
                            fret_num = position.fret_num

                    if string_num is None or fret_num is None:
                        string_num, fret_num = _calculate_correct_string_fret(
                            midi_val, getattr(song, "tuning_type", DEFAULT_TUNING_TYPE)
                        )

                    string_num = max(1, min(num_strings, int(string_num)))
                    fret_num = max(0, min(MAX_FRETBOARD_FRETS, int(fret_num)))

                    # Ensure <pitch> matches <string>/<fret> tab position exactly
                    string_open_midi = open_strings.get(string_num, 28)
                    expected_midi = string_open_midi + fret_num
                    if expected_midi != midi_val:
                        e_step, e_octave, e_alter = _midi_to_pitch_components(expected_midi)
                        _set_or_create_ordered(pitch_elem, "step", e_step, PITCH_SCHEMA_ORDER, ns)
                        _set_or_create_ordered(pitch_elem, "octave", e_octave, PITCH_SCHEMA_ORDER, ns)
                        if e_alter != 0:
                            _set_or_create_ordered(pitch_elem, "alter", e_alter, PITCH_SCHEMA_ORDER, ns)
                        else:
                            e_alt_elem = pitch_elem.find(f"{ns}alter")
                            if e_alt_elem is not None:
                                pitch_elem.remove(e_alt_elem)
                        step_text = e_step
                        alter = e_alter

                    # Accidental evaluation after final pitch step/alter is guaranteed
                    acc_elem = note_elem.find(f"{ns}accidental")
                    if alter != measure_alters.get(step_text, 0):
                        accidental_names = {1: "sharp", -1: "flat", 0: "natural", 2: "double-sharp", -2: "flat-flat"}
                        acc_text = accidental_names.get(alter, "natural")
                        _set_or_create_ordered(
                            note_elem,
                            "accidental",
                            acc_text,
                            NOTE_SCHEMA_ORDER,
                            ns,
                        )
                        measure_alters[step_text] = alter
                    else:
                        if acc_elem is not None:
                            note_elem.remove(acc_elem)

                    notations = note_elem.find(f"{ns}notations")
                    if notations is None:
                        notations = _insert_schema_compliant(note_elem, "notations", NOTE_SCHEMA_ORDER, ns)
                    technical = notations.find(f"{ns}technical")
                    if technical is None:
                        technical = _insert_schema_compliant(notations, "technical", NOTATIONS_SCHEMA_ORDER, ns)
                    _set_or_create_ordered(technical, "string", string_num, TECHNICAL_SCHEMA_ORDER, ns)
                    _set_or_create_ordered(technical, "fret", fret_num, TECHNICAL_SCHEMA_ORDER, ns)

                    # Inject idiomatic bass articulations if available
                    if pitch_note_counter < len(song.notes):
                        curr_note = song.notes[pitch_note_counter]
                        if getattr(curr_note, "is_slide", False) or getattr(curr_note, "tag", "") == "slide":
                            _set_or_create_ordered(technical, "slide", "", TECHNICAL_SCHEMA_ORDER, ns)
                        if getattr(curr_note, "is_legato", False) or getattr(curr_note, "tag", "") in [
                            "hammer_on",
                            "pull_off",
                        ]:
                            _set_or_create_ordered(technical, "hammer-on", "", TECHNICAL_SCHEMA_ORDER, ns)
                        if getattr(curr_note, "tag", "") == "palm_mute":
                            _set_or_create_ordered(technical, "other-technical", "P.M.", TECHNICAL_SCHEMA_ORDER, ns)
                        elif getattr(curr_note, "tag", "") == "slap" or getattr(curr_note, "is_slap", False):
                            _set_or_create_ordered(technical, "other-technical", "S", TECHNICAL_SCHEMA_ORDER, ns)
                        elif getattr(curr_note, "tag", "") == "pop" or getattr(curr_note, "is_pop", False):
                            _set_or_create_ordered(technical, "other-technical", "P", TECHNICAL_SCHEMA_ORDER, ns)

                    pitch_note_counter += 1
                except (ValueError, TypeError, IndexError) as err:
                    raise ValueError(f"Could not assign tablature to MusicXML pitch: {err}") from err

    temp_path = xml_path + ".tmp"
    tree.write(temp_path, encoding="utf-8", xml_declaration=True)
    os.replace(temp_path, xml_path)


# --- Rhythm Engine Operations ---


def idiomatic_rhythm_snap(dur_q, level=5, is_compound=False):
    """Snaps fractional quarter note duration to idiomatic musical grid based on level/compound mode."""
    if isinstance(dur_q, (float, int)):
        dur_q = round(float(dur_q), 4)

    if is_compound:
        max_denom = 24
        units = max(1, round(float(dur_q) * max_denom))
        dur_q = fractions.Fraction(units, max_denom)
    else:
        max_denom = 16 if level <= 2 else 32
        units = max(1, round(float(dur_q) * max_denom))
        dur_q = fractions.Fraction(units, max_denom)

    MIN_DURATION = fractions.Fraction(1, 16) if level <= 3 else fractions.Fraction(1, 32)
    if dur_q < MIN_DURATION:
        return MIN_DURATION
    return dur_q


def decompose_duration_engraver_rules(dur_q, curr_m_fill, measure_capacity, is_compound=False):
    """
    Decomposes a long note or rest duration across beat/measure boundaries into standard
    note values tied together according to standard engraver rules.
    """
    dur_q = fractions.Fraction(dur_q)
    curr_m_fill = fractions.Fraction(curr_m_fill)
    measure_capacity = fractions.Fraction(measure_capacity)

    MIN_CHUNK = fractions.Fraction(1, 32)
    if 0 < dur_q < MIN_CHUNK:
        dur_q = MIN_CHUNK
    else:
        if is_compound:
            units = round(float(dur_q) * 24)
            dur_q = fractions.Fraction(units, 24)
        else:
            units = round(float(dur_q) * 32)
            dur_q = fractions.Fraction(units, 32)

    if is_compound:
        allowed_values = [
            fractions.Fraction(3, 1),
            fractions.Fraction(3, 2),
            fractions.Fraction(1, 1),
            fractions.Fraction(3, 4),
            fractions.Fraction(2, 3),
            fractions.Fraction(1, 2),
            fractions.Fraction(3, 8),
            fractions.Fraction(1, 3),
            fractions.Fraction(1, 4),
            fractions.Fraction(1, 6),
            fractions.Fraction(1, 8),
            fractions.Fraction(1, 12),
            fractions.Fraction(1, 16),
            fractions.Fraction(1, 24),
        ]
    else:
        allowed_values = [
            fractions.Fraction(4, 1),
            fractions.Fraction(3, 1),
            fractions.Fraction(2, 1),
            fractions.Fraction(3, 2),
            fractions.Fraction(1, 1),
            fractions.Fraction(3, 4),
            fractions.Fraction(1, 2),
            fractions.Fraction(3, 8),
            fractions.Fraction(1, 4),
            fractions.Fraction(3, 16),
            fractions.Fraction(1, 8),
            fractions.Fraction(1, 16),
            fractions.Fraction(1, 32),
        ]

    chunks = []
    max_iterations = 128
    iterations = 0

    while dur_q > 0 and iterations < max_iterations:
        iterations += 1

        rem_in_m = measure_capacity - curr_m_fill
        if rem_in_m <= 0 or rem_in_m < MIN_CHUNK:
            curr_m_fill = fractions.Fraction(0, 1)
            rem_in_m = measure_capacity

        if 0 < dur_q < MIN_CHUNK:
            dur_q = MIN_CHUNK

        cap_boundary = rem_in_m
        if measure_capacity == fractions.Fraction(4, 1):  # Standard 4/4 meter
            half_bar = fractions.Fraction(2, 1)
            if curr_m_fill < half_bar and (curr_m_fill + dur_q) > half_bar:
                if curr_m_fill > 0 or dur_q < fractions.Fraction(3, 1):
                    cap_boundary = min(rem_in_m, half_bar - curr_m_fill)
        elif is_compound or measure_capacity in (
            fractions.Fraction(3, 2),
            fractions.Fraction(3, 1),
        ):  # 6/8, 12/8
            beat_step = fractions.Fraction(3, 2)  # Dotted quarter
            next_beat = ((curr_m_fill // beat_step) + 1) * beat_step
            if curr_m_fill < next_beat and (curr_m_fill + dur_q) > next_beat:
                if curr_m_fill % beat_step != 0 or dur_q < beat_step:
                    cap_boundary = min(rem_in_m, next_beat - curr_m_fill)

        best_val = None

        # 1. Try allowed values within metric boundary
        if cap_boundary > 0:
            for val in allowed_values:
                if val <= dur_q and val <= cap_boundary:
                    best_val = val
                    break

        # 2. Try allowed values within remaining measure space
        if best_val is None:
            for val in allowed_values:
                if val <= dur_q and val <= rem_in_m:
                    best_val = val
                    break

        # 3. Robust Fallback: Guarantee progress by forcing a valid positive chunk
        if best_val is None or best_val <= 0:
            candidate = min(dur_q, rem_in_m)
            if candidate <= 0 or candidate < MIN_CHUNK:
                curr_m_fill = fractions.Fraction(0, 1)
                rem_in_m = measure_capacity
                candidate = min(dur_q, rem_in_m)
            if candidate <= 0:
                candidate = MIN_CHUNK
            best_val = candidate

        chunks.append(best_val)
        dur_q -= best_val
        curr_m_fill += best_val
        if curr_m_fill >= measure_capacity:
            curr_m_fill = fractions.Fraction(0, 1)

    return [c for c in chunks if c > 0]


def consolidate_measure_notation(curr_measure, measure_capacity, is_compound=False):
    """Ensures every note and rest element inside curr_measure has explicit duration metadata set."""
    if curr_measure is None:
        return

    for elem in list(curr_measure.notesAndRests):
        if not hasattr(elem, "duration") or elem.duration is None:
            elem.duration = build_m21_duration(1.0)
        elif not elem.duration.type or elem.duration.type == "inexpressible":
            elem.duration = build_m21_duration(elem.duration.quarterLength)


def decompose_note_to_atoms(
    note_obj: Note | None,
    q_dur: fractions.Fraction,
    curr_m_fill: fractions.Fraction,
    measure_capacity: fractions.Fraction,
    measure_index: int = 1,
    is_compound: bool = False,
    level: int = 5,
) -> list[RhythmicAtom]:
    """
    Decomposes a thick Note or Rest event into fine-grained RhythmicAtom instances
    adhering to engraver rules, measure boundaries, minimum duration guards, and tie linking.
    Preserves parent event ID and assigned (string_num, fret_num) tablature positions.
    """
    MIN_ATOM = fractions.Fraction(1, 24) if is_compound else fractions.Fraction(1, 32)

    if note_obj is None or getattr(note_obj, "is_rest", False):
        chunks = decompose_duration_engraver_rules(q_dur, curr_m_fill, measure_capacity, is_compound)
        atoms = []
        for chunk_dur in chunks:
            if chunk_dur < MIN_ATOM:
                continue
            atom = RhythmicAtom.from_note(
                note_obj=note_obj,
                duration_q=chunk_dur,
                measure_index=measure_index,
                is_rest=True,
            )
            atoms.append(atom)
        return atoms

    q_dur = max(MIN_ATOM, idiomatic_rhythm_snap(q_dur, level=level, is_compound=is_compound))
    rem_in_m = measure_capacity - curr_m_fill

    if q_dur <= rem_in_m:
        sub_chunks = decompose_duration_engraver_rules(q_dur, curr_m_fill, measure_capacity, is_compound)
        atoms = []
        for sc in sub_chunks:
            if sc < MIN_ATOM:
                continue
            tie_t = (
                "start"
                if getattr(note_obj, "is_tied_start", False)
                else ("stop" if getattr(note_obj, "is_tied_stop", False) else None)
            )
            atom = RhythmicAtom.from_note(
                note_obj=note_obj,
                duration_q=sc,
                measure_index=measure_index,
                is_rest=False,
                tie_type=tie_t,
            )
            atoms.append(atom)
        return atoms

    # Cross-measure boundary splitting with ties
    atoms = []
    remaining_note_dur = q_dur
    is_first = True
    curr_fill = curr_m_fill
    curr_m_idx = measure_index
    max_loops = 1000
    loop_cnt = 0

    while remaining_note_dur > 0 and loop_cnt < max_loops:
        loop_cnt += 1
        rem_m = measure_capacity - curr_fill
        if rem_m <= 0 or rem_m < MIN_ATOM:
            curr_m_idx += 1
            curr_fill = fractions.Fraction(0, 1)
            rem_m = measure_capacity

        chunk_q = min(remaining_note_dur, rem_m)
        sub_chunks = decompose_duration_engraver_rules(chunk_q, curr_fill, measure_capacity, is_compound)

        if not sub_chunks:
            if chunk_q < MIN_ATOM:
                break
            sub_chunks = [chunk_q]

        consumed = fractions.Fraction(0, 1)
        for sc in sub_chunks:
            if sc < MIN_ATOM:
                continue
            rem_after = remaining_note_dur - sc
            tie_t = "start" if is_first else ("stop" if rem_after <= MIN_ATOM else "continue")

            atom = RhythmicAtom.from_note(
                note_obj=note_obj,
                duration_q=sc,
                measure_index=curr_m_idx,
                is_rest=False,
                tie_type=tie_t,
            )
            atoms.append(atom)
            curr_fill += sc
            consumed += sc
            is_first = False

            if curr_fill >= measure_capacity:
                curr_m_idx += 1
                curr_fill = fractions.Fraction(0, 1)

        if consumed <= 0:
            break
        remaining_note_dur -= consumed

    return atoms


# --- Core Engraver Logic ---


def stream_quantized_events(
    note_layer: list[Note],
    bpm: float,
    measure_capacity: fractions.Fraction,
    is_compound: bool = False,
    level: int = 5,
):
    """
    Unified stream generator that quantizes events, calculates gap rests, splits notes
    across measure boundaries with ties, and applies metric engraver rules using RhythmicAtoms.
    """
    quarter_sec = 60.0 / bpm if bpm > 0 else 0.5
    sorted_layer = sorted([e for e in note_layer if getattr(e, "duration", 0) > 0], key=lambda x: x.start)

    curr_m_num = 1
    curr_m_fill = fractions.Fraction(0, 1)

    def _yield_rests(target_dur):
        nonlocal curr_m_num, curr_m_fill
        target_dur = fractions.Fraction(target_dur).limit_denominator(32)
        max_loops = 1000
        loop_cnt = 0

        while target_dur > 0 and loop_cnt < max_loops:
            loop_cnt += 1
            rem_in_m = measure_capacity - curr_m_fill
            if rem_in_m <= 0:
                curr_m_num += 1
                curr_m_fill = fractions.Fraction(0, 1)
                rem_in_m = measure_capacity

            chunk_q = min(target_dur, rem_in_m)
            sub_chunks = decompose_duration_engraver_rules(chunk_q, curr_m_fill, measure_capacity, is_compound)

            if not sub_chunks:
                if chunk_q < fractions.Fraction(1, 32):
                    break
                sub_chunks = [chunk_q]

            consumed = fractions.Fraction(0, 1)
            for sc in sub_chunks:
                if sc <= 0:
                    continue
                yield MeasureChunk(curr_m_num, None, sc, is_rest=True)
                curr_m_fill += sc
                consumed += sc
                if curr_m_fill >= measure_capacity:
                    curr_m_num += 1
                    curr_m_fill = fractions.Fraction(0, 1)

            if consumed <= 0:
                consumed = chunk_q
                if consumed > 0:
                    yield MeasureChunk(curr_m_num, None, consumed, is_rest=True)
                    curr_m_fill += consumed
                    if curr_m_fill >= measure_capacity:
                        curr_m_num += 1
                        curr_m_fill = fractions.Fraction(0, 1)
                else:
                    break

            target_dur -= consumed

    for evt in sorted_layer:
        if is_compound:
            target_units = round(float(evt.start / quarter_sec) * 24)
            target_q = fractions.Fraction(target_units, 24)
        else:
            target_units = round(float(evt.start / quarter_sec) * 32)
            target_q = fractions.Fraction(target_units, 32)

        if target_q < 0:
            target_q = fractions.Fraction(0, 1)

        current_total_q = (curr_m_num - 1) * measure_capacity + curr_m_fill

        if target_q > current_total_q:
            yield from _yield_rests(target_q - current_total_q)

        q_dur = fractions.Fraction(evt.duration / quarter_sec)

        if getattr(evt, "is_rest", False):
            if q_dur > 0:
                yield from _yield_rests(q_dur)
        else:
            atoms = decompose_note_to_atoms(
                note_obj=evt,
                q_dur=q_dur,
                curr_m_fill=curr_m_fill,
                measure_capacity=measure_capacity,
                measure_index=curr_m_num,
                is_compound=is_compound,
                level=level,
            )
            for atom in atoms:
                chunk = MeasureChunk(
                    measure_index=atom.measure_index,
                    event_or_start=evt,
                    duration_q_or_end=atom.duration_q,
                    is_rest=atom.is_rest,
                    tie_type=atom.tie_type,
                )
                chunk.atoms = [atom]
                yield chunk
                curr_m_fill += atom.duration_q
                while curr_m_fill >= measure_capacity:
                    curr_m_num += 1
                    curr_m_fill -= measure_capacity

    if 0 < curr_m_fill < measure_capacity:
        yield from _yield_rests(measure_capacity - curr_m_fill)


def _amplitude_to_dynamic(amplitude: float):
    """Maps a 0-1 note amplitude/velocity to a conventional dynamic marking, or None."""
    if amplitude is None:
        return None
    if amplitude < 0.2:
        return "pp"
    if amplitude < 0.35:
        return "p"
    if amplitude < 0.5:
        return "mp"
    if amplitude < 0.65:
        return "mf"
    if amplitude < 0.8:
        return "f"
    return "ff"


def _make_note_or_rest(evt, q_dur, detected_key):
    """Instantiates a music21 Note or Rest with articulations."""
    if evt is None or getattr(evt, "is_rest", False):
        r = note.Rest()
        r.duration = build_m21_duration(q_dur)
        return r

    p_str = get_directional_enharmonic_pitch(evt.pitch, detected_key)
    m21_note = note.Note(p_str)
    m21_note.duration = build_m21_duration(q_dur)

    if getattr(evt, "is_accent", False):
        m21_note.articulations.append(articulations.Accent())
    if getattr(evt, "is_staccato", False) or getattr(evt, "tag", "") == "staccato":
        m21_note.articulations.append(articulations.Staccato())
    if getattr(evt, "is_tenuto", False):
        m21_note.articulations.append(articulations.Tenuto())

    if getattr(evt, "is_fermata", False):
        m21_note.expressions.append(expressions.Fermata())
    if getattr(evt, "is_slap", False) or getattr(evt, "tag", "") == "slap":
        m21_note.expressions.append(expressions.TextExpression("S"))
    elif getattr(evt, "is_pop", False) or getattr(evt, "tag", "") == "pop":
        m21_note.expressions.append(expressions.TextExpression("P"))
    elif getattr(evt, "is_palm_mute", False) or getattr(evt, "tag", "") == "palm_mute":
        m21_note.expressions.append(expressions.TextExpression("P.M."))

    if getattr(evt, "is_ghost", False) or getattr(evt, "tag", "") == "ghost":
        m21_note.notehead = "x"
        m21_note.noteheadParenthesis = True
        m21_note.articulations.append(articulations.Staccato())
    elif getattr(evt, "is_harmonic", False) or getattr(evt, "tag", "") == "harmonic":
        m21_note.notehead = "diamond"

    return m21_note


def build_and_export_song(song: Song, output_xml_path: str = None, **kwargs):
    """Export a Song without unpacking its state at every service boundary."""
    destination = output_xml_path or getattr(song, "output_xml_path", None)
    if not destination:
        raise ValueError("Song.output_xml_path must be set before exporting")

    target_level = getattr(song, "target_level", 5)

    if kwargs.pop("filter_level", True):
        filter_song_for_level(song, target_level)

    return _build_score(
        note_layer=song.notes,
        fretboard_path=getattr(song, "fretboard_path", None),
        detected_key=song.key_obj,
        song_title=song.song_title,
        artist_name=song.artist_name,
        bpm=song.bpm,
        is_compound=song.is_compound,
        target_level=target_level,
        output_xml_path=destination,
        time_sig_str=song.time_sig,
        tuning_type=song.tuning_type,
        enable_dual_stave=kwargs.pop("enable_dual_stave", True),
        song=song,
        **kwargs,
    )


def _build_score(
    note_layer: list[Note],
    fretboard_path: list,
    detected_key,
    song_title: str,
    artist_name: str,
    bpm: float,
    is_compound: bool,
    target_level: int,
    output_xml_path: str,
    time_sig_str: str = DEFAULT_TIME_SIGNATURE,
    tuning_type: str = DEFAULT_TUNING_TYPE,
    enable_dual_stave: bool = True,
    song: Song = None,
    **kwargs,
):
    os.makedirs(os.path.dirname(os.path.abspath(output_xml_path)), exist_ok=True)

    if song is None:
        song = Song(
            bass_notes=note_layer,
            bpm=bpm,
            time_sig=time_sig_str,
            tuning_type=tuning_type,
            song_title=song_title,
            artist_name=artist_name,
            target_level=target_level,
        )

    m21_score = stream.Score()
    m21_score.insert(
        0,
        metadata.Metadata(title=song_title or "Bass Transcription", composer=artist_name or "Transcribed Score"),
    )

    m21_part = stream.Part(id="P1")
    m21_part.partName = "Electric Bass"
    bass_instrument = instrument.ElectricBass()
    tuning = BASS_STRING_TUNINGS.get(tuning_type, BASS_STRING_TUNINGS[DEFAULT_TUNING_TYPE])
    bass_instrument.stringPitches = [f"{step}{octave}" for step, octave, _midi in tuning]
    m21_part.append(bass_instrument)

    ts = meter.TimeSignature(time_sig_str)
    measure_capacity = fractions.Fraction(ts.numerator * (4.0 / ts.denominator)).limit_denominator(64)

    measures_map = {}
    event_nodes = {}  # id(evt) -> {'evt': Note, 'first': Note, 'last': Note}
    fret_path_list = fretboard_path or []
    if not fret_path_list and note_layer:
        solver = ErgonomicFretboardHMMSolver(tuning_type=tuning_type, song=song)
        if song:
            solver.solve_song(song)
            fret_path_list = song.fretboard_path
        else:
            fret_path_list, _, _, _ = solver._solve_notes(note_layer, bpm=bpm)

    positions_by_note_id = {id(note_event): position for note_event, position in zip(note_layer, fret_path_list)}
    emitted_fretboard_path = []
    last_dynamic = None

    for chunk in stream_quantized_events(note_layer, bpm, measure_capacity, is_compound, level=target_level):
        if chunk.duration_q <= 0:
            continue

        m_num = chunk.measure_num
        if m_num not in measures_map:
            m = stream.Measure(number=m_num)
            m.insert(0, meter.TimeSignature(time_sig_str))
            if m_num == 1:
                if detected_key:
                    key_obj = parse_key_object(detected_key)
                    m.append(key_obj)
                if bpm and bpm > 0:
                    m.append(tempo.MetronomeMark(number=round(bpm)))
            measures_map[m_num] = m

        m = measures_map[m_num]

        if chunk.is_rest:
            m.append(_make_note_or_rest(None, chunk.duration_q, detected_key))
        else:
            is_new_onset = chunk.tie_type in (None, "start")
            if is_new_onset:
                new_dynamic = _amplitude_to_dynamic(getattr(chunk.event, "amplitude", None))
                if new_dynamic and new_dynamic != last_dynamic:
                    m.append(make_dynamic(new_dynamic))
                    last_dynamic = new_dynamic

            m21_n = _make_note_or_rest(chunk.event, chunk.duration_q, detected_key)
            if chunk.tie_type:
                m21_n.tie = tie.Tie(chunk.tie_type)
            m.append(m21_n)

            evt_id = id(chunk.event)
            if evt_id not in event_nodes:
                event_nodes[evt_id] = {"evt": chunk.event, "first": m21_n, "last": m21_n}
            else:
                event_nodes[evt_id]["last"] = m21_n

            emitted_fretboard_path.append(
                getattr(chunk.event, "fret_position", None) or positions_by_note_id.get(evt_id)
            )

    for m_num in sorted(measures_map.keys()):
        m = measures_map[m_num]
        consolidate_measure_notation(m, measure_capacity, is_compound)
        m21_part.append(m)

    if not measures_map:
        empty_measure = stream.Measure(number=1)
        empty_measure.insert(0, meter.TimeSignature(time_sig_str))
        empty_measure.append(_make_note_or_rest(None, measure_capacity, None))
        m21_part.append(empty_measure)

    m21_score.append(m21_part)

    node_list = list(event_nodes.values())
    if len(node_list) > 1:
        for i in range(1, len(node_list)):
            prev_info, curr_info = node_list[i - 1], node_list[i]
            _, prev_last = prev_info["evt"], prev_info["last"]
            curr_evt, curr_first = curr_info["evt"], curr_info["first"]

            if getattr(curr_evt, "is_legato", False):
                m21_score.insert(0, spanner.Slur(prev_last, curr_first))
            if getattr(curr_evt, "is_slide", False):
                m21_score.insert(0, spanner.Glissando(prev_last, curr_first))

    for m in m21_part.getElementsByClass(stream.Measure):
        consolidate_measure_notation(m, measure_capacity, is_compound)
        zero_dur_elements = [el for el in m.notesAndRests if el.duration.quarterLength <= 0]
        for el in zero_dur_elements:
            m.remove(el)
        for el in m.notesAndRests:
            if not el.duration.type or el.duration.type == "inexpressible":
                el.duration = build_m21_duration(el.duration.quarterLength)

    m21_score.write("musicxml", fp=output_xml_path, makeNotation=False)

    if song is None:
        raise RuntimeError("Internal score builder requires a Song state")
    _inject_tablature_technical(song, output_xml_path, emitted_fretboard_path)