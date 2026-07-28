import fractions
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections.abc import Mapping

try:
    import numpy as np
except ImportError:
    np = None

# Ensure src directory is in Python path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from subtone.settings_loader import BASS_STRING_TUNINGS as BASS_TUNINGS
from subtone.fretboard import ErgonomicFretboardHMMSolver
from subtone.schemas import AudioEvent, Genre, MeasureChunk, Note, RhythmicAtom, Song
from subtone.dsp import apply_bass_bandpass
from subtone.pitch_theory import snap_song_to_scale, parse_key_object
from subtone.engraver import build_and_export_song, decompose_note_to_atoms, filter_song_for_level, stream_quantized_events
from subtone.settings_loader import (
    load_genre_configs,
    load_midi_folder_to_event_streams,
    parse_metadata_from_path,
    resolve_artist,
    resolve_genre,
    tomllib,
)
try:
    from music21 import key
except ImportError:
    key = None

# Schema definition constants
NOTE_SCHEMA_ORDER = [
    "grace",
    "cue",
    "chord",
    "pitch",
    "rest",
    "unpitched",
    "duration",
    "tie",
    "instrument",
    "footnote",
    "level",
    "voice",
    "type",
    "dot",
    "accidental",
    "time-modification",
    "stem",
    "notehead",
    "notehead-text",
    "staff",
    "beam",
    "notations",
    "lyric",
]

MEASURE_SCHEMA_ORDER = [
    "print",
    "attributes",
    "harmony",
    "first-beat",
    "direction",
    "note",
    "forward",
    "backup",
    "figured-bass",
    "sound",
    "listening",
    "barline",
]


def pitch_to_midi(step, octave, alter=0):
    offsets = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    return (octave + 1) * 12 + offsets[step.upper()] + alter


def export_score(**kwargs):
    song = Song.from_transcription(
        source_events=kwargs.pop("note_layer"),
        beat_times=[],
        bpm=kwargs.pop("bpm"),
        time_sig=kwargs.pop("time_sig_str", "4/4"),
        is_compound=kwargs.pop("is_compound", False),
        key_obj=kwargs.pop("detected_key", None),
        artist_name=kwargs.pop("artist_name", ""),
        song_title=kwargs.pop("song_title", ""),
        tuning_type=kwargs.pop("tuning_type", "4_string_standard"),
        target_level=kwargs.pop("target_level", 5),
    )
    song.fretboard_path = kwargs.pop("fretboard_path", [])
    song.output_xml_path = kwargs.pop("output_xml_path")
    build_and_export_song(song)


class TestMusicXmlOutput(unittest.TestCase):
    """
    Robust test suite validating MusicXML generation for VexFlow, OpenSheetMusicDisplay (OSMD),
    SoundSlice, and AlphaTab score engine compatibility.
    """

    def setUp(self):
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testoutput"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.xml_path = os.path.join(self.output_dir, "test_output.xml")

    def tearDown(self):
        pass

    def parse_music_xml(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"
        return root, ns

    def verify_measure_durations_sum(self, root, ns):
        parts = root.findall(f"{ns}part")
        self.assertGreater(len(parts), 0, "MusicXML contains no <part> elements")

        for part in parts:
            measures = part.findall(f"{ns}measure")
            self.assertGreater(len(measures), 0, "Part contains no <measure> elements")

            current_beats = 4
            current_beat_type = 4
            current_divisions = 8

            for measure in measures:
                attrs = measure.find(f"{ns}attributes")
                if attrs is not None:
                    time_elem = attrs.find(f"{ns}time")
                    if time_elem is not None:
                        beats_e = time_elem.find(f"{ns}beats")
                        beat_type_e = time_elem.find(f"{ns}beat-type")
                        if beats_e is not None and beats_e.text:
                            current_beats = int(beats_e.text)
                        if beat_type_e is not None and beat_type_e.text:
                            current_beat_type = int(beat_type_e.text)

                    divs_elem = attrs.find(f"{ns}divisions")
                    if divs_elem is not None and divs_elem.text:
                        current_divisions = int(divs_elem.text)

                expected_capacity = int(current_beats * (4.0 / current_beat_type) * current_divisions)

                notes = measure.findall(f"{ns}note")
                self.assertGreater(len(notes), 0, f"Measure {measure.get('number')} is completely empty")

                total_duration = 0
                for note in notes:
                    dur_elem = note.find(f"{ns}duration")
                    self.assertIsNotNone(dur_elem, f"Note in measure {measure.get('number')} missing <duration>")
                    total_duration += int(dur_elem.text)

                self.assertEqual(
                    total_duration,
                    expected_capacity,
                    f"Measure {measure.get('number')} duration sum ({total_duration}) does not match capacity ({expected_capacity})",
                )

    def verify_schema_child_ordering(self, root, ns):
        for part in root.findall(f"{ns}part"):
            for measure in part.findall(f"{ns}measure"):
                measure_children = [c.tag.replace(ns, "") for c in measure]
                known_m = [t for t in measure_children if t in MEASURE_SCHEMA_ORDER]
                sorted_m = sorted(known_m, key=lambda t: MEASURE_SCHEMA_ORDER.index(t))
                self.assertEqual(
                    known_m, sorted_m, f"Measure {measure.get('number')} schema ordering violation: {measure_children}"
                )

                for note in measure.findall(f"{ns}note"):
                    note_children = [c.tag.replace(ns, "") for c in note]
                    known_n = [t for t in note_children if t in NOTE_SCHEMA_ORDER]
                    sorted_n = sorted(known_n, key=lambda t: NOTE_SCHEMA_ORDER.index(t))
                    self.assertEqual(
                        known_n,
                        sorted_n,
                        f"Note schema ordering violation in measure {measure.get('number')}: {note_children}",
                    )

    def verify_voice_and_staff_assignments(self, root, ns):
        for part in root.findall(f"{ns}part"):
            for measure in part.findall(f"{ns}measure"):
                for note in measure.findall(f"{ns}note"):
                    voice = note.find(f"{ns}voice")
                    self.assertIsNotNone(voice, "Note missing <voice> tag")
                    self.assertTrue(len(voice.text) > 0, "<voice> tag is empty")

                    staff = note.find(f"{ns}staff")
                    self.assertIsNotNone(staff, "Note missing <staff> tag")
                    self.assertTrue(len(staff.text) > 0, "<staff> tag is empty")

    def verify_dotted_notes_have_dot_tag(self, root, ns):
        for part in root.findall(f"{ns}part"):
            for measure in part.findall(f"{ns}measure"):
                divs = 8
                attrs = measure.find(f"{ns}attributes")
                if attrs is not None:
                    d_el = attrs.find(f"{ns}divisions")
                    if d_el is not None and d_el.text:
                        divs = int(d_el.text)

                for note in measure.findall(f"{ns}note"):
                    dur_val = int(note.find(f"{ns}duration").text)
                    q_len = dur_val / float(divs)

                    is_dotted = any(abs(q_len - target) < 0.02 for target in [3.0, 1.5, 0.75, 0.375])
                    if is_dotted:
                        dot_elem = note.find(f"{ns}dot")
                        self.assertIsNotNone(
                            dot_elem,
                            f"Dotted note ({q_len} quarters) in measure {measure.get('number')} missing <dot/> tag",
                        )

    def verify_accidentals_integrity(self, root, ns):
        for part in root.findall(f"{ns}part"):
            for measure in part.findall(f"{ns}measure"):
                for note in measure.findall(f"{ns}note"):
                    pitch = note.find(f"{ns}pitch")
                    if pitch is not None:
                        alt = pitch.find(f"{ns}alter")
                        if alt is not None and alt.text and alt.text != "0":
                            acc = note.find(f"{ns}accidental")
                            self.assertIsNotNone(acc, f"Altered pitch ({alt.text}) missing <accidental> element")
                            if alt.text == "1":
                                self.assertEqual(acc.text, "sharp")
                            elif alt.text == "-1":
                                self.assertEqual(acc.text, "flat")

    def verify_tablature_pitch_alignment(self, root, ns, tuning_type="4_string_standard"):
        tuning = BASS_TUNINGS.get(tuning_type, BASS_TUNINGS["4_string_standard"])
        num_strings = len(tuning)

        for part in root.findall(f"{ns}part"):
            for measure in part.findall(f"{ns}measure"):
                for note in measure.findall(f"{ns}note"):
                    pitch = note.find(f"{ns}pitch")
                    if pitch is not None:
                        notations = note.find(f"{ns}notations")
                        self.assertIsNotNone(notations, "Pitch note missing <notations> element")

                        tech = notations.find(f"{ns}technical")
                        self.assertIsNotNone(tech, "Pitch note missing <technical> element for tablature")

                        str_elem = tech.find(f"{ns}string")
                        fret_elem = tech.find(f"{ns}fret")
                        self.assertIsNotNone(str_elem, "Tablature missing <string>")
                        self.assertIsNotNone(fret_elem, "Tablature missing <fret>")

                        string_num = int(str_elem.text)
                        fret_num = int(fret_elem.text)

                        self.assertTrue(
                            1 <= string_num <= num_strings,
                            f"String number {string_num} out of range [1, {num_strings}]",
                        )
                        self.assertTrue(0 <= fret_num <= 24, f"Fret number {fret_num} out of valid range [0, 24]")

                        string_idx = num_strings - string_num
                        base_string_midi = tuning[string_idx][2]
                        tab_midi = base_string_midi + fret_num

                        step = pitch.find(f"{ns}step").text
                        octave = int(pitch.find(f"{ns}octave").text)
                        alt_elem = pitch.find(f"{ns}alter")
                        alter = int(alt_elem.text) if (alt_elem is not None and alt_elem.text) else 0

                        notation_midi = pitch_to_midi(step, octave, alter)
                        self.assertEqual(
                            tab_midi,
                            notation_midi,
                            f"Tab MIDI ({tab_midi}) does not match notation MIDI ({notation_midi}) for string {string_num}, fret {fret_num}",
                        )

    def verify_ties_and_notations_consistency(self, root, ns):
        for part in root.findall(f"{ns}part"):
            for measure in part.findall(f"{ns}measure"):
                for note in measure.findall(f"{ns}note"):
                    tie_elem = note.find(f"{ns}tie")
                    if tie_elem is not None:
                        tie_type = tie_elem.get("type")
                        notations = note.find(f"{ns}notations")
                        self.assertIsNotNone(notations, "Tied note missing <notations> tag")
                        tied_notation = notations.find(f"{ns}tied")
                        self.assertIsNotNone(tied_notation, "Tied note missing <notations><tied> tag")
                        self.assertEqual(tied_notation.get("type"), tie_type)

    def verify_rest_cleanliness(self, root, ns):
        for part in root.findall(f"{ns}part"):
            for measure in part.findall(f"{ns}measure"):
                for note in measure.findall(f"{ns}note"):
                    rest = note.find(f"{ns}rest")
                    if rest is not None:
                        self.assertIsNone(note.find(f"{ns}pitch"), "Rest note must NOT contain <pitch> element")
                        self.assertIsNone(note.find(f"{ns}beam"), "Rest note must NOT contain <beam> element")
                        self.assertIsNotNone(note.find(f"{ns}duration"), "Rest note missing <duration>")
                        self.assertIsNotNone(note.find(f"{ns}type"), "Rest note missing <type>")

    def test_syncopated_melodic_funk_4_4(self):
        notes = [
            AudioEvent(start=0.0, end=0.3, pitch=28),
            AudioEvent(start=0.3, end=0.45, pitch=0, is_rest=True),
            AudioEvent(start=0.45, end=0.6, pitch=33),
            AudioEvent(start=0.6, end=1.05, pitch=38),
            AudioEvent(start=1.05, end=1.2, pitch=43),
            AudioEvent(start=1.2, end=1.8, pitch=28),
            AudioEvent(start=1.8, end=3.0, pitch=33),
            AudioEvent(start=3.0, end=3.6, pitch=0, is_rest=True),
        ]

        out_xml = os.path.join(self.output_dir, "syncopated_funk_4_4.xml")
        export_score(
            note_layer=notes,
            fretboard_path=[],
            detected_key="G",
            song_title="Melodic Funk Jam",
            artist_name="Funkmeister",
            bpm=100.0,
            is_compound=False,
            target_level=5,
            output_xml_path=out_xml,
            time_sig_str="4/4",
            tuning_type="4_string_standard",
            enable_dual_stave=True,
        )

        self.assertTrue(os.path.exists(out_xml))
        root, ns = self.parse_music_xml(out_xml)

        self.verify_measure_durations_sum(root, ns)
        self.verify_schema_child_ordering(root, ns)
        self.verify_voice_and_staff_assignments(root, ns)
        self.verify_dotted_notes_have_dot_tag(root, ns)
        self.verify_tablature_pitch_alignment(root, ns, "4_string_standard")
        self.verify_rest_cleanliness(root, ns)

    def test_expressive_slap_pop_drop_d(self):
        notes = [
            AudioEvent(start=0.0, end=0.25, pitch=26, is_slap=True, is_accent=True),
            AudioEvent(start=0.25, end=0.5, pitch=38, is_pop=True),
            AudioEvent(start=0.5, end=0.625, pitch=26, is_slap=True, is_ghost=True),
            AudioEvent(start=0.625, end=0.875, pitch=41, is_slide=True),
            AudioEvent(start=0.875, end=1.125, pitch=43, is_slide=True),
            AudioEvent(start=1.125, end=2.125, pitch=33),
        ]

        out_xml = os.path.join(self.output_dir, "slap_pop_drop_d.xml")
        export_score(
            note_layer=notes,
            fretboard_path=[],
            detected_key="D",
            song_title="Slap Attack Drop-D",
            artist_name="Thump Bassist",
            bpm=120.0,
            is_compound=False,
            target_level=5,
            output_xml_path=out_xml,
            time_sig_str="4/4",
            tuning_type="drop_d",
            enable_dual_stave=True,
        )

        self.assertTrue(os.path.exists(out_xml))
        root, ns = self.parse_music_xml(out_xml)

        self.verify_measure_durations_sum(root, ns)
        self.verify_schema_child_ordering(root, ns)
        self.verify_voice_and_staff_assignments(root, ns)
        self.verify_tablature_pitch_alignment(root, ns, "drop_d")

        part = root.find(f"{ns}part")
        first_m = part.find(f"{ns}measure")
        first_note = first_m.find(f"{ns}note")
        tech = first_note.find(f"{ns}notations").find(f"{ns}technical")
        self.assertEqual(int(tech.find(f"{ns}string").text), 4)
        self.assertEqual(int(tech.find(f"{ns}fret").text), 0)

    def test_compound_prog_rock_6_8(self):
        notes = [
            AudioEvent(start=0.0, end=1.0, pitch=28),
            AudioEvent(start=1.0, end=2.5, pitch=31),
            AudioEvent(start=2.5, end=3.0, pitch=33),
        ]

        out_xml = os.path.join(self.output_dir, "compound_prog_rock_6_8.xml")
        export_score(
            note_layer=notes,
            fretboard_path=[],
            detected_key="Em",
            song_title="Prog Odyssey",
            artist_name="Genesis fan",
            bpm=120.0,
            is_compound=True,
            target_level=5,
            output_xml_path=out_xml,
            time_sig_str="6/8",
            tuning_type="4_string_standard",
            enable_dual_stave=True,
        )

        self.assertTrue(os.path.exists(out_xml))
        root, ns = self.parse_music_xml(out_xml)

        self.verify_measure_durations_sum(root, ns)
        self.verify_schema_child_ordering(root, ns)
        self.verify_voice_and_staff_assignments(root, ns)
        self.verify_ties_and_notations_consistency(root, ns)
        self.verify_tablature_pitch_alignment(root, ns, "4_string_standard")

    def test_five_string_bass_odd_meter_5_4(self):
        notes = [
            AudioEvent(start=0.0, end=0.5, pitch=23),
            AudioEvent(start=0.5, end=1.0, pitch=28),
            AudioEvent(start=1.0, end=1.5, pitch=33),
            AudioEvent(start=1.5, end=2.0, pitch=35),
            AudioEvent(start=2.0, end=2.5, pitch=38),
        ]

        out_xml = os.path.join(self.output_dir, "five_string_bass_5_4.xml")
        export_score(
            note_layer=notes,
            fretboard_path=[],
            detected_key="B",
            song_title="Sub-Zero Riff",
            artist_name="Djent Master",
            bpm=120.0,
            is_compound=False,
            target_level=5,
            output_xml_path=out_xml,
            time_sig_str="5/4",
            tuning_type="5_string_standard",
            enable_dual_stave=True,
        )

        self.assertTrue(os.path.exists(out_xml))
        root, ns = self.parse_music_xml(out_xml)

        self.verify_measure_durations_sum(root, ns)
        self.verify_schema_child_ordering(root, ns)
        self.verify_voice_and_staff_assignments(root, ns)
        self.verify_tablature_pitch_alignment(root, ns, "5_string_standard")

        part = root.find(f"{ns}part")
        first_m = part.find(f"{ns}measure")
        first_note = first_m.find(f"{ns}note")
        tech = first_note.find(f"{ns}notations").find(f"{ns}technical")
        self.assertEqual(int(tech.find(f"{ns}string").text), 5)
        self.assertEqual(int(tech.find(f"{ns}fret").text), 0)

    def test_jazz_swing_walking_3_4_accidentals(self):
        notes = [
            AudioEvent(start=0.0, end=0.5, pitch=29),
            AudioEvent(start=0.5, end=1.0, pitch=30),
            AudioEvent(start=1.0, end=1.5, pitch=31),
            AudioEvent(start=1.5, end=2.0, pitch=32),
            AudioEvent(start=2.0, end=2.5, pitch=33),
            AudioEvent(start=2.5, end=3.0, pitch=34),
        ]

        out_xml = os.path.join(self.output_dir, "jazz_swing_3_4.xml")
        export_score(
            note_layer=notes,
            fretboard_path=[],
            detected_key="F",
            song_title="Waltz for Debby",
            artist_name="Jazz Trio",
            bpm=120.0,
            is_compound=False,
            target_level=5,
            output_xml_path=out_xml,
            time_sig_str="3/4",
            tuning_type="4_string_standard",
            enable_dual_stave=True,
        )

        self.assertTrue(os.path.exists(out_xml))
        root, ns = self.parse_music_xml(out_xml)

        self.verify_measure_durations_sum(root, ns)
        self.verify_schema_child_ordering(root, ns)
        self.verify_voice_and_staff_assignments(root, ns)
        self.verify_accidentals_integrity(root, ns)
        self.verify_tablature_pitch_alignment(root, ns, "4_string_standard")


class TestTomlConfigurations(unittest.TestCase):
    """
    Comprehensive unit test validating all TOML files in subtone/config.
    """

    def setUp(self):
        self.config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "subtone", "config"))
        self.assertTrue(
            os.path.isdir(self.config_dir),
            f"Config directory does not exist: {self.config_dir}",
        )

    def test_all_toml_files_parseable(self):
        toml_files = [f for f in os.listdir(self.config_dir) if f.endswith(".toml")]
        self.assertGreater(len(toml_files), 0, "No TOML files found in config directory.")

        for filename in toml_files:
            filepath = os.path.join(self.config_dir, filename)
            with self.subTest(file=filename):
                with open(filepath, "rb") as f:
                    data = tomllib.load(f)
                self.assertIsInstance(data, dict, f"Parsed TOML root in {filename} must be a dictionary.")

    def test_load_all_genre_configs_merge_and_validate(self):
        all_configs = load_genre_configs(self.config_dir)
        self.assertIn("genres", all_configs)
        self.assertIn("categories", all_configs)
        self.assertIn("patterns", all_configs)
        self.assertIn("artists", all_configs)

    def test_genres_extend_valid_parents_and_patterns(self):
        all_configs = load_genre_configs(self.config_dir)
        genres = all_configs.get("genres", {})
        categories = all_configs.get("categories", {})
        patterns = all_configs.get("patterns", {})

        for genre_name, genre_data in genres.items():
            with self.subTest(genre=genre_name):
                if not isinstance(genre_data, dict):
                    continue
                if "extends" in genre_data:
                    parent = genre_data["extends"]
                    self.assertTrue(
                        parent in categories or parent in genres,
                        f"Genre '{genre_name}' extends '{parent}', which is not in categories or genres.",
                    )

                if (
                    "rhythmic_anchor" in genre_data
                    and isinstance(genre_data["rhythmic_anchor"], dict)
                    and "pattern" in genre_data["rhythmic_anchor"]
                ):
                    pat_name = genre_data["rhythmic_anchor"]["pattern"]
                    if isinstance(pat_name, str):
                        self.assertIn(
                            pat_name,
                            patterns,
                            f"Genre '{genre_name}' references pattern '{pat_name}' which does not exist in patterns.",
                        )

                name_res, obj_res = resolve_genre(genre_name, all_configs)
                self.assertIsNotNone(obj_res)

    def test_artists_extend_valid_parents_and_patterns(self):
        all_configs = load_genre_configs(self.config_dir)
        artists = all_configs.get("artists", {})
        categories = all_configs.get("categories", {})
        genres = all_configs.get("genres", {})
        patterns = all_configs.get("patterns", {})

        for artist_name, artist_data in artists.items():
            with self.subTest(artist=artist_name):
                if not isinstance(artist_data, dict):
                    continue
                if "extends" in artist_data:
                    parent = artist_data["extends"]
                    self.assertTrue(
                        parent in categories or parent in genres,
                        f"Artist '{artist_name}' extends '{parent}', which is not in categories or genres.",
                    )

                if (
                    "rhythmic_anchor" in artist_data
                    and isinstance(artist_data["rhythmic_anchor"], dict)
                    and "pattern" in artist_data["rhythmic_anchor"]
                ):
                    pat_name = artist_data["rhythmic_anchor"]["pattern"]
                    if isinstance(pat_name, str):
                        self.assertIn(
                            pat_name,
                            patterns,
                            f"Artist '{artist_name}' references pattern '{pat_name}' which does not exist in patterns.",
                        )

                resolved = resolve_artist(artist_name, all_configs)
                self.assertIsNotNone(resolved)

    def test_patterns_and_groove_structures(self):
        all_configs = load_genre_configs(self.config_dir)
        patterns = all_configs.get("patterns", {})
        self.assertGreater(len(patterns), 0, "patterns.toml should contain pattern definitions.")

        for pat_key, pat_val in patterns.items():
            with self.subTest(pattern=pat_key):
                if isinstance(pat_val, dict):
                    has_accents = "accents" in pat_val
                    has_pattern = "pattern" in pat_val
                    self.assertTrue(
                        has_accents or has_pattern,
                        f"Pattern '{pat_key}' must have 'accents' or 'pattern' key.",
                    )

    def test_rule_configurations_match_known_categories_or_genres(self):
        all_configs = load_genre_configs(self.config_dir)
        categories = set(all_configs.get("categories", {}).keys())
        genres = set(all_configs.get("genres", {}).keys())
        known_parents = categories.union(genres)

        rule_keys = [
            "micro_timing",
            "articulation_intent",
            "fretboard_navigation",
            "notation_engraving",
            "harmonic_hysteresis",
        ]

        for rule_key in rule_keys:
            rule_dict = all_configs.get(rule_key, {})
            self.assertIsInstance(rule_dict, (dict, Mapping), f"Rule config '{rule_key}' must be a dict.")
            for section_name in rule_dict.keys():
                with self.subTest(rule=rule_key, section=section_name):
                    self.assertIn(
                        section_name,
                        known_parents,
                        f"Rule '{rule_key}' contains section '{section_name}' which is not in categories or genres.",
                    )

    def test_parse_metadata_from_path_hysteria(self):
        path = "midi/alternative rock_A minor_Muse_Hysteria"
        artist, title, clean_name, key_str, resolved_genre_name, genre_config = parse_metadata_from_path(path)
        self.assertEqual(artist, "Muse")
        self.assertEqual(title, "Hysteria")
        self.assertEqual(key_str, "A minor")
        self.assertIsNotNone(genre_config)


class TestPublishingExport(unittest.TestCase):
    def test_genre_strict_validation(self):
        with self.assertRaises(TypeError):
            Genre.from_dict("test_invalid_genre", {"tuning": 12345})

        with self.assertRaises(TypeError):
            Genre.from_dict("test_invalid_genre", {"preferred_key_signatures": "not_a_list"})

    def test_audio_engine_strict_validation(self):
        if np is None:
            self.skipTest("numpy is not installed")
        empty_audio = np.array([], dtype=np.float32)
        with self.assertRaises(ValueError):
            apply_bass_bandpass(empty_audio, 100, 300)

        valid_audio = np.zeros(44100, dtype=np.float32)
        with self.assertRaises(ValueError):
            apply_bass_bandpass(valid_audio, 1000, lowcut=500, highcut=200)

    def test_quantized_event_streaming(self):
        notes = [
            Note(start=0.0, end=0.5, pitch=36, amplitude=0.8),
            Note(start=0.5, end=1.0, pitch=38, amplitude=0.7),
        ]
        bpm = 120.0
        capacity = fractions.Fraction(4, 1)

        chunks = list(stream_quantized_events(notes, bpm=bpm, measure_capacity=capacity))
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(chunks[0].event.pitch, 36)
        self.assertFalse(chunks[0].is_rest)

    def test_publishing_export_pipeline(self):
        song = Song(
            artist_name="Test Artist",
            song_title="Test Bassline",
            bpm=120.0,
            genres=["rock_punk_alternative"],
            time_sig="4/4",
            target_level=5,
        )

        song.notes = [
            Note(start=0.0, end=0.5, pitch=36, amplitude=0.8),
            Note(start=0.5, end=1.0, pitch=38, amplitude=0.7),
            Note(start=1.0, end=2.0, pitch=40, amplitude=0.9),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_xml = os.path.join(tmp_dir, "test_output.xml")
            song.output_xml_path = output_xml

            build_and_export_song(song)

            self.assertTrue(os.path.exists(output_xml))
            self.assertTrue(os.path.getsize(output_xml) > 0)

            with open(output_xml, encoding="utf-8") as f:
                content = f.read()
                self.assertTrue("score-partwise" in content or "xml" in content)


class TestNoteStreamPipeline(unittest.TestCase):
    def test_note_clone_isolated_but_keeps_source_provenance(self):
        source = AudioEvent(start=0.0, end=0.25, pitch=40, pitches=[40, 47], bends=[0.2])
        note = Note.from_event(source)
        cloned = note.clone()

        cloned.pitches.append(52)
        cloned.bends.append(0.4)
        cloned.associated_events.clear()

        self.assertEqual(note.pitches, [40, 47])
        self.assertEqual(note.bends, [0.2])
        self.assertIs(note.original_event, source)
        self.assertEqual(cloned.associated_events, [])

    def test_note_from_note_preserves_original_source_event(self):
        source = AudioEvent(start=0.0, end=0.25, pitch=40)
        note = Note.from_event(source)
        converted = Note.from_event(note)

        self.assertIs(converted.original_event, source)
        self.assertIsNot(converted.associated_events, note.associated_events)

    def test_later_stages_preserve_and_export_processed_notes(self):
        source_event = AudioEvent(start=0.0, end=0.3, pitch=32)
        song = Song.from_transcription(
            source_events=[source_event],
            beat_times=[0.0, 0.5, 1.0, 1.5],
            bpm=120.0,
            key_obj=key.Key("C"),
            target_level=5,
        )

        original_note = song.notes[0]
        original_note.end = 0.5
        filter_song_for_level(song)

        self.assertEqual(song.audio_events[0].end, 0.3)
        self.assertEqual(song.notes[0].end, 0.5)
        self.assertIsNot(song.notes[0], original_note)

        snap_song_to_scale(song)
        self.assertEqual(song.audio_events[0].pitch, 32)
        self.assertEqual(song.notes[0].pitch, 31)
        self.assertEqual(song.notes[0].end, 0.5)

        ErgonomicFretboardHMMSolver().solve_song(song)
        self.assertIsNotNone(song.notes[0].fret_position)

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = os.path.join(output_dir, "note-stream.musicxml")
            song.output_xml_path = output_path
            build_and_export_song(song)

            root = ET.parse(output_path).getroot()
            pitch = next(elem for elem in root.iter() if elem.tag.endswith("pitch"))
            step = next(elem.text for elem in pitch if elem.tag.endswith("step"))
            octave = int(next(elem.text for elem in pitch if elem.tag.endswith("octave")))
            alter = next((int(elem.text) for elem in pitch if elem.tag.endswith("alter")), 0)
            midi = (octave + 1) * 12 + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step] + alter
            self.assertEqual(midi, 31)


class TestMusicXmlPublishingExport(unittest.TestCase):
    """
    Robust test suite verifying W3C MusicXML specification compliance for professional publishing houses
    and VexFlow rendering compatibility on bass parts.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.xml_path = os.path.join(self.temp_dir.name, "long_bass_publishing_score.musicxml")

    def tearDown(self):
        self.temp_dir.cleanup()

    def parse_xml(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"
        return root, ns

    def test_long_song_publishing_musicxml(self):
        notes = []
        bpm = 120.0

        curr_time = 0.0
        pitch_sequence = [28, 31, 33, 35, 36, 38, 40, 43]

        for i in range(128):
            pitch = pitch_sequence[i % len(pitch_sequence)]
            dur = 1.5 if i % 8 == 7 else 0.45
            n = Note(start=curr_time, end=curr_time + dur, pitch=pitch)
            if i % 5 == 0:
                n.is_staccato = True
            if i % 7 == 0:
                n.is_accent = True
            notes.append(n)
            curr_time += 0.5

        song = Song.from_transcription(
            source_events=notes,
            bpm=bpm,
            time_sig="4/4",
            artist_name="Subtone Publishing House",
            song_title="Long Bass Symphony",
            tuning_type="4_string_standard",
            target_level=5,
        )
        song.output_xml_path = self.xml_path
        build_and_export_song(song)

        self.assertTrue(os.path.exists(self.xml_path), "Exported MusicXML file must exist")

        root, ns = self.parse_xml(self.xml_path)

        part = root.find(f"{ns}part")
        self.assertIsNotNone(part, "<part> element must exist in MusicXML")
        measures = part.findall(f"{ns}measure")
        self.assertGreaterEqual(len(measures), 32, "Song must not be truncated; expected >= 32 measures")

        div_elem = root.find(f".//{ns}divisions")
        divisions_per_quarter = int(div_elem.text) if div_elem is not None and div_elem.text else 8

        total_note_and_rest_count = 0
        for m in measures:
            notes_in_m = m.findall(f"{ns}note")
            total_note_and_rest_count += len(notes_in_m)
            dur_sum = 0
            for n_elem in notes_in_m:
                dur_e = n_elem.find(f"{ns}duration")
                if dur_e is not None and dur_e.text:
                    dur_sum += int(dur_e.text)
            self.assertEqual(
                dur_sum,
                4 * divisions_per_quarter,
                f"Measure {m.get('number')} total duration must equal {4 * divisions_per_quarter} divisions",
            )

        self.assertGreater(
            total_note_and_rest_count, 100, "Must contain all notes/rests across long song without truncation"
        )

        m1 = measures[0]
        attrs1 = m1.find(f"{ns}attributes")
        self.assertIsNotNone(attrs1, "Measure 1 must contain <attributes>")

        trans_elem = attrs1.find(f"{ns}transpose")
        self.assertIsNotNone(trans_elem, "Measure 1 <attributes> must include <transpose> directive for bass")

        diatonic_e = trans_elem.find(f"{ns}diatonic")
        self.assertIsNotNone(diatonic_e, "<transpose> must contain <diatonic>")
        self.assertEqual(diatonic_e.text, "0", "<diatonic> value must be 0")

        chromatic_e = trans_elem.find(f"{ns}chromatic")
        self.assertIsNotNone(chromatic_e, "<transpose> must contain <chromatic>")
        self.assertEqual(chromatic_e.text, "0", "<chromatic> value must be 0")

        octave_e = trans_elem.find(f"{ns}octave-change")
        self.assertIsNotNone(octave_e, "<transpose> must contain <octave-change>")
        self.assertEqual(octave_e.text, "-1", "Octave-transposing bass must specify <octave-change>-1</octave-change>")

        part_list = root.find(f"{ns}part-list")
        self.assertIsNotNone(part_list, "MusicXML header must contain <part-list>")
        score_part = part_list.find(f"{ns}score-part")
        self.assertIsNotNone(score_part, "<part-list> must contain <score-part>")
        self.assertEqual(score_part.get("id"), part.get("id"), "<score-part> id must match <part> id")

        p_name = score_part.find(f"{ns}part-name")
        self.assertIsNotNone(p_name, "<score-part> must contain <part-name>")
        self.assertEqual(p_name.text, "Electric Bass")

        p_abbr = score_part.find(f"{ns}part-abbreviation")
        self.assertIsNotNone(p_abbr, "<score-part> must contain <part-abbreviation>")
        self.assertEqual(p_abbr.text, "B. Pass")

        score_inst = score_part.find(f"{ns}score-instrument")
        self.assertIsNotNone(score_inst, "<score-part> must contain <score-instrument>")
        inst_name = score_inst.find(f"{ns}instrument-name")
        self.assertIsNotNone(inst_name, "<score-instrument> must contain <instrument-name>")
        self.assertEqual(inst_name.text, "Electric Bass")

        midi_dev = score_part.find(f"{ns}midi-device")
        self.assertIsNotNone(midi_dev, "<score-part> must contain <midi-device>")
        self.assertEqual(midi_dev.get("port"), "1")

        midi_inst = score_part.find(f"{ns}midi-instrument")
        self.assertIsNotNone(midi_inst, "<score-part> must contain <midi-instrument>")

        midi_chan = midi_inst.find(f"{ns}midi-channel")
        self.assertIsNotNone(midi_chan, "<midi-instrument> must contain <midi-channel>")
        self.assertEqual(midi_chan.text, "1", "MIDI channel must default to 1")

        midi_prog = midi_inst.find(f"{ns}midi-program")
        self.assertIsNotNone(midi_prog, "<midi-instrument> must contain <midi-program>")
        self.assertEqual(midi_prog.text, "34", "Electric Bass MIDI program must be 34")

        staff_det = attrs1.find(f"{ns}staff-details")
        self.assertIsNotNone(staff_det, "Measure 1 <attributes> must contain <staff-details>")

        staff_lines = staff_det.find(f"{ns}staff-lines")
        self.assertIsNotNone(staff_lines, "<staff-details> must contain <staff-lines>")
        self.assertEqual(staff_lines.text, "4", "Standard 4-string bass must specify 4 staff lines")

        staff_tunings = staff_det.findall(f"{ns}staff-tuning")
        self.assertEqual(
            len(staff_tunings), 4, "<staff-details> must contain 4 <staff-tuning> elements for 4-string bass"
        )

        expected_tuning = [
            ("1", "E", "1"),
            ("2", "A", "1"),
            ("3", "D", "2"),
            ("4", "G", "2"),
        ]

        for st_elem, (exp_line, exp_step, exp_oct) in zip(staff_tunings, expected_tuning):
            self.assertEqual(st_elem.get("line"), exp_line, f"Staff tuning line attribute must be {exp_line}")
            t_step = st_elem.find(f"{ns}tuning-step")
            t_oct = st_elem.find(f"{ns}tuning-octave")
            self.assertIsNotNone(t_step, "<staff-tuning> must contain <tuning-step>")
            self.assertIsNotNone(t_oct, "<staff-tuning> must contain <tuning-octave>")
            self.assertEqual(t_step.text, exp_step, f"String {exp_line} tuning step must be {exp_step}")
            self.assertEqual(t_oct.text, exp_oct, f"String {exp_line} tuning octave must be {exp_oct}")

        found_technical_tab = False
        found_staccato = False
        found_accent = False

        for m in measures:
            for note_elem in m.findall(f"{ns}note"):
                if note_elem.find(f"{ns}rest") is not None:
                    continue

                notations = note_elem.find(f"{ns}notations")
                if notations is not None:
                    tech = notations.find(f"{ns}technical")
                    if tech is not None:
                        string_e = tech.find(f"{ns}string")
                        fret_e = tech.find(f"{ns}fret")
                        if string_e is not None and fret_e is not None:
                            found_technical_tab = True
                            string_val = int(string_e.text)
                            fret_val = int(fret_e.text)
                            self.assertTrue(1 <= string_val <= 4, f"Bass string {string_val} must be 1..4")
                            self.assertTrue(0 <= fret_val <= 24, f"Fret {fret_val} must be 0..24")

                    artic = notations.find(f"{ns}articulations")
                    if artic is not None:
                        if artic.find(f"{ns}staccato") is not None:
                            found_staccato = True
                        if artic.find(f"{ns}accent") is not None:
                            found_accent = True

        self.assertTrue(found_technical_tab, "Notes must contain <notations><technical> with <string> and <fret>")
        self.assertTrue(found_staccato, "Articulations must include <staccato/>")
        self.assertTrue(found_accent, "Articulations must include <accent/>")

        for m in measures:
            for note_elem in m.findall(f"{ns}note"):
                voice_elem = note_elem.find(f"{ns}voice")
                self.assertIsNotNone(voice_elem, "Every note/rest must contain explicit <voice> element")
                self.assertEqual(voice_elem.text, "1", "<voice> must be set to 1")

                staff_elem = note_elem.find(f"{ns}staff")
                self.assertIsNotNone(staff_elem, "Every note/rest must contain explicit <staff> element")
                self.assertEqual(staff_elem.text, "1", "<staff> must be set to 1")

        found_tie = False
        for m in measures:
            for note_elem in m.findall(f"{ns}note"):
                tie_elems = note_elem.findall(f"{ns}tie")
                if tie_elems:
                    found_tie = True
                    notations = note_elem.find(f"{ns}notations")
                    self.assertIsNotNone(notations, "Notes with <tie> tags must also contain <notations>")
                    for te in tie_elems:
                        t_type = te.get("type")
                        self.assertIn(
                            t_type, ["start", "stop", "continue"], "<tie> type must be start, stop, or continue"
                        )
                        tied_elems = [td for td in notations.findall(f"{ns}tied") if td.get("type") == t_type]
                        self.assertTrue(
                            len(tied_elems) > 0,
                            f"Must contain corresponding <tied type='{t_type}'/> inside <notations>",
                        )

        self.assertTrue(found_tie, "Score must contain tied notes across measure boundaries")


class TestMeasureChunkAndRhythmicAtom(unittest.TestCase):
    def test_rhythmic_atom_from_note(self):
        event = AudioEvent(start=0.0, end=0.5, pitch=40)
        note = Note.from_event(event)
        note.fret_position = (4, 2, 1)
        note.is_accent = True

        atom = RhythmicAtom.from_note(
            note_obj=note,
            duration_q=fractions.Fraction(1, 4),
            measure_index=1,
            start_q=fractions.Fraction(0, 1),
            tie_type=None,
        )

        self.assertEqual(atom.pitch, 40)
        self.assertEqual(atom.duration_q, fractions.Fraction(1, 4))
        self.assertFalse(atom.is_rest)
        self.assertEqual(atom.string_num, 4)
        self.assertEqual(atom.fret_num, 2)
        self.assertEqual(atom.finger_num, 1)
        self.assertIn("accent", atom.articulations)
        self.assertIsNotNone(atom.parent_event_id)

    def test_measure_chunk_encapsulation(self):
        event = AudioEvent(start=0.0, end=1.0, pitch=48)
        note = Note.from_event(event)

        chunk = MeasureChunk(
            measure_index=1,
            event_or_start=0.0,
            duration_q_or_end=2.0,
            measure_capacity=fractions.Fraction(4, 1),
            events=[note],
        )

        atom = RhythmicAtom.from_note(
            note_obj=note,
            duration_q=fractions.Fraction(1, 2),
            measure_index=1,
        )
        chunk.add_atom(atom)

        self.assertEqual(chunk.measure_index, 1)
        self.assertEqual(len(chunk.atoms), 1)
        self.assertEqual(chunk.atoms[0].pitch, 48)

    def test_decompose_note_to_atoms_with_ties(self):
        note = Note(start=0.0, end=2.0, pitch=40)
        atoms = decompose_note_to_atoms(
            note_obj=note,
            q_dur=fractions.Fraction(6, 1),
            curr_m_fill=fractions.Fraction(0, 1),
            measure_capacity=fractions.Fraction(4, 1),
            measure_index=1,
        )

        self.assertTrue(len(atoms) >= 2)
        self.assertEqual(atoms[0].measure_index, 1)
        self.assertEqual(atoms[0].tie_type, "start")
        self.assertEqual(atoms[-1].measure_index, 2)
        self.assertEqual(atoms[-1].tie_type, "stop")

    def test_minimum_duration_filtering(self):
        note = Note(start=0.0, end=0.01, pitch=40)
        atoms = decompose_note_to_atoms(
            note_obj=note,
            q_dur=fractions.Fraction(1, 100),
            curr_m_fill=fractions.Fraction(0, 1),
            measure_capacity=fractions.Fraction(4, 1),
            measure_index=1,
        )

        self.assertTrue(all(a.duration_q >= fractions.Fraction(1, 32) for a in atoms))


class TestFretboardAndAudioEvents(unittest.TestCase):
    def test_key_object_auto_initialization(self):
        song = Song(parsed_key_str="A minor", song_title="Test Track")
        self.assertIsNotNone(song.key_obj)
        self.assertEqual(str(song.key_obj), "a minor")

    def test_sub_bass_pitch_folding(self):
        events = [AudioEvent(start=0.0, end=0.5, pitch=21, amplitude=0.8)]
        song = Song(bass_audio_events=events, tuning_type="4_string_standard")
        self.assertEqual(song.bass_notes[0].pitch, 33)
        self.assertEqual(song.bass_audio_events[0].pitch, 33)

    def test_fretboard_high_fret_penalty_and_positioning(self):
        solver = ErgonomicFretboardHMMSolver(tuning_type="4_string_standard")
        notes = [Note(start=float(i) * 0.5, end=float(i) * 0.5 + 0.4, pitch=48) for i in range(5)]
        positions, rakes, legatos, slides = solver._solve_notes(notes, bpm=120)

        for pos in positions:
            string_num, fret, finger = pos
            self.assertLessEqual(fret, 12, f"Expected fret <= 12, got fret {fret} on string {string_num}")

    def test_alternate_downpicking_for_fast_runs(self):
        solver = ErgonomicFretboardHMMSolver()
        solver.downpicking_pref = True
        notes = [Note(start=float(i) * 0.1, end=float(i) * 0.1 + 0.08, pitch=38) for i in range(8)]
        solver._solve_notes(notes, bpm=120)

        downpicks = [n.is_downpick for n in notes]
        self.assertIn(True, downpicks)
        self.assertIn(False, downpicks)

    def test_measure_chunk_measure_num_property(self):
        chunk = MeasureChunk(measure_index=5)
        self.assertEqual(chunk.measure_num, 5)
        chunk.measure_num = 10
        self.assertEqual(chunk.measure_index, 10)

    def test_audio_event_flag_synchronization(self):
        evt = AudioEvent(start=0.0, end=0.5, pitch=40, tag="slap")
        self.assertTrue(evt.is_slap)

        evt2 = AudioEvent(start=0.0, end=0.5, pitch=40, is_pop=True)
        self.assertEqual(evt2.tag, "pop")


class TestEngineAndBassCollections(unittest.TestCase):
    def test_audio_event_engine(self):
        event = AudioEvent(start=0.0, end=0.5, pitch=36, engine="basicPitch")
        self.assertEqual(event.engine, "basicPitch")
        self.assertEqual(event.engine_type, "basicPitch")

        event.engine_type = "pyin"
        self.assertEqual(event.engine, "pyin")

        d = event.to_dict()
        self.assertEqual(d["engine"], "pyin")
        self.assertEqual(d["engine_type"], "pyin")

    def test_note_engine_inheritance(self):
        event = AudioEvent(start=0.0, end=0.5, pitch=36, engine="customEngine")
        note = Note.from_event(event)
        self.assertEqual(note.engine, "customEngine")
        self.assertEqual(note.engine_type, "customEngine")

    def test_song_engine_and_bass_collections(self):
        evt1 = AudioEvent(start=0.0, end=0.5, pitch=28, engine="basicPitch")
        evt2 = AudioEvent(start=0.5, end=1.0, pitch=33, engine="basicPitch")

        song = Song.from_transcription(
            source_events=[evt1, evt2], beat_times=[0.0, 0.5, 1.0], bpm=120.0, engine="basicPitch"
        )

        self.assertEqual(song.engine, "basicPitch")

        self.assertEqual(len(song.bassAudioEvents), 2)
        self.assertEqual(len(song.bass_audio_events), 2)
        self.assertEqual(len(song.audioEvents), 2)
        self.assertEqual(song.bassAudioEvents[0].pitch, 28)

        self.assertEqual(len(song.bassNotes), 2)
        self.assertEqual(len(song.bass_notes), 2)
        self.assertEqual(song.bassNotes[0].pitch, 28)

        new_note = Note(start=0.0, end=1.0, pitch=40, engine="basicPitch")
        song.bassNotes = [new_note]
        self.assertEqual(len(song.notes), 1)
        self.assertEqual(song.notes[0].pitch, 40)

        sd = song.to_dict()
        self.assertEqual(sd["engine"], "basicPitch")
        self.assertIn("bassNotes", sd)
        self.assertIn("bassAudioEvents", sd)
        self.assertEqual(len(sd["bassNotes"]), 1)
        self.assertEqual(sd["bassNotes"][0]["pitch"], 40)

    def test_from_event_streams_engine(self):
        stream_data = {
            "bass_primary": {
                "stream_name": "bass_primary",
                "source_stem": "bass",
                "stream_type": "primary",
                "engine": "basicPitch",
                "events": [AudioEvent(start=0.0, end=0.5, pitch=36)],
                "metadata": {"bpm": 120.0, "time_sig": "4/4", "engine": "basicPitch"},
            }
        }
        song = Song.from_event_streams(stream_data)
        self.assertEqual(song.engine, "basicPitch")
        self.assertEqual(len(song.bassAudioEvents), 1)
        self.assertEqual(song.bassNotes[0].pitch, 36)

    def test_parse_key_object_minor_and_major(self):
        k1 = parse_key_object("c minor")
        self.assertEqual(k1.tonic.name.lower(), "c")
        self.assertEqual(k1.mode, "minor")

        k2 = parse_key_object("C major")
        self.assertEqual(k2.tonic.name, "C")
        self.assertEqual(k2.mode, "major")

    def test_midi_folder_loading(self):
        try:
            import pretty_midi
        except ImportError:
            self.skipTest("pretty_midi not installed")

        with tempfile.TemporaryDirectory() as midi_dir:
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=33)
            inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.5))
            pm.instruments.append(inst)

            bass_midi_path = os.path.join(midi_dir, "bass.midi")
            pm.write(bass_midi_path)

            streams = load_midi_folder_to_event_streams(midi_dir)
            self.assertIn("bass", streams)
            events = streams["bass"]["events"]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].pitch, 36)


if __name__ == "__main__":
    unittest.main()
