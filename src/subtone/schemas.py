import copy
from dataclasses import dataclass, field
import fractions
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except ImportError:
    class ValidationError(Exception):
        pass

    def Field(default=..., default_factory=None, **kwargs):
        if default_factory is not None:
            return field(default_factory=default_factory)
        if default is not ...:
            return field(default=default)
        return field()

    def ConfigDict(**kwargs):
        return kwargs

    _dataclass_field_type = type(field())

    class BaseModel:
        model_fields = {}

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            mf = {}
            for base in reversed(cls.__mro__):
                for k in getattr(base, "__annotations__", {}).keys():
                    mf[k] = None
            cls.model_fields = mf

        def __init__(self, **data):
            cls = self.__class__
            if not cls.model_fields:
                mf = {}
                for base in reversed(cls.__mro__):
                    for k in getattr(base, "__annotations__", {}).keys():
                        mf[k] = None
                cls.model_fields = mf

            for name, typ in cls.model_fields.items():
                if name in data:
                    setattr(self, name, data[name])
                else:
                    val = getattr(cls, name, None)
                    if hasattr(val, "default_factory") and val.default_factory is not None:
                        setattr(self, name, val.default_factory())
                    elif hasattr(val, "default") and val.default is not ...:
                        setattr(self, name, copy.deepcopy(val.default))
                    elif not hasattr(val, "default_factory") and not isinstance(val, _dataclass_field_type) and val is not None:
                        setattr(self, name, copy.deepcopy(val))
                    else:
                        setattr(self, name, None)

            for k, v in data.items():
                if not hasattr(self, k):
                    setattr(self, k, v)

        def keys(self):
            return [k for k in self.__dict__.keys() if not k.startswith("_")]

        def items(self):
            return [(k, v) for k, v in self.__dict__.items() if not k.startswith("_")]

        def values(self):
            return [v for k, v in self.__dict__.items() if not k.startswith("_")]

        def __getitem__(self, item):
            return getattr(self, item)

        def get(self, key, default=None):
            return getattr(self, key, default)

        def __contains__(self, item):
            return hasattr(self, item) and not item.startswith("_")

        def __len__(self):
            return len(self.keys())

        def __iter__(self):
            return iter(self.keys())

        def model_dump(self, *args, **kwargs):
            res = {}
            for k, v in self.__dict__.items():
                if not k.startswith("_"):
                    if hasattr(v, "model_dump"):
                        res[k] = v.model_dump(*args, **kwargs)
                    elif isinstance(v, list):
                        res[k] = [x.model_dump(*args, **kwargs) if hasattr(x, "model_dump") else x for x in v]
                    elif isinstance(v, dict):
                        res[k] = {dk: (dv.model_dump(*args, **kwargs) if hasattr(dv, "model_dump") else dv) for dk, dv in v.items()}
                    else:
                        res[k] = v
            return res

        def dict(self, *args, **kwargs):
            return self.model_dump(*args, **kwargs)

        def copy(self, update=None, deep=False):
            c = copy.deepcopy(self) if deep else copy.copy(self)
            if update:
                for k, v in update.items():
                    setattr(c, k, v)
            return c


class AudioEvent(BaseModel):
    model_config = ConfigDict(
        strict=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",
    )

    start: float
    end: float
    pitch: int
    engine: str = "torchcrepe"  # e.g., torchcrepe, basicPitch, pyin, etc.
    pitches: list[int] = Field(default_factory=list)
    amplitude: float = 0.5
    bends: list[float] = Field(default_factory=list)
    microtone_cents: float = 0.0
    tag: str = "normal"  # normal, staccato, slap, pop, palm_mute, ghost, harmonic, hammer_on, pull_off, slide
    duty_cycle: float = 1.0
    is_triplet: bool = False
    is_accent: bool = False
    dynamic_mark: str = "mf"  # p, mp, mf, f
    is_pickup: bool = False
    is_harmonic: bool = False
    slide_from: int | None = None

    # Rest & Tie status flags
    is_rest: bool = False
    is_tied_start: bool = False
    is_tied_stop: bool = False
    is_tied_continue: bool = False

    # Articulation & Technique Flags
    is_rake: bool = False
    is_legato: bool = False
    is_slide: bool = False
    is_slap: bool = False
    is_pop: bool = False
    is_ghost: bool = False
    is_palm_mute: bool = False
    is_staccato: bool = False
    is_tenuto: bool = False
    is_fermata: bool = False

    # Finger position assignment (String, Fret, Finger)
    fret_position: tuple[int, int, int] | None = None
    string: int | None = None
    fret: int | None = None
    is_downpick: bool = False

    # Category and Anchor Pattern Encoding Attributes
    category: str = "melodic"  # groove_anchor, percussive, expressive, melodic
    anchor_pattern: str | None = None
    anchor_fret: int | None = None
    is_anchor: bool = False
    confidence: float = 1.0

    def model_post_init(self, __context: Any) -> None:
        if not self.pitches:
            self.pitches = [self.pitch]
        self.sync_flags()
        if self.category == "melodic":
            self.determine_category()

    def sync_flags(self) -> None:
        """Synchronizes performance tag string and boolean technique flags."""
        if self.tag == "slap":
            self.is_slap = True
        elif self.tag == "pop":
            self.is_pop = True
        elif self.tag == "ghost":
            self.is_ghost = True
        elif self.tag == "palm_mute":
            self.is_palm_mute = True
        elif self.tag == "staccato":
            self.is_staccato = True
        elif self.tag == "harmonic":
            self.is_harmonic = True
        elif self.tag == "slide":
            self.is_slide = True

        if getattr(self, "is_slap", False) and self.tag in ["normal", "rest"]:
            self.tag = "slap"
        elif getattr(self, "is_pop", False) and self.tag in ["normal", "rest"]:
            self.tag = "pop"
        elif getattr(self, "is_ghost", False) and self.tag in ["normal", "rest"]:
            self.tag = "ghost"
        elif getattr(self, "is_palm_mute", False) and self.tag in ["normal", "rest"]:
            self.tag = "palm_mute"
        elif getattr(self, "is_staccato", False) and self.tag in ["normal", "rest"]:
            self.tag = "staccato"
        elif getattr(self, "is_harmonic", False) and self.tag in ["normal", "rest"]:
            self.tag = "harmonic"
        elif getattr(self, "is_slide", False) and self.tag in ["normal", "rest"]:
            self.tag = "slide"

    @classmethod
    def make_rest(cls, start: float, end: float, engine: str = "torchcrepe") -> "AudioEvent":
        """Constructs an explicit Rest event spanning start to end time."""
        return cls(
            start=float(start),
            end=float(end),
            pitch=0,
            engine=engine,
            pitches=[0],
            amplitude=0.0,
            is_rest=True,
            tag="rest",
            category="rest",
        )

    @property
    def engine_type(self) -> str:
        return self.engine

    @engine_type.setter
    def engine_type(self, value: str):
        self.engine = value

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def update_pitch(self, new_pitch: int):
        self.pitch = new_pitch
        self.pitches = [new_pitch]

    def determine_category(self) -> str:
        """Categorizes audio events based on performance tags, dynamics, and structural anchor attributes."""
        if (
            self.tag in ["ghost", "slap", "pop", "palm_mute", "staccato"]
            or self.is_ghost
            or self.is_slap
            or self.is_pop
            or self.is_staccato
        ):
            self.category = "percussive"
        elif (
            self.tag in ["hammer_on", "pull_off", "slide"]
            or self.is_harmonic
            or len(self.bends) > 0
            or abs(self.microtone_cents) > 10.0
            or self.is_slide
            or self.is_legato
            or self.is_rake
        ):
            self.category = "expressive"
        elif self.is_anchor or self.is_pickup or self.is_accent:
            self.category = "groove_anchor"
        else:
            self.category = "melodic"
        return self.category

    def to_dict(self) -> dict:
        d = self.model_dump()
        d["duration"] = self.duration
        d["engine"] = self.engine
        d["engine_type"] = self.engine
        return d

    def clone(self, **overrides) -> "AudioEvent":
        """Creates a deep copy of the AudioEvent instance with optional field overrides."""
        copied = copy.deepcopy(self)
        for key, value in overrides.items():
            setattr(copied, key, value)
        return copied

    def copy(self, **overrides) -> "AudioEvent":
        """Alias for clone()."""
        return self.clone(**overrides)


class Note(AudioEvent):
    """
    A Note element represents a logical musical note inside a Bassline.
    Unlike AudioEvent which represents raw onset/offset transcriptions,
    Note element has an explicit decided duration to avoid micro rests
    at later pipeline stages.
    """

    associated_events: list[AudioEvent] = Field(default_factory=list)

    @property
    def original_event(self) -> AudioEvent | None:
        """Returns the first/primary associated AudioEvent for calibration and timing queries."""
        return self.associated_events[0] if self.associated_events else None

    @classmethod
    def make_rest(cls, start: float, end: float, engine: str = "torchcrepe") -> "Note":
        """Construct an explicit rest without creating a throwaway AudioEvent."""
        return cls(
            start=float(start),
            end=float(end),
            pitch=0,
            engine=engine,
            pitches=[0],
            amplitude=0.0,
            is_rest=True,
            tag="rest",
            category="rest",
        )

    @classmethod
    def from_event(cls, event: AudioEvent) -> "Note":
        """Create an editable Note while retaining its source-event provenance."""
        if not isinstance(event, AudioEvent):
            raise TypeError(f"Note.from_event requires AudioEvent, got {type(event).__name__}")

        values = {field_name: copy.copy(getattr(event, field_name)) for field_name in AudioEvent.model_fields.keys()}
        source_events = (
            list(event.associated_events) if isinstance(event, Note) and event.associated_events else [event]
        )
        return cls(**values, associated_events=source_events)

    def clone(self, **overrides) -> "Note":
        """Copy editable note state without deep-copying immutable source provenance."""
        values = {field_name: copy.copy(getattr(self, field_name)) for field_name in AudioEvent.model_fields.keys()}
        values.update(overrides)
        if "associated_events" in overrides:
            values["associated_events"] = overrides["associated_events"]
        else:
            values["associated_events"] = list(self.associated_events)
        return type(self)(**values)


class RhythmicAtom(BaseModel):
    """
    The fundamental, indivisible notation building block representing a discrete visual
    sound duration (e.g., a sixteenth note, an eighth note, a dotted quarter) anchored
    to standard engraver rules.
    """

    model_config = ConfigDict(
        strict=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",
    )

    pitch: int  # MIDI pitch (0 for rest)
    duration_q: fractions.Fraction  # Fraction duration in quarter-note units
    start_q: fractions.Fraction = fractions.Fraction(0, 1)  # Metric start in measure/song quarters
    measure_index: int = 1
    is_rest: bool = False
    tie_type: str | None = None  # None, 'start', 'continue', 'stop'

    # Fingering / Tablature Coordinates
    string_num: int | None = None
    fret_num: int | None = None
    finger_num: int | None = None

    # Provenance
    parent_event_id: Any | None = None  # ID or reference of parent thick Note/AudioEvent
    source_note: Note | None = None  # Original Note instance

    # Notation & Engraver Details
    articulations: list[str] = Field(default_factory=list)  # e.g., ['accent', 'staccato', 'tenuto']
    expressions: list[str] = Field(default_factory=list)  # e.g., ['slap', 'pop', 'palm_mute', 'fermata']
    notehead: str = "normal"  # 'normal', 'x', 'diamond'
    notehead_parenthesis: bool = False
    amplitude: float = 0.5
    dynamic_mark: str | None = None
    is_triplet: bool = False

    def model_post_init(self, __context: Any) -> None:
        if not isinstance(self.duration_q, fractions.Fraction):
            self.duration_q = fractions.Fraction(self.duration_q).limit_denominator(64)
        if not isinstance(self.start_q, fractions.Fraction):
            self.start_q = fractions.Fraction(self.start_q).limit_denominator(64)
        if self.is_rest:
            self.pitch = 0

    @classmethod
    def from_note(
        cls,
        note_obj: Note | None,
        duration_q: fractions.Fraction,
        measure_index: int = 1,
        start_q: fractions.Fraction = fractions.Fraction(0, 1),
        is_rest: bool = False,
        tie_type: str | None = None,
    ) -> "RhythmicAtom":
        if is_rest or note_obj is None or getattr(note_obj, "is_rest", False):
            return cls(
                pitch=0,
                duration_q=duration_q,
                start_q=start_q,
                measure_index=measure_index,
                is_rest=True,
                tie_type=None,
                source_note=note_obj,
                parent_event_id=id(note_obj) if note_obj else None,
            )

        fret_pos = getattr(note_obj, "fret_position", None)
        string_n = fret_pos[0] if fret_pos else None
        fret_n = fret_pos[1] if fret_pos else None
        finger_n = fret_pos[2] if fret_pos and len(fret_pos) > 2 else None

        articulations = []
        if getattr(note_obj, "is_accent", False) or getattr(note_obj, "tag", "") == "accent":
            articulations.append("accent")
        if getattr(note_obj, "is_staccato", False) or getattr(note_obj, "tag", "") == "staccato":
            articulations.append("staccato")
        if getattr(note_obj, "is_tenuto", False):
            articulations.append("tenuto")

        expressions = []
        if getattr(note_obj, "is_fermata", False):
            expressions.append("fermata")
        if getattr(note_obj, "is_slap", False) or getattr(note_obj, "tag", "") == "slap":
            expressions.append("slap")
        elif getattr(note_obj, "is_pop", False) or getattr(note_obj, "tag", "") == "pop":
            expressions.append("pop")
        elif getattr(note_obj, "is_palm_mute", False) or getattr(note_obj, "tag", "") == "palm_mute":
            expressions.append("palm_mute")

        nh = "normal"
        nh_parent = False
        if getattr(note_obj, "is_ghost", False) or getattr(note_obj, "tag", "") == "ghost":
            nh = "x"
            nh_parent = True
            if "staccato" not in articulations:
                articulations.append("staccato")
        elif getattr(note_obj, "is_harmonic", False) or getattr(note_obj, "tag", "") == "harmonic":
            nh = "diamond"

        return cls(
            pitch=note_obj.pitch,
            duration_q=duration_q,
            start_q=start_q,
            measure_index=measure_index,
            is_rest=False,
            tie_type=tie_type,
            string_num=string_n,
            fret_num=fret_n,
            finger_num=finger_n,
            parent_event_id=id(note_obj),
            source_note=note_obj,
            articulations=articulations,
            expressions=expressions,
            notehead=nh,
            notehead_parenthesis=nh_parent,
            amplitude=getattr(note_obj, "amplitude", 0.5),
            dynamic_mark=getattr(note_obj, "dynamic_mark", None),
            is_triplet=getattr(note_obj, "is_triplet", False),
        )

    def clone(self, **overrides) -> "RhythmicAtom":
        values = {
            "pitch": self.pitch,
            "duration_q": self.duration_q,
            "start_q": self.start_q,
            "measure_index": self.measure_index,
            "is_rest": self.is_rest,
            "tie_type": self.tie_type,
            "string_num": self.string_num,
            "fret_num": self.fret_num,
            "finger_num": self.finger_num,
            "parent_event_id": self.parent_event_id,
            "source_note": self.source_note,
            "articulations": list(self.articulations),
            "expressions": list(self.expressions),
            "notehead": self.notehead,
            "notehead_parenthesis": self.notehead_parenthesis,
            "amplitude": self.amplitude,
            "dynamic_mark": self.dynamic_mark,
            "is_triplet": self.is_triplet,
        }
        values.update(overrides)
        return RhythmicAtom(**values)


@dataclass
class MeasureChunk:
    """
    A temporal, windowed slice of time corresponding to the exact duration of a single bar
    within a piece of music. Acts as an isolated processing boundary during audio engine
    and transcription stages, and as an intermediate measure container for score generation.
    """

    measure_index: int = 1
    start_time: float = 0.0
    end_time: float = 0.0
    start_q: fractions.Fraction = fractions.Fraction(0, 1)
    end_q: fractions.Fraction = fractions.Fraction(4, 1)
    measure_capacity: fractions.Fraction = fractions.Fraction(4, 1)
    events: list[Any] = field(default_factory=list)
    atoms: list[RhythmicAtom] = field(default_factory=list)
    bpm: float = 120.0
    time_sig: str = "4/4"
    is_compound: bool = False
    subdivisions: int = 4

    # Backward compatibility attributes for single-atom or split-event wrapper instantiation
    # e.g., MeasureChunk(measure_num, event, duration_q, is_rest, tie_type)
    _legacy_event: Any | None = None
    _legacy_duration_q: fractions.Fraction | None = None
    _legacy_is_rest: bool = False
    _legacy_tie_type: str | None = None

    def __init__(
        self,
        measure_index: int = 1,
        event_or_start: Any | None = None,
        duration_q_or_end: Any | None = None,
        is_rest: bool = False,
        tie_type: str | None = None,
        **kwargs,
    ):
        if isinstance(event_or_start, (Note, AudioEvent, type(None))) and (
            isinstance(duration_q_or_end, (fractions.Fraction, int, float)) or duration_q_or_end is None
        ):
            self.measure_index = measure_index
            self._legacy_event = event_or_start
            self._legacy_duration_q = (
                fractions.Fraction(duration_q_or_end) if duration_q_or_end is not None else fractions.Fraction(0, 1)
            )
            self._legacy_is_rest = is_rest or (event_or_start is None or getattr(event_or_start, "is_rest", False))
            self._legacy_tie_type = tie_type
            self.events = [event_or_start] if event_or_start else []
            self.atoms = []

            atom = RhythmicAtom.from_note(
                note_obj=event_or_start if isinstance(event_or_start, Note) else None,
                duration_q=self._legacy_duration_q,
                measure_index=measure_index,
                is_rest=self._legacy_is_rest,
                tie_type=self._legacy_tie_type,
            )
            if not isinstance(event_or_start, Note) and event_or_start is not None:
                atom.pitch = getattr(event_or_start, "pitch", 0)
                atom.source_note = event_or_start
            self.atoms.append(atom)
            self.measure_capacity = kwargs.get("measure_capacity", fractions.Fraction(4, 1))
            self.start_time = kwargs.get("start_time", 0.0)
            self.end_time = kwargs.get("end_time", 0.0)
            self.start_q = kwargs.get("start_q", fractions.Fraction(0, 1))
            self.end_q = kwargs.get("end_q", self.measure_capacity)
            self.bpm = kwargs.get("bpm", 120.0)
            self.time_sig = kwargs.get("time_sig", "4/4")
            self.is_compound = kwargs.get("is_compound", False)
            self.subdivisions = kwargs.get("subdivisions", 4)
        else:
            self.measure_index = measure_index
            self.start_time = float(event_or_start) if event_or_start is not None else kwargs.get("start_time", 0.0)
            self.end_time = float(duration_q_or_end) if duration_q_or_end is not None else kwargs.get("end_time", 0.0)
            self.measure_capacity = kwargs.get("measure_capacity", fractions.Fraction(4, 1))
            self.start_q = kwargs.get("start_q", fractions.Fraction(0, 1))
            self.end_q = kwargs.get("end_q", self.measure_capacity)
            self.events = list(kwargs.get("events", []))
            self.atoms = list(kwargs.get("atoms", []))
            self.bpm = kwargs.get("bpm", 120.0)
            self.time_sig = kwargs.get("time_sig", "4/4")
            self.is_compound = kwargs.get("is_compound", False)
            self.subdivisions = kwargs.get("subdivisions", 4)
            self._legacy_event = None
            self._legacy_duration_q = None
            self._legacy_is_rest = False
            self._legacy_tie_type = None

    @property
    def measure_num(self) -> int:
        return self.measure_index

    @measure_num.setter
    def measure_num(self, val: int):
        self.measure_index = val

    @property
    def event(self) -> Any | None:
        if self._legacy_event is not None:
            return self._legacy_event
        if self.atoms and self.atoms[0].source_note is not None:
            return self.atoms[0].source_note
        if self.events:
            return self.events[0]
        return None

    @event.setter
    def event(self, val: Any):
        self._legacy_event = val

    @property
    def duration_q(self) -> fractions.Fraction:
        if self._legacy_duration_q is not None:
            return self._legacy_duration_q
        if self.atoms:
            return sum((a.duration_q for a in self.atoms), fractions.Fraction(0, 1))
        return fractions.Fraction(0, 1)

    @duration_q.setter
    def duration_q(self, val: Any):
        self._legacy_duration_q = fractions.Fraction(val)

    @property
    def is_rest(self) -> bool:
        if self._legacy_event is not None or self._legacy_duration_q is not None:
            return self._legacy_is_rest
        if self.atoms:
            return all(a.is_rest for a in self.atoms)
        return True

    @is_rest.setter
    def is_rest(self, val: bool):
        self._legacy_is_rest = val

    @property
    def tie_type(self) -> str | None:
        if self._legacy_event is not None or self._legacy_duration_q is not None:
            return self._legacy_tie_type
        if self.atoms:
            return self.atoms[0].tie_type
        return None

    @tie_type.setter
    def tie_type(self, val: str | None):
        self._legacy_tie_type = val

    def add_atom(self, atom: RhythmicAtom):
        self.atoms.append(atom)

    def add_event(self, event: Any):
        self.events.append(event)


@dataclass
class MusicalPhrase:
    phrase_id: int
    start_time: float
    end_time: float
    notes: list[Note] = field(default_factory=list)
    key: str | None = None
    harmony: str | None = None


class Song:
    """
    Representation of a song and its transcribed bass event streams/notes.
    Maintains bass_audio_events and bass_notes as primary collections.
    """

    def __init__(
        self,
        artist_name: str = "Unknown Artist",
        song_title: str = "Untitled Track",
        genres: list[str] | None = None,
        genre_config: Any = None,
        parsed_key_str: str | None = None,
        key_obj: Any = None,
        stem_folder: str = "",
        bpm: float = 120.0,
        time_sig: str = "4/4",
        is_compound: bool = False,
        tuning_type: str = "4_string_standard",
        target_level: int = 5,
        engine: str = "pyin",
        sr: int = 22050,
        bass_audio_events: list[AudioEvent] | None = None,
        bass_notes: list[Note] | None = None,
        **kwargs,
    ):
        self.artist_name = artist_name
        self.song_title = song_title
        self.genres = genres or []
        self.genre_config = genre_config
        self.parsed_key_str = parsed_key_str
        self.key_obj = key_obj
        if self.key_obj is None and self.parsed_key_str:
            try:
                from subtone.pitch_theory import parse_key_object

                self.key_obj = parse_key_object(self.parsed_key_str)
            except Exception:
                pass

        self.stem_folder = stem_folder
        self.bpm = bpm
        self.time_sig = time_sig
        self.time_signature = time_sig
        self.is_compound = is_compound
        self.tuning_type = tuning_type
        self.target_level = target_level
        self.engine = engine
        self.sr = sr
        self.output_xml_path = kwargs.get("output_xml_path", "")
        self.fretboard_path = kwargs.get("fretboard_path", [])
        self.rakes = kwargs.get("rakes", [])
        self.legatos = kwargs.get("legatos", [])
        self.slides = kwargs.get("slides", [])
        self.measures = kwargs.get("measures", [])
        self.phrases = kwargs.get("phrases", [])

        events = bass_audio_events or kwargs.get("audio_events") or kwargs.get("source_events") or []
        self._bass_audio_events = [e if isinstance(e, AudioEvent) else AudioEvent(**e) for e in events]

        if bass_notes is not None:
            self._bass_notes = [
                n if isinstance(n, Note) else (Note.from_event(n) if isinstance(n, AudioEvent) else Note(**n))
                for n in bass_notes
            ]
        else:
            self._bass_notes = [Note.from_event(e) for e in self._bass_audio_events]

        # Fold pitches to active tuning bounds to prevent out-of-range sub-bass artifacts
        try:
            from subtone.pitch_theory import fold_pitch_to_bass_range
            from subtone.settings_loader import STANDARD_BASS_TUNING_MIDIS

            tuning_midis = STANDARD_BASS_TUNING_MIDIS.get(self.tuning_type, [28, 33, 38, 43])
            min_p = min(tuning_midis)
            max_p = max(tuning_midis) + 24

            for evt in self._bass_audio_events:
                if getattr(evt, "pitch", 0) > 0:
                    folded = fold_pitch_to_bass_range(evt.pitch, min_pitch=min_p, max_pitch=max_p)
                    evt.update_pitch(folded)

            for n in self._bass_notes:
                if getattr(n, "pitch", 0) > 0:
                    folded = fold_pitch_to_bass_range(n.pitch, min_pitch=min_p, max_pitch=max_p)
                    n.update_pitch(folded)
        except Exception:
            pass

    @property
    def bass_audio_events(self) -> list[AudioEvent]:
        return self._bass_audio_events

    @bass_audio_events.setter
    def bass_audio_events(self, value: list[AudioEvent]):
        self._bass_audio_events = [e if isinstance(e, AudioEvent) else AudioEvent(**e) for e in value]

    @property
    def bassAudioEvents(self) -> list[AudioEvent]:
        return self._bass_audio_events

    @bassAudioEvents.setter
    def bassAudioEvents(self, value: list[AudioEvent]):
        self.bass_audio_events = value

    @property
    def audioEvents(self) -> list[AudioEvent]:
        return self._bass_audio_events

    @audioEvents.setter
    def audioEvents(self, value: list[AudioEvent]):
        self.bass_audio_events = value

    @property
    def audio_events(self) -> list[AudioEvent]:
        return self._bass_audio_events

    @audio_events.setter
    def audio_events(self, value: list[AudioEvent]):
        self.bass_audio_events = value

    @property
    def bass_notes(self) -> list[Note]:
        return self._bass_notes

    @bass_notes.setter
    def bass_notes(self, value: list[Note]):
        self._bass_notes = [
            n if isinstance(n, Note) else (Note.from_event(n) if isinstance(n, AudioEvent) else Note(**n))
            for n in value
        ]

    @property
    def bassNotes(self) -> list[Note]:
        return self._bass_notes

    @bassNotes.setter
    def bassNotes(self, value: list[Note]):
        self.bass_notes = value

    @property
    def notes(self) -> list[Note]:
        return self._bass_notes

    @notes.setter
    def notes(self, value: list[Note]):
        self.bass_notes = value

    def replace_notes(self, new_notes: list[Note]):
        self.bass_notes = new_notes

    @property
    def beat_times(self) -> list[float]:
        if hasattr(self, "_beat_times") and self._beat_times is not None:
            return self._beat_times
        if self.bpm > 0:
            bpm_val = float(self.bpm) if self.bpm and self.bpm > 0 else 120.0
            sec_per_beat = 60.0 / bpm_val
            max_end = max((n.end for n in self.bass_notes), default=60.0)
            return [i * sec_per_beat for i in range(int(max_end / sec_per_beat) + 2)]
        return []

    @beat_times.setter
    def beat_times(self, value: list[float] | None):
        self._beat_times = value

    @classmethod
    def from_transcription(
        cls,
        source_events: list[AudioEvent],
        beat_times: list[float] | None = None,
        bpm: float = 120.0,
        time_sig: str = "4/4",
        is_compound: bool = False,
        key_obj: Any = None,
        artist_name: str = "Unknown Artist",
        song_title: str = "Untitled Track",
        tuning_type: str = "4_string_standard",
        target_level: int = 5,
        engine: str = "pyin",
        **kwargs,
    ) -> "Song":
        audio_events = [e if isinstance(e, AudioEvent) else AudioEvent(**e) for e in source_events]
        notes = [Note.from_event(e) for e in audio_events]

        song = cls(
            artist_name=artist_name,
            song_title=song_title,
            bpm=bpm,
            time_sig=time_sig,
            is_compound=is_compound,
            key_obj=key_obj,
            tuning_type=tuning_type,
            target_level=target_level,
            engine=engine,
            bass_audio_events=audio_events,
            bass_notes=notes,
            **kwargs,
        )
        if beat_times and not song.measures:
            song.measures = list(range(1, len(beat_times) // 4 + 2))
        return song

    @classmethod
    def from_event_streams(
        cls,
        event_streams: dict[str, dict[str, Any]],
        active_stream_name: str | None = None,
        artist_name: str = "Unknown Artist",
        song_title: str = "Untitled Track",
        genres: list[str] | None = None,
        genre_config: Any = None,
        parsed_key_str: str | None = None,
        stem_folder: str = "",
        **kwargs,
    ) -> "Song":
        selected_key = active_stream_name
        if not selected_key or selected_key not in event_streams:
            selected_key = next(
                (k for k in event_streams if "bass" in k or event_streams[k].get("stream_type") == "primary"),
                None,
            )
            if not selected_key and event_streams:
                selected_key = list(event_streams.keys())[0]

        stream_data = event_streams.get(selected_key, {}) if event_streams else {}
        events = (
            stream_data.get("events")
            or stream_data.get("bassAudioEvents")
            or stream_data.get("bass_audio_events")
            or []
        )
        metadata = stream_data.get("metadata", {})

        bpm = metadata.get("bpm", 120.0)
        time_sig = metadata.get("time_sig", "4/4")
        is_compound = metadata.get("is_compound", False)
        engine = stream_data.get("engine") or metadata.get("engine", "pyin")

        beat_times = metadata.get("beat_times", [])
        song = cls(
            artist_name=artist_name,
            song_title=song_title,
            genres=genres or [],
            genre_config=genre_config,
            parsed_key_str=parsed_key_str,
            stem_folder=stem_folder,
            bpm=bpm,
            time_sig=time_sig,
            is_compound=is_compound,
            engine=engine,
            bass_audio_events=events,
            **kwargs,
        )
        if beat_times:
            song.beat_times = beat_times
            if not song.measures:
                song.measures = list(range(1, max(1, len(beat_times) // 4 + 1)))
        return song

    def to_dict(self) -> dict[str, Any]:
        return {
            "artist_name": self.artist_name,
            "song_title": self.song_title,
            "genres": self.genres,
            "bpm": self.bpm,
            "time_sig": self.time_sig,
            "is_compound": self.is_compound,
            "tuning_type": self.tuning_type,
            "target_level": self.target_level,
            "engine": self.engine,
            "bassAudioEvents": [e.to_dict() for e in self.bass_audio_events],
            "bassNotes": [n.to_dict() for n in self.bass_notes],
        }


class Level(BaseModel):
    model_config = ConfigDict(
        strict=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",
    )

    level_id: int
    name: str
    description: str
    ghost_notes_allowed: bool
    min_duration: float
    snaps_to_scale: bool
    downbeat_only: bool
    enabled_techniques: list[str] = Field(default_factory=list)
    enabled_articulations: list[str] = Field(default_factory=list)

    @property
    def ghost_notes(self) -> bool:
        return self.ghost_notes_allowed

    @classmethod
    def from_id(cls, level_id: int, level_profile: dict = None) -> "Level":
        clamped_id = max(0, min(5, level_id))

        if level_profile and "levels" in level_profile:
            levels_dict = level_profile["levels"]
            level_cfg = levels_dict.get(clamped_id) or levels_dict.get(str(clamped_id))
            if level_cfg:
                return cls(
                    level_id=clamped_id,
                    name=level_cfg.get("name", f"Level {clamped_id}"),
                    description=level_cfg.get("description", ""),
                    ghost_notes_allowed=level_cfg.get("ghost_notes_allowed", True),
                    min_duration=float(level_cfg.get("min_duration", 0.0)),
                    snaps_to_scale=level_cfg.get("snaps_to_scale", False),
                    downbeat_only=level_cfg.get("downbeat_only", False),
                    enabled_techniques=level_cfg.get(
                        "enabled_techniques",
                        ["fingerstyle", "plectrum", "slap_pop", "thumb_mute", "upright_sim", "synth_emulation"],
                    ),
                    enabled_articulations=level_cfg.get(
                        "enabled_articulations",
                        [
                            "normal",
                            "staccato",
                            "slap",
                            "pop",
                            "palm_mute",
                            "ghost",
                            "harmonic",
                            "hammer_on",
                            "pull_off",
                            "slide",
                            "accent",
                        ],
                    ),
                )

        configs = {
            0: {
                "name": "Minimalist Roots",
                "description": "Downbeats and half-measure anchors only, creating a highly spacious foundational root note layout.",
                "ghost_notes_allowed": False,
                "min_duration": 0.20,
                "snaps_to_scale": True,
                "downbeat_only": True,
                "enabled_techniques": [
                    "fingerstyle",
                    "plectrum",
                    "slap_pop",
                    "thumb_mute",
                    "upright_sim",
                    "synth_emulation",
                ],
                "enabled_articulations": ["normal", "accent"],
            },
            1: {
                "name": "Fundamental Anchors",
                "description": "Retains core groove anchors and primary subdivisions to establish the main structural chord progression.",
                "ghost_notes_allowed": False,
                "min_duration": 0.20,
                "snaps_to_scale": True,
                "downbeat_only": False,
                "enabled_techniques": [
                    "fingerstyle",
                    "plectrum",
                    "slap_pop",
                    "thumb_mute",
                    "upright_sim",
                    "synth_emulation",
                ],
                "enabled_articulations": ["normal", "accent"],
            },
            2: {
                "name": "Laid-Back Simple",
                "description": "Brings in eighth-note pulses and on-beat subdivisions, filtering out complex rapid fills and ghost notes.",
                "ghost_notes_allowed": False,
                "min_duration": 0.20,
                "snaps_to_scale": False,
                "downbeat_only": False,
                "enabled_techniques": [
                    "fingerstyle",
                    "plectrum",
                    "slap_pop",
                    "thumb_mute",
                    "upright_sim",
                    "synth_emulation",
                ],
                "enabled_articulations": ["normal", "staccato", "accent"],
            },
            3: {
                "name": "Authentic Direct",
                "description": "Original transcription minus soft percussive clicks and ghost notes, resulting in a clean melodic line.",
                "ghost_notes_allowed": False,
                "min_duration": 0.12,
                "snaps_to_scale": False,
                "downbeat_only": False,
                "enabled_techniques": [
                    "fingerstyle",
                    "plectrum",
                    "slap_pop",
                    "thumb_mute",
                    "upright_sim",
                    "synth_emulation",
                ],
                "enabled_articulations": [
                    "normal",
                    "staccato",
                    "slap",
                    "pop",
                    "palm_mute",
                    "harmonic",
                    "hammer_on",
                    "pull_off",
                    "slide",
                    "accent",
                ],
            },
            4: {
                "name": "Unfiltered Dynamic",
                "description": "Matches the original recording's full notation, including syncopated microtones and selective ghost notes.",
                "ghost_notes_allowed": True,
                "min_duration": 0.0,
                "snaps_to_scale": False,
                "downbeat_only": False,
                "enabled_techniques": [
                    "fingerstyle",
                    "plectrum",
                    "slap_pop",
                    "thumb_mute",
                    "upright_sim",
                    "synth_emulation",
                ],
                "enabled_articulations": [
                    "normal",
                    "staccato",
                    "slap",
                    "pop",
                    "palm_mute",
                    "ghost",
                    "harmonic",
                    "hammer_on",
                    "pull_off",
                    "slide",
                    "accent",
                ],
            },
            5: {
                "name": "Complete Original",
                "description": "The exact high-fidelity transcription featuring all expressive micro-dynamics, slides, and ghost notes.",
                "ghost_notes_allowed": True,
                "min_duration": 0.0,
                "snaps_to_scale": False,
                "downbeat_only": False,
                "enabled_techniques": [
                    "fingerstyle",
                    "plectrum",
                    "slap_pop",
                    "thumb_mute",
                    "upright_sim",
                    "synth_emulation",
                ],
                "enabled_articulations": [
                    "normal",
                    "staccato",
                    "slap",
                    "pop",
                    "palm_mute",
                    "ghost",
                    "harmonic",
                    "hammer_on",
                    "pull_off",
                    "slide",
                    "accent",
                ],
            },
        }

        cfg = configs[clamped_id]
        return cls(
            level_id=clamped_id,
            name=cfg["name"],
            description=cfg["description"],
            ghost_notes_allowed=cfg["ghost_notes_allowed"],
            min_duration=float(cfg["min_duration"]),
            snaps_to_scale=cfg["snaps_to_scale"],
            downbeat_only=cfg["downbeat_only"],
            enabled_techniques=cfg["enabled_techniques"],
            enabled_articulations=cfg["enabled_articulations"],
        )

    def to_dict(self) -> dict:
        return self.model_dump()

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        d = self.to_dict()
        return d.get(key, default)


class Genre(BaseModel):
    model_config = ConfigDict(
        strict=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",
    )

    name: str = "default"
    extends: str = "default"
    tuning: str = "4_string_standard"
    technique: str = "fingerstyle"
    rhythmic_grid: str = "16th_syncopated"
    rhythmic_anchor: dict[str, Any] = Field(
        default_factory=lambda: {
            "pattern": "downbeat_one",
            "lock_strength": "moderate",
        }
    )
    features: dict[str, bool] = Field(
        default_factory=lambda: {
            "ghost_notes": True,
            "compound_meter": False,
            "downpicking_preference": False,
            "synth_emulation": False,
        }
    )
    costs: dict[str, float] = Field(
        default_factory=lambda: {
            "pop_non_treble_penalty": 25.0,
            "slap_non_bass_penalty": 18.0,
            "fret_stretch_penalty": 10.0,
            "position_shift_multiplier": 2.0,
            "open_string_bonus": -1.5,
        }
    )
    micro_timing: dict[str, Any] = Field(default_factory=dict)
    articulation_intent: dict[str, Any] = Field(default_factory=dict)
    fretboard_navigation: dict[str, Any] = Field(default_factory=dict)
    notation_engraving: dict[str, Any] = Field(default_factory=dict)
    harmonic_hysteresis: dict[str, Any] = Field(default_factory=dict)
    cross_stem_validation: dict[str, Any] = Field(default_factory=dict)
    level_profile: dict[Any, Any] = Field(default_factory=dict)
    structural: dict[str, Any] = Field(
        default_factory=lambda: {
            "phrase_length_measures": 8,
            "cross_section_healing": True,
        }
    )
    sub_genres: list[Any] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if isinstance(self.rhythmic_anchor, str):
            self.rhythmic_anchor = {"pattern": self.rhythmic_anchor}

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "Genre":
        if "tuning" in data and not isinstance(data["tuning"], str):
            raise TypeError(f"tuning must be a string, got {type(data['tuning']).__name__}")
        if "preferred_key_signatures" in data and not isinstance(data["preferred_key_signatures"], list):
            raise TypeError(
                f"preferred_key_signatures must be a list, got {type(data['preferred_key_signatures']).__name__}"
            )
        rhythmic_anchor = data.get(
            "rhythmic_anchor",
            {"pattern": "downbeat_one", "lock_strength": "moderate"},
        )
        if isinstance(rhythmic_anchor, str):
            rhythmic_anchor = {"pattern": rhythmic_anchor}

        try:
            return cls(
                name=data.get("name", name),
                extends=data.get("extends", "default"),
                tuning=data.get("tuning", "4_string_standard"),
                technique=data.get("technique", "fingerstyle"),
                rhythmic_grid=data.get("rhythmic_grid", "16th_syncopated"),
                rhythmic_anchor=rhythmic_anchor,
                features=data.get(
                    "features",
                    {
                        "ghost_notes": True,
                        "compound_meter": False,
                        "downpicking_preference": False,
                        "synth_emulation": False,
                    },
                ),
                costs=data.get(
                    "costs",
                    {
                        "pop_non_treble_penalty": 25.0,
                        "slap_non_bass_penalty": 18.0,
                        "fret_stretch_penalty": 10.0,
                        "position_shift_multiplier": 2.0,
                        "open_string_bonus": -1.5,
                    },
                ),
                micro_timing=data.get("micro_timing", {}),
                articulation_intent=data.get("articulation_intent", {}),
                fretboard_navigation=data.get("fretboard_navigation", {}),
                notation_engraving=data.get("notation_engraving", {}),
                harmonic_hysteresis=data.get("harmonic_hysteresis", {}),
                cross_stem_validation=data.get("cross_stem_validation", {}),
                level_profile=data.get("level_profile", {}),
                structural=data.get(
                    "structural",
                    {"phrase_length_measures": 8, "cross_section_healing": True},
                ),
                sub_genres=[
                    cls.from_dict(sg.get("name", "sub"), sg) if isinstance(sg, dict) else sg
                    for sg in data.get("sub_genres", [])
                ],
            )
        except ValidationError as err:
            raise TypeError(f"Invalid type data for Genre: {err}") from err

    def to_dict(self) -> dict:
        return self.model_dump()

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        d = self.to_dict()
        return d.get(key, default)

    def keys(self):
        return list(self.model_fields.keys())