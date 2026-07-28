"""Subtone Idiomatic Bass Transcription & MusicXML Engraver Engine."""

from subtone.schemas import AudioEvent, Genre, Level, MeasureChunk, Note, RhythmicAtom, Song
from subtone.dsp import (
    stage1_stem_separation_and_audio_to_midi,
    stage2_multistem_f0_tracking,
    stage3_drum_percussive_grid_mining,
    stage4_frame_to_symbolic_bounding,
    stage5_drum_pocket_and_groove_audit,
    stage6_melodic_counterpoint_register_audit,
    stage10_songwide_multistem_audit,
    process_audio_target_to_events,
    transcribe_song,
)
from subtone.musicality import (
    stage7_harmonic_context_validation,
    snap_song_to_scale,
    midi_to_frequency,
    frequency_to_midi,
)
from subtone.biomechanics import (
    stage8_genre_pattern_and_fretboard_hmm,
    ErgonomicFretboardHMMSolver,
)
from subtone.tabs import (
    stage9_pedagogical_abstraction_and_partitioning,
    stage11_rest_synthesis_and_reconciliation,
    stage12_musicxml_dom_serialization,
    build_and_export_song,
    stream_quantized_events,
)
from subtone.settings import AudioTranscriptionPipeline

__all__ = [
    "AudioEvent",
    "Genre",
    "Level",
    "MeasureChunk",
    "Note",
    "RhythmicAtom",
    "Song",
    "AudioTranscriptionPipeline",
    "ErgonomicFretboardHMMSolver",
    "build_and_export_song",
    "stream_quantized_events",
    "snap_song_to_scale",
    "transcribe_song",
    "process_audio_target_to_events",
    "stage1_stem_separation_and_audio_to_midi",
    "stage2_multistem_f0_tracking",
    "stage3_drum_percussive_grid_mining",
    "stage4_frame_to_symbolic_bounding",
    "stage5_drum_pocket_and_groove_audit",
    "stage6_melodic_counterpoint_register_audit",
    "stage7_harmonic_context_validation",
    "stage8_genre_pattern_and_fretboard_hmm",
    "stage9_pedagogical_abstraction_and_partitioning",
    "stage10_songwide_multistem_audit",
    "stage11_rest_synthesis_and_reconciliation",
    "stage12_musicxml_dom_serialization",
]
