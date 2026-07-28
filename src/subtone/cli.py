import argparse

from subtone.settings import AudioTranscriptionPipeline


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
        song = pipeline.run(
            target_input=target,
            generate_all_levels=args.all_levels,
            level=args.level,
            use_gpu=args.gpu,
            genre_override=args.genre,
        )
        # pipeline.run() always returns a fully-populated Song whose `measures`
        # collection always holds MeasureChunk objects once transcription
        # succeeds, so no defensive hasattr/getattr fallback is needed here.
        total_measures = max((m.measure_num for m in song.measures), default=0)
        print(f"Transcription Complete: {song.song_title} | Total Measures: {total_measures}")


if __name__ == "__main__":
    main()