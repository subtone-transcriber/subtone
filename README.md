            +===================================================+
            |          RAW SONG INPUT (MP3 / Metadata)          |
            +===================================================+
                                     ↓
+---------------------------------------------------------------------------+
| PHASE I: EXTERNAL SCRIPT STEM SEPARATION & AUDIO-TO-MIDI                  |
|                                                                           |
|  +---------------------------------------------------------------------+  |
|  | Stage 1: External Script Stem Separation & Audio-to-MIDI            |  |
|  | • Multi-Stem Separation using Demucs                                |  |
|  | • Audio-to-MIDI Extraction with Basic-Pitch / Librosa               |  |
|  +---------------------------------------------------------------------+  |
+---------------------------------------------------------------------------+
                                     ↓
            +===================================================+
            |                 MIDI Audio Stems                  |
            +===================================================+
                                     ↓
+---------------------------------------------------------------------------+
| PHASE II: SIGNAL INGESTION & GENRE-DRIVEN DSP FEATURE EXTRACTION          |
|                                                                           |
|  +---------------------------------------------------------------------+  |
|  | Stage 2: Multi-Stem Source Separation & Genre Policy F0 Tracking    |  |
|  | • Demucs HPSS (Bass, Drums, Vocals, Guitar, Piano, Other)           |  |
|  | • Genre Policy Injection (Tuning, Technique, DSP Bounds)            |  |
|  | • Dynamic F0 Tracking (Sub-bass, Drop Tuning, Slap Attacks)         |  |
|  +---------------------------------------------------------------------+  |
|                                    ↓                                      |
|  +---------------------------------------------------------------------+  |
|  | Stage 3: Genre-Aware Percussive Grid & Rhythmic Anchor Mining       |  |
|  |           [DRUMS]                                                   |  |
|  | • Transient Energy Mining (Kick/Snare/Hi-Hat Maps)                  |  |
|  | • Dynamic Swing Ratio & Clave/Syncopation Grid Extraction           |  |
|  +---------------------------------------------------------------------+  |
+---------------------------------------------------------------------------+
                                     ↓
            +===================================================+
            |     Continuous F0 Trajectories + AudioEvents      |
            +===================================================+
                                     ↓
+---------------------------------------------------------------------------+
| PHASE III: SYMBOLIC CONVERSION, RHYTHMIC & MELODIC STEM VALIDATION        |
|                                                                           |
|  +---------------------------------------------------------------------+  |
|  | Stage 4: Frame-to-Symbolic Bounding & Quantization Grid Mapping     |  |
|  +---------------------------------------------------------------------+  |
|                                    ↓                                      |
|  +---------------------------------------------------------------------+  |
|  | Stage 5: Genre-Conditioned Rhythmic Pocket & Groove Audit           |  |
|  |           [DRUMS STEM]                                              |  |
|  | • Transient Attack Alignment & Pocket Determination                 |  |
|  | • Technique Ghost Note Tagging (Slap Clicks / Palm Mutes)           |  |
|  +---------------------------------------------------------------------+  |
|                                    ↓                                      |
|  +---------------------------------------------------------------------+  |
|  | Stage 6: Melodic Counterpoint & Register Audit                      |  |
|  |           [VOCALS / GUITAR STEMS]                                   |  |
|  | • Spectral Masking Resolution & Pitch Cutoff Filtering              |  |
|  +---------------------------------------------------------------------+  |
+---------------------------------------------------------------------------+
                                     ↓
            +===================================================+
            |    Rhythmically & Melodically Validated Notes     |
            +===================================================+
                                     ↓
+---------------------------------------------------------------------------+
| PHASE IV: HARMONIC VALIDATION, ERGONOMICS & MEASURE PARTITIONING          |
|                                                                           |
|  +---------------------------------------------------------------------+  |
|  | Stage 7: Polyphonic Harmonic Context Validation                     |  |
|  |           [GUITAR / PIANO / OTHER]                                  |  |
|  | • Root vs. Inversion Resolution via Chroma/CQT Matrices             |  |
|  | • Directional Enharmonic Pitch Spelling (Key Signature Tonal)       |  |
|  +---------------------------------------------------------------------+  |
|                                    ↓                                      |
|  +---------------------------------------------------------------------+  |
|  | Stage 8: Genre Pattern Engine & Biomechanical Ergonomic Solver      |  |
|  | • Genre Pattern Matching (Tumbao, Walking, Gallop, Slap)            |  |
|  | • Fretboard HMM / Viterbi Path (Genre Cost Parameter Matrix)        |  |
|  +---------------------------------------------------------------------+  |
|                                    ↓                                      |
|  +---------------------------------------------------------------------+  |
|  | Stage 9: Pedagogical Abstraction (Levels 1-5) & Metric Partitioning |  |
|  | • 5-Level Pedagogical Filter Matrix Application                     |  |
|  | • Measure Capacity Partitioning & Beat Boundary Note Tying          |  |
|  +---------------------------------------------------------------------+  |
+---------------------------------------------------------------------------+
                                     ↓
            +===================================================+
            |    MeasureChunks with Unvalidated Note Atoms      |
            +===================================================+
                                     ↓
+---------------------------------------------------------------------------+
| PHASE V: HOLISTIC MULTI-STEM VALIDATION & REST SYNTHESIS ENGINE           |
|                                                                           |
|  +---------------------------------------------------------------------+  |
|  | Stage 10: Song-Wide Multi-Stem Audit, Outlier Pruning & Coherence   |  |
|  | • Cross-Scan Bass against ALL STEMS (Drums/Guitar/Keys/Vocals)      |  |
|  | • Section Healing & Melodic Strictest Bounds Enforcement            |  |
|  +---------------------------------------------------------------------+  |
|                                    ↓                                      |
|  +---------------------------------------------------------------------+  |
|  | Stage 11: First-Class Rest Synthesis & Measure Reconciliation       |  |
|  | • Instantiate Explicit Rest Objects (Duration, Position)            |  |
|  | • Strict Measure Capacity Lock: Sum(Notes) + Sum(Rests) = Bar       |  |
|  +---------------------------------------------------------------------+  |
+---------------------------------------------------------------------------+
                                     ↓
            +===================================================+
            | Fully Audited Score Object Tree (Notes + Rests)   |
            +===================================================+
                                     ↓
+---------------------------------------------------------------------------+
| PHASE VI: PURE SCORE SERIALIZATION & MUSICXML ENGRAVING                   |
|                                                                           |
|  +---------------------------------------------------------------------+  |
|  | Stage 12: Pure 1:1 Score Object to MusicXML DOM Serialization       |  |
|  +---------------------------------------------------------------------+  |
+---------------------------------------------------------------------------+
