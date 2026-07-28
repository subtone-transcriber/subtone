flowchart TD
    %% Global Styles
    classDef input fill:#2d3748,stroke:#cbd5e0,color:#fff,stroke-width:2px;
    classDef phase fill:#1a202c,stroke:#4a5568,color:#fff,stroke-width:2px;
    classDef stage fill:#2b6cb0,stroke:#63b3ed,color:#fff,stroke-width:1px;
    classDef data fill:#2f855a,stroke:#68d391,color:#fff,stroke-width:1px,stroke-dasharray: 5 5;

    RawInput["🎵 Raw Song Input (MP3 / Metadata)"] :::input
    RawInput --> Phase1

    subgraph Phase1 ["PHASE I: SIGNAL INGESTION & GENRE-DRIVEN DSP FEATURE EXTRACTION"]
        direction TB
        S1["<b>Stage 1: Multi-Stem Source Separation & Genre Policy F0 Tracking</b><br/>• Demucs HPSS (Bass, Drums, Vocals, Guitar, Piano, Other)<br/>• Genre Policy Injection (Tuning, Technique, DSP Bounds)<br/>• Dynamic F0 Tracking (Sub-bass, Drop Tuning, Slap Attacks)"] :::stage
        S2["<b>Stage 2: Genre-Aware Percussive Grid & Rhythmic Anchor Mining [DRUMS]</b><br/>• Transient Energy Mining (Kick/Snare/Hi-Hat Maps)<br/>• Dynamic Swing Ratio & Clave/Syncopation Grid Extraction"] :::stage
        S1 --> S2
    end

    Data1[/"⚡ Continuous F0 Trajectories + AudioEvents"/] :::data
    Phase1 --> Data1 --> Phase2

    subgraph Phase2 ["PHASE II: SYMBOLIC CONVERSION, RHYTHMIC & MELODIC STEM VALIDATION"]
        direction TB
        S3["<b>Stage 3: Frame-to-Symbolic Bounding & Quantization Grid Mapping</b>"] :::stage
        S4["<b>Stage 4: Genre-Conditioned Rhythmic Pocket & Groove Audit [DRUMS STEM]</b><br/>• Transient Attack Alignment & Pocket Determination<br/>• Technique Ghost Note Tagging (Slap Clicks / Palm Mutes)"] :::stage
        S5["<b>Stage 5: Melodic Counterpoint & Register Audit [VOCALS / GUITAR STEMS]</b><br/>• Spectral Masking Resolution & Pitch Cutoff Filtering"] :::stage
        S3 --> S4 --> S5
    end

    Data2[/"🎼 Rhythmically & Melodically Validated Notes"/] :::data
    Phase2 --> Data2 --> Phase3

    subgraph Phase3 ["PHASE III: HARMONIC VALIDATION, ERGONOMICS & MEASURE PARTITIONING"]
        direction TB
        S6["<b>Stage 6: Polyphonic Harmonic Context Validation [GUITAR/PIANO/OTHER]</b><br/>• Root vs. Inversion Resolution via Chroma/CQT Matrices<br/>• Directional Enharmonic Pitch Spelling (Key Signature Tonal)"] :::stage
        S7["<b>Stage 7: Genre Pattern Engine & Biomechanical Ergonomic Solver</b><br/>• Genre Pattern Matching (Tumbao, Walking, Gallop, Slap)<br/>• Fretboard HMM / Viterbi Path (Genre Cost Parameter Matrix)"] :::stage
        S8["<b>Stage 8: Pedagogical Abstraction (Levels 1-5) & Metric Partitioning</b><br/>• 5-Level Pedagogical Filter Matrix Application<br/>• Measure Capacity Partitioning & Beat Boundary Note Tying"] :::stage
        S6 --> S7 --> S8
    end

    Data3[/"🧩 MeasureChunks with Unvalidated Note Atoms"/] :::data
    Phase3 --> Data3 --> Phase4

    subgraph Phase4 ["PHASE IV: HOLISTIC MULTI-STEM VALIDATION & REST SYNTHESIS ENGINE"]
        direction TB
        S9["<b>Stage 9: Song-Wide Multi-Stem Audit, Outlier Pruning & Coherence Check</b><br/>• Cross-Scan Bass against ALL STEMS (Drums/Guitar/Keys/Vocals)<br/>• Section Healing & Melodic Strictest Bounds Enforcement"] :::stage
        S10["<b>Stage 10: First-Class Rest Synthesis & Measure Reconciliation</b><br/>• Instantiate Explicit Rest Objects (Duration, Position)<br/>• Strict Measure Capacity Lock: Sum(Notes) + Sum(Rests) = Bar"] :::stage
        S9 --> S10
    end

    Data4[/"🌳 Fully Audited Score Object Tree (Notes + Rests)"/] :::data
    Phase4 --> Data4 --> Phase5

    subgraph Phase5 ["PHASE V: PURE SCORE SERIALIZATION & MUSICXML ENGRAVING"]
        direction TB
        S11["<b>Stage 11: Pure 1:1 Score Object to MusicXML DOM Serialization</b>"] :::stage
    end