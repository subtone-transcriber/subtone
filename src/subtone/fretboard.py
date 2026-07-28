import math

from subtone.schemas import Genre, Note, Song
from subtone.settings_loader import (
    DEFAULT_BPM,
    DEFAULT_TUNING_TYPE,
    FRETBOARD_TUNING_PROFILES,
    MAX_FRETBOARD_FRETS,
)
from subtone.pitch_theory import fold_pitch_to_bass_range


class ErgonomicFretboardHMMSolver:
    def __init__(self, tuning_type=DEFAULT_TUNING_TYPE, beam_width=8, genre_config=None, song: Song = None):
        if song is not None:
            tuning_type = song.tuning_type or tuning_type
            genre_config = song.genre_config or genre_config

        if genre_config is None:
            self.genre_config = Genre()
        elif isinstance(genre_config, dict):
            self.genre_config = Genre.from_dict("default", genre_config)
        else:
            self.genre_config = genre_config

        self.tuning_type = tuning_type if tuning_type in FRETBOARD_TUNING_PROFILES else DEFAULT_TUNING_TYPE
        self.beam_width = beam_width
        self.strings = FRETBOARD_TUNING_PROFILES[self.tuning_type]
        self.num_frets = MAX_FRETBOARD_FRETS

        # Read technique penalty costs dynamically from genre config
        costs = self.genre_config.costs
        # `or` treats an explicitly-configured 0.0 penalty as falsy and silently overrides
        # it with the default, ignoring a legitimate genre-config value of zero.
        self.pop_penalty = costs.get("pop_non_treble_penalty", 30.0)
        self.slap_penalty = costs.get("slap_non_bass_penalty", 20.0)
        self.fret_stretch_penalty = costs.get("fret_stretch_penalty", 20.0)
        self.shift_multiplier = costs.get("position_shift_multiplier", 3.0)
        self.open_bonus = costs.get("open_string_bonus", -2.0)

        # Genre-specific technique flags
        features = self.genre_config.features
        self.downpicking_pref = features.get("downpicking_preference", False)
        self.is_synth_emulation = (
            features.get("synth_emulation", False)
            or (self.genre_config.technique == "synth_emulation")
            or (self.genre_config.extends == "four_on_floor_dance")
        )
        if self.is_synth_emulation:
            self.shift_multiplier *= 0.5  # Encourage wide linear movement for synth basslines

    def get_valid_positions(self, midi_pitch: int):
        string_pitches = list(self.strings.values()) if isinstance(self.strings, dict) else self.strings
        min_p = min(string_pitches)
        max_p = max(string_pitches) + self.num_frets
        midi_pitch = fold_pitch_to_bass_range(midi_pitch, min_pitch=min_p, max_pitch=max_p)

        positions = []
        string_items = self.strings.items() if isinstance(self.strings, dict) else [(len(self.strings) - i, open_p) for i, open_p in enumerate(self.strings)]
        for s, open_p in string_items:
            fret = midi_pitch - open_p
            if 0 <= fret <= self.num_frets:
                if fret == 0:
                    positions.append((s, 0, 0))
                else:
                    fingers = [1, 2, 4] if fret <= 5 else [1, 2, 3, 4]
                    for f in fingers:
                        positions.append((s, fret, f))

        if not positions:
            # Fallback for extreme out-of-range pitches: clamp to closest string & fret
            if isinstance(self.strings, dict):
                string_keys = list(self.strings.keys())
                get_open = lambda s: self.strings[s]
            else:
                num_s = len(self.strings)
                string_keys = list(range(1, num_s + 1))
                get_open = lambda s: self.strings[num_s - s]

            closest_string = min(string_keys, key=lambda s: abs(get_open(s) - midi_pitch))
            open_p = get_open(closest_string)
            clamped_fret = max(0, min(self.num_frets, midi_pitch - open_p))
            positions = [(closest_string, clamped_fret, 1)]

        return positions

    def _get_local_anchor_fret(self, notes: list[Note], t: int, window=8) -> float:
        start = max(0, t - 2)
        end = min(len(notes), t + window + 1)

        string_pitches = list(self.strings.values()) if isinstance(self.strings, dict) else self.strings
        open_pitches = sorted(string_pitches)
        median_open = open_pitches[len(open_pitches) // 2]

        weighted_sum = 0.0
        total_weight = 0.0

        for idx in range(start, end):
            note = notes[idx]

            # Base expected fret position for this note's pitch (clamped to ergonomic box limits)
            expected_fret = max(0, min(12, note.pitch - median_open))

            # 1. Weight by duration (non-linear scaling to let longer notes dominate)
            dur = getattr(note, "duration", 0.25)
            dur_weight = math.pow(max(0.01, dur), 1.8) * 4.0

            # 2. Weight by accent and dynamic marks (stronger emphasis on accents)
            accent_mult = 1.0
            if getattr(note, "is_accent", False) or note.tag == "accent":
                accent_mult *= 3.0

            dynamic_mark = getattr(note, "dynamic_mark", "mf")
            if dynamic_mark in ["f", "ff"]:
                accent_mult *= 2.0
            elif dynamic_mark in ["p", "pp"]:
                accent_mult *= 0.5

            # 3. Look-ahead distance discount
            dist = abs(idx - t)
            if idx > t:
                dist_discount = math.pow(0.88, dist)
            else:
                dist_discount = math.pow(0.75, dist)

            weight = dur_weight * accent_mult * dist_discount

            weighted_sum += expected_fret * weight
            total_weight += weight

        if total_weight > 0:
            avg_fret = weighted_sum / total_weight
        else:
            avg_fret = 5.0  # safe default

        return float(max(1, min(self.num_frets, avg_fret)))

    def _solve_notes(self, notes: list[Note], bpm=DEFAULT_BPM):
        if not notes:
            return [], [], [], []

        notes = sorted(notes, key=lambda x: x.start)
        T = len(notes)
        sec_per_beat = 60.0 / bpm if bpm > 0 else 0.5

        first_str = list(self.strings.keys())[0] if isinstance(self.strings, dict) else 1
        sequence_states = [
            self.get_valid_positions(n.pitch)
            or self.get_valid_positions(n.pitch - 12)
            or [(first_str, 0, 0)]
            for n in notes
        ]

        V = [{} for _ in range(T)]
        backpointer = [{} for _ in range(T)]

        initial_anchor = self._get_local_anchor_fret(notes, 0)
        for state in sequence_states[0]:
            string_num, fret, finger = state
            tag = notes[0].tag
            note_dur = notes[0].duration

            open_cost = (self.open_bonus if note_dur > 0.3 else 1.5) if fret == 0 else 0.0
            high_fret_penalty = (0.5 * math.pow(fret - 12, 1.8)) if fret > 12 else 0.0
            box_cost = (fret * 0.08 if fret <= 7 else fret * 0.20) + open_cost + high_fret_penalty

            tech_cost = 0.0
            if tag == "pop":
                tech_cost = 0.0 if string_num in [1, 2] else self.pop_penalty
            elif tag == "slap":
                tech_cost = 0.0 if string_num >= 3 else self.slap_penalty

            anchor_dist = abs(fret - initial_anchor) if fret > 0 else 0.0
            anchor_cost = anchor_dist * 0.15

            V[0][state] = box_cost + tech_cost + anchor_cost
            backpointer[0][state] = None

        if len(V[0]) > self.beam_width:
            V[0] = dict(sorted(V[0].items(), key=lambda x: x[1])[: self.beam_width])

        for t in range(1, T):
            prev_onset, prev_offset = notes[t - 1].start, notes[t - 1].end
            curr_onset, _ = notes[t].start, notes[t].end

            onset_dt_sec = max(0.01, curr_onset - prev_onset)
            onset_dt_beats = max(0.125, onset_dt_sec / sec_per_beat)

            curr_dur = notes[t].duration
            overlap_dur = max(0.0, prev_offset - curr_onset)
            tag = notes[t].tag

            local_anchor = self._get_local_anchor_fret(notes, t)

            for c_state in sequence_states[t]:
                c_string, c_fret, c_finger = c_state
                best_cost, best_prev = float("inf"), None

                for p_state in V[t - 1]:
                    p_string, p_fret, p_finger = p_state

                    is_transition_legato_or_slide = (
                        getattr(notes[t], "is_legato", False)
                        or getattr(notes[t], "is_slide", False)
                        or tag in ["slide", "hammer_on", "pull_off", "legato"]
                    )

                    string_diff = c_string - p_string
                    fret_span = abs(c_fret - p_fret) if (p_fret > 0 and c_fret > 0) else 0

                    if is_transition_legato_or_slide:
                        if string_diff != 0:
                            string_shift = 10000.0
                            overlap_penalty = 5000.0
                            inertia_penalty = 5000.0
                            stretch_penalty = 5000.0
                            transition_step_cost = 5000.0
                        else:
                            string_shift = -10.0
                            overlap_penalty = 0.0
                            inertia_penalty = 0.0
                            stretch_penalty = 0.0
                            transition_step_cost = 0.0
                    else:
                        if overlap_dur > 0.08 and c_string == p_string and c_fret != p_fret:
                            overlap_penalty = 150.0
                        else:
                            overlap_penalty = overlap_dur * 20.0

                        if fret_span > 4:
                            inertia_penalty = 80.0 + (25.0 * (fret_span - 4))
                        else:
                            inertia_penalty = fret_span * 1.2

                        if onset_dt_beats < 0.5 and fret_span >= 4:
                            inertia_penalty += 120.0

                        if p_fret == 0 or c_fret == 0:
                            fret_dist = 0.2
                            stretch_penalty = 0.0
                        else:
                            d_prev = 1.0 - math.pow(2, -p_fret / 12.0)
                            d_curr = 1.0 - math.pow(2, -c_fret / 12.0)
                            fret_dist = abs(d_curr - d_prev) * 25.0

                            if min(p_fret, c_fret) <= 5 and fret_span > 3:
                                stretch_penalty = self.fret_stretch_penalty * 1.75
                            elif fret_span > 4:
                                stretch_penalty = self.fret_stretch_penalty
                            else:
                                stretch_penalty = 0.0

                        p_anchor = p_fret - (p_finger - 1) if p_finger > 0 else p_fret
                        c_anchor = c_fret - (c_finger - 1) if c_finger > 0 else c_fret
                        anchor_shift = abs(c_anchor - p_anchor)

                        if anchor_shift == 0:
                            finger_diff = c_finger - p_finger
                            fret_diff = c_fret - p_fret
                            strain = (
                                8.0
                                if (fret_diff > 0 and finger_diff < 0) or (fret_diff < 0 and finger_diff > 0)
                                else 0.0
                            )
                            transition_step_cost = (fret_dist * 0.3) + strain
                        else:
                            transition_step_cost = (anchor_shift * self.shift_multiplier) / (onset_dt_beats + 0.1)

                        if string_diff == 0:
                            string_shift = -2.0
                        elif string_diff > 0:
                            string_shift = math.pow(string_diff, 1.3) * 1.8 + (80.0 if fret_span >= 4 else 0.0) + 15.0
                        else:
                            string_shift = math.pow(abs(string_diff), 1.4) * 2.5 + 15.0

                    open_cost = (
                        (self.open_bonus if (onset_dt_beats > 0.5 or curr_dur > 0.4) else 2.0) if c_fret == 0 else 0.0
                    )
                    high_fret_penalty = (0.5 * math.pow(c_fret - 12, 1.8)) if c_fret > 12 else 0.0

                    tech_cost = 0.0
                    if tag == "pop":
                        tech_cost = 0.0 if c_string in [1, 2] else self.pop_penalty
                    elif tag == "slap":
                        tech_cost = 0.0 if c_string >= 3 else self.slap_penalty

                    anchor_dist = abs(c_fret - local_anchor) if c_fret > 0 else 0.0
                    anchor_cost = anchor_dist * 0.15

                    local_cost = (
                        transition_step_cost
                        + stretch_penalty
                        + inertia_penalty
                        + string_shift
                        + open_cost
                        + high_fret_penalty
                        + tech_cost
                        + anchor_cost
                        + overlap_penalty
                    )
                    # Apply non-linear scaling penalty only to positive costs to preserve negative hysteresis bonuses
                    total_score = V[t - 1][p_state] + local_cost + (0.1 * math.pow(max(0.0, local_cost), 2))

                    if total_score < best_cost:
                        best_cost, best_prev = total_score, p_state

                if best_prev is not None:
                    V[t][c_state] = best_cost
                    backpointer[t][c_state] = best_prev

            if not V[t]:
                fallback_prev = min(V[t - 1], key=V[t - 1].get) if V[t - 1] else sequence_states[t - 1][0]
                for c_state in sequence_states[t]:
                    V[t][c_state] = V[t - 1].get(fallback_prev, 0.0) + 100.0
                    backpointer[t][c_state] = fallback_prev

            if len(V[t]) > self.beam_width:
                V[t] = dict(sorted(V[t].items(), key=lambda x: x[1])[: self.beam_width])

        optimal_states_full = [None] * T
        best_last_state = min(V[-1], key=V[-1].get) if V[-1] else sequence_states[-1][0]
        optimal_states_full[-1] = best_last_state

        for t in range(T - 1, 0, -1):
            curr_state = optimal_states_full[t]
            optimal_states_full[t - 1] = backpointer[t].get(curr_state, sequence_states[t - 1][0])

        optimal_positions = optimal_states_full

        rakes = [False] * T
        legatos = [False] * T
        slides = [False] * T
        for i in range(T):
            local_anc = self._get_local_anchor_fret(notes, i)
            c_pos = optimal_positions[i]
            notes[i].fret_position = c_pos
            notes[i].anchor_fret = int(round(local_anc))
            notes[i].anchor_pattern = f"Box-Fret-{int(round(local_anc))}" if local_anc > 0 else "Open-Box"
            notes[i].is_anchor = c_pos[1] == 0 or abs(c_pos[1] - local_anc) <= 2

            # Store string and fret assignments as metadata on the Note object without mutating note.pitch
            c_string, c_fret = c_pos[0], c_pos[1]
            notes[i].string = c_string
            notes[i].fret = c_fret

            if self.downpicking_pref and notes[i].tag not in ["slap", "pop"]:
                prev_start = notes[i - 1].start if i > 0 else 0.0
                onset_dt = notes[i].start - prev_start if i > 0 else 1.0
                if onset_dt < 0.18:
                    notes[i].is_downpick = (i % 2 == 0) or getattr(notes[i], "is_accent", False)
                else:
                    notes[i].is_downpick = True

            notes[i].determine_category()

            if i > 0:
                onset_dt = notes[i].start - notes[i - 1].start
                p_string, p_fret = optimal_states_full[i - 1][0], optimal_states_full[i - 1][1]
                c_string, c_fret = optimal_states_full[i][0], optimal_states_full[i][1]

                if (c_string - p_string) == 1 and onset_dt < 0.12:
                    rakes[i] = True

                if c_string == p_string and p_fret > 0 and c_fret > 0 and p_fret != c_fret:
                    fret_diff = abs(c_fret - p_fret)
                    if fret_diff in [1, 2, 3] and onset_dt < 0.08:
                        legatos[i] = True
                    elif fret_diff >= 3 and onset_dt < 0.18:
                        slides[i] = True

        return optimal_positions, rakes, legatos, slides

    def solve_song(self, song: Song) -> Song:
        """Solve fretboard state from, and write it back to, processed Notes."""
        song.notes.sort(key=lambda n: n.start)
        positions, rakes, legatos, slides = self._solve_notes(song.notes, bpm=song.bpm)
        song.fretboard_path = positions
        song.rakes = rakes
        song.legatos = legatos
        song.slides = slides

        for note, is_rake, is_legato, is_slide in zip(song.notes, rakes, legatos, slides):
            note.is_rake = note.is_rake or is_rake
            note.is_legato = note.is_legato or is_legato
            note.is_slide = note.is_slide or is_slide
            note.determine_category()
        return song
