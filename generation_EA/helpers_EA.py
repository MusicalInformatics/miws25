import partitura as pt
import numpy as np
import string
import random


def randomword(length):
    """
    a random character generator
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))

def sanitize_chord_progression(prog) -> None:

    prev_soprano = None
    prev_alto = None
    prev_tenor = None
    prev_bass = None

    for chord in prog.chords:

        if len(chord.soprano) == 0:
            chord.soprano = prev_soprano
            chord.alto = prev_alto
            chord.tenor = prev_tenor
            chord.bass = prev_bass
        else:
            prev_soprano = chord.soprano
            prev_alto = chord.alto
            prev_tenor = chord.tenor
            prev_bass = chord.bass


def partFromFourPartProgression(prog, part=None, quarter_duration=4, time_offset=0):

    sanitize_chord_progression(prog)
    if part is None:
        part = pt.score.Part(
            "P0", "part from progression", quarter_duration=quarter_duration
        )
        part.add(pt.score.TimeSignature(4, 4), start=0)
        part.add(pt.score.Clef(1, "G", line=3, octave_change=0), start=0)
        part.add(pt.score.Clef(2, "F", line=4, octave_change=0), start=0)

    part_id = "".join(np.random.randint(0, 10, 4).astype(str))
    rhythm = [
        (i * quarter_duration + time_offset, (i + 1) * quarter_duration + time_offset)
        for i in range(len(prog.chords))
    ]
    for i, chord in enumerate(prog.chords):

        if len(chord.soprano) > 0:
            addnote(
                chord.soprano,
                part,
                1,
                rhythm[i][0],
                rhythm[i][1],
                part_id + "_s" + str(i),
                staff=1,
            )
            addnote(
                chord.alto,
                part,
                2,
                rhythm[i][0],
                rhythm[i][1],
                part_id + "_a" + str(i),
                staff=1,
            )
            addnote(
                chord.tenor,
                part,
                3,
                rhythm[i][0],
                rhythm[i][1],
                part_id + "_t" + str(i),
                staff=2,
            )
            addnote(
                chord.bass,
                part,
                4,
                rhythm[i][0],
                rhythm[i][1],
                part_id + "_b" + str(i),
                staff=2,
            )
    return part


def addnote(
    midipitch: int,
    part: pt.score.Part,
    voice: int,
    start: int,
    end: int,
    idx: int,
    staff: int = None,
) -> None:
    """
    adds a single note by midipitch to a part
    """
    step, alter, octave = pt.utils.music.midi_pitch_to_pitch_spelling(int(midipitch))
    part.add(
        pt.score.Note(
            id="n{}".format(idx),
            step=step,
            octave=int(octave),
            alter=alter,
            voice=voice,
            staff=staff,
        ),
        start=start,
        end=end,
    )


def addrest(
    part: pt.score.Part,
    voice: int,
    start: int,
    end: int,
    idx: int,
    staff: int = None,
) -> None:
    part.add(
        pt.score.Rest(
            id="n{}".format(idx),
            voice=voice,
            staff=staff,
        ),
        start=start,
        end=end,
    )
