#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains methods to compute features from MIDI signals.
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import partitura as pt
from numpy.typing import NDArray
from partitura.performance import PerformanceLike, PerformedPart, Performance
from partitura.score import Part, Score, ScoreLike, merge_parts
from partitura.utils.music import performance_from_part
import mido

# from matchmaker.utils.symbolic import (
#     framed_midi_messages_from_performance,
#     midi_messages_from_performance,
# )

from mido import Message

# Alias for typing arrays of a specific numerical dtype
NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]


# Type hint for Input MIDI frame. A frame is a tuple
# consisting of a list with the MIDI messages corresponding
# to the frame (List[Tuple[Message, float]]) and the
# time associated to the frame
InputMIDIFrame = Tuple[List[Tuple[Message, float]], float]

class Processor(object):
    """
    Abstract class for a processor.
    """

    def __call__(
        self,
        data: Any,
        **kwargs,
    ) -> Any:
        """
        Parameters
        ----------
        data : Any
            Input data to the processor
        **kwargs: Dict[str, Any]
            Optional keyword arguments

        Returns
        -------
        output: Any
            The output of the processor
        """

        raise NotImplementedError

    def reset(self):
        """
        Resets the processor, if it has any internal states.

        This method needs to be implemented in derived classes if needed.
        """
        pass

class Buffer(object):
    """
    A Buffer for MIDI input

    This class is a buffer to collect MIDI messages
    within a specified time window.

    Parameters
    ----------
    polling_period : float
        Polling period in seconds

    Attributes
    ----------
    polling_period : float
        Polling period in seconds.

    frame : list of tuples of (mido.Message and float)
        A list of tuples containing MIDI messages and
        the absolute time at which the messages arrived

    start : float
        The starting time of the buffer
    """

    polling_period: float
    frame: List[Tuple[mido.Message, float]]
    start: Optional[float]

    def __init__(self, polling_period: float) -> None:
        self.polling_period = polling_period
        self.frame = []
        self.start = None
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Logic to return the next item
        if self.index < len(self.frame):
            result = self.frame[self.index]
            self.index += 1
            return result
        else:
            # Raises StopIteration when the iteration is complete
            raise StopIteration

    def append(self, input: mido.Message, time: float) -> None:
        self.frame.append((input, time))

    # def set_start(self) -> None:
    #     if len(self.frame) > 0:
    #         self.start = np.min([time for _, time in self.frame])

    def reset(self, time: float) -> None:
        self.frame = []
        self.start = time

    @property
    def end(self) -> float:
        """
        Maximal end time of the frame
        """
        return self.start + self.polling_period

    @property
    def time(self) -> float:
        """
        Time of the middle of the frame
        """
        return self.start + 0.5 * self.polling_period

    def __len__(self) -> int:
        """
        Number of MIDI messages in the frame
        """
        return len(self.frame)

    def __str__(self) -> str:
        return str(self.frame)


def midi_messages_from_performance(
    perf: Union[PerformanceLike, str],
) -> Tuple[NDArray, NDArray]:
    """
    Get a list of MIDI messages and message times from
    a PerformedPart or a Performance object.

    The method ignores Meta messages, since they
    are not "streamed" live (see documentation for
    mido.Midifile.play)

    Parameters
    ----------
    perf : PerformanceLike
        A partitura PerformedPart or Performance object.

    Returns
    -------
    message_array : np.ndarray of mido.Message
        An array containing MIDI messages

    message_times : np.ndarray
        An array containing the times of the messages
        in seconds.
    """

    if isinstance(perf, str):
        # from a MIDI/Match file
        perf = pt.load_performance(perf)

    elif isinstance(perf, np.ndarray):
        # From a Note array
        perf = PerformedPart.from_note_array(perf)

    if isinstance(perf, Performance):
        pparts = perf.performedparts
    elif isinstance(perf, PerformedPart):
        pparts = [perf]

    messages = []
    message_times = []
    for ppart in pparts:
        # Get note on and note off info
        for note in ppart.notes:
            channel = note.get("channel", 0)
            note_on = mido.Message(
                type="note_on",
                note=note["pitch"],
                velocity=note["velocity"],
                channel=channel,
            )
            note_off = mido.Message(
                type="note_off",
                note=note["pitch"],
                velocity=0,
                channel=channel,
            )
            messages += [
                note_on,
                note_off,
            ]
            message_times += [
                note["note_on"],
                note["note_off"],
            ]

        # get control changes
        for control in ppart.controls:
            channel = control.get("channel", 0)
            msg = mido.Message(
                type="control_change",
                control=int(control["number"]),
                value=int(control["value"]),
                channel=channel,
            )
            messages.append(msg)
            message_times.append(control["time"])

        # Get program changes
        for program in ppart.programs:
            channel = program.get("channel", 0)
            msg = mido.Message(
                type="program_change",
                program=int(program["program"]),
                channel=channel,
            )
            messages.append(msg)
            message_times.append(program["time"])

    message_array = np.array(messages)
    message_times_array = np.array(message_times)

    sort_idx = np.argsort(message_times_array)
    # sort messages by time
    message_array = message_array[sort_idx]
    message_times_array = message_times_array[sort_idx]

    return message_array, message_times_array


def midi_messages_to_framed_midi(
    midi_msgs: NDArray,
    msg_times: NDArray,
    polling_period: float,
) -> Tuple[NDArray, NDArray]:
    """
    Convert a list of MIDI messages to a framed MIDI representation
    Parameters
    ----------
    midi_msgs: list of mido.Message
        List of MIDI messages.

    msg_times: list of float
        List of times (in seconds) at which the MIDI messages were received.

    polling_period:
        Polling period (in seconds) used to convert the MIDI messages.

    Returns
    -------
    frames_array: np.ndarray
        An array of MIDI frames.
    frame_times:
    """
    n_frames = int(np.ceil(msg_times.max() / polling_period))
    frame_times = (np.arange(n_frames) + 0.5) * polling_period
    start_times = np.arange(n_frames) * polling_period

    frames = []

    for cursor, s_time in enumerate(start_times):
        buffer = Buffer(polling_period)
        if cursor == 0:
            # do not leave messages starting at 0 behind!
            idxs = np.where(msg_times <= polling_period)[0]
        else:
            idxs = np.where(
                np.logical_and(
                    msg_times > cursor * polling_period,
                    msg_times <= (cursor + 1) * polling_period,
                )
            )[0]

        buffer.frame = list(
            zip(
                midi_msgs[idxs],
                msg_times[idxs],
            )
        )
        buffer.start = s_time
        frames.append(buffer)

    frames_array = np.array(
        frames,
        dtype=object,
    )

    return frames_array, frame_times


def framed_midi_messages_from_midi(
    filename: str,
    polling_period: float,
) -> Tuple[NDArray, NDArray]:
    """
    Get a list of framed MIDI messages and frame times from
    a MIDI file.

    This is a convenience method
    """

    midi_messages, message_times = midi_messages_from_midi(
        filename=filename,
    )

    frames_array, frame_times = midi_messages_to_framed_midi(
        midi_msgs=midi_messages,
        msg_times=message_times,
        polling_period=polling_period,
    )

    return frames_array, frame_times


def framed_midi_messages_from_performance(
    perf: PerformanceLike,
    polling_period: float,
) -> Tuple[NDArray, NDArray]:
    """
    Get a list of framed MIDI messages and frame times from
    a partitura Performance or PerformedPart object.

    This is a convenience method
    """
    midi_messages, message_times = midi_messages_from_performance(perf=perf)

    frames_array, frame_times = midi_messages_to_framed_midi(
        midi_msgs=midi_messages,
        msg_times=message_times,
        polling_period=polling_period,
    )

    return frames_array, frame_times



class PitchProcessor(Processor):
    """
    A class to process pitch information from MIDI input.

    Parameters
    ----------
    piano_range : bool
        If True, the pitch range will be limited to the piano range (21-108).

    return_pitch_list: bool
        If True, it will return an array of MIDI pitch values, instead of
        a "piano roll" slice.
    """

    prev_time: float
    piano_range: bool

    def __init__(
        self,
        piano_range: bool = False,
        return_pitch_list: bool = False,
    ) -> None:
        super().__init__()
        self.piano_range = piano_range
        self.return_pitch_list = return_pitch_list
        self.piano_shift = 21 if piano_range else 0

    def __call__(
        self,
        frame: InputMIDIFrame,
    ) -> Optional[Tuple[NDArrayFloat, float]]:
        if isinstance(frame, tuple):
            data, f_time = frame
        else:
            data = frame
        # pitch_obs = []
        pitch_obs = np.zeros(
            128,
            dtype=np.float32,
        )

        # TODO: Replace the for loop with list comprehension
        pitch_obs_list = []

        for msg, _ in data:
            if (
                getattr(msg, "type", "other") == "note_on"
                and getattr(msg, "velocity", 0) > 0
            ):
                pitch_obs[msg.note] = 1
                pitch_obs_list.append(msg.note - self.piano_shift)

        if pitch_obs.sum() > 0:
            if self.piano_range:
                pitch_obs = pitch_obs[21:109]

            if self.return_pitch_list:
                return np.array(
                    pitch_obs_list,
                    dtype=np.float32,
                )
            return pitch_obs
        else:
            return None

    def reset(self) -> None:
        pass


class PitchIOIProcessor(Processor):
    """
    A class to process pitch and IOI information from MIDI files

    Parameters
    ----------
    piano_range : bool
        If True, the pitch range will be limited to the piano range (21-108).

    return_pitch_list: bool
        If True, it will return an array of MIDI pitch values, instead of
        a "piano roll" slice.
    """

    prev_time: Optional[float]
    piano_range: bool

    def __init__(
        self,
        piano_range: bool = False,
        return_pitch_list: bool = False,
    ) -> None:
        super().__init__()
        self.prev_time = None
        self.piano_range = piano_range
        self.return_pitch_list = return_pitch_list
        self.piano_shift = 21 if piano_range else 0

    def __call__(
        self,
        frame: InputMIDIFrame,
    ) -> Optional[Tuple[NDArrayFloat, float]]:
        if isinstance(frame, tuple):
            data, f_time = frame
        else:
            data = frame
        # pitch_obs = []
        pitch_obs = np.zeros(
            128,
            dtype=np.float32,
        )

        # TODO: Replace the for loop with list comprehension
        pitch_obs_list = []
        for msg, _ in data:
            if (
                getattr(msg, "type", "other") == "note_on"
                and getattr(msg, "velocity", 0) > 0
            ):
                pitch_obs[msg.note] = 1
                pitch_obs_list.append(msg.note - self.piano_shift)

        if pitch_obs.sum() > 0:
            if self.prev_time is None:
                # There is no IOI for the first observed note
                ioi_obs = 0.0
            else:
                ioi_obs = f_time - self.prev_time
            self.prev_time = f_time
            if self.piano_range:
                pitch_obs = pitch_obs[21:109]

            if self.return_pitch_list:
                return (
                    np.array(
                        pitch_obs_list,
                        dtype=np.float32,
                    ),
                    ioi_obs,
                )
            return (pitch_obs, ioi_obs)
        else:
            return None

    def reset(self) -> None:
        pass


class PianoRollProcessor(Processor):
    """
    A class to convert a MIDI file time slice to a piano roll representation.

    Parameters
    ----------
    use_velocity : bool
        If True, the velocity of the note is used as the value in the piano
        roll. Otherwise, the value is 1.
    piano_range : bool
        If True, the piano roll will only contain the notes in the piano.
        Otherwise, the piano roll will contain all 128 MIDI notes.
    dtype : type
        The data type of the piano roll. Default is float.
    """

    def __init__(
        self,
        use_velocity: bool = False,
        piano_range: bool = False,
        dtype: type = np.float32,
    ):
        Processor.__init__(self)
        self.active_notes: Dict = dict()
        self.piano_roll_slices: List[np.ndarray] = []
        self.use_velocity: bool = use_velocity
        self.piano_range: bool = piano_range
        self.dtype: type = dtype

    def __call__(
        self,
        frame: InputMIDIFrame,
    ) -> np.ndarray:
        # initialize piano roll
        piano_roll_slice: np.ndarray = np.zeros(128, dtype=self.dtype)
        if isinstance(frame, tuple):
            data, f_time = frame
        else:
            data = frame
        for msg, m_time in data:
            if msg.type in ("note_on", "note_off"):
                if msg.type == "note_on" and msg.velocity > 0:
                    self.active_notes[msg.note] = (msg.velocity, m_time)
                else:
                    try:
                        del self.active_notes[msg.note]
                    except KeyError:
                        pass

        for note, (vel, m_time) in self.active_notes.items():
            if self.use_velocity:
                piano_roll_slice[note] = vel
            else:
                piano_roll_slice[note] = 1

        if self.piano_range:
            piano_roll_slice = piano_roll_slice[21:109]
        self.piano_roll_slices.append(piano_roll_slice)

        return piano_roll_slice

    def reset(self) -> None:
        self.piano_roll_slices = []
        self.active_notes = dict()


class PitchClassPianoRollProcessor(Processor):
    """
    A class to convert a MIDI file time slice to a piano roll representation.

    Parameters
    ----------
    use_velocity : bool
        If True, the velocity of the note is used as the value in the piano
        roll. Otherwise, the value is 1.
    piano_range : bool
        If True, the piano roll will only contain the notes in the piano.
        Otherwise, the piano roll will contain all 128 MIDI notes.
    dtype : type
        The data type of the piano roll. Default is float.
    """

    def __init__(
        self,
        use_velocity: bool = False,
        dtype: type = np.float32,
    ):
        Processor.__init__(self)
        self.active_notes: Dict = dict()
        self.pitch_class_slices: List[np.ndarray] = []
        self.use_velocity: bool = use_velocity
        self.dtype: type = dtype

    def __call__(
        self,
        frame: InputMIDIFrame,
    ) -> np.ndarray:
        # initialize pitch class
        pitch_class_slice: np.ndarray = np.zeros(12, dtype=self.dtype)
        if isinstance(frame, tuple):
            data, f_time = frame
        else:
            data = frame
        for msg, m_time in data:
            if msg.type in ("note_on", "note_off"):
                if msg.type == "note_on" and msg.velocity > 0:
                    self.active_notes[msg.note] = (msg.velocity, m_time)
                else:
                    try:
                        del self.active_notes[msg.note]
                    except KeyError:
                        pass

        for note, (vel, m_time) in self.active_notes.items():
            if self.use_velocity:
                pitch_class_slice[note % 12] = max(vel, pitch_class_slice[note % 12])
            else:
                pitch_class_slice[note % 12] = 1

        self.pitch_class_slices.append(pitch_class_slice)

        return pitch_class_slice

    def reset(self) -> None:
        self.pitch_class_slices = []
        self.active_notes = dict()


def compute_features_from_symbolic(
    ref_info: Union[ScoreLike, PerformanceLike, NDArray, str],
    processor_name: str,
    processor_kwargs: Optional[dict] = None,
    polling_period: Optional[float] = 0.01,
    bpm: Optional[float] = 120,
):
    processor_mapping = {
        "pitch": PitchProcessor,
        "pitch_ioi": PitchIOIProcessor,
        "pianoroll": PianoRollProcessor,
        "pitch_class_pianoroll": PitchClassPianoRollProcessor,
    }

    if processor_kwargs is None:
        processor_kwargs = {}

    feature_processor = processor_mapping[processor_name](**processor_kwargs)

    if isinstance(ref_info, Score):
        ref_info = performance_from_part(
            part=merge_parts(ref_info) if len(ref_info) > 1 else ref_info[0],
            bpm=bpm,
        )
    elif isinstance(ref_info, Part):
        ref_info = performance_from_part(
            part=ref_info,
            bpm=bpm,
        )
    elif isinstance(ref_info, str):
        # This method assumes that all paths are to
        # performance files.
        ref_info = pt.load_performance(ref_info)

    elif isinstance(ref_info, np.ndarray):
        ref_info = PerformedPart.from_note_array(ref_info)

    if polling_period is not None:
        frames_array, frame_times = framed_midi_messages_from_performance(
            perf=ref_info, polling_period=polling_period
        )
    else:
        frames_array, frame_times = midi_messages_from_performance(
            perf=ref_info,
        )
        # Get same format as expected by the input processors
        frames_array = np.array([list(zip(frames_array, frame_times))])

    outputs = []
    for frame, f_time in zip(frames_array, frame_times):
        output = feature_processor((frame, f_time))

        outputs.append(output)

    return outputs


if __name__ == "__main__":  # pragma: no cover
    pass
