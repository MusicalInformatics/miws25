#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import warnings
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import scipy.spatial.distance as sp_dist
from hiddenmarkov import (
    ConstantTransitionModel,
    HiddenMarkovModel,
    ObservationModel,
    TransitionModel,
)
from numpy.typing import NDArray
from scipy.signal import convolve
from scipy.stats import gumbel_l, norm

import partitura as pt
from partitura.utils.synth import SAMPLE_RATE

try:
    import librosa
    import soundfile as sf

    HAS_LIBROSA = True
except ImportError:
    print("Librosa not found! Try installing it with `pip install librosa`")

    HAS_LIBROSA = False

warnings.filterwarnings("ignore")
# Alias for typing arrays
NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]


class TempoModel(object):
    """
    Base class for tempo models.

    Parameters
    ----------
    init_beat_period: float
        Initial tempo in seconds per beat
    init_score_onset: float
        Initial score onset time in beats.

    Attributes
    ----------
    beat_period : float
        The current tempo in beats per second
    prev_score_onset: float
        Latest covered score onset (in beats)
    prev_perf_onset : float
        Last performed onset time in seconds
    asynchrony : float
        Asynchrony of the estimated onset time and the actually performed onset time.
    has_tempo_expectations : bool
        Whether the model includes tempo expectations
    counter: int
        The number of times that the model has been updated. Useful for debugging
        purposes.
    """

    beat_period: float
    prev_score_onset: float
    prev_perf_onset: float
    est_onset: float
    asynchrony: float
    has_tempo_expectations: bool
    counter: int
    score_onsets: np.ndarray

    def __init__(
        self,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0,
        score_onsets: Optional[np.ndarray] = None,
    ) -> None:
        self.beat_period = init_beat_period
        self.prev_score_onset = init_score_onset
        self.prev_perf_onset = None
        self.est_onset = None
        self.asynchrony = 0.0
        self.has_tempo_expectations = False
        self.score_onsets = score_onsets
        # Count how many times has the tempo model been
        # called
        self.counter = 0

    def __call__(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Update beat period and compute estimated onset time

        Parameters
        ----------
        performed_onset : float
            Latest performed onset
        score_onset: float
            Latest score onset corresponding to the performed onset

        Returns
        -------
        beat_period : float
            Tempo in beats per second
        est_onsete : float
            Estimated onset given the current beat period
        """
        self.update_beat_period(performed_onset, score_onset, *args, **kwargs)
        self.counter += 1
        return self.beat_period, self.est_onset

    def update_beat_period(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Internal method for updating the beat period.
        Needs to be implemented for each variant of the model
        """
        raise NotImplementedError


class BaseHMM(HiddenMarkovModel):
    """
    Base class for Hidden Markov Model alignment methods.

    Parameters
    ----------
    observation_model: ObservationModel
        An observation (data) model for computing the observation probabilities.

    transition_model: TransitionModel
        A transition model for computing the transition probabilities.

    state_space: np.ndarray
        The hidden states (positions in reference time).

    tempo_model: Optional[TempoModel]
        A tempo model instance

    has_insertions: bool
        A boolean indicating whether the state space consider inserted notes.
    """

    observation_model: ObservationModel
    transition_model: TransitionModel
    state_space: Union[NDArrayFloat, NDArrayInt]
    tempo_model: Optional[TempoModel]
    has_insertions: bool
    _warping_path: List[Tuple[int, int]]

    def __init__(
        self,
        observation_model: ObservationModel,
        transition_model: TransitionModel,
        state_space: Optional[Union[NDArrayFloat, NDArrayInt]] = None,
        tempo_model: Optional[TempoModel] = None,
        has_insertions: bool = False,
        **kwargs,
    ) -> None:
        HiddenMarkovModel.__init__(
            self,
            observation_model=observation_model,
            transition_model=transition_model,
            state_space=state_space,
        )
        self.tempo_model = tempo_model
        self.has_insertions = has_insertions
        self.input_index = 0
        self._warping_path = []
        self.current_state = 0

    @property
    def warping_path(self) -> NDArrayInt:
        return (np.array(self._warping_path).T).astype(np.int32)

    def __call__(self, input: NDArrayFloat) -> float:
        current_state = self.forward_algorithm_step(
            observation=input,
            log_probabilities=False,
        )

        self._warping_path.append((current_state, self.input_index))
        self.input_index += 1
        self.current_state = current_state

        return current_state


## TEMPO MODELS
class ReactiveTempoModel(TempoModel):
    """
    Reactive tempo model.

    This sync model computes the tempo as the direct (raw) value of the performed
    ioi divided by the notated ioi. This method is mostly intended for as a baseline
    and is generally a poor choice of a tempo model.

    Parameters
    ----------
    init_beat_period: float
        Initial tempo in seconds per beat
    init_score_onset: float
        Initial score onset time in beats.
    """

    def __init__(
        self,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0.0,
        update_on_valid_onsets_only: bool = False,
    ) -> None:
        super().__init__(
            init_beat_period=init_beat_period,
            init_score_onset=init_score_onset,
        )
        self.update_on_valid_onsets_only = update_on_valid_onsets_only

    def update_beat_period(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> None:
        """
        See documentation in SyncModel above.
        """

        self.est_onset = performed_onset
        if self.prev_perf_onset:
            s_ioi = score_onset - self.prev_score_onset
            p_ioi = performed_onset - self.prev_perf_onset

            if s_ioi > 0 and p_ioi > 0:
                self.beat_period = p_ioi / s_ioi
            else:
                self.beat_period *= 0.5

        self.prev_score_onset = score_onset
        self.prev_perf_onset = performed_onset


class MovingAverageTempoModel(TempoModel):
    """
    Moving average tempo model

    This sync model computes the tempo as moving average value of the raw tempo
    (performed ioi divided by the notated ioi). This method is mostly intended as a
    baseline and is generally a poor choice of a tempo model.

    Parameters
    ----------
    init_beat_period: float
        Initial tempo in seconds per beat
    init_score_onset: float
        Initial score onset time in beats.
    alpha: float
        Smoothing factor (must be between 0 and 1).
        A value closer to 1 changes the MA value very slowly, while a
        value closer to 0 "forgets" the previous estimate and
        takes always the most recent value.
    predict_onset: bool
        If True, computes the expected next performed onset time using the current
        tempo estimation. Otherwise, it takes the observed performed onset.
        This option is for testing purposes.
    """

    alpha: float
    predict_onset: bool

    def __init__(
        self,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0,
        alpha: float = 0.5,
        predict_onset: bool = True,
    ):
        super().__init__(
            init_beat_period=init_beat_period,
            init_score_onset=init_score_onset,
        )
        self.alpha = alpha
        self.predict_onset = predict_onset

    def update_beat_period(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> None:
        """
        See documentation in TempoModel above.
        """
        if self.prev_perf_onset:
            s_ioi = score_onset - self.prev_score_onset
            p_ioi = performed_onset - self.prev_perf_onset

            if s_ioi > 0:
                beat_period = p_ioi / s_ioi
            else:
                beat_period = self.beat_period

            if self.predict_onset:
                self.est_onset = self.est_onset + self.beat_period * s_ioi
            else:
                self.est_onset = performed_onset
            self.beat_period = (
                self.alpha * self.beat_period + (1 - self.alpha) * beat_period
            )
        else:
            self.est_onset = performed_onset

        self.prev_score_onset = score_onset
        self.prev_perf_onset = performed_onset


class KalmanTempoModel(TempoModel):
    """
    A Tempo model using a linear Kalman filter.
    """

    trans_par: float
    trans_var: float
    obs_var: float
    var_est: float

    def __init__(
        self,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0,
        trans_par: float = 1,
        trans_var: float = 0.03,
        obs_var: float = 0.0213,  # values from old ACCompanion
        init_var: float = 1,
    ) -> None:
        super().__init__(
            init_beat_period=init_beat_period,
            init_score_onset=init_score_onset,
        )
        # Assign the parameters:
        self.trans_par = trans_par
        self.trans_var = trans_var
        self.obs_var = obs_var

        # Assing the initial values:
        self.var_est = init_var

    # A function to compute one step of the predict-update cycle:
    def update_beat_period(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Updates the model, when a new IOI observation is made. Computes
        one step of the predict-update cycle in the Kalman Filter.

        Parameters
        ----------
        performed_onset: float
            The latest performed onset

        score_onset : float
           Latest score onset corresponding to the performed onset
        """

        if self.counter == 0:
            score_ioi = 0
            performed_ioi = 0
            self.est_onset = performed_onset
            self.prev_perf_onset = performed_onset
            self.prev_score_onset = score_onset
            self.counter += 1
            return  # skip update on first observation
        else:
            performed_ioi = abs(performed_onset - self.prev_perf_onset)

            score_ioi = abs(score_onset - self.prev_score_onset)

        self.prev_score_onset = score_onset
        self.prev_perf_onset = performed_onset
        # First, compute the prediction step:
        period_pred = self.beat_period * self.trans_par
        var_pred = (self.trans_par**2) * self.var_est + self.trans_var

        # Compute the error (innovation), between the estimation and obs:
        err = performed_ioi - score_ioi * period_pred
        # Compute the Kalman gain with the new predictions:
        kalman_gain = float(var_pred * score_ioi) / (
            (score_ioi**2) * var_pred + self.obs_var
        )
        # Compute the estimations after the update step:
        self.beat_period = period_pred + kalman_gain * err
        self.var_est = (1 - kalman_gain * score_ioi) * var_pred
        self.est_onset += score_ioi * self.beat_period
        # print("tempo", self.beat_period)
        self.counter += 1


class LinearTempoModel(TempoModel):
    """
    Linear synchronization model.

    The sensorimotor synch model to use if there are no tempo expectations.

    Parameters
    ----------
    init_beat_period : float
        Initial beat period in seconds
    init_score_onset : float
        Initial score onset in beats (can be negative)
    eta_t : float
        Learning rate for the tempo. This parameter serves a similar function
        to the alpha parameter in the MASM.
    eta_o : float
        Learning rate for the onset. This parameter serves a similar function to
        the alpha parameter in the MASM.
    """

    eta_t: float
    eta_p: float

    def __init__(
        self,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0,
        eta_t: float = 0.3,
        eta_p: float = 0.7,
        min_beat_period: float = 0.25,
        max_beat_period: float = 3,
    ) -> None:
        super().__init__(
            init_beat_period=init_beat_period,
            init_score_onset=init_score_onset,
        )
        self.eta_t = eta_t
        self.eta_p = eta_p
        self.min_beat_period = min_beat_period
        self.max_beat_period = max_beat_period

    def update_beat_period(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> None:
        if self.prev_perf_onset:
            s_ioi = abs(score_onset - self.prev_score_onset)
            self.est_onset = (
                self.est_onset + self.beat_period * s_ioi - self.eta_p * self.asynchrony
            )
            self.asynchrony = self.est_onset - performed_onset

        else:
            s_ioi = 0
            self.est_onset = performed_onset

        tempo_correction_term = (
            self.asynchrony if self.asynchrony != 0 and s_ioi != 0 else 0
        )

        self.prev_perf_onset = performed_onset
        self.prev_score_onset = score_onset

        if tempo_correction_term < 0:
            beat_period = self.beat_period - self.eta_t * tempo_correction_term
        else:
            beat_period = self.beat_period - 2 * self.eta_t * tempo_correction_term

        if beat_period >= self.min_beat_period and beat_period <= self.max_beat_period:
            self.beat_period = beat_period


## Utils
def interleave_with_constant(
    array: np.array,
    constant_row: float = 0,
) -> np.ndarray:
    """
    Interleave a matrix with rows of a constant value.

    Parameters
    -----------
    array : np.ndarray
    """
    # Determine the shape of the input array
    num_rows, num_cols = array.shape

    # Create an output array with interleaved rows (double the number of rows)
    interleaved_array = np.zeros((num_rows * 2, num_cols), dtype=array.dtype)

    # Set the odd rows to the original array and even rows to the constant_row
    interleaved_array[0::2] = array
    interleaved_array[1::2] = constant_row

    return interleaved_array


def compute_ioi_matrix(
    unique_onsets: np.ndarray,
    inserted_states: bool = False,
) -> np.ndarray:
    # Construct unique onsets with skips:
    if inserted_states:
        unique_onsets_s = np.insert(
            unique_onsets,
            np.arange(1, len(unique_onsets)),
            (unique_onsets[:-1] + 0.5 * np.diff(unique_onsets)),
        )
        ioi_matrix = sp_dist.squareform(sp_dist.pdist(unique_onsets_s.reshape(-1, 1)))

    # ... or without skips:
    else:
        unique_onsets_s = unique_onsets
        ioi_matrix = sp_dist.squareform(sp_dist.pdist(unique_onsets.reshape(-1, 1)))

    return ioi_matrix


def compute_chord_matrix(chord_pitches: List[NDArrayInt]) -> NDArrayFloat:
    num_rows = len(chord_pitches)
    matrix = np.zeros((num_rows, 128), dtype=float)

    # Flatten the chord pitches and create corresponding row indices
    row_indices = np.repeat(np.arange(num_rows), [len(p) for p in chord_pitches])
    col_indices = np.concatenate(chord_pitches)

    # Set the specified indices to 1
    matrix[row_indices, col_indices] = 1

    return matrix



def save_mixed_audio(
    audio: Union[np.ndarray, str, os.PathLike],
    annots: np.ndarray,
    save_path: Union[str, os.PathLike],
    sr: int = SAMPLE_RATE,
):
    
    if HAS_LIBROSA:
        if not isinstance(audio, np.ndarray):
            audio, _ = librosa.load(audio, sr=sr)

        annots_audio = librosa.clicks(
            times=annots,
            sr=sr,
            click_freq=1000,
            length=len(audio),
        )
        audio_mixed = audio + annots_audio
        sf.write(str(save_path), audio_mixed, sr, subtype="PCM_24")


def transfer_positions(wp, ref_anns, frame_rate, reverse=False):
    """
    Transfer the positions of the reference annotations to the target annotations using the warping path.
    Parameters
    ----------
    wp : np.array with shape (2, T)
        array of warping path.
        warping_path[0] is the index of the reference (score) feature and warping_path[1] is the index of the target(input) feature.
    ref_ann : List[float]
        reference annotations in seconds.
    frame_rate : int
        frame rate of the audio.

    Returns
    -------
    predicted_targets : np.array with shape (T,)
        predicted target positions in seconds.
    """
    # Causal nearest neighbor interpolation
    if reverse:
        x, y = wp[1], wp[0]
    else:
        x, y = wp[0], wp[1]
    ref_anns_frame = np.round(ref_anns * frame_rate)
    predicted_targets = np.ones(len(ref_anns)) * np.nan

    for i, r in enumerate(ref_anns_frame):
        # 1) Scan all x values less than or equal to r and find the largest x value
        past_indices = np.where(x <= r)[0]
        if past_indices.size > 0:
            # Find indices corresponding to the largest x value
            max_x_val = x[past_indices[-1]]
            max_x_indices = np.where(x == max_x_val)[0]

            # 2) Among all y values mapped to this x value, select the minimum y value
            corresponding_y_values = y[max_x_indices]
            min_y_val = np.min(corresponding_y_values)

            # predicted_targets.append(min_y_val)
            predicted_targets[i] = min_y_val

    return np.array(predicted_targets) / frame_rate