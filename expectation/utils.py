from abc import ABC, abstractmethod
import logging
import os

from typing import List, Tuple, Union, Optional

import numpy as np
import glob
import partitura as pt
import warnings

try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False

HOME_DIR = "."

if IN_COLAB:
    HOME_DIR = "/content/miws2024/expectation"

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


QUANTIZED_DURATIONS = np.array(
    [
        2,
        1,
        0.75,
        0.5,
        0.25,
    ]
)
QUANTIZED_DURATIONS.sort()


RNG = np.random.RandomState(1984)


def load_data(
    min_seq_length: int = 10,
    add_random_pitch_transform: Optional[int] = None,
) -> List[np.ndarray]:
    # load data
    files = glob.glob(os.path.join(HOME_DIR, "data", "*.mid"))
    files.sort()
    sequences = []
    for fn in files:
        seq = pt.load_performance_midi(fn)[0]
        if len(seq.notes) > min_seq_length:
            na = seq.note_array()
            sequences.append(na)

            if add_random_pitch_transform is not None:

                for tranposition in RNG.choice(
                    [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6],
                    add_random_pitch_transform,
                ):
                    transposed_na = na.copy()
                    transposed_na["pitch"] += tranposition
                    sequences.append(transposed_na)
    return sequences


def find_nearest(array: np.ndarray, value: float) -> np.ndarray:
    """
    From https://stackoverflow.com/a/26026189
    """
    idx = np.clip(np.searchsorted(array, value, side="left"), 0, len(array) - 1)
    idx = idx - (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))
    return idx


def get_indices_cartesian_product(
    elem: Union[List[Tuple], np.ndarray],
    cartesian_product: np.ndarray,
) -> np.ndarray:
    # Convert elem to a numpy array for vectorized comparison
    elem_array = np.array(elem)

    # Find indices where all elements match
    indices = np.where((cartesian_product[:, None] == elem_array).all(-1))[1]

    return indices.astype(np.int32)
