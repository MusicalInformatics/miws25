#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline submission for the piece difficulty estimation challenge 
for Musical Informatics WS25
"""
import warnings

import pandas as pd
import partitura as pt
import os
import numpy as np

from typing import Callable, Tuple, Union, List

from sklearn.model_selection import train_test_split

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="partitura.*",
)

warnings.filterwarnings("ignore", module="sklearn")


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.tree import DecisionTreeClassifier


def compute_note_density(score: Union[pt.score.ScoreLike, np.ndarray]) -> float:

    if isinstance(score, (pt.score.Score, pt.score.Part)):

        note_array = score.note_array()
    elif isinstance(score, np.ndarray):
        note_array = score

    piece_duration_beats = (
        note_array["onset_beat"] + note_array["duration_beat"]
    ).max() - note_array["onset_beat"].min()

    number_of_notes = len(note_array)

    note_density = number_of_notes / piece_duration_beats

    return note_density


def compute_piece_length(score: Union[pt.score.ScoreLike, np.ndarray]) -> float:

    if isinstance(score, (pt.score.Score, pt.score.Part)):

        note_array = score.note_array()
    elif isinstance(score, np.ndarray):
        note_array = score

    piece_duration_beats = (
        note_array["onset_beat"] + note_array["duration_beat"]
    ).max() - note_array["onset_beat"].min()

    return piece_duration_beats


def compute_vertical_neighbors(
    score: Union[pt.score.ScoreLike, np.ndarray]
) -> np.ndarray:
    """Vertical neighbor feature.

    Describes various aspects of simultaneously starting notes.

    Returns:
    * n_total :
    * n_above :
    * n_below :
    * highest_pitch :
    * lowest_pitch :
    * pitch_range :

    """

    if isinstance(score, (pt.score.Score, pt.score.Part)):

        na = score.note_array()
    elif isinstance(score, np.ndarray):
        na = score
    # the list of descriptors
    names = [
        "n_total",
        "n_above",
        "n_below",
        "highest_pitch",
        "lowest_pitch",
        "pitch_range",
    ]
    W = np.zeros((len(na), len(names)))
    for i, n in enumerate(na):
        neighbors = na[np.where(na["onset_beat"] == n["onset_beat"])]["pitch"]
        max_pitch = np.max(neighbors)
        min_pitch = np.min(neighbors)
        W[i, 0] = len(neighbors) - 1
        W[i, 1] = np.sum(neighbors > n["pitch"])
        W[i, 2] = np.sum(neighbors < n["pitch"])
        W[i, 3] = max_pitch
        W[i, 4] = min_pitch
        W[i, 5] = max_pitch - min_pitch

    vertical_neighbors = W.mean(0)
    return vertical_neighbors


def compute_score_features(
    score: Union[pt.score.Score, pt.score.Part, np.ndarray]
) -> np.ndarray:

    if isinstance(score, (pt.score.Score, pt.score.Part)):
        na = score.note_array()
    elif isinstance(score, np.ndarray):
        na = score

    ## TODO: Compute all features here
    # Decide which features might be more relevant

    note_density = compute_note_density(na)
    piece_duration = compute_piece_length(na)
    vertical_neighbors = compute_vertical_neighbors(na)

    features = np.r_[note_density, piece_duration, vertical_neighbors]

    return features


def load_piece_difficulty_dataset(datadir: str) -> np.ndarray:

    train_labels_fn = os.path.join(
        os.path.dirname(__file__),
        "difficulty_classification_training.csv",
    )

    train_data_ = pd.read_csv(train_labels_fn, delimiter=",")

    train_data = {}
    for bn, difficulty in zip(train_data_["file"], train_data_["difficulty"]):
        fn = os.path.join(os.path.abspath(datadir), bn)
        if not os.path.exists(fn):
            raise ValueError(f"{fn} not found!")
        train_data[bn] = (difficulty, fn)

    # Test files without labels
    test_labels_fn = "./difficulty_classification_test_gt.csv"

    if not os.path.exists(test_labels_fn):
        test_labels_fn = "./difficulty_classification_test_no_labels.csv"

        test_data_ = pd.read_csv(test_labels_fn, delimiter=",")
        test_data = {}
        for bn in test_data_["file"]:

            fn = os.path.join(os.path.abspath(datadir), bn)

            if not os.path.exists(fn):
                raise ValueError(f"{fn} not found!")
            test_data[bn] = (np.nan, fn)

    else:

        test_data_ = pd.read_csv(test_labels_fn, delimiter=",")

        test_data = {}
        for bn, difficulty in zip(test_data_["file"], test_data_["difficulty"]):

            fn = os.path.join(os.path.abspath(datadir), bn)

            if not os.path.exists(fn):
                raise ValueError(f"{fn} not found!")
            test_data[bn] = (difficulty, fn)

    return train_data, test_data


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Difficulty estimation")

    parser.add_argument(
        "--datadir",
        "-i",
        help="path to the input files",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--outfn",
        "-o",
        help="Output file with results",
        type=str,
        default="piece_difficulty_results.csv",
    )

    args = parser.parse_args()

    if args.datadir is None:
        raise ValueError("No data directory given")

    train_data, test_data = load_piece_difficulty_dataset(args.datadir)

    train_scores = []
    train_note_arrays = []
    train_features = []
    train_labels = []
    for i, (score_name, (dif, fn)) in enumerate(train_data.items()):

        print(f"Processing training score {i + 1}/{len(train_data)}: {score_name}")
        score = pt.load_musicxml(fn)
        score[0].use_musical_beat()

        # Adapt this part as needed
        note_array = pt.utils.music.ensure_notearray(
            score,
            include_pitch_spelling=True,  # adds 3 fields: step, alter, octave
            include_key_signature=True,  # adds 2 fields: ks_fifths, ks_mode
            include_time_signature=True,  # adds 2 fields: ts_beats, ts_beat_type
            include_metrical_position=True,  # adds 3 fields: is_downbeat, rel_onset_div, tot_measure_div
            include_grace_notes=True,  # adds 2 fields: is_grace, grace_type
        )
        features = compute_score_features(note_array)

        train_scores.append(train_scores)
        train_note_arrays.append(note_array)
        train_features.append(features)
        train_labels.append(dif)

    # if playing with hyper parameters of the classifier/learning algorithm,
    # you might want to split these data into training and validation
    # set `use_validation_set` to False, if you want to use the entire
    # training set.
    X_train = np.array(train_features)
    Y_train = np.array(train_labels)

    use_validation_set = True

    if use_validation_set:
        val_size = 0.2
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train,
            Y_train,
            test_size=val_size,
            stratify=Y_train,
            random_state=42,
        )

    print("Training classifier...")
    # We are using the classifier that performed
    # best in our tests in the piece_difficulty_estimation.ipynb
    # notebook
    classifier = DecisionTreeClassifier()

    classifier.fit(X_train, Y_train)

    if use_validation_set:
        Y_val_pred = classifier.predict(X_val)

        print("#" * 55)
        print("Piece Difficulty Estimation Results on Test Set\n")
        acc = accuracy_score(Y_val, Y_val_pred)
        f1 = f1_score(Y_val, Y_val_pred, average="macro")
        print(f"    Accuracy (test set): {acc:.2f}")
        print(f"    Macro F1-score (test set): {f1:.2f}")
        print("#" * 55)

    # We load the scores in the test set.
    test_scores = []
    test_note_arrays = []
    test_features = []
    test_labels = []
    score_names = []
    for i, (score_name, (dif, fn)) in enumerate(test_data.items()):

        print(f"Processing test score {i + 1}/{len(test_data)}: {score_name}")
        score = pt.load_musicxml(fn)
        score[0].use_musical_beat()

        # Adapt this part as needed
        note_array = pt.utils.music.ensure_notearray(
            score,
            include_pitch_spelling=True,
            include_key_signature=True,
            include_time_signature=True,
            include_metrical_position=True,
            include_grace_notes=True,
        )
        features = compute_score_features(note_array)

        test_scores.append(test_scores)
        test_note_arrays.append(note_array)
        test_features.append(features)
        test_labels.append(dif)
        score_names.append(score_name)

    X_test = np.array(test_features)
    Y_test = np.array(test_labels)
    score_names_test = np.array(score_names)

    Y_pred = classifier.predict(X_test)

    # Run evaluation on the test labels if available
    if not any(np.isnan(Y_test)):
        print("#" * 55)
        print("Piece Difficulty Estimation Results on Test Set\n")
        acc = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred, average="macro")
        print(f"    Accuracy (test set): {acc:.2f}")
        print(f"    Macro F1-score (test set): {f1:.2f}")
        print("#" * 55)

    # This part will only save results for the test set!
    with open(args.outfn, "w") as f:

        f.write("file,difficulty\n")

        for basename, pred_dif in zip(score_names_test, Y_pred):
            f.write(f"{basename},{pred_dif}\n")
