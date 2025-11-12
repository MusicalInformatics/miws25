#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluates a submission on the challenge server,
computing the F1 score of piece difficulty estimation
"""
import argparse
import re
import numpy as np
from sklearn.metrics import f1_score


def load_submission(fn: str) -> dict:
    """
    Load a submission
    """

    gt = np.loadtxt(
        fn,
        dtype=str,
        delimiter=",",
        comments="//",
        skiprows=1,
    )

    predictions = dict([(g[0], g[1]) for g in gt])

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    tonal_distance = []

    submission = load_submission(args.submission)

    target = load_submission(args.target)

    targets = []
    predictions = []
    for piece, difficulty in target.items():
        targets.append(difficulty)
        if piece in submission:
            predictions.append(submission[piece])
        else:
            # There is no -1 class
            # so this will always be an incorrect
            # classification.
            predictions.append(-1)

    mean_score = f1_score(
        y_true=np.array(targets),
        y_pred=np.array(predictions),
        average="macro",
    )
    print(mean_score)
